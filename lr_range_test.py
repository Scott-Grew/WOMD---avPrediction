import womd.runtime_env
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from womd import contract, loss, pipeline
from womd.model import MotionPredictor


def swept_learning_rate(batch_index, batch_count, lowest_rate, highest_rate):
    sweep_fraction = batch_index / max(batch_count - 1, 1)
    return lowest_rate * (highest_rate / lowest_rate) ** sweep_fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("anchors_path", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--lowest-rate", type=float, required=True)
    parser.add_argument("--highest-rate", type=float, required=True)
    parser.add_argument("--report-groups", type=int, required=True)
    parser.add_argument("--heading-loss-weight", type=float, required=True)
    parser.add_argument("--classification-loss-weight", type=float, required=True)
    parser.add_argument("--neighbour-future-loss-weight", type=float, required=True)
    parser.add_argument("--speed-loss-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    with np.load(arguments.anchors_path) as anchors_file:
        contract.check_artifact_provenance(
            anchors_file["provenance"] if "provenance" in anchors_file else None,
            arguments.anchors_path, "Refit them with fit_anchors.py.",
        )
        unit_anchors = torch.from_numpy(anchors_file["unit_anchors"])
    predictor = MotionPredictor(unit_anchors)
    optimizer = torch.optim.AdamW(predictor.parameters())
    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))

    swept_rates = []
    regression_values = []
    total_values = []
    for batch_index, batch in enumerate(pipeline.batches(
        scenario_paths, 0, arguments.batch_size, 2, arguments.seed, True
    )):
        if batch_index >= arguments.batches:
            break
        learning_rate = swept_learning_rate(
            batch_index, arguments.batches, arguments.lowest_rate, arguments.highest_rate
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        (
            trajectories, heading_cosine_sine, position_log_standard_deviation,
            confidence_logits, predicted_speed, selected_unit_anchors,
            neighbour_future_positions,
        ) = predictor.predict_with_heading(batch)
        total, regression, _, _, _ = loss.prediction_loss(
            trajectories, heading_cosine_sine, position_log_standard_deviation,
            confidence_logits, predicted_speed,
            batch["future_positions"], batch["future_headings"], batch["future_mask"],
            selected_unit_anchors, arguments.heading_loss_weight,
            arguments.classification_loss_weight, arguments.speed_loss_weight,
        )
        total = total + arguments.neighbour_future_loss_weight * loss.neighbour_future_loss(
            neighbour_future_positions, batch["neighbour_future_positions"],
            batch["neighbour_future_mask"], batch["neighbour_history_mask"].any(dim=-1),
        )
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        swept_rates.append(learning_rate)
        regression_values.append(float(regression.detach()))
        total_values.append(float(total.detach()))

    print(
        f"{arguments.staged_directory}, batch {arguments.batch_size},"
        f" {len(swept_rates)} batches sweeping {arguments.lowest_rate:.1e}"
        f" to {arguments.highest_rate:.1e}, seed {arguments.seed}"
    )
    print(f"{'rate':>12s}{'regression':>14s}{'total':>12s}")
    group_size = max(len(swept_rates) // arguments.report_groups, 1)
    group_regression_means = []
    for group_start in range(0, len(swept_rates), group_size):
        group_rates = swept_rates[group_start:group_start + group_size]
        group_regression = regression_values[group_start:group_start + group_size]
        group_total = total_values[group_start:group_start + group_size]
        finite_regression = [value for value in group_regression if math.isfinite(value)]
        group_regression_means.append(
            (float(np.mean(group_rates)), np.mean(finite_regression) if finite_regression else float("nan"))
        )
        print(
            f"{np.mean(group_rates):12.2e}"
            f"{np.mean(finite_regression) if finite_regression else float('nan'):14.3f}"
            f"{np.mean([value for value in group_total if math.isfinite(value)]) if any(math.isfinite(value) for value in group_total) else float('nan'):12.3f}"
        )

    finite_groups = [pair for pair in group_regression_means if math.isfinite(pair[1])]
    best_rate, best_regression = min(finite_groups, key=lambda pair: pair[1])
    print(
        f"regression term bottoms at rate {best_rate:.2e} (value {best_regression:.3f});"
        f" non-finite batches {sum(1 for value in total_values if not math.isfinite(value))}"
    )


if __name__ == "__main__":
    main()
