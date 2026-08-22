import womd.runtime_env
import argparse
from pathlib import Path

import numpy as np
import torch

from womd import contract, loss, pipeline
from womd.model import MotionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("anchors_path", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--batches", type=int, required=True)
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
    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))

    norms = []
    for batch_index, batch in enumerate(pipeline.batches(
        scenario_paths, 0, arguments.batch_size, 2, arguments.seed, True
    )):
        if batch_index >= arguments.batches:
            break
        (
            trajectories, heading_cosine_sine, position_log_standard_deviation,
            confidence_logits, predicted_speed, selected_unit_anchors,
            neighbour_future_positions,
        ) = predictor.predict_with_heading(batch)
        total, _, _, _, _ = loss.prediction_loss(
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
        predictor.zero_grad()
        total.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), float("inf")
        )))

    measured = np.array(norms)
    print(
        f"{arguments.staged_directory}, batch {arguments.batch_size}, {len(measured)} steps,"
        f" untrained model, seed {arguments.seed}"
    )
    for name, value in (
        ("p50", np.percentile(measured, 50)),
        ("p90", np.percentile(measured, 90)),
        ("p99", np.percentile(measured, 99)),
        ("max", measured.max()),
        ("max / p50", measured.max() / np.percentile(measured, 50)),
    ):
        print(f"  {name:12s}{value:12.3f}")


if __name__ == "__main__":
    main()
