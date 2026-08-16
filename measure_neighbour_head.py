# > Auxiliary neighbour-future head: what it costs and what weight puts it level with the position
# term. The weight is measured the way the heading weight was - both terms evaluated on an UNTRAINED
# model, weight = regression / neighbour_future, the ratio that makes the two contributions equal -
# and it is measured over several batches at more than one batch size because a single-batch ratio is
# a statistic of one draw. Cost is timed against the same step WITHOUT the head, so the number is the
# head's own forward, loss and backward rather than the whole step.

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from womd import contract, loader, loss, pipeline
from womd.model import QUERY_COUNT, MotionPredictor


def read_batches(scenario_paths, batch_size, batch_count):
    neighbour_counts = []
    batches = []
    samples = []
    for scenario_path in scenario_paths:
        scenario_array = loader.read_scenario(scenario_path)
        for track_index in loader.eligible_track_indices(
            scenario_array["track_rows"], scenario_array["track_valid"],
            scenario_array["is_designated_target"], True,
        ):
            sample = loader.build_sample(scenario_array, int(track_index))
            neighbour_counts.append(sample["neighbour_history"].shape[0])
            if len(batches) < batch_count:
                samples.append(sample)
                if len(samples) == batch_size:
                    batches.append(pipeline.collate_samples(samples))
                    samples = []
    return batches, np.array(neighbour_counts)


def equal_footing_weight(predictor, batch):
    with torch.no_grad():
        (
            trajectories, heading_cosine_sine, position_log_standard_deviation, confidence_logits,
            reachable_distance_metres, selected_unit_anchors, neighbour_future_positions,
        ) = predictor.predict_with_heading(batch)
        _, regression, _, heading = loss.prediction_loss(
            trajectories, heading_cosine_sine, position_log_standard_deviation, confidence_logits,
            batch["future_positions"], batch["future_headings"], batch["future_mask"],
            selected_unit_anchors, reachable_distance_metres, 1.0,
        )
        neighbour_future = loss.neighbour_future_loss(
            neighbour_future_positions,
            batch["neighbour_future_positions"],
            batch["neighbour_future_mask"],
            batch["neighbour_history_mask"].any(dim=-1),
        )
    return (
        float(regression), float(neighbour_future),
        float(regression / neighbour_future), float(regression / heading),
    )


def step_seconds(predictor, batch, with_neighbour_head, repeats):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        if with_neighbour_head:
            (
                trajectories, heading_cosine_sine, position_log_standard_deviation,
                confidence_logits, reachable_distance_metres, selected_unit_anchors,
                neighbour_future_positions,
            ) = predictor.predict_with_heading(batch)
        else:
            (
                _, trajectories, heading_cosine_sine, position_log_standard_deviation,
                confidence_logits, reachable_distance_metres, selected_unit_anchors,
            ) = predictor.encode_scene_and_modes(batch)
        total, _, _, _ = loss.prediction_loss(
            trajectories, heading_cosine_sine, position_log_standard_deviation, confidence_logits,
            batch["future_positions"], batch["future_headings"], batch["future_mask"],
            selected_unit_anchors, reachable_distance_metres, 1.0,
        )
        if with_neighbour_head:
            total = total + loss.neighbour_future_loss(
                neighbour_future_positions,
                batch["neighbour_future_positions"],
                batch["neighbour_future_mask"],
                batch["neighbour_history_mask"].any(dim=-1),
            )
        predictor.zero_grad()
        total.backward()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("--scenarios", type=int, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--timing-repeats", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    with np.load(arguments.anchors) as anchors_file:
        initial_unit_anchors = torch.from_numpy(anchors_file["unit_anchors"])
    assert initial_unit_anchors.shape == (contract.NUM_OBJECT_TYPES, QUERY_COUNT, 2)
    predictor = MotionPredictor(initial_unit_anchors)

    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))[: arguments.scenarios]
    assert scenario_paths, f"no .npz scenarios in {arguments.staged_directory}"

    print(f"{len(scenario_paths)} scenarios of {arguments.staged_directory}, untrained model, CPU")
    all_ratios = []
    for batch_size in arguments.batch_sizes:
        batches, neighbour_counts = read_batches(scenario_paths, batch_size, arguments.batches)
        ratios = [equal_footing_weight(predictor, batch) for batch in batches]
        all_ratios.extend(ratio for _, _, ratio, _ in ratios)
        heading_ratios = [heading_ratio for _, _, _, heading_ratio in ratios]
        print(
            f"batch size {batch_size}, {len(batches)} batches |"
            f" regression {np.mean([value for value, _, _, _ in ratios]):.3f} nats |"
            f" neighbour future {np.mean([value for _, value, _, _ in ratios]):.3f} m |"
            f" equal-footing weight per batch "
            + " ".join(f"{ratio:.3f}" for _, _, ratio, _ in ratios)
        )
        print(
            f"  equal-footing HEADING weight mean {np.mean(heading_ratios):.3f},"
            f" per batch " + " ".join(f"{ratio:.3f}" for ratio in heading_ratios)
        )
        print(
            f"  neighbours per sample over {len(neighbour_counts)} samples:"
            f" p50 {np.percentile(neighbour_counts, 50):.0f}"
            f" p90 {np.percentile(neighbour_counts, 90):.0f}"
            f" max {neighbour_counts.max()}"
        )
        head_output_bytes = 64 * neighbour_counts.max() * contract.FUTURE_STEPS * 2 * 4
        print(
            f"  neighbour future tensor at batch 64, padded to this max:"
            f" {head_output_bytes / 1e6:.1f} MB, and"
            f" {64 * np.percentile(neighbour_counts, 50) * contract.FUTURE_STEPS * 2 * 4 / 1e6:.1f} MB"
            f" at the p50 neighbour count"
        )
        with_head = step_seconds(predictor, batches[0], True, arguments.timing_repeats)
        without_head = step_seconds(predictor, batches[0], False, arguments.timing_repeats)
        print(
            f"  CPU step (forward + loss + backward), median of {arguments.timing_repeats}:"
            f" {with_head:.3f} s with the head, {without_head:.3f} s without,"
            f" added {with_head - without_head:+.3f} s"
            f" ({(with_head - without_head) / without_head:+.1%})"
        )
    print(
        f"equal-footing weight over all {len(all_ratios)} batches:"
        f" mean {np.mean(all_ratios):.3f}, min {np.min(all_ratios):.3f},"
        f" max {np.max(all_ratios):.3f}, spread {np.max(all_ratios) / np.min(all_ratios):.2f}x"
    )


if __name__ == "__main__":
    main()
