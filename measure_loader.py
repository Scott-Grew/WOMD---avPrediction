import sys
import time
from pathlib import Path

import numpy as np

from womd import contract, loader, pipeline


def timed_epoch(staged_directory, designated_targets_only):
    scenario_paths = sorted(Path(staged_directory).glob("*.npz"))
    read_seconds = 0.0
    build_seconds = 0.0
    batch_seconds = 0.0
    sample_dot_counts = []
    sample_neighbour_counts = []

    epoch_start = time.perf_counter()
    for scenario_path in scenario_paths:
        start = time.perf_counter()
        scenario_arrays = loader.read_scenario(scenario_path)
        read_seconds += time.perf_counter() - start

        sample_track_indices = loader.eligible_track_indices(
            scenario_arrays["track_rows"],
            scenario_arrays["track_valid"],
            scenario_arrays["is_designated_target"],
            designated_targets_only,
        )
        start = time.perf_counter()
        samples = [
            loader.build_sample(scenario_arrays, int(track_index))
            for track_index in sample_track_indices
        ]
        build_seconds += time.perf_counter() - start

        start = time.perf_counter()
        if samples:
            loader.build_batch(samples)
        batch_seconds += time.perf_counter() - start

        for sample in samples:
            sample_dot_counts.append(sample["map_rows"].shape[0])
            sample_neighbour_counts.append(sample["neighbour_history"].shape[0])
    epoch_seconds = time.perf_counter() - epoch_start

    sample_count = len(sample_dot_counts)
    print(f"scenarios {len(scenario_paths)} | samples {sample_count}")
    print(
        f"epoch {epoch_seconds:.1f} s | read {read_seconds:.1f} s | "
        f"build {build_seconds:.1f} s ({build_seconds / sample_count * 1000:.2f} ms/sample) | "
        f"batch-per-scenario {batch_seconds:.1f} s"
    )
    print(
        f"map dots/sample p50 {int(np.percentile(sample_dot_counts, 50))} | "
        f"p90 {int(np.percentile(sample_dot_counts, 90))} | max {max(sample_dot_counts)} | "
        f"neighbours max {max(sample_neighbour_counts)}"
    )
    return np.array(sample_dot_counts), epoch_seconds, sample_count


def padding_waste_curve(sample_dot_counts, batch_sizes):
    print("padding waste by batch size (diagnostic, stream order, map block only):")
    for batch_size in batch_sizes:
        padded_slots = 0
        real_slots = 0
        for batch_start in range(0, len(sample_dot_counts), batch_size):
            batch_counts = sample_dot_counts[batch_start:batch_start + batch_size]
            padded_slots += batch_counts.max() * len(batch_counts)
            real_slots += batch_counts.sum()
        sorted_counts = np.sort(sample_dot_counts)
        sorted_padded = 0
        for batch_start in range(0, len(sorted_counts), batch_size):
            batch_counts = sorted_counts[batch_start:batch_start + batch_size]
            sorted_padded += batch_counts.max() * len(batch_counts)
        print(
            f"  batch {batch_size:>3}: waste {100 * (1 - real_slots / padded_slots):.1f}% | "
            f"fully bucketed floor {100 * (1 - real_slots / sorted_padded):.1f}%"
        )


def timed_pipeline_epoch(staged_directory, worker_count, batch_size, prefetch_batches, designated_targets_only):
    scenario_paths = sorted(Path(staged_directory).glob("*.npz"))
    batch_count = 0
    sample_count = 0
    start = time.perf_counter()
    stream = pipeline.batches(
        scenario_paths, worker_count, batch_size, prefetch_batches, 0, designated_targets_only
    )
    for batch in stream:
        batch_count += 1
        sample_count += batch["agent_history"].shape[0]
    epoch_seconds = time.perf_counter() - start
    print(
        f"workers {worker_count} | batch {batch_size} | prefetch {prefetch_batches} | "
        f"epoch {epoch_seconds:.1f} s | samples {sample_count} | batches {batch_count}"
    )


if __name__ == "__main__":
    designated_targets_only = "--all-eligible-agents" not in sys.argv
    positional_arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    if len(positional_arguments) == 4:
        timed_pipeline_epoch(
            positional_arguments[0], int(positional_arguments[1]), int(positional_arguments[2]),
            int(positional_arguments[3]), designated_targets_only,
        )
    else:
        dot_counts, epoch_seconds, sample_count = timed_epoch(
            positional_arguments[0], designated_targets_only
        )
        padding_waste_curve(dot_counts, (8, 16, 32, 64, 128, 256))
