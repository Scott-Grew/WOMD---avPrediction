import argparse
import multiprocessing
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from waymo_open_dataset.protos import scenario_pb2
from womd import dataset, features, synthetic, tfrecord


def synthetic_scenarios(count, seed):
    random_generator = np.random.default_rng(seed)
    for scenario_index in range(count):
        yield synthetic.random_scenario(scenario_index, random_generator)


def record_paths(input_directory):
    paths = sorted(path for path in pathlib.Path(input_directory).glob("*") if path.is_file())
    if len(paths) == 0:
        raise FileNotFoundError(f"no record files under {input_directory}")
    return paths


def build_from_scenarios(scenarios, output_directory, name_prefix, shard_size, benchmark_targets_only):
    output_directory = pathlib.Path(output_directory)
    pending, written, total = [], [], 0
    for scenario in scenarios:
        pending.extend(
            features.build_scenario_samples(
                scenario, benchmark_targets_only=benchmark_targets_only
            )
        )
        while len(pending) >= shard_size:
            path = output_directory / f"{name_prefix}-{len(written):04d}.npz"
            dataset.write_shard(path, pending[:shard_size])
            pending = pending[shard_size:]
            written.append(path)
            total += shard_size
    if pending:
        path = output_directory / f"{name_prefix}-{len(written):04d}.npz"
        dataset.write_shard(path, pending)
        written.append(path)
        total += len(pending)
    return written, total


def stage_record_file(task):
    path, output_directory, shard_size, benchmark_targets_only = task
    scenarios = tfrecord.read_scenarios(path, scenario_pb2.Scenario)
    return build_from_scenarios(
        scenarios, output_directory, path.name, shard_size, benchmark_targets_only
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("synthetic", "tfrecord"), required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--benchmark-targets-only", action="store_true")
    arguments = parser.parse_args()

    if arguments.source == "synthetic":
        results = [
            build_from_scenarios(
                synthetic_scenarios(arguments.count, arguments.seed),
                arguments.output,
                "synthetic",
                arguments.shard_size,
                arguments.benchmark_targets_only,
            )
        ]
    else:
        if not arguments.input:
            parser.error("--input is required when --source tfrecord")
        tasks = [
            (path, arguments.output, arguments.shard_size, arguments.benchmark_targets_only)
            for path in record_paths(arguments.input)
        ]
        if arguments.workers > 1:
            with multiprocessing.Pool(arguments.workers) as pool:
                results = pool.map(stage_record_file, tasks)
        else:
            results = [stage_record_file(task) for task in tasks]

    written = [path for paths, _ in results for path in paths]
    sample_count = sum(samples for _, samples in results)
    if len(set(written)) != len(written):
        raise RuntimeError(
            f"shard name collision: {len(written)} writes produced {len(set(written))} files"
        )

    print(f"wrote {sample_count} samples across {len(written)} shards to {arguments.output}")


if __name__ == "__main__":
    main()
