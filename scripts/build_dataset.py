import argparse
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


def tfrecord_scenarios(input_directory):
    paths = sorted(pathlib.Path(input_directory).glob("*"))
    if len(paths) == 0:
        raise FileNotFoundError(f"no record files under {input_directory}")
    for path in paths:
        yield from tfrecord.read_scenarios(path, scenario_pb2.Scenario)


def build(scenarios, output_directory, shard_size, benchmark_targets_only):
    output_directory = pathlib.Path(output_directory)
    pending, shard_index, total = [], 0, 0
    for scenario in scenarios:
        pending.extend(
            features.build_scenario_samples(
                scenario, benchmark_targets_only=benchmark_targets_only
            )
        )
        while len(pending) >= shard_size:
            dataset.write_shard(
                output_directory / f"shard-{shard_index:06d}.npz", pending[:shard_size]
            )
            pending = pending[shard_size:]
            shard_index += 1
            total += shard_size
    if pending:
        dataset.write_shard(output_directory / f"shard-{shard_index:06d}.npz", pending)
        total += len(pending)
        shard_index += 1
    return shard_index, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("synthetic", "tfrecord"), required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--benchmark-targets-only", action="store_true")
    arguments = parser.parse_args()

    if arguments.source == "synthetic":
        scenarios = synthetic_scenarios(arguments.count, arguments.seed)
    else:
        if not arguments.input:
            parser.error("--input is required when --source tfrecord")
        scenarios = tfrecord_scenarios(arguments.input)

    shard_count, sample_count = build(
        scenarios, arguments.output, arguments.shard_size, arguments.benchmark_targets_only
    )
    print(f"wrote {sample_count} samples across {shard_count} shards to {arguments.output}")


if __name__ == "__main__":
    main()
