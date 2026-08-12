# > Staging entry point: raw WOMD shards in, one .npz per scenario out
# stage.py is what a staging session runs. It walks each downloaded shard, hands every parsed
# scenario to store.write_scenario, and names each output file by WOMD's globally unique
# scenario id. Returns paths, never counts - a count can hide a filename collision (B4).

import sys
from pathlib import Path

from womd import store, tfrecord
from waymo_open_dataset.protos import scenario_pb2

# One shard (about 300-600 scenarios) -> its .npz files. Streams scenarios off disk one at a time
# (never loads the shard whole), writes <scenario_id>.npz into output_directory. Returns every
# path written plus each scenario's worst timestep-spacing deviation, index-aligned.
def stage_shard(shard_path, output_directory):
    written_paths = []
    spacing_deviations = []
    for scenario in tfrecord.read_scenarios(shard_path, scenario_pb2.Scenario):
        output_path = Path(output_directory) / f"{scenario.scenario_id}.npz"
        spacing_deviations.append(store.write_scenario(scenario, output_path))
        written_paths.append(output_path)
    return written_paths, spacing_deviations

# The whole run: all shards through stage_shard, then the B4 guard - if any two scenarios
# claimed the same filename the set shrinks below the list and staging dies loudly,
# naming the collision count. Guard spans ALL shards because B4's collision was cross-file.
def stage_shards(shard_paths, output_directory):
    written_paths = []
    spacing_deviations = []
    for shard_path in shard_paths:
        shard_written_paths, shard_deviations = stage_shard(shard_path, output_directory)
        written_paths.extend(shard_written_paths)
        spacing_deviations.extend(shard_deviations)
    distinct_paths = set(written_paths)
    assert len(distinct_paths) == len(written_paths), (
        f"{len(written_paths) - len(distinct_paths)} colliding output paths"
    )
    return written_paths, spacing_deviations

# Run: python3 stage.py <output_directory> <shard> [<shard> ...]
# Irregular timestep spacing (deviation over 0.005 s from the 0.1 s tick) is reported, not
# fatal: Waymo ships such scenarios and scores them like any other, so we stage them all.
def main():
    output_directory = Path(sys.argv[1])
    shard_paths = [Path(argument) for argument in sys.argv[2:]]
    output_directory.mkdir(parents=True, exist_ok=True)
    written_paths, spacing_deviations = stage_shards(shard_paths, output_directory)
    irregular_count = sum(deviation >= 0.005 for deviation in spacing_deviations)
    print(f"{len(written_paths)} scenarios staged into {output_directory}")
    print(
        f"{irregular_count} with irregular timestep spacing"
        f" (worst deviation {max(spacing_deviations, default=0.0):.4f} s)"
    )


if __name__ == "__main__":
    main()
