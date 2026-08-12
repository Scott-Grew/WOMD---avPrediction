# > Staging entry point: raw WOMD shards in, one .npz per scenario out
# stage.py is what a staging session runs. It walks each downloaded shard, hands every parsed
# scenario to store.write_scenario, and names each output file by WOMD's globally unique
# scenario id. Returns paths, never counts - a count can hide a filename collision (B4).

import sys
from pathlib import Path

from womd import store, tfrecord
from waymo_open_dataset.protos import scenario_pb2

# One shard (about 1k scenarios) -> its .npz files. Streams scenarios off disk one at a time (never loads the
# shard whole), writes <scenario_id>.npz into output_directory, returns every path written.
def stage_shard(shard_path, output_directory):
    written_paths = []
    for scenario in tfrecord.read_scenarios(shard_path, scenario_pb2.Scenario):
        output_path = Path(output_directory) / f"{scenario.scenario_id}.npz"
        store.write_scenario(scenario, output_path)
        written_paths.append(output_path)
    return written_paths

# The whole run: all shards through stage_shard, then the B4 guard - if any two scenarios
# claimed the same filename the set shrinks below the list and staging dies loudly,
# naming the collision count. Guard spans ALL shards because B4's collision was cross-file.
def stage_shards(shard_paths, output_directory):
    written_paths = []
    for shard_path in shard_paths:
        written_paths.extend(stage_shard(shard_path, output_directory))
    distinct_paths = set(written_paths)
    assert len(distinct_paths) == len(written_paths), (
        f"{len(written_paths) - len(distinct_paths)} colliding output paths"
    )
    return written_paths

# Run: python3 stage.py <output_directory> <shard> [<shard> ...]
def main():
    output_directory = Path(sys.argv[1])
    shard_paths = [Path(argument) for argument in sys.argv[2:]]
    output_directory.mkdir(parents=True, exist_ok=True)
    written_paths = stage_shards(shard_paths, output_directory)
    print(f"{len(written_paths)} scenarios staged into {output_directory}")


if __name__ == "__main__":
    main()
