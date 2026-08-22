# > Staging entry point: raw WOMD shards in, one .npz per scenario out
# stage.py is what a staging session runs. It walks each downloaded shard, hands every parsed
# scenario to store.write_scenario, and names each output file by WOMD's globally unique
# scenario id. Returns paths, never counts - a count can hide a filename collision (B4).

import womd.runtime_env
import argparse
from pathlib import Path

from womd import store, tfrecord
from womd_protos import scenario_pb2

# One shard (about 300-600 scenarios) -> its .npz files. Streams scenarios off disk one at a time
# (never loads the shard whole), writes <scenario_id>.npz into output_directory. Returns every
# path written plus each scenario's worst timestep-spacing deviation, index-aligned, and the
# paths it declined to rewrite. An output file that already exists is never overwritten: the
# staging job is ~74 core-hours and gets run in pieces, so a second piece landing on a directory
# an earlier piece already filled is either a resume - which asks for it with skip_existing -
# or a mistake, and a mistake must not silently replace data that is already on disk.
def stage_shard(shard_path, output_directory, skip_existing):
    written_paths = []
    spacing_deviations = []
    skipped_paths = []
    for scenario in tfrecord.read_scenarios(shard_path, scenario_pb2.Scenario):
        output_path = Path(output_directory) / f"{scenario.scenario_id}.npz"
        if output_path.exists():
            assert skip_existing, f"output file already exists: {output_path}"
            skipped_paths.append(output_path)
            continue
        spacing_deviations.append(store.write_scenario(scenario, output_path))
        written_paths.append(output_path)
    return written_paths, spacing_deviations, skipped_paths

# The whole run: all shards through stage_shard, then the B4 guard - if any two scenarios
# claimed the same filename the set shrinks below the list and staging dies loudly,
# naming the collision count. Guard spans ALL shards because B4's collision was cross-file;
# the per-scenario existence check in stage_shard is what spans separate staging sessions.
def stage_shards(shard_paths, output_directory, skip_existing=False):
    written_paths = []
    spacing_deviations = []
    skipped_paths = []
    for shard_path in shard_paths:
        shard_written_paths, shard_deviations, shard_skipped_paths = stage_shard(
            shard_path, output_directory, skip_existing
        )
        written_paths.extend(shard_written_paths)
        spacing_deviations.extend(shard_deviations)
        skipped_paths.extend(shard_skipped_paths)
    distinct_paths = set(written_paths)
    assert len(distinct_paths) == len(written_paths), (
        f"{len(written_paths) - len(distinct_paths)} colliding output paths"
    )
    if skip_existing:
        print(f"{len(skipped_paths)} scenarios already staged, skipped")
    return written_paths, spacing_deviations

# Run: python3 stage.py <output_directory> <shard> [<shard> ...] [--skip-existing]
# Irregular timestep spacing (deviation over 0.005 s from the 0.1 s tick) is reported, not
# fatal: Waymo ships such scenarios and scores them like any other, so we stage them all.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("shard_paths", type=Path, nargs="+")
    parser.add_argument("--skip-existing", action="store_true")
    arguments = parser.parse_args()

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    written_paths, spacing_deviations = stage_shards(
        arguments.shard_paths, arguments.output_directory, arguments.skip_existing
    )
    irregular_count = sum(deviation >= 0.005 for deviation in spacing_deviations)
    print(f"{len(written_paths)} scenarios staged into {arguments.output_directory}")
    print(
        f"{irregular_count} with irregular timestep spacing"
        f" (worst deviation {max(spacing_deviations, default=0.0):.4f} s)"
    )


if __name__ == "__main__":
    main()
