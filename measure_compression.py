# > Storage-format measurement: over a staged directory, rewrites every scenario's arrays both
# ways - np.savez_compressed, as store.py writes today, and plain np.savez - into a temporary
# directory that is deleted at the end, and reports bytes on disk, write time and read time for
# each format. Read time is the loader's own read: np.load followed by materialising every array
# into a dict, exactly womd.loader.read_scenario's np.load block, without the two derived arrays
# that cost the same in either format. Each scenario's write and read timings discard
# DISCARDED_WARMUP_PASSES passes and report the median of WRITE_REPEAT_COUNT / READ_REPEAT_COUNT
# timed passes. Reads are WARM: every file is read moments after being written by this process,
# and dropping the macOS page cache needs privileges this script does not take, so the read
# column is a page-cache-resident decompression cost, not a cold-disk cost. Projections to the
# full training split use FULL_SPLIT_SCENARIO_COUNT; the read core-hours per epoch assume one
# read of every scenario per epoch at the measured per-scenario read time. Per-array bytes come
# from the zip member sizes inside each .npz.
# Reports only - no constant is chosen, derived or written anywhere here, and no staged file is
# modified.
# Run: python3 measure_compression.py ../data/staged

import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

FULL_SPLIT_SCENARIO_COUNT = 579_000
READ_REPEAT_COUNT = 5
WRITE_REPEAT_COUNT = 3
DISCARDED_WARMUP_PASSES = 1
PER_ARRAY_SCENARIO_COUNT = 3
BYTES_PER_GIGABYTE = 1024.0 ** 3
SECONDS_PER_HOUR = 3600.0
FORMAT_NAMES = ("compressed", "uncompressed")
SAVE_FUNCTIONS = {"compressed": np.savez_compressed, "uncompressed": np.savez}


def materialise_arrays(scenario_path):
    with np.load(scenario_path) as scenario_file:
        return {name: scenario_file[name] for name in scenario_file.files}


def median_write_seconds(save_function, scenario_path, arrays):
    for _ in range(DISCARDED_WARMUP_PASSES):
        save_function(scenario_path, **arrays)
    durations = []
    for _ in range(WRITE_REPEAT_COUNT):
        start_seconds = time.perf_counter()
        save_function(scenario_path, **arrays)
        durations.append(time.perf_counter() - start_seconds)
    return float(np.median(durations))


def median_read_seconds(scenario_path):
    for _ in range(DISCARDED_WARMUP_PASSES):
        materialise_arrays(scenario_path)
    durations = []
    for _ in range(READ_REPEAT_COUNT):
        start_seconds = time.perf_counter()
        materialise_arrays(scenario_path)
        durations.append(time.perf_counter() - start_seconds)
    return float(np.median(durations))


def array_bytes_on_disk(scenario_path):
    with zipfile.ZipFile(scenario_path) as archive:
        return {Path(member.filename).stem: member.compress_size for member in archive.infolist()}


class FormatTally:
    def __init__(self, name):
        self.name = name
        self.total_bytes = 0
        self.write_seconds = []
        self.read_seconds = []
        self.array_totals = {}

    def add(self, scenario_bytes, write_seconds, read_seconds, per_array_bytes):
        self.total_bytes += scenario_bytes
        self.write_seconds.append(write_seconds)
        self.read_seconds.append(read_seconds)
        for array_name, array_size in per_array_bytes.items():
            self.array_totals[array_name] = self.array_totals.get(array_name, 0) + array_size

    def mean_bytes(self):
        return self.total_bytes / len(self.read_seconds)

    def mean_write_seconds(self):
        return float(np.mean(self.write_seconds))

    def mean_read_seconds(self):
        return float(np.mean(self.read_seconds))

    def projected_gigabytes(self):
        return self.mean_bytes() * FULL_SPLIT_SCENARIO_COUNT / BYTES_PER_GIGABYTE

    def projected_read_core_hours(self):
        return self.mean_read_seconds() * FULL_SPLIT_SCENARIO_COUNT / SECONDS_PER_HOUR


def main():
    staged_directory = Path(sys.argv[1])
    scenario_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    scenario_paths = sorted(staged_directory.glob("*.npz"))[:scenario_limit]
    tallies = {name: FormatTally(name) for name in FORMAT_NAMES}
    per_scenario_array_bytes = []
    staged_total_bytes = 0

    start_seconds = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary_directory:
        rewrite_paths = {
            name: Path(temporary_directory) / f"{name}.npz" for name in FORMAT_NAMES
        }
        for scenario_path in scenario_paths:
            arrays = materialise_arrays(scenario_path)
            staged_total_bytes += scenario_path.stat().st_size
            scenario_array_bytes = {}
            for name in FORMAT_NAMES:
                rewrite_path = rewrite_paths[name]
                write_seconds = median_write_seconds(SAVE_FUNCTIONS[name], rewrite_path, arrays)
                read_seconds = median_read_seconds(rewrite_path)
                per_array_bytes = array_bytes_on_disk(rewrite_path)
                tallies[name].add(
                    rewrite_path.stat().st_size, write_seconds, read_seconds, per_array_bytes
                )
                scenario_array_bytes[name] = per_array_bytes
            if len(per_scenario_array_bytes) < PER_ARRAY_SCENARIO_COUNT:
                per_scenario_array_bytes.append((scenario_path.name, scenario_array_bytes))
    elapsed_seconds = time.perf_counter() - start_seconds

    scenario_count = len(scenario_paths)
    compressed = tallies["compressed"]
    uncompressed = tallies["uncompressed"]

    print(f"staged directory {staged_directory} | scenarios {scenario_count} | "
          f"{elapsed_seconds:.1f} s")
    print(f"staged files as they sit on disk: {staged_total_bytes / BYTES_PER_GIGABYTE:.3f} GB | "
          f"{staged_total_bytes / scenario_count / 1024.0:.1f} KiB per scenario")
    print(f"repeats per scenario: write {WRITE_REPEAT_COUNT} timed after {DISCARDED_WARMUP_PASSES} "
          f"discarded, read {READ_REPEAT_COUNT} timed after {DISCARDED_WARMUP_PASSES} discarded, "
          f"median of each")
    print("reads are WARM - the file was written by this process moments earlier and stays in the "
          "page cache; no cold-disk read is measured")
    print()

    print(f"{'format':<14}{'total MiB':>12}{'KiB/scenario':>15}{'write ms':>11}{'read ms':>10}")
    for name in FORMAT_NAMES:
        tally = tallies[name]
        print(f"{name:<14}{tally.total_bytes / 1024.0 / 1024.0:>12.1f}"
              f"{tally.mean_bytes() / 1024.0:>15.1f}"
              f"{tally.mean_write_seconds() * 1000.0:>11.2f}"
              f"{tally.mean_read_seconds() * 1000.0:>10.2f}")
    print()

    print(f"bytes ratio uncompressed / compressed  {uncompressed.mean_bytes() / compressed.mean_bytes():.2f}x")
    print(f"read ratio  compressed / uncompressed  "
          f"{compressed.mean_read_seconds() / uncompressed.mean_read_seconds():.2f}x")
    print(f"write ratio compressed / uncompressed  "
          f"{compressed.mean_write_seconds() / uncompressed.mean_write_seconds():.2f}x")
    print()

    print(f"projected to the full training split, {FULL_SPLIT_SCENARIO_COUNT} scenarios")
    print(f"{'format':<14}{'total GB':>11}{'read core-hours per epoch':>28}")
    for name in FORMAT_NAMES:
        tally = tallies[name]
        print(f"{name:<14}{tally.projected_gigabytes():>11.1f}"
              f"{tally.projected_read_core_hours():>28.1f}")
    print(f"difference: {uncompressed.projected_gigabytes() - compressed.projected_gigabytes():.1f} GB "
          f"more on disk uncompressed | "
          f"{compressed.projected_read_core_hours() - uncompressed.projected_read_core_hours():.1f} "
          f"core-hours per epoch more to read compressed")
    print()

    print(f"per-array bytes on disk, summed over {scenario_count} scenarios")
    print(f"{'array':<24}{'compressed KiB':>17}{'share':>9}{'uncompressed KiB':>19}{'share':>9}"
          f"{'ratio':>8}")
    for array_name in sorted(uncompressed.array_totals, key=uncompressed.array_totals.get, reverse=True):
        compressed_bytes = compressed.array_totals[array_name]
        uncompressed_bytes = uncompressed.array_totals[array_name]
        print(f"{array_name:<24}{compressed_bytes / 1024.0:>17.1f}"
              f"{compressed_bytes / compressed.total_bytes:>9.1%}"
              f"{uncompressed_bytes / 1024.0:>19.1f}"
              f"{uncompressed_bytes / uncompressed.total_bytes:>9.1%}"
              f"{uncompressed_bytes / compressed_bytes:>8.2f}")
    print()

    for scenario_name, scenario_array_bytes in per_scenario_array_bytes:
        compressed_bytes_by_array = scenario_array_bytes["compressed"]
        uncompressed_bytes_by_array = scenario_array_bytes["uncompressed"]
        compressed_scenario_bytes = sum(compressed_bytes_by_array.values())
        uncompressed_scenario_bytes = sum(uncompressed_bytes_by_array.values())
        print(f"{scenario_name}")
        print(f"{'  array':<24}{'compressed KiB':>17}{'share':>9}{'uncompressed KiB':>19}{'share':>9}"
              f"{'ratio':>8}")
        for array_name in sorted(
            uncompressed_bytes_by_array, key=uncompressed_bytes_by_array.get, reverse=True
        ):
            compressed_bytes = compressed_bytes_by_array[array_name]
            uncompressed_bytes = uncompressed_bytes_by_array[array_name]
            print(f"  {array_name:<22}{compressed_bytes / 1024.0:>17.1f}"
                  f"{compressed_bytes / compressed_scenario_bytes:>9.1%}"
                  f"{uncompressed_bytes / 1024.0:>19.1f}"
                  f"{uncompressed_bytes / uncompressed_scenario_bytes:>9.1%}"
                  f"{uncompressed_bytes / max(compressed_bytes, 1):>8.2f}")
        print()


if __name__ == "__main__":
    main()
