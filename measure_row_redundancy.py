# > Per-row redundancy measurement: for every column block of map_rows and track_rows, measures how
# many times each distinct value is repeated inside its owning group - a map feature's block of dots
# for map_rows, a track's 91 timesteps for track_rows - and what that repetition costs on disk and in
# decompression time. The redundancy factor is MEASURED, not assumed: for each group the script
# counts the distinct rows the block actually takes over that group, so a block that genuinely varies
# comes out at a factor near 1 and a block copied onto every row of its group comes out at the group
# size. track_rows blocks get a second factor counted over that track's VALID steps only, because an
# unobserved step is a zero row and a run of identical zeros is padding, not redundancy.
# Cost is measured by rewriting each scenario the way store.py writes it, np.savez_compressed, once
# as it stands and once per block with that block MOVED to per-group storage - the columns deleted
# from the row array and a (groups, block width) array holding each group's first row added in their
# place - then differencing total file size and read time. The per-group array is included in every
# variant, so the reported saving is NET of what per-group storage costs to add. Two combined
# variants move all seven claimed-redundant blocks at once: one adding seven separate per-group
# arrays, one packing them into a single per-group array per row array, because a .npz pays per zip
# member and an implementation would pack.
# A per-group block can be moved on disk WITHOUT changing what the model reads, by broadcasting it
# back onto the dots at load time, so the packed variant is also timed for that reconstruction: the
# gather of each dot's polyline row and the concatenate that rebuilds the full-width row arrays.
# np.repeat of the dot-to-polyline index is EXCLUDED from that timing because womd.loader.read_scenario
# already computes it today, so it is not new cost. Read plus reconstruct is the honest total for a
# storage-only move; read alone is the total for a move that also feeds the model per token.
# Read time is the loader's own read: np.load followed by materialising every array into a dict,
# exactly womd.loader.read_scenario's np.load block. Reads are WARM - each file is read moments after
# this process wrote it and dropping the macOS page cache needs privileges this script does not take
# - so the read column is a page-cache-resident decompression cost, not a cold-disk cost. Each
# variant's read time is the median of READ_REPEAT_COUNT timed passes after DISCARDED_WARMUP_PASSES
# discarded, and the reported delta is PAIRED per scenario against the baseline, reported with the
# standard error of that paired mean so a delta smaller than its own error reads as no effect.
# The compressed column carries a confound, so it is measured against a control: a variant that
# deletes two map_rows columns which are IDENTICALLY ZERO in every row of that scenario and so carry
# no information whatever. Any file-size drop that control shows is bought by narrowing the row -
# deflate matching differently across a shorter float32 stride - not by the deleted content. A block
# whose compressed saving does not clear that floor has not been shown to cost any compressed bytes.
# Uncompressed bytes are exact: the arrays on disk are already float32, so a block occupies
# rows x columns x 4 bytes and the corpus total is the sum of every array's nbytes.
# Reports only - no constant is chosen, derived or written anywhere here, no staged file is modified,
# and no variant is ever written back into the staged directory.
# Run: python3 measure_row_redundancy.py ../data/staged

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from womd import contract

READ_REPEAT_COUNT = 5
DISCARDED_WARMUP_PASSES = 1
BYTES_PER_MEBIBYTE = 1024.0 ** 2

ROW_ARRAY_WIDTHS = {
    "map_rows": contract.MAP_FEATURE_DIM,
    "track_rows": contract.AGENT_FEATURE_DIM,
}

BLOCKS = (
    ("map kind", "map_rows", contract.MAP_KIND),
    ("map lane type", "map_rows", contract.MAP_LANE_TYPE),
    ("map speed limit", "map_rows", slice(contract.MAP_SPEED_LIMIT, contract.MAP_SPEED_LIMIT + 1)),
    ("map boundary type", "map_rows", contract.MAP_BOUNDARY_TYPE),
    ("map stop point", "map_rows", contract.MAP_STOP_POINT),
    ("map position", "map_rows", contract.MAP_POSITION),
    ("map direction", "map_rows", contract.MAP_DIRECTION),
    ("agent type", "track_rows", contract.AGENT_TYPE),
    ("agent is sdc", "track_rows", slice(contract.AGENT_IS_SDC, contract.AGENT_IS_SDC + 1)),
    ("agent position", "track_rows", contract.AGENT_POSITION),
    ("agent heading", "track_rows", slice(contract.AGENT_HEADING_COSINE, contract.AGENT_HEADING_SINE + 1)),
    ("agent velocity", "track_rows", contract.AGENT_VELOCITY),
    ("agent dimensions", "track_rows", contract.AGENT_DIMENSIONS),
)

CLAIMED_REDUNDANT_BLOCK_NAMES = (
    "map kind",
    "map lane type",
    "map speed limit",
    "map boundary type",
    "map stop point",
    "agent type",
    "agent is sdc",
)
SEPARATE_ARRAYS_VARIANT_NAME = "move all seven, separate arrays"
PACKED_ARRAYS_VARIANT_NAME = "move all seven, packed arrays"
ZERO_COLUMN_CONTROL_WIDTH = 2


def materialise_arrays(scenario_path):
    with np.load(scenario_path) as scenario_file:
        return {name: scenario_file[name] for name in scenario_file.files}


def block_columns(column_slice, row_width):
    return np.arange(row_width)[column_slice]


def block_groups(scenario_array, row_array_name, column_slice):
    if row_array_name == "track_rows":
        return [track[:, column_slice] for track in scenario_array["track_rows"]]
    block = scenario_array["map_rows"][:, column_slice]
    group_boundaries = np.cumsum(scenario_array["feature_lengths"])[:-1]
    return np.split(block, group_boundaries)


def group_validity(scenario_array, row_array_name):
    if row_array_name == "track_rows":
        return list(scenario_array["track_valid"])
    return None


def per_group_first_rows(groups, column_count):
    if not groups:
        return np.zeros((0, column_count), dtype=np.float32)
    return np.stack([group[0] for group in groups]).astype(np.float32)


def per_group_array_name(block_name):
    return f"{block_name.replace(' ', '_')}_per_group"


def rewrite_with_blocks_moved(scenario_array, moved_blocks, pack_per_group_arrays):
    arrays = dict(scenario_array)
    columns_to_delete = {}
    packed_first_rows = {}
    for block_name, row_array_name, column_slice in moved_blocks:
        columns = block_columns(column_slice, ROW_ARRAY_WIDTHS[row_array_name])
        first_rows = per_group_first_rows(
            block_groups(scenario_array, row_array_name, column_slice), len(columns)
        )
        if pack_per_group_arrays:
            packed_first_rows.setdefault(row_array_name, []).append(first_rows)
        else:
            arrays[per_group_array_name(block_name)] = first_rows
        columns_to_delete.setdefault(row_array_name, []).append(columns)
    for row_array_name, first_rows_list in packed_first_rows.items():
        arrays[f"{row_array_name}_per_group"] = np.concatenate(first_rows_list, axis=1)
    for row_array_name, column_lists in columns_to_delete.items():
        arrays[row_array_name] = np.delete(
            scenario_array[row_array_name], np.concatenate(column_lists), axis=-1
        )
    return arrays


def moved_and_kept_columns(moved_blocks, row_array_name):
    moved = np.concatenate([
        block_columns(column_slice, ROW_ARRAY_WIDTHS[row_array_name])
        for _, block_array_name, column_slice in moved_blocks
        if block_array_name == row_array_name
    ])
    return moved, np.setdiff1d(np.arange(ROW_ARRAY_WIDTHS[row_array_name]), moved)


def assert_reconstruction_is_lossless(scenario_array, reconstructed, moved_blocks, row_array_name):
    moved, kept = moved_and_kept_columns(moved_blocks, row_array_name)
    original = scenario_array[row_array_name][..., np.concatenate([kept, moved])]
    assert np.array_equal(original, reconstructed), (
        f"per-group storage of {row_array_name} is lossy for scenario "
        f"{scenario_array['scenario_id']}"
    )


def identically_zero_map_columns(scenario_array):
    return np.flatnonzero(~np.any(scenario_array["map_rows"] != 0.0, axis=0))


def uncompressed_bytes(arrays):
    return sum(int(np.asarray(array).nbytes) for array in arrays.values())


def reconstruct_row_arrays(arrays, map_dot_polyline_index):
    map_rows = np.concatenate(
        [arrays["map_rows"], arrays["map_rows_per_group"][map_dot_polyline_index]], axis=1
    )
    per_track = arrays["track_rows_per_group"]
    track_rows = np.concatenate(
        [
            arrays["track_rows"],
            np.broadcast_to(
                per_track[:, None, :],
                (per_track.shape[0], contract.TOTAL_STEPS, per_track.shape[1]),
            ),
        ],
        axis=2,
    )
    return map_rows, track_rows


def median_reconstruct_seconds(arrays, map_dot_polyline_index):
    for _ in range(DISCARDED_WARMUP_PASSES):
        reconstruct_row_arrays(arrays, map_dot_polyline_index)
    durations = []
    for _ in range(READ_REPEAT_COUNT):
        start_seconds = time.perf_counter()
        reconstruct_row_arrays(arrays, map_dot_polyline_index)
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


class VariantTally:
    def __init__(self, name):
        self.name = name
        self.compressed_bytes = 0
        self.uncompressed_bytes = 0
        self.read_seconds = []

    def add(self, compressed_bytes, uncompressed_bytes, read_seconds):
        self.compressed_bytes += compressed_bytes
        self.uncompressed_bytes += uncompressed_bytes
        self.read_seconds.append(read_seconds)

    def mean_read_milliseconds(self):
        return float(np.mean(self.read_seconds)) * 1000.0

    def paired_read_saving_milliseconds(self, baseline_tally):
        paired = (np.array(baseline_tally.read_seconds) - np.array(self.read_seconds)) * 1000.0
        return float(paired.mean()), float(paired.std(ddof=1) / np.sqrt(paired.size))


class RedundancyTally:
    def __init__(self, name):
        self.name = name
        self.group_sizes = []
        self.distinct_row_counts = []
        self.valid_group_sizes = []
        self.valid_distinct_row_counts = []
        self.constant_group_count = 0
        self.valid_constant_group_count = 0
        self.all_zero_group_count = 0
        self.block_uncompressed_bytes = 0
        self.per_group_uncompressed_bytes = 0

    def add(self, groups, validity, per_group_bytes):
        for group_index, group in enumerate(groups):
            distinct_row_count = len(np.unique(group, axis=0))
            self.group_sizes.append(len(group))
            self.distinct_row_counts.append(distinct_row_count)
            self.constant_group_count += int(distinct_row_count == 1)
            self.all_zero_group_count += int(not np.any(group))
            self.block_uncompressed_bytes += int(group.nbytes)
            valid_rows = group if validity is None else group[validity[group_index]]
            if len(valid_rows):
                valid_distinct_row_count = len(np.unique(valid_rows, axis=0))
                self.valid_group_sizes.append(len(valid_rows))
                self.valid_distinct_row_counts.append(valid_distinct_row_count)
                self.valid_constant_group_count += int(valid_distinct_row_count == 1)
        self.per_group_uncompressed_bytes += per_group_bytes

    def group_size_percentiles(self):
        sizes = np.array(self.group_sizes, dtype=np.float64)
        return np.percentile(sizes, (50.0, 90.0)), float(sizes.max())

    def redundancy_percentiles(self):
        factors = np.array(self.group_sizes, dtype=np.float64) / np.array(
            self.distinct_row_counts, dtype=np.float64
        )
        return np.percentile(factors, (50.0, 90.0)), float(factors.max())

    def valid_redundancy_percentiles(self):
        factors = np.array(self.valid_group_sizes, dtype=np.float64) / np.array(
            self.valid_distinct_row_counts, dtype=np.float64
        )
        return np.percentile(factors, (50.0, 90.0)), float(factors.max())

    def constant_group_share(self):
        return self.constant_group_count / len(self.group_sizes)

    def valid_constant_group_share(self):
        return self.valid_constant_group_count / len(self.valid_group_sizes)

    def all_zero_group_share(self):
        return self.all_zero_group_count / len(self.group_sizes)


def main():
    staged_directory = Path(sys.argv[1])
    scenario_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    scenario_paths = sorted(staged_directory.glob("*.npz"))[:scenario_limit]

    block_by_name = {block[0]: block for block in BLOCKS}
    claimed_redundant_blocks = tuple(
        block_by_name[block_name] for block_name in CLAIMED_REDUNDANT_BLOCK_NAMES
    )
    variants = [(block[0], (block,), False) for block in BLOCKS]
    variants.append((SEPARATE_ARRAYS_VARIANT_NAME, claimed_redundant_blocks, False))
    variants.append((PACKED_ARRAYS_VARIANT_NAME, claimed_redundant_blocks, True))

    baseline_tally = VariantTally("staged as written today")
    variant_tallies = {variant_name: VariantTally(variant_name) for variant_name, _, _ in variants}
    redundancy_tallies = {block[0]: RedundancyTally(block[0]) for block in BLOCKS}
    packed_reconstruct_seconds = []
    zero_column_control_tally = VariantTally("control: drop 2 all-zero map columns")
    zero_column_control_baseline_bytes = 0
    all_zero_column_counts = []
    staged_bytes_on_disk = 0

    start_seconds = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary_directory:
        baseline_path = Path(temporary_directory) / "baseline.npz"
        variant_path = Path(temporary_directory) / "variant.npz"
        for scenario_path in scenario_paths:
            scenario_array = materialise_arrays(scenario_path)
            staged_bytes_on_disk += scenario_path.stat().st_size

            np.savez_compressed(baseline_path, **scenario_array)
            baseline_tally.add(
                baseline_path.stat().st_size,
                uncompressed_bytes(scenario_array),
                median_read_seconds(baseline_path),
            )

            all_zero_columns = identically_zero_map_columns(scenario_array)
            all_zero_column_counts.append(len(all_zero_columns))
            if len(all_zero_columns) >= ZERO_COLUMN_CONTROL_WIDTH:
                control_arrays = dict(scenario_array)
                control_arrays["map_rows"] = np.delete(
                    scenario_array["map_rows"], all_zero_columns[:ZERO_COLUMN_CONTROL_WIDTH], axis=1
                )
                np.savez_compressed(variant_path, **control_arrays)
                zero_column_control_tally.add(
                    variant_path.stat().st_size,
                    uncompressed_bytes(control_arrays),
                    median_read_seconds(variant_path),
                )
                zero_column_control_baseline_bytes += baseline_path.stat().st_size

            for variant_name, moved_blocks, pack_per_group_arrays in variants:
                arrays = rewrite_with_blocks_moved(
                    scenario_array, moved_blocks, pack_per_group_arrays
                )
                np.savez_compressed(variant_path, **arrays)
                variant_tallies[variant_name].add(
                    variant_path.stat().st_size,
                    uncompressed_bytes(arrays),
                    median_read_seconds(variant_path),
                )
                if variant_name == PACKED_ARRAYS_VARIANT_NAME:
                    map_dot_polyline_index = np.repeat(
                        np.arange(len(scenario_array["feature_lengths"])),
                        scenario_array["feature_lengths"],
                    )
                    reconstructed_map_rows, reconstructed_track_rows = reconstruct_row_arrays(
                        arrays, map_dot_polyline_index
                    )
                    assert_reconstruction_is_lossless(
                        scenario_array, reconstructed_map_rows, moved_blocks, "map_rows"
                    )
                    assert_reconstruction_is_lossless(
                        scenario_array, reconstructed_track_rows, moved_blocks, "track_rows"
                    )
                    packed_reconstruct_seconds.append(
                        median_reconstruct_seconds(arrays, map_dot_polyline_index)
                    )
                if len(moved_blocks) == 1:
                    block_name, row_array_name, column_slice = moved_blocks[0]
                    redundancy_tallies[block_name].add(
                        block_groups(scenario_array, row_array_name, column_slice),
                        group_validity(scenario_array, row_array_name),
                        int(arrays[per_group_array_name(block_name)].nbytes),
                    )
    elapsed_seconds = time.perf_counter() - start_seconds

    scenario_count = len(scenario_paths)
    print(f"staged directory {staged_directory} | scenarios {scenario_count} | "
          f"{elapsed_seconds:.1f} s")
    print(f"staged files as they sit on disk {staged_bytes_on_disk / BYTES_PER_MEBIBYTE:.1f} MiB | "
          f"rewritten compressed {baseline_tally.compressed_bytes / BYTES_PER_MEBIBYTE:.1f} MiB | "
          f"uncompressed {baseline_tally.uncompressed_bytes / BYTES_PER_MEBIBYTE:.1f} MiB")
    print(f"baseline read {baseline_tally.mean_read_milliseconds():.3f} ms per scenario, median of "
          f"{READ_REPEAT_COUNT} timed passes after {DISCARDED_WARMUP_PASSES} discarded, WARM page cache")
    print(f"row widths today: map_rows {contract.MAP_FEATURE_DIM} | track_rows "
          f"{contract.AGENT_FEATURE_DIM}")
    print()

    ranking = sorted(
        (block[0] for block in BLOCKS),
        key=lambda block_name: redundancy_tallies[block_name].block_uncompressed_bytes
        - redundancy_tallies[block_name].per_group_uncompressed_bytes,
        reverse=True,
    )

    zero_column_control_saving = (
        zero_column_control_baseline_bytes - zero_column_control_tally.compressed_bytes
    )
    compressed_artefact_floor = zero_column_control_saving / zero_column_control_baseline_bytes
    print(f"compressed artefact floor: deleting {ZERO_COLUMN_CONTROL_WIDTH} map_rows columns that are "
          f"IDENTICALLY ZERO in every row saves {compressed_artefact_floor:.1%} of compressed bytes")
    print(f"that content carries no information, so it is bought by the narrower row stride alone; "
          f"scenarios with at least {ZERO_COLUMN_CONTROL_WIDTH} such columns "
          f"{len(zero_column_control_tally.read_seconds)} of {scenario_count}, median all-zero map "
          f"columns {np.median(all_zero_column_counts):.0f} of {contract.MAP_FEATURE_DIM}")
    print("a block whose netCmp% does not clear that floor has NOT been shown to cost compressed bytes")
    print()

    print("cost of each block, ranked by uncompressed bytes moving it would save")
    print("saving is NET: the block's columns leave the row array and a (groups, width) array arrives")
    print("readms is the PAIRED per-scenario saving, +- the standard error of that paired mean")
    print(f"{'block':<20}{'cols':>6}{'uncompMiB':>11}{'corpus%':>9}{'netUncMiB':>11}{'netUnc%':>9}"
          f"{'netCmpKiB':>11}{'netCmp%':>9}{'overFloor':>11}{'readms':>9}{'stderr':>8}{'newWidth':>10}")
    for block_name in ranking:
        _, row_array_name, column_slice = block_by_name[block_name]
        redundancy = redundancy_tallies[block_name]
        variant = variant_tallies[block_name]
        column_count = len(block_columns(column_slice, ROW_ARRAY_WIDTHS[row_array_name]))
        net_uncompressed_saved = (
            redundancy.block_uncompressed_bytes - redundancy.per_group_uncompressed_bytes
        )
        net_compressed_saved = baseline_tally.compressed_bytes - variant.compressed_bytes
        read_saved, read_saved_error = variant.paired_read_saving_milliseconds(baseline_tally)
        print(f"{block_name:<20}{column_count:>6}"
              f"{redundancy.block_uncompressed_bytes / BYTES_PER_MEBIBYTE:>11.1f}"
              f"{redundancy.block_uncompressed_bytes / baseline_tally.uncompressed_bytes:>9.1%}"
              f"{net_uncompressed_saved / BYTES_PER_MEBIBYTE:>11.1f}"
              f"{net_uncompressed_saved / baseline_tally.uncompressed_bytes:>9.1%}"
              f"{net_compressed_saved / 1024.0:>11.1f}"
              f"{net_compressed_saved / baseline_tally.compressed_bytes:>9.1%}"
              f"{net_compressed_saved / baseline_tally.compressed_bytes - compressed_artefact_floor:>11.1%}"
              f"{read_saved:>9.3f}{read_saved_error:>8.3f}"
              f"{ROW_ARRAY_WIDTHS[row_array_name] - column_count:>10}")
    for variant_name in (SEPARATE_ARRAYS_VARIANT_NAME, PACKED_ARRAYS_VARIANT_NAME):
        variant = variant_tallies[variant_name]
        net_uncompressed_saved = baseline_tally.uncompressed_bytes - variant.uncompressed_bytes
        net_compressed_saved = baseline_tally.compressed_bytes - variant.compressed_bytes
        read_saved, read_saved_error = variant.paired_read_saving_milliseconds(baseline_tally)
        print(f"{variant_name:<20}{'':>6}{'':>11}{'':>9}"
              f"{net_uncompressed_saved / BYTES_PER_MEBIBYTE:>11.1f}"
              f"{net_uncompressed_saved / baseline_tally.uncompressed_bytes:>9.1%}"
              f"{net_compressed_saved / 1024.0:>11.1f}"
              f"{net_compressed_saved / baseline_tally.compressed_bytes:>9.1%}"
              f"{net_compressed_saved / baseline_tally.compressed_bytes - compressed_artefact_floor:>11.1%}"
              f"{read_saved:>9.3f}{read_saved_error:>8.3f}{'4 / 8':>10}")
    print()

    print("measured redundancy: rows per group against DISTINCT rows the block takes over that group")
    print("a block that genuinely varies per row lands near 1.0; a block copied onto every row lands "
          "at the group size")
    print("factValid re-counts over VALID steps only for track_rows blocks, so zero-padding is not "
          "counted as repetition; it equals fact for map_rows, which has no validity")
    print("const% is the decisive column: the share of groups over which the block takes exactly ONE "
          "distinct row, so per-group storage would lose nothing at all")
    print(f"{'block':<20}{'groups':>9}{'rowsP50':>9}{'rowsP90':>9}{'rowsMax':>9}"
          f"{'factP50':>9}{'factP90':>9}{'factMax':>9}{'const%':>9}{'vFactP50':>10}{'vConst%':>9}"
          f"{'allZero%':>10}{'perGrpMiB':>11}")
    for block_name in ranking:
        redundancy = redundancy_tallies[block_name]
        (group_size_p50, group_size_p90), group_size_max = redundancy.group_size_percentiles()
        (factor_p50, factor_p90), factor_max = redundancy.redundancy_percentiles()
        (valid_factor_p50, _), _ = redundancy.valid_redundancy_percentiles()
        print(f"{block_name:<20}{len(redundancy.group_sizes):>9}"
              f"{group_size_p50:>9.1f}{group_size_p90:>9.1f}{group_size_max:>9.0f}"
              f"{factor_p50:>9.2f}{factor_p90:>9.2f}{factor_max:>9.1f}"
              f"{redundancy.constant_group_share():>9.1%}"
              f"{valid_factor_p50:>10.2f}{redundancy.valid_constant_group_share():>9.1%}"
              f"{redundancy.all_zero_group_share():>10.1%}"
              f"{redundancy.per_group_uncompressed_bytes / BYTES_PER_MEBIBYTE:>11.2f}")
    print()

    print(f"corpus size under each move, {scenario_count} scenarios")
    print(f"{'variant':<38}{'compressed MiB':>16}{'uncompressed MiB':>18}{'read ms':>10}")
    print(f"{baseline_tally.name:<38}"
          f"{baseline_tally.compressed_bytes / BYTES_PER_MEBIBYTE:>16.2f}"
          f"{baseline_tally.uncompressed_bytes / BYTES_PER_MEBIBYTE:>18.2f}"
          f"{baseline_tally.mean_read_milliseconds():>10.3f}")
    for variant_name in ranking + [SEPARATE_ARRAYS_VARIANT_NAME, PACKED_ARRAYS_VARIANT_NAME]:
        variant = variant_tallies[variant_name]
        label = variant_name if variant_name.startswith("move") else f"move {variant_name}"
        print(f"{label:<38}"
              f"{variant.compressed_bytes / BYTES_PER_MEBIBYTE:>16.2f}"
              f"{variant.uncompressed_bytes / BYTES_PER_MEBIBYTE:>18.2f}"
              f"{variant.mean_read_milliseconds():>10.3f}")
    print()

    packed_tally = variant_tallies[PACKED_ARRAYS_VARIANT_NAME]
    reconstruct_milliseconds = float(np.mean(packed_reconstruct_seconds)) * 1000.0
    packed_read_saving, packed_read_saving_error = packed_tally.paired_read_saving_milliseconds(
        baseline_tally
    )
    print(f"storage-only move, verified lossless on all {scenario_count} scenarios: the moved columns "
          f"broadcast back to per-dot reproduce map_rows and track_rows exactly")
    print(f"read {packed_tally.mean_read_milliseconds():.3f} ms + reconstruct "
          f"{reconstruct_milliseconds:.3f} ms = {packed_tally.mean_read_milliseconds() + reconstruct_milliseconds:.3f} ms "
          f"against a {baseline_tally.mean_read_milliseconds():.3f} ms baseline")
    print(f"net change {packed_read_saving - reconstruct_milliseconds:+.3f} ms per scenario "
          f"({(packed_read_saving - reconstruct_milliseconds) / baseline_tally.mean_read_milliseconds():+.1%}), "
          f"against a read-alone saving of {packed_read_saving:+.3f} +- {packed_read_saving_error:.3f} ms "
          f"if the model is fed per token instead")


if __name__ == "__main__":
    main()
