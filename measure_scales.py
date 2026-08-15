# > Feature-scale measurement: over a staged directory, build every eligible sample through
# womd.loader.build_sample and report count, mean, standard deviation, p1/p50/p99 and the
# maximum absolute value of every feature family the model reads or emits, in the agent frame
# the model actually sees. Masked history and future steps are excluded. Map speed limit and
# map stop point are measured over non-zero entries only - both are zero on rows where they do
# not apply, and those zeros would drag the statistic. Count, mean, standard deviation and the
# maximum use every valid value; percentiles come from a uniformly decimated retention of at
# most PERCENTILE_SAMPLE_BUDGET values per family (the "kept" column), because the map block
# alone runs to about 10^8 dots. Families whose maximum exceeds FP16_SQUARE_SAFETY_LIMIT are
# flagged: squaring such a value approaches the 65504 ceiling of half precision.
# Reports only - no normalisation constant is chosen, derived or written anywhere here.
# Run: python3 measure_scales.py ../data/staged

import math
import sys
import time
from pathlib import Path

import numpy as np

from womd import contract, loader

PERCENTILE_SAMPLE_BUDGET = 2_000_000
FP16_SQUARE_SAFETY_LIMIT = 255.0


class FeatureStatistics:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0.0
        self.total_of_squares = 0.0
        self.maximum_absolute = 0.0
        self.retained_chunks = []
        self.retained_count = 0
        self.retention_stride = 1

    def add(self, values):
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        first_index = self.count
        self.count += flat.size
        self.total += float(flat.sum())
        self.total_of_squares += float(np.square(flat).sum())
        self.maximum_absolute = max(self.maximum_absolute, float(np.abs(flat).max()))

        retention_offset = (-first_index) % self.retention_stride
        retained = flat[retention_offset::self.retention_stride]
        self.retained_chunks.append(retained)
        self.retained_count += retained.size
        while self.retained_count > PERCENTILE_SAMPLE_BUDGET:
            decimated = np.concatenate(self.retained_chunks)[::2]
            self.retained_chunks = [decimated]
            self.retained_count = decimated.size
            self.retention_stride *= 2

    def mean(self):
        return self.total / self.count

    def standard_deviation(self):
        return math.sqrt(max(self.total_of_squares / self.count - self.mean() ** 2, 0.0))

    def percentiles(self):
        return np.percentile(np.concatenate(self.retained_chunks), (1.0, 50.0, 99.0))


def record(family_statistics, name, values):
    if name not in family_statistics:
        family_statistics[name] = FeatureStatistics(name)
    family_statistics[name].add(values)


def record_agent_families(family_statistics, block_name, valid_rows):
    positions = valid_rows[:, contract.AGENT_POSITION]
    velocities = valid_rows[:, contract.AGENT_VELOCITY]
    record(family_statistics, f"{block_name} position x", positions[:, 0])
    record(family_statistics, f"{block_name} position y", positions[:, 1])
    record(family_statistics, f"{block_name} position xy", positions)
    record(family_statistics, f"{block_name} velocity x", velocities[:, 0])
    record(family_statistics, f"{block_name} velocity y", velocities[:, 1])
    record(family_statistics, f"{block_name} velocity xy", velocities)
    record(family_statistics, f"{block_name} dimensions", valid_rows[:, contract.AGENT_DIMENSIONS])
    record(family_statistics, f"{block_name} heading cosine", valid_rows[:, contract.AGENT_HEADING_COSINE])
    record(family_statistics, f"{block_name} heading sine", valid_rows[:, contract.AGENT_HEADING_SINE])


def record_map_families(family_statistics, map_rows):
    positions = map_rows[:, contract.MAP_POSITION]
    directions = map_rows[:, contract.MAP_DIRECTION]
    speed_limits = map_rows[:, contract.MAP_SPEED_LIMIT]
    stop_points = map_rows[:, contract.MAP_STOP_POINT]
    record(family_statistics, "map position x", positions[:, 0])
    record(family_statistics, "map position y", positions[:, 1])
    record(family_statistics, "map position xy", positions)
    record(family_statistics, "map direction xy", directions)
    record(family_statistics, "map direction length", np.linalg.norm(directions, axis=1))
    record(family_statistics, "map speed limit non-zero", speed_limits[speed_limits != 0.0])
    record(family_statistics, "map stop point non-zero", stop_points[np.any(stop_points != 0.0, axis=1)])


def main():
    staged_directory = Path(sys.argv[1])
    scenario_paths = sorted(staged_directory.glob("*.npz"))
    family_statistics = {}
    sample_count = 0
    agent_history_steps = 0
    masked_agent_history_steps = 0
    neighbour_history_steps = 0
    masked_neighbour_history_steps = 0
    future_steps = 0
    masked_future_steps = 0
    map_row_count = 0
    map_rows_with_speed_limit = 0
    map_rows_with_stop_point = 0

    start_seconds = time.perf_counter()
    for scenario_path in scenario_paths:
        scenario_array = loader.read_scenario(scenario_path)
        eligible = loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            True,
        )
        for track_index in eligible:
            sample = loader.build_sample(scenario_array, int(track_index))
            sample_count += 1

            agent_history_mask = sample["agent_history_mask"]
            agent_history_steps += agent_history_mask.size
            masked_agent_history_steps += int((~agent_history_mask).sum())
            record_agent_families(
                family_statistics, "agent", sample["agent_history"][agent_history_mask]
            )

            neighbour_history_mask = sample["neighbour_history_mask"]
            neighbour_history_steps += neighbour_history_mask.size
            masked_neighbour_history_steps += int((~neighbour_history_mask).sum())
            record_agent_families(
                family_statistics, "neighbour", sample["neighbour_history"][neighbour_history_mask]
            )

            map_rows = sample["map_rows"]
            map_row_count += map_rows.shape[0]
            map_rows_with_speed_limit += int(np.count_nonzero(map_rows[:, contract.MAP_SPEED_LIMIT]))
            map_rows_with_stop_point += int(
                np.count_nonzero(np.any(map_rows[:, contract.MAP_STOP_POINT] != 0.0, axis=1))
            )
            record_map_families(family_statistics, map_rows)

            future_mask = sample["future_mask"]
            future_steps += future_mask.size
            masked_future_steps += int((~future_mask).sum())
            valid_future_positions = sample["future_positions"][future_mask]
            record(family_statistics, "future position x", valid_future_positions[:, 0])
            record(family_statistics, "future position y", valid_future_positions[:, 1])
            record(family_statistics, "future position xy", valid_future_positions)
            if valid_future_positions.shape[0]:
                record(
                    family_statistics,
                    "future final displacement",
                    np.linalg.norm(valid_future_positions[-1]),
                )
    elapsed_seconds = time.perf_counter() - start_seconds

    print(f"scenarios {len(scenario_paths)} | samples {sample_count} | {elapsed_seconds:.1f} s")
    print(
        f"agent history steps {agent_history_steps} | masked out {masked_agent_history_steps} "
        f"({masked_agent_history_steps / agent_history_steps:.1%})"
    )
    print(
        f"neighbour history steps {neighbour_history_steps} | masked out {masked_neighbour_history_steps} "
        f"({masked_neighbour_history_steps / neighbour_history_steps:.1%})"
    )
    print(
        f"future steps {future_steps} | masked out {masked_future_steps} "
        f"({masked_future_steps / future_steps:.1%})"
    )
    print(
        f"map rows {map_row_count} | non-zero speed limit {map_rows_with_speed_limit} "
        f"({map_rows_with_speed_limit / map_row_count:.1%}) | non-zero stop point "
        f"{map_rows_with_stop_point} ({map_rows_with_stop_point / map_row_count:.1%})"
    )
    print()

    print(
        f"{'family':<27}{'count':>12}{'mean':>10}{'std':>10}{'p1':>10}"
        f"{'p50':>10}{'p99':>10}{'max abs':>10}{'kept':>10}"
    )
    for statistics in family_statistics.values():
        p1, p50, p99 = statistics.percentiles()
        print(
            f"{statistics.name:<27}{statistics.count:>12}{statistics.mean():>10.3f}"
            f"{statistics.standard_deviation():>10.3f}{p1:>10.3f}{p50:>10.3f}{p99:>10.3f}"
            f"{statistics.maximum_absolute:>10.3f}{statistics.retained_count:>10}"
        )
    print()

    flagged = [
        statistics.name
        for statistics in family_statistics.values()
        if statistics.maximum_absolute > FP16_SQUARE_SAFETY_LIMIT
    ]
    print(f"max abs over {FP16_SQUARE_SAFETY_LIMIT:.0f} (squaring nears the fp16 ceiling 65504): "
          f"{', '.join(flagged) if flagged else 'none'}")


if __name__ == "__main__":
    main()
