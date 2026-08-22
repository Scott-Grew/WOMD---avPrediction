# > Staging-parameter measurement: reads a RAW WOMD shard, never the staged .npz - staging has
# already applied its crop radius, so staged files cannot answer whether that radius is too
# small. For WOMD's designated targets and for every eligible agent separately, reports how far
# each sits from the self-driving car at "now" and at its last logged future step, how far the
# loader's speed-stretched crop reaches from the self-driving car, and how many of the map dots
# that crop wants survive a staging radius of 150/200/250/300/400 m. Reports how much map
# Waymo's own scenario extent supplies at 100-500 m, so a larger radius can be judged against
# what is actually there. Reports what resampling the same polylines at --coarse-spacing-metres
# instead of --fine-spacing-metres costs: dots per scenario, dots inside a target's crop, and the
# distance from each dropped dot to the coarser polyline that replaces it.
# Crop reach is the maximum distance from the self-driving car reached by any point of the crop
# region, taken over CROP_BOUNDARY_SAMPLES points on its boundary; the region is convex, so its
# farthest point from an outside origin lies on that boundary.
# Reports only - no constant is chosen, derived or written anywhere here.
# Run: python3 measure_staging.py ../data/validation_raw/validation.tfrecord-00000-of-00150 --fine-spacing-metres
# 0.5 --coarse-spacing-metres 1.0

import womd.runtime_env
import argparse
import time
from pathlib import Path

import numpy as np

from womd import contract, frame_ops, loader, store, tfrecord
from womd_protos import scenario_pb2

STAGING_RADIUS_PROBES = (150.0, 200.0, 250.0, 300.0, 400.0)
MAP_EXTENT_RADII_METRES = (100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0)
CROP_BOUNDARY_SAMPLES = 2048
DESIGNATED_TARGETS = "designated targets"
ALL_ELIGIBLE_AGENTS = "all eligible agents"
POPULATION_NAMES = (DESIGNATED_TARGETS, ALL_ELIGIBLE_AGENTS)


class PopulationTallies:
    def __init__(self):
        self.current_step_distances = []
        self.final_future_distances = []
        self.crop_reach_distances = []
        self.wanted_dot_count = 0
        self.kept_dot_counts = {radius: 0 for radius in STAGING_RADIUS_PROBES}
        self.lost_dot_fractions = {radius: [] for radius in STAGING_RADIUS_PROBES}

    def record(self, current_step_distance, final_future_distance, crop_reach_distance, wanted_dot_distances):
        self.current_step_distances.append(current_step_distance)
        self.final_future_distances.append(final_future_distance)
        self.crop_reach_distances.append(crop_reach_distance)
        self.wanted_dot_count += wanted_dot_distances.size
        for radius in STAGING_RADIUS_PROBES:
            kept_dot_count = int(np.count_nonzero(wanted_dot_distances <= radius))
            self.kept_dot_counts[radius] += kept_dot_count
            self.lost_dot_fractions[radius].append(
                1.0 - kept_dot_count / wanted_dot_distances.size if wanted_dot_distances.size else 0.0
            )


def point_to_segment_distances(query_points, segment_starts, segment_ends):
    segment_vectors = segment_ends - segment_starts
    squared_lengths = np.sum(segment_vectors ** 2, axis=1)
    squared_lengths[squared_lengths == 0.0] = 1.0
    projected_fractions = np.clip(
        np.sum((query_points - segment_starts) * segment_vectors, axis=1) / squared_lengths, 0.0, 1.0
    )
    closest_points = segment_starts + projected_fractions[:, None] * segment_vectors
    return np.linalg.norm(query_points - closest_points, axis=1)


def dropped_dot_deviations(fine_points, coarse_points):
    if len(fine_points) < 3 or len(coarse_points) < 2:
        return np.zeros(0)
    dropped_points = fine_points[1:len(fine_points) - 1:2]
    if len(dropped_points) == 0:
        return np.zeros(0)
    bracketing_indices = np.arange(len(dropped_points))
    deviations = np.full(len(dropped_points), np.inf)
    for neighbour_offset in (-1, 0, 1):
        segment_indices = np.clip(bracketing_indices + neighbour_offset, 0, len(coarse_points) - 2)
        deviations = np.minimum(
            deviations,
            point_to_segment_distances(
                dropped_points, coarse_points[segment_indices], coarse_points[segment_indices + 1]
            ),
        )
    return deviations


def scenario_map_geometry(scenario, origin, heading, fine_spacing_metres, coarse_spacing_metres):
    fine_blocks = []
    coarse_blocks = []
    deviation_blocks = []
    for feature in scenario.map_features:
        raw_points = store.map_feature_points(feature)[0]
        if raw_points is None:
            continue
        fine_points = store.points_along_polyline(raw_points, fine_spacing_metres)
        coarse_points = store.points_along_polyline(raw_points, coarse_spacing_metres)
        fine_blocks.append(frame_ops.positions_to_agent_frame(fine_points, origin, heading))
        coarse_blocks.append(frame_ops.positions_to_agent_frame(coarse_points, origin, heading))
        deviation_blocks.append(dropped_dot_deviations(fine_points, coarse_points))
    if not fine_blocks:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    return np.concatenate(fine_blocks), np.concatenate(coarse_blocks), np.concatenate(deviation_blocks)


def crop_boundary_points(base_radius, forward_stretch):
    half_turn_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, CROP_BOUNDARY_SAMPLES)
    rear_arc = np.stack(
        [-base_radius * np.cos(half_turn_angles), base_radius * np.sin(half_turn_angles)], axis=1
    )
    forward_arc = np.stack(
        [base_radius * forward_stretch * np.cos(half_turn_angles), base_radius * np.sin(half_turn_angles)],
        axis=1,
    )
    return np.concatenate([rear_arc, forward_arc])


def distribution_row(label, values, exceeded_radius):
    finite_values = np.asarray(values)
    finite_values = finite_values[np.isfinite(finite_values)]
    p50, p90, p99 = np.percentile(finite_values, (50.0, 90.0, 99.0))
    exceeding_fraction = np.count_nonzero(finite_values > exceeded_radius) / finite_values.size
    return (f"{label:<17}{finite_values.size:>9}{p50:>9.1f}{p90:>9.1f}{p99:>9.1f}"
            f"{finite_values.max():>10.1f}{exceeding_fraction:>11.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_path", type=Path)
    parser.add_argument("--fine-spacing-metres", type=float, required=True)
    parser.add_argument("--coarse-spacing-metres", type=float, required=True)
    arguments = parser.parse_args()
    shard_path = arguments.shard_path
    fine_spacing_metres = arguments.fine_spacing_metres
    coarse_spacing_metres = arguments.coarse_spacing_metres

    tallies = {name: PopulationTallies() for name in POPULATION_NAMES}
    map_dots_within_radius = {radius: [] for radius in MAP_EXTENT_RADII_METRES}
    map_extent_maxima = []
    spacings = (fine_spacing_metres, coarse_spacing_metres)
    dots_per_scenario = {spacing: [] for spacing in spacings}
    crop_dots_per_target = {spacing: [] for spacing in spacings}
    staged_crop_dots_per_target = {spacing: [] for spacing in spacings}
    deviation_blocks = []
    scenario_count = 0
    designated_but_not_eligible = 0
    targets_without_logged_future = 0

    start_seconds = time.perf_counter()
    for scenario in tfrecord.read_scenarios(shard_path, scenario_pb2.Scenario):
        scenario_count += 1
        origin, heading = store.scenario_storage_frame(scenario)
        track_rows, track_valid = store.scenario_track_arrays(scenario)
        is_designated_target = store.scenario_track_labels(scenario)[1]
        fine_map_positions, coarse_map_positions, feature_deviations = scenario_map_geometry(
            scenario, origin, heading, fine_spacing_metres, coarse_spacing_metres
        )
        deviation_blocks.append(feature_deviations)
        dots_per_scenario[fine_spacing_metres].append(len(fine_map_positions))
        dots_per_scenario[coarse_spacing_metres].append(len(coarse_map_positions))

        fine_map_distances = np.linalg.norm(fine_map_positions, axis=1)
        coarse_map_distances = np.linalg.norm(coarse_map_positions, axis=1)
        for radius in MAP_EXTENT_RADII_METRES:
            map_dots_within_radius[radius].append(int(np.count_nonzero(fine_map_distances <= radius)))
        map_extent_maxima.append(float(fine_map_distances.max()))

        eligible_indices = loader.eligible_track_indices(
            track_rows, track_valid, is_designated_target, False
        )
        designated_but_not_eligible += int(
            np.count_nonzero(is_designated_target) - np.count_nonzero(is_designated_target[eligible_indices])
        )

        for track_index in eligible_indices:
            agent_origin, agent_heading = loader.sample_frame(track_rows, track_index)
            current_step_distance = float(np.linalg.norm(agent_origin))

            future_valid = track_valid[track_index, contract.CURRENT_STEP_INDEX + 1:]
            future_positions = track_rows[
                track_index, contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION
            ][future_valid]
            if len(future_positions):
                final_future_distance = float(np.linalg.norm(future_positions[-1]))
            else:
                final_future_distance = float("nan")
                targets_without_logged_future += 1

            speed = float(np.linalg.norm(track_rows[track_index, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY]))
            forward_stretch = 1.0 + loader.STRETCH_GAIN * speed
            crop_boundary = frame_ops.positions_to_world_frame(
                crop_boundary_points(loader.BASE_RADIUS_METRES, forward_stretch), agent_origin, agent_heading
            )
            crop_reach_distance = float(np.linalg.norm(crop_boundary, axis=1).max())

            fine_agent_frame_map = frame_ops.positions_to_agent_frame(
                fine_map_positions, agent_origin, agent_heading
            )
            fine_wanted = loader.inside_crop(
                fine_agent_frame_map, loader.BASE_RADIUS_METRES, forward_stretch
            )
            fine_wanted_distances = fine_map_distances[fine_wanted]

            tallies[ALL_ELIGIBLE_AGENTS].record(
                current_step_distance, final_future_distance, crop_reach_distance, fine_wanted_distances
            )
            if not is_designated_target[track_index]:
                continue

            tallies[DESIGNATED_TARGETS].record(
                current_step_distance, final_future_distance, crop_reach_distance, fine_wanted_distances
            )
            coarse_agent_frame_map = frame_ops.positions_to_agent_frame(
                coarse_map_positions, agent_origin, agent_heading
            )
            coarse_wanted = loader.inside_crop(
                coarse_agent_frame_map, loader.BASE_RADIUS_METRES, forward_stretch
            )
            coarse_wanted_distances = coarse_map_distances[coarse_wanted]
            crop_dots_per_target[fine_spacing_metres].append(fine_wanted_distances.size)
            crop_dots_per_target[coarse_spacing_metres].append(coarse_wanted_distances.size)
            staged_crop_dots_per_target[fine_spacing_metres].append(
                int(np.count_nonzero(fine_wanted_distances <= contract.STAGING_CROP_RADIUS_METRES))
            )
            staged_crop_dots_per_target[coarse_spacing_metres].append(
                int(np.count_nonzero(coarse_wanted_distances <= contract.STAGING_CROP_RADIUS_METRES))
            )
    elapsed_seconds = time.perf_counter() - start_seconds

    designated_count = len(tallies[DESIGNATED_TARGETS].current_step_distances)
    eligible_count = len(tallies[ALL_ELIGIBLE_AGENTS].current_step_distances)
    print(f"shard {shard_path.name} | scenarios {scenario_count} | {elapsed_seconds:.1f} s")
    print(
        f"designated targets {designated_count} ({designated_count / scenario_count:.2f} per scenario)"
        f" | eligible agents {eligible_count} ({eligible_count / scenario_count:.2f} per scenario)"
    )
    print(
        f"designated tracks not eligible at now {designated_but_not_eligible}"
        f" | eligible agents with no logged future step {targets_without_logged_future}"
    )
    print(
        f"staging radius in force {contract.STAGING_CROP_RADIUS_METRES:.0f} m"
        f" | loader crop {loader.BASE_RADIUS_METRES:.0f} m base, stretch 1 + {loader.STRETCH_GAIN} * speed"
    )
    print()

    print("distance from the self-driving car in the storage frame, metres")
    print(f"{'population':<24}{'quantity':<17}{'count':>9}{'p50':>9}{'p90':>9}{'p99':>9}"
          f"{'max':>10}{'over 250 m':>11}")
    for population_name in POPULATION_NAMES:
        population = tallies[population_name]
        for label, values in (
            ("at now", population.current_step_distances),
            ("at last future", population.final_future_distances),
            ("crop reach", population.crop_reach_distances),
        ):
            print(f"{population_name:<24}"
                  + distribution_row(label, values, contract.STAGING_CROP_RADIUS_METRES))
    print()

    print("map dots the loader crop wants, against the staging radius that would hold them")
    print(f"{'population':<24}{'radius':>8}{'dots kept':>12}{'dots wanted':>13}{'dots lost':>11}"
          f"{'per-target lost p50':>21}{'p90':>8}{'max':>8}{'losing any':>12}")
    for population_name in POPULATION_NAMES:
        population = tallies[population_name]
        for radius in STAGING_RADIUS_PROBES:
            lost_fractions = np.array(population.lost_dot_fractions[radius])
            kept_dot_count = population.kept_dot_counts[radius]
            p50, p90 = np.percentile(lost_fractions, (50.0, 90.0))
            print(f"{population_name:<24}{radius:>8.0f}{kept_dot_count:>12}"
                  f"{population.wanted_dot_count:>13}"
                  f"{1.0 - kept_dot_count / population.wanted_dot_count:>11.1%}"
                  f"{p50:>21.1%}{p90:>8.1%}{lost_fractions.max():>8.1%}"
                  f"{np.count_nonzero(lost_fractions > 0.0) / lost_fractions.size:>12.1%}")
    print()

    print(f"map dots Waymo's own scenario extent supplies, per scenario, "
          f"{fine_spacing_metres} m spacing")
    print(f"{'radius':>8}{'dots p50':>11}{'dots mean':>12}{'share of 500 m':>16}")
    dots_within_500 = np.mean(map_dots_within_radius[MAP_EXTENT_RADII_METRES[-1]])
    for radius in MAP_EXTENT_RADII_METRES:
        counts = np.array(map_dots_within_radius[radius])
        print(f"{radius:>8.0f}{np.percentile(counts, 50):>11.0f}{counts.mean():>12.0f}"
              f"{counts.mean() / dots_within_500:>16.1%}")
    extent_maxima = np.array(map_extent_maxima)
    print(f"farthest map dot from the self-driving car: p50 {np.percentile(extent_maxima, 50):.1f} m"
          f" | p90 {np.percentile(extent_maxima, 90):.1f} m | max {extent_maxima.max():.1f} m")
    print()

    print("map point spacing")
    all_deviations = np.concatenate(deviation_blocks)
    print(f"{'spacing':>8}{'dots/scenario p50':>19}{'mean':>10}{'crop dots p50':>15}{'p90':>9}{'max':>9}"
          f"{'crop dots within 250 m p50':>28}{'p90':>9}{'max':>9}")
    for spacing in spacings:
        scenario_counts = np.array(dots_per_scenario[spacing])
        crop_counts = np.array(crop_dots_per_target[spacing])
        staged_crop_counts = np.array(staged_crop_dots_per_target[spacing])
        print(f"{spacing:>8.1f}{np.percentile(scenario_counts, 50):>19.0f}{scenario_counts.mean():>10.0f}"
              f"{np.percentile(crop_counts, 50):>15.0f}{np.percentile(crop_counts, 90):>9.0f}"
              f"{crop_counts.max():>9}"
              f"{np.percentile(staged_crop_counts, 50):>28.0f}"
              f"{np.percentile(staged_crop_counts, 90):>9.0f}{staged_crop_counts.max():>9}")
    print(f"dropped dots {all_deviations.size} | distance to the {coarse_spacing_metres} m polyline "
          f"that replaces them: p50 {np.percentile(all_deviations, 50):.4f} m | "
          f"p99 {np.percentile(all_deviations, 99):.4f} m | max {all_deviations.max():.4f} m")


if __name__ == "__main__":
    main()
