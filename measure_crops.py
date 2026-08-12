# > Loader crop measurement: over a staged directory, for each eligible sample (predicted type,
# valid at "now"), count map dots inside the speed-stretched half-ellipse crop and measure how
# much of the agent's logged future the crop covers, across a grid of base radii and stretch
# gains. Also reports neighbour-count distribution (informative only - neighbours ship uncapped).
# Peeks at logged futures: legitimate for an input-selection knob the model never receives.
# Run: python3 measure_crops.py ../data/staged

import sys
from pathlib import Path

import numpy as np

from womd import contract, frame_ops
from womd.loader import inside_crop, eligible_track_indices, sample_frame

BASE_RADII_METRES = (40.0, 60.0, 80.0, 100.0)
STRETCH_GAINS = (0.0, 0.5, 0.75, 1.0)
SPEED_BANDS = ((0.0, 1.0), (1.0, 8.0), (8.0, np.inf))


def main():
    staged_directory = Path(sys.argv[1])
    dot_counts = {(radius, gain): [] for radius in BASE_RADII_METRES for gain in STRETCH_GAINS}
    covered = {(radius, gain, band): [0, 0] for radius in BASE_RADII_METRES for gain in STRETCH_GAINS for band in SPEED_BANDS}
    neighbour_counts = []

    for scenario_path in sorted(staged_directory.glob("*.npz")):
        scenario_file = np.load(scenario_path)
        track_rows = scenario_file["track_rows"]
        track_valid = scenario_file["track_valid"]
        map_points = scenario_file["map_rows"][:, contract.MAP_POSITION]
        neighbour_counts.append(len(track_rows))

        for track_index in eligible_track_indices(track_rows, track_valid):
            now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
            speed = float(np.linalg.norm(now_row[contract.AGENT_VELOCITY]))
            band = next(b for b in SPEED_BANDS if b[0] <= speed < b[1])

            origin, heading = sample_frame(track_rows, track_index)
            agent_map = frame_ops.positions_to_agent_frame(map_points, origin, heading)
            future_valid = track_valid[track_index, contract.CURRENT_STEP_INDEX + 1:]
            future_positions = track_rows[track_index, contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION][future_valid]
            agent_future = frame_ops.positions_to_agent_frame(future_positions, origin, heading)

            for radius in BASE_RADII_METRES:
                for gain in STRETCH_GAINS:
                    stretch = 1.0 + gain * speed
                    dot_counts[(radius, gain)].append(int(np.sum(inside_crop(agent_map, radius, stretch))))
                    inside_future = inside_crop(agent_future, radius, stretch)
                    covered[(radius, gain, band)][0] += int(np.sum(inside_future))
                    covered[(radius, gain, band)][1] += len(inside_future)

    sample_count = len(dot_counts[(BASE_RADII_METRES[0], STRETCH_GAINS[0])])
    print(f"samples: {sample_count} | scenarios: {len(neighbour_counts)}")
    print(f"agents per scenario: p50 {np.percentile(neighbour_counts, 50):.0f}"
          f" | p90 {np.percentile(neighbour_counts, 90):.0f} | max {max(neighbour_counts)}")
    print()
    header = "radius  k     dots p50   dots p90   dots max   cov stationary  cov 1-8m/s  cov >8m/s  cov all"
    print(header)
    for radius in BASE_RADII_METRES:
        for gain in STRETCH_GAINS:
            counts = np.array(dot_counts[(radius, gain)])
            band_fractions = []
            total_inside, total_points = 0, 0
            for band in SPEED_BANDS:
                inside, points = covered[(radius, gain, band)]
                band_fractions.append(inside / points if points else float("nan"))
                total_inside += inside
                total_points += points
            print(f"{radius:5.0f}  {gain:4.2f}  {np.percentile(counts, 50):9.0f}"
                  f"  {np.percentile(counts, 90):9.0f}  {counts.max():9d}"
                  f"  {band_fractions[0]:14.1%}  {band_fractions[1]:10.1%}"
                  f"  {band_fractions[2]:9.1%}  {total_inside / total_points:7.1%}")


if __name__ == "__main__":
    main()
