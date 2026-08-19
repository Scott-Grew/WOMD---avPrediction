import sys
from pathlib import Path

import numpy as np

from womd import contract, loader

MOVING_SPEED_FLOOR_METRES_PER_SECOND = 0.5
CURVATURE_BUCKET_COUNT = 6


def logged_step_kinematics(sample):
    positions = sample["future_positions"]
    valid = sample["future_mask"].astype(bool)
    unbroken = valid[1:] & valid[:-1]
    steps = np.diff(positions, axis=0)
    speeds = np.linalg.norm(steps, axis=1) / contract.TIMESTEP_SECONDS
    directions = np.arctan2(steps[:, 1], steps[:, 0])
    turn = np.diff(directions)
    turn = np.arctan2(np.sin(turn), np.cos(turn))
    yaw_rates = turn / contract.TIMESTEP_SECONDS
    usable = unbroken[1:] & unbroken[:-1] & (speeds[1:] >= MOVING_SPEED_FLOOR_METRES_PER_SECOND)
    return speeds[1:][usable], np.abs(yaw_rates[usable])


def main():
    staged_directory = Path(sys.argv[1])
    speeds, yaw_rates = [], []
    for scenario_path in sorted(staged_directory.glob("*.npz")):
        scenario_array = loader.read_scenario(scenario_path)
        for track_index in loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            True,
        ):
            sample = loader.build_sample(scenario_array, int(track_index))
            step_speeds, step_yaw_rates = logged_step_kinematics(sample)
            speeds.append(step_speeds)
            yaw_rates.append(step_yaw_rates)

    speed = np.concatenate(speeds)
    yaw_rate = np.concatenate(yaw_rates)
    lateral_acceleration = speed * yaw_rate
    radius = np.divide(speed, yaw_rate, out=np.full_like(speed, np.inf), where=yaw_rate > 0.0)

    print(f"logged futures in {staged_directory}, designated targets only")
    print(f"{speed.size} steps at or above {MOVING_SPEED_FLOOR_METRES_PER_SECOND} m/s,"
          f" the same moving-step floor the longitudinal limit in contract.py was measured under")
    print(f"\nLATERAL ACCELERATION, speed x yaw rate, metres per second squared")
    for name, value in (
        ("p50", np.percentile(lateral_acceleration, 50)),
        ("p90", np.percentile(lateral_acceleration, 90)),
        ("p99", np.percentile(lateral_acceleration, 99)),
        ("p99.9", np.percentile(lateral_acceleration, 99.9)),
        ("max", lateral_acceleration.max()),
    ):
        print(f"  {name:8s}{value:10.3f}")
    print(f"  longitudinal limit already in contract.py for comparison:"
          f" {contract.MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED}")

    print(f"\nHOW TIGHTLY RADIUS DETERMINES SPEED — logged steps bucketed by turn radius")
    finite = np.isfinite(radius)
    bucket_edges = np.quantile(radius[finite], np.linspace(0.0, 1.0, CURVATURE_BUCKET_COUNT + 1))
    bucket_edges[-1] = np.inf
    print(f"  {'radius bucket':28s}{'steps':>10s}{'speed p10':>11s}{'speed p50':>11s}"
          f"{'speed p90':>11s}{'lat acc p90':>13s}")
    for bucket_index in range(CURVATURE_BUCKET_COUNT):
        selected = finite & (radius >= bucket_edges[bucket_index])
        selected &= radius < bucket_edges[bucket_index + 1]
        if not selected.any():
            continue
        label = (f"{bucket_edges[bucket_index]:.1f}-{bucket_edges[bucket_index + 1]:.1f} m"
                 if np.isfinite(bucket_edges[bucket_index + 1])
                 else f"{bucket_edges[bucket_index]:.1f}+ m")
        print(f"  {label:28s}{int(selected.sum()):10d}"
              f"{np.percentile(speed[selected], 10):11.2f}"
              f"{np.percentile(speed[selected], 50):11.2f}"
              f"{np.percentile(speed[selected], 90):11.2f}"
              f"{np.percentile(lateral_acceleration[selected], 90):13.3f}")

    tight = finite & (radius < np.percentile(radius[finite], 20.0))
    print(f"\ncorrelation of speed with radius over finite-radius steps:"
          f" {np.corrcoef(np.log(radius[finite]), speed[finite])[0, 1]:.3f} on log radius")
    print(f"speed p90 in the tightest fifth of radii: {np.percentile(speed[tight], 90):.2f} m/s"
          f" against {np.percentile(speed[finite], 90):.2f} m/s over all turning steps")


if __name__ == "__main__":
    main()
