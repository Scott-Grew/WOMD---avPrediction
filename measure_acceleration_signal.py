import womd.runtime_env
import sys
from pathlib import Path

import numpy as np

from womd import contract, loader

TURN_QUANTILE_COUNT = 5
LATE_WINDOW_START_STEP = 60


def history_acceleration(agent_history, agent_history_mask):
    velocities = agent_history[:, contract.AGENT_VELOCITY]
    speeds = np.linalg.norm(velocities, axis=1)
    valid = np.flatnonzero(agent_history_mask)
    if valid.size < 2:
        return 0.0
    span_seconds = (valid[-1] - valid[0]) * contract.TIMESTEP_SECONDS
    return float((speeds[valid[-1]] - speeds[valid[0]]) / max(span_seconds, contract.TIMESTEP_SECONDS))


def lane_speed_limit(sample):
    speed_limits = sample["map_rows"][:, contract.MAP_SPEED_LIMIT]
    present = speed_limits[speed_limits != 0.0]
    return float(np.median(present)) if present.size else 0.0


def turn_magnitude(sample):
    current = sample["agent_history"][contract.CURRENT_STEP_INDEX]
    current_angle = np.arctan2(
        current[contract.AGENT_HEADING_SINE], current[contract.AGENT_HEADING_COSINE]
    )
    valid = np.flatnonzero(sample["future_mask"])
    final = sample["future_headings"][valid[-1]]
    difference = np.arctan2(final[1], final[0]) - current_angle
    return float(abs(np.arctan2(np.sin(difference), np.cos(difference))))


def late_window_speed(sample):
    positions = sample["future_positions"]
    valid = sample["future_mask"].astype(bool)
    steps = np.diff(positions[LATE_WINDOW_START_STEP:], axis=0)
    unbroken = valid[LATE_WINDOW_START_STEP + 1:] & valid[LATE_WINDOW_START_STEP:-1]
    if not unbroken.any():
        return None
    return float(np.linalg.norm(steps[unbroken], axis=1).mean() / contract.TIMESTEP_SECONDS)


def main():
    staged_directory = Path(sys.argv[1])
    rows = []
    for scenario_path in sorted(staged_directory.glob("*.npz")):
        scenario_array = loader.read_scenario(scenario_path)
        for track_index in loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            True,
        ):
            sample = loader.build_sample(scenario_array, int(track_index))
            if not sample["future_mask"].all():
                continue
            late_speed = late_window_speed(sample)
            if late_speed is None:
                continue
            current = sample["agent_history"][contract.CURRENT_STEP_INDEX]
            rows.append((
                float(np.linalg.norm(current[contract.AGENT_VELOCITY])),
                history_acceleration(sample["agent_history"], sample["agent_history_mask"]),
                lane_speed_limit(sample),
                turn_magnitude(sample),
                late_speed,
            ))

    table = np.array(rows)
    current_speed, history_accel, speed_limit, turn, late_speed = table.T
    turn_edges = np.quantile(turn, np.linspace(0.0, 1.0, TURN_QUANTILE_COUNT + 1))
    turn_edges[-1] = np.inf

    print(f"{staged_directory}, {len(rows)} complete-track designated targets")
    print(f"target = mean speed over the last 2 s of the logged future"
          f" (steps {LATE_WINDOW_START_STEP}-79)")
    print(f"\n{'turn bucket':22s}{'n':>6s}{'late speed':>12s}{'r vs v0':>10s}"
          f"{'r vs accel':>12s}{'r vs limit':>12s}{'r all three':>13s}")
    for bucket in range(TURN_QUANTILE_COUNT):
        selected = (turn >= turn_edges[bucket]) & (turn < turn_edges[bucket + 1])
        if selected.sum() < 3:
            continue
        target = late_speed[selected]
        predictors = np.column_stack([
            current_speed[selected], history_accel[selected], speed_limit[selected]
        ])
        fitted = predictors @ np.linalg.lstsq(
            np.column_stack([predictors, np.ones(selected.sum())]), target, rcond=None
        )[0][:3]
        fitted = fitted + np.linalg.lstsq(
            np.column_stack([predictors, np.ones(selected.sum())]), target, rcond=None
        )[0][3]
        print(f"  {turn_edges[bucket]:.2f}-{turn_edges[bucket + 1]:.2f} rad"
              f"{int(selected.sum()):>6d}{target.mean():12.2f}"
              f"{np.corrcoef(current_speed[selected], target)[0, 1]:10.3f}"
              f"{np.corrcoef(history_accel[selected], target)[0, 1]:12.3f}"
              f"{np.corrcoef(speed_limit[selected], target)[0, 1]:12.3f}"
              f"{np.corrcoef(fitted, target)[0, 1]:13.3f}")


if __name__ == "__main__":
    main()
