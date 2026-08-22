# > Reachable-distance DIAGNOSTIC over the logged data: how much of its own reachable distance each
# logged future actually spends, read two ways - the ENDPOINT form (how far the 8 s position lands
# from the start) and the PATH form (how far the agent drives to get there). Nothing in the model
# bounds a predicted coordinate, so this is the reference distribution a prediction's own use is set
# beside, never a limit anything is held to.
# The reachable distance is the agent's own logged speed carried for the horizon plus the
# acceleration allowance §LIMITS measured, so nothing here is chosen. Only tracks valid at every
# future step are measured: a partial track's path length is missing the segments it did not record,
# which would read as an agent that drove less than it did.
# Run: python3 measure_fence.py ../data/staged

import womd.runtime_env
import sys
from pathlib import Path

import numpy as np

from womd import contract
from womd.loader import build_sample, eligible_track_indices, read_scenario


def agent_reachable_metres(agent_history):
    speed = np.linalg.norm(agent_history[contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY])
    return (
        speed * contract.FUTURE_HORIZON_SECONDS
        + 0.5 * contract.MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED
        * contract.FUTURE_HORIZON_SECONDS ** 2
    )


def main():
    staged_directory = Path(sys.argv[1])
    endpoint_use = []
    path_use = []
    partial_tracks = 0

    for scenario_path in sorted(staged_directory.glob("*.npz")):
        scenario_array = read_scenario(scenario_path)
        track_indices = eligible_track_indices(
            scenario_array["track_rows"], scenario_array["track_valid"],
            scenario_array["is_designated_target"], True,
        )
        for track_index in track_indices:
            sample = build_sample(scenario_array, track_index)
            if not sample["future_mask"].all():
                partial_tracks += 1
                continue
            reachable = agent_reachable_metres(sample["agent_history"])
            future = sample["future_positions"]
            walked = np.linalg.norm(
                np.diff(np.vstack([np.zeros(2), future]), axis=0), axis=1
            ).sum()
            endpoint_use.append(np.linalg.norm(future[-1]) / reachable)
            path_use.append(walked / reachable)

    endpoint_use = np.array(endpoint_use)
    path_use = np.array(path_use)
    print(f"{staged_directory}: {len(path_use)} designated targets with a complete 8 s track"
          f" ({partial_tracks} partial tracks skipped)")
    print(f"{'form':10s}{'p50':>8s}{'p90':>8s}{'p99':>8s}{'max':>8s}{'over 1.0':>10s}")
    for name, use in (("endpoint", endpoint_use), ("path", path_use)):
        print(f"{name:10s}{np.percentile(use, 50):8.3f}{np.percentile(use, 90):8.3f}"
              f"{np.percentile(use, 99):8.3f}{use.max():8.3f}"
              f"{int((use > 1.0).sum()):10d}")


if __name__ == "__main__":
    main()
