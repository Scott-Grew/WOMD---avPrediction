import womd.runtime_env
import sys
from pathlib import Path

import numpy as np
import torch

from womd import contract, loader, model


def main():
    staged_directory = Path(sys.argv[1])
    breakers_path = staged_directory.parent / f"{staged_directory.name}_budget_breakers.npz"

    fraction_endpoints = []
    type_indices = []
    breakers = []
    for scenario_path in sorted(staged_directory.glob("*.npz")):
        scenario_array = loader.read_scenario(scenario_path)
        eligible = loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            True,
        )
        for track_index in eligible:
            sample = loader.build_sample(scenario_array, int(track_index))
            valid_future_steps = np.flatnonzero(sample["future_mask"])
            if valid_future_steps.size == 0:
                continue
            agent_history = torch.from_numpy(sample["agent_history"][None])
            budget = float(model.agent_reachable_distance(agent_history)[0])
            endpoint = sample["future_positions"][valid_future_steps[-1]]
            fraction = endpoint / budget
            fraction_endpoints.append(fraction)
            type_indices.append(int(model.predicted_type_index(agent_history)[0]))
            if float(np.linalg.norm(fraction)) > 1.0:
                speed = float(np.linalg.norm(
                    sample["agent_history"][contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY]
                ))
                breakers.append({
                    "scenario_id": str(sample["scenario_id"]),
                    "track_id": int(sample["track_id"]),
                    "type_index": type_indices[-1],
                    "speed": speed,
                    "budget": budget,
                    "endpoint_fraction": float(np.linalg.norm(fraction)),
                    "future": sample["future_positions"][valid_future_steps],
                })

    print(f"{len(fraction_endpoints)} endpoints measured")

    print(f"\n{len(breakers)} samples end past their own reachable budget:")
    for breaker in sorted(breakers, key=lambda entry: -entry["endpoint_fraction"]):
        future = breaker["future"]
        step_jumps = np.linalg.norm(np.diff(future, axis=0), axis=1)
        print(
            f"  scenario {breaker['scenario_id']} track {breaker['track_id']}"
            f" {contract.PREDICTED_OBJECT_TYPES[breaker['type_index']]}"
            f" | speed {breaker['speed']:.2f} m/s | budget {breaker['budget']:.1f} m"
            f" | endpoint {breaker['endpoint_fraction']:.3f}x budget"
            f" | largest single-step jump {step_jumps.max():.1f} m"
            f" | steps recorded {len(future)}"
        )
    np.savez(
        breakers_path,
        provenance=contract.artifact_provenance("measure_budget_breakers.py", staged_directory),
        **{
            f"future_{rank}": breaker["future"]
            for rank, breaker in enumerate(sorted(breakers, key=lambda entry: -entry["endpoint_fraction"]))
        },
    )
    print(f"trajectories saved: {breakers_path}")


if __name__ == "__main__":
    main()
