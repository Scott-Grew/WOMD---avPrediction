import sys
from pathlib import Path

import torch

from womd import contract, pipeline
from womd.model import MotionPredictor

BATCH_SIZE = 32
FIXED_SEED = 0
CONTROL_NOISE_METRES = 1e-6


def count_modified_rows(original, modified):
    changed = (original != modified).reshape(-1, original.shape[-1]).any(dim=-1)
    return int(changed.sum())


def perturb_control_noise(batch):
    modified = dict(batch)
    modified["map_rows"] = batch["map_rows"] + CONTROL_NOISE_METRES
    return modified, "map_rows", count_modified_rows(batch["map_rows"], modified["map_rows"])


def perturb_delete_map(batch):
    modified = dict(batch)
    modified["map_rows"] = torch.zeros_like(batch["map_rows"])
    return modified, "map_rows", count_modified_rows(batch["map_rows"], modified["map_rows"])


def perturb_agent_speed(batch):
    modified = dict(batch)
    modified_agent_history = batch["agent_history"].clone()
    modified_agent_history[..., contract.AGENT_VELOCITY] += 1.0
    modified["agent_history"] = modified_agent_history
    return modified, "agent_history", count_modified_rows(batch["agent_history"], modified_agent_history)


def perturb_delete_neighbours(batch):
    modified = dict(batch)
    modified["neighbour_history_mask"] = torch.zeros_like(batch["neighbour_history_mask"])
    return modified, "neighbour_history_mask", count_modified_rows(
        batch["neighbour_history_mask"], modified["neighbour_history_mask"]
    )


def perturb_shuffle_map_positions(batch, random_generator):
    modified = dict(batch)
    modified_map_rows = batch["map_rows"].clone()
    permutation = torch.randperm(batch["map_rows"].shape[0], generator=random_generator)
    modified_map_rows[:, contract.MAP_POSITION] = batch["map_rows"][permutation][:, contract.MAP_POSITION]
    modified["map_rows"] = modified_map_rows
    return modified, "map_rows", count_modified_rows(
        batch["map_rows"][:, contract.MAP_POSITION], modified_map_rows[:, contract.MAP_POSITION]
    )


def perturb_force_signal_state(batch, signal_state_name):
    modified = dict(batch)
    modified_signal_history = batch["map_chunk_signal_history"].clone()
    live_chunk_mask = batch["map_chunk_signal_history"].any(dim=(-2, -1))
    forced_one_hot = torch.zeros(contract.NUM_TRAFFIC_SIGNAL_STATES, dtype=modified_signal_history.dtype)
    forced_one_hot[contract.TRAFFIC_SIGNAL_STATES.index(signal_state_name)] = 1.0
    modified_signal_history[live_chunk_mask] = forced_one_hot
    modified["map_chunk_signal_history"] = modified_signal_history
    return modified, "map_chunk_signal_history", count_modified_rows(
        batch["map_chunk_signal_history"], modified_signal_history
    )


def perturb_delete_signal_history(batch):
    modified = dict(batch)
    modified["map_chunk_signal_history"] = torch.zeros_like(batch["map_chunk_signal_history"])
    return modified, "map_chunk_signal_history", count_modified_rows(
        batch["map_chunk_signal_history"], modified["map_chunk_signal_history"]
    )


def mean_metres_moved(base_trajectories, perturbed_trajectories):
    return float((perturbed_trajectories - base_trajectories).norm(dim=-1).mean())


def main():
    checkpoint_path = Path(sys.argv[1])
    staged_directory = Path(sys.argv[2])

    torch.manual_seed(FIXED_SEED)
    predictor = MotionPredictor()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    predictor.load_state_dict(checkpoint["model_state"])
    predictor.eval()

    scenario_paths = sorted(staged_directory.glob("*.npz"))
    assert scenario_paths, f"no .npz scenarios in {staged_directory}"
    batch = next(iter(pipeline.batches(scenario_paths, 0, BATCH_SIZE, 0, FIXED_SEED, True)))
    random_generator = torch.Generator().manual_seed(FIXED_SEED)

    with torch.no_grad():
        base_trajectories, _ = predictor(batch)

    perturbations = (
        ("control noise 1e-6 on map_rows", lambda: perturb_control_noise(batch)),
        ("delete map", lambda: perturb_delete_map(batch)),
        ("agent speed +1 m/s", lambda: perturb_agent_speed(batch)),
        ("delete neighbours", lambda: perturb_delete_neighbours(batch)),
        ("shuffle map positions", lambda: perturb_shuffle_map_positions(batch, random_generator)),
        ("force signals to LANE_STATE_STOP", lambda: perturb_force_signal_state(batch, "LANE_STATE_STOP")),
        ("force signals to LANE_STATE_GO", lambda: perturb_force_signal_state(batch, "LANE_STATE_GO")),
        ("delete signal history", lambda: perturb_delete_signal_history(batch)),
    )

    print(f"{'perturbation':<34}{'rows modified':>15}{'mean metres moved':>20}")
    for name, perturbation in perturbations:
        modified_batch, target_tensor_name, rows_modified = perturbation()
        assert not torch.equal(batch[target_tensor_name], modified_batch[target_tensor_name])
        assert rows_modified > 0
        with torch.no_grad():
            perturbed_trajectories, _ = predictor(modified_batch)
        print(
            f"{name:<34}{rows_modified:>15}"
            f"{mean_metres_moved(base_trajectories, perturbed_trajectories):>20.6f}"
        )


if __name__ == "__main__":
    main()
