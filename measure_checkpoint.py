# > Checkpoint measurement: the comparable read on a trained model, so two runs can be set beside
# each other without a throwaway script in between. Reports the three numbers that separate what the
# model KNOWS from what it SUBMITS - the best of all its modes, the best of the six the confidence
# head chooses, and the best of six drawn at random - plus two diagnostics over an output nothing
# bounds: how far apart the modes spread, and how far the longest of them drives against the agent's
# own reachable distance. The second is a ratio to be read beside measure_fence.py's reading of the
# same ratio over the logged futures; no value of it is a failure by itself.
# The random draw is the control the confidence number is read against: a confidence head that
# cannot beat it has learned nothing about which mode to trust, whatever the loss curve says.
# Four nulls are the controls the MODEL is read against, on the same samples through the same path.
# Constant velocity carries the logged velocity for the whole horizon and CTRV carries the logged
# yaw rate with it: a model that cannot beat those has learned nothing about the agent's own motion.
# The anchor null drives the six most-used anchors fitted for the agent's OWN OBJECT TYPE in a
# straight line with the scene unread: a model that cannot beat it has learned nothing the anchors
# did not already carry. The lane null
# drives the lane the agent is already in, at the speed it is already going, out to six routes
# through the lane graph's forks: a model that cannot beat it has learned nothing from the map,
# the lane graph, the traffic lights or the neighbours it is handed.
# Every distance is masked to the steps the logged track actually recorded, and every one is reported
# TWICE: over all 80 logged future steps, which is what train.py's ade_80step monitor averages, and
# over the 16 decimated steps a submission actually carries. They are not the same number - the 16
# steps a submission sends are weighted toward the far end of the horizon, where the error is - so
# neither is printed without the other beside it.
# Every designated target in the directory is scored; there is no sample cap, because the per-sample
# error has a standard deviation over 2 m and a few hundred samples cannot separate two checkpoints.
# Run: python3 measure_checkpoint.py checkpoint.pt ../data/staged fitted_anchors.npz

import sys
from pathlib import Path

import numpy as np
import torch

from womd import baseline, contract, loader, model, pipeline

RANDOM_DRAW_SEED = 0
SCENARIO_ORDER_SEED = 0
BATCH_SIZE = 16
SUBMITTED_STEPS = torch.tensor(contract.SUBMISSION_FUTURE_INDICES)
STEP_SETS = ("ade_80step", "ade_16step")


def mode_mean_distances(trajectories, future_positions, future_mask):
    step_distances = (trajectories - future_positions.unsqueeze(1)).norm(dim=-1)
    validity = future_mask.unsqueeze(1).to(step_distances.dtype)
    return (step_distances * validity).sum(dim=-1) / validity.sum(dim=-1).clamp_min(1.0)


def best_mode_distance(step_set, trajectories, future_positions, future_mask):
    if step_set == "ade_16step":
        trajectories = trajectories[..., SUBMITTED_STEPS, :]
        future_positions = future_positions[:, SUBMITTED_STEPS]
        future_mask = future_mask[:, SUBMITTED_STEPS]
    return mode_mean_distances(trajectories, future_positions, future_mask).min(dim=1).values


def batches_with_lane_null(staged_directory, batch_size, seed):
    scenario_paths = sorted(staged_directory.glob("*.npz"))
    samples, lane_null_trajectories, followed_a_lane = [], [], []

    def collated():
        return (
            pipeline.collate_samples(samples),
            torch.from_numpy(np.stack(lane_null_trajectories)),
            torch.tensor(followed_a_lane),
        )

    for scenario_index in np.random.default_rng(seed).permutation(len(scenario_paths)):
        scenario_array = loader.read_scenario(scenario_paths[scenario_index])
        for track_index in loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            True,
        ):
            samples.append(loader.build_sample(scenario_array, int(track_index)))
            trajectories, lane_found = baseline.follow_the_lane_predictions(
                scenario_array, int(track_index)
            )
            lane_null_trajectories.append(trajectories)
            followed_a_lane.append(lane_found)
            if len(samples) < batch_size:
                continue
            yield collated()
            samples.clear()
            lane_null_trajectories.clear()
            followed_a_lane.clear()
    if samples:
        yield collated()


def main():
    checkpoint_path = Path(sys.argv[1])
    staged_directory = Path(sys.argv[2])
    fitted_anchors_path = Path(sys.argv[3])
    with np.load(fitted_anchors_path) as fitted_anchors_file:
        unit_anchors = torch.from_numpy(fitted_anchors_file["unit_anchors"])
    predictor = model.MotionPredictor(unit_anchors)
    predictor.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state"])
    predictor.eval()
    generator = torch.Generator().manual_seed(RANDOM_DRAW_SEED)

    best_distances = {}
    widest_gap, reachable_distance_use = [], []
    sample_count = 0
    lane_fallback_count = 0
    for batch, lane_null_trajectories, followed_a_lane in batches_with_lane_null(
        staged_directory, BATCH_SIZE, SCENARIO_ORDER_SEED
    ):
        with torch.no_grad():
            trajectories, confidence_logits = predictor(batch)
        scoreable = batch["future_mask"].any(dim=-1)
        if not scoreable.any():
            continue
        scored_trajectories = trajectories[scoreable]
        future_positions = batch["future_positions"][scoreable]
        future_mask = batch["future_mask"][scoreable]
        pruned, _ = model.prune_modes_batched(scored_trajectories, confidence_logits[scoreable])
        draw = torch.stack([
            torch.randperm(scored_trajectories.shape[1], generator=generator)[
                : contract.NUM_PREDICTED_MODES
            ]
            for _ in range(scored_trajectories.shape[0])
        ])
        anchor_trajectories, _ = baseline.straight_lines_to_most_used_anchors(batch, unit_anchors)
        constant_velocity_trajectories, _ = baseline.constant_velocity(batch)
        turn_rate_trajectories, _ = baseline.constant_turn_rate_and_velocity(batch)

        mode_sets = (
            ("best of ALL modes", scored_trajectories),
            ("best of 6 by CONFIDENCE", pruned),
            ("best of 6 at RANDOM", scored_trajectories.gather(
                1, draw[:, :, None, None].expand(-1, -1, contract.FUTURE_STEPS, 2)
            )),
            ("best of 6 ANCHOR NULL", anchor_trajectories[scoreable]),
            ("best of 6 LANE NULL", lane_null_trajectories[scoreable]),
            ("1 mode CONSTANT VELOCITY", constant_velocity_trajectories[scoreable]),
            ("1 mode CONSTANT TURN RATE", turn_rate_trajectories[scoreable]),
        )
        for name, modes in mode_sets:
            for step_set in STEP_SETS:
                best_distances.setdefault((name, step_set), []).append(
                    best_mode_distance(step_set, modes, future_positions, future_mask)
                )

        endpoints = scored_trajectories[:, :, -1]
        reachable = model.agent_reachable_distance(batch["agent_history"][scoreable])
        steps = torch.cat(
            [scored_trajectories[..., :1, :], scored_trajectories.diff(dim=-2)], dim=-2
        )
        widest_gap.append(torch.cdist(endpoints, endpoints).amax(dim=(1, 2)))
        reachable_distance_use.append(steps.norm(dim=-1).sum(dim=-1).amax(dim=1) / reachable)
        sample_count += int(scoreable.sum())
        lane_fallback_count += int((~followed_a_lane[scoreable]).sum())

    rows = [
        (name, step_set, torch.cat(values).numpy())
        for (name, step_set), values in best_distances.items()
    ]
    rows.extend(
        (name, "", torch.cat(values).numpy())
        for name, values in (
            ("widest gap between modes", widest_gap),
            ("path / reachable distance", reachable_distance_use),
        )
    )
    print(f"{checkpoint_path} on {staged_directory}, anchors {fitted_anchors_path}")
    print(f"every designated target in the directory scored: {sample_count} samples")
    print("ade_80step averages over all 80 logged future steps at 10 Hz, the quantity train.py"
          " monitors; ade_16step averages over contract.SUBMISSION_FUTURE_INDICES, the 16 steps a"
          " submission carries")
    print(f"{'quantity':27s}{'metric':12s}{'mean':>10s}{'p50':>10s}{'p90':>10s}{'max':>10s}")
    for name, step_set, values in rows:
        print(f"{name:27s}{step_set:12s}{values.mean():10.3f}{np.percentile(values, 50):10.3f}"
              f"{np.percentile(values, 90):10.3f}{values.max():10.3f}")
    print(f"lane null had no lane to follow and fell back to constant velocity for"
          f" {lane_fallback_count} of {sample_count} agents")


if __name__ == "__main__":
    main()
