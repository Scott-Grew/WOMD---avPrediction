import womd.runtime_env
import sys
from pathlib import Path

import numpy as np
import torch

from womd import contract, loss, metrics, model, pipeline

BATCH_SIZE = 16
BUCKET_COUNT = 5
WORST_SAMPLE_COUNT = 20


def batches_with_identifiers(staged_directory, batch_size):
    for batch in pipeline.track_sample_batches(staged_directory, batch_size, True):
        samples = [sample for _, _, sample in batch]
        yield (
            pipeline.collate_samples(samples),
            [str(sample["scenario_id"]) for sample in samples],
            [int(sample["track_id"]) for sample in samples],
        )


def best_possible_curve_error(future_positions, future_mask):
    basis = model.bernstein_curve_basis(
        model.TRAJECTORY_CONTROL_POINTS, contract.FUTURE_STEPS
    )[:, 1:]
    weights = future_mask.to(future_positions.dtype).unsqueeze(-1)
    weighted_basis = basis.unsqueeze(0) * weights
    weighted_targets = future_positions * weights
    control_points = torch.linalg.lstsq(weighted_basis, weighted_targets).solution
    fitted = weighted_basis.new_empty(future_positions.shape)
    torch.matmul(basis.unsqueeze(0), control_points, out=fitted)
    step_distances = (fitted - future_positions).norm(dim=-1)
    validity = future_mask.to(step_distances.dtype)
    return (step_distances * validity).sum(dim=-1) / validity.sum(dim=-1).clamp_min(1.0)


def logged_path_length(future_positions, future_mask):
    steps = torch.cat([future_positions[:, :1], future_positions.diff(dim=-2)], dim=-2)
    previous_valid = torch.cat(
        [torch.ones_like(future_mask[:, :1]), future_mask[:, :-1]], dim=-1
    )
    unbroken = (future_mask & previous_valid).to(steps.dtype)
    return (steps.norm(dim=-1) * unbroken).sum(dim=-1)


def final_heading_change(agent_history, future_headings, future_mask):
    current = agent_history[:, contract.CURRENT_STEP_INDEX]
    current_angle = torch.atan2(
        current[:, contract.AGENT_HEADING_SINE], current[:, contract.AGENT_HEADING_COSINE]
    )
    last_valid_step = (
        future_mask.to(torch.long) * torch.arange(future_mask.shape[1])
    ).argmax(dim=-1)
    final = future_headings[torch.arange(future_headings.shape[0]), last_valid_step]
    final_angle = torch.atan2(final[:, 1], final[:, 0])
    difference = final_angle - current_angle
    return torch.atan2(difference.sin(), difference.cos())


def path_turn_against_current_heading(trajectories, agent_history):
    current = agent_history[:, contract.CURRENT_STEP_INDEX]
    current_angle = torch.atan2(
        current[:, contract.AGENT_HEADING_SINE], current[:, contract.AGENT_HEADING_COSINE]
    )
    final_step = trajectories[..., -1, :] - trajectories[..., -2, :]
    final_angle = torch.atan2(final_step[..., 1], final_step[..., 0])
    difference = final_angle - current_angle.unsqueeze(-1)
    return torch.atan2(difference.sin(), difference.cos())


def along_and_cross_track_error(trajectories, future_positions, future_headings, future_mask):
    offset = trajectories - future_positions
    forward = future_headings
    sideways = torch.stack([-future_headings[..., 1], future_headings[..., 0]], dim=-1)
    validity = future_mask.to(offset.dtype)
    step_count = validity.sum(dim=-1).clamp_min(1.0)
    along_per_step = (offset * forward).sum(dim=-1)
    along = (along_per_step.abs() * validity).sum(dim=-1) / step_count
    signed_along = (along_per_step * validity).sum(dim=-1) / step_count
    cross = ((offset * sideways).sum(dim=-1).abs() * validity).sum(dim=-1) / step_count
    return along, cross, signed_along


def path_distances_both_directions(trajectories, future_positions, future_mask):
    pairwise = torch.cdist(trajectories, future_positions)
    pairwise = pairwise.masked_fill(~future_mask.unsqueeze(1), float("inf"))
    predicted_to_logged = pairwise.amin(dim=2).mean(dim=1)
    validity = future_mask.to(pairwise.dtype)
    nearest_to_each_logged_step = torch.where(
        future_mask, pairwise.amin(dim=1), torch.zeros_like(validity)
    )
    logged_to_predicted = nearest_to_each_logged_step.sum(dim=1) / validity.sum(
        dim=1
    ).clamp_min(1.0)
    return predicted_to_logged, logged_to_predicted


def signal_is_present(map_chunk_signal_history, map_chunk_lane_context):
    reachable_chunk = map_chunk_lane_context[..., contract.LANE_CONTEXT_REACHABLE] > 0.0
    chunk_signal_at_now = map_chunk_signal_history[:, :, contract.CURRENT_STEP_INDEX]
    chunk_has_signal = chunk_signal_at_now.argmax(dim=-1) > 0
    return (reachable_chunk & chunk_has_signal).any(dim=-1)


def reachable_lane_counts(map_chunk_lane_context):
    forward = map_chunk_lane_context[..., contract.LANE_CONTEXT_REACHABLE] > 0.0
    by_lane_change = (
        map_chunk_lane_context[..., contract.LANE_CONTEXT_REACHABLE_BY_LANE_CHANGE] > 0.0
    )
    return forward.sum(dim=-1), by_lane_change.sum(dim=-1)


def quantile_bucket_labels(values, bucket_count, unit):
    edges = np.quantile(values, np.linspace(0.0, 1.0, bucket_count + 1))
    edges[-1] = np.inf
    assignment = np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, bucket_count - 1)
    labels = [
        f"{edges[index]:.2f}-{edges[index + 1]:.2f} {unit}"
        if np.isfinite(edges[index + 1])
        else f"{edges[index]:.2f}+ {unit}"
        for index in range(bucket_count)
    ]
    return assignment, labels


def print_breakdown(title, assignment, labels, errors):
    print(f"\n{title}")
    print(f"  {'bucket':26s}{'samples':>9s}{'share of':>10s}{'mean':>9s}{'p50':>9s}{'p90':>9s}"
          f"{'max':>9s}")
    print(f"  {'':26s}{'':>9s}{'total err':>10s}{'':>9s}{'':>9s}{'':>9s}{'':>9s}")
    total_error = errors.sum()
    for bucket_index, label in enumerate(labels):
        selected = errors[assignment == bucket_index]
        if selected.size == 0:
            print(f"  {label:26s}{0:9d}")
            continue
        print(f"  {label:26s}{selected.size:9d}{selected.sum() / total_error:9.1%}"
              f"{selected.mean():9.3f}{np.percentile(selected, 50):9.3f}"
              f"{np.percentile(selected, 90):9.3f}{selected.max():9.3f}")


def main():
    checkpoint_path = Path(sys.argv[1])
    staged_directory = Path(sys.argv[2])
    fitted_anchors_path = Path(sys.argv[3])
    with np.load(fitted_anchors_path) as fitted_anchors_file:
        contract.check_artifact_provenance(
            fitted_anchors_file["provenance"] if "provenance" in fitted_anchors_file else None,
            fitted_anchors_path, "Refit them with fit_anchors.py.",
        )
        unit_anchors = torch.from_numpy(fitted_anchors_file["unit_anchors"])
    predictor = model.MotionPredictor(unit_anchors)
    predictor.load_state_dict(model.load_checkpoint_state(checkpoint_path)["model_state"])
    predictor.eval()

    columns = {name: [] for name in (
        "confidence_error", "all_mode_error", "curve_ceiling", "winning_mode", "type_index", "speed",
        "heading_change", "path_length", "reachable_lanes", "lane_change_lanes", "signal_present",
        "winning_mode_turn", "widest_mode_turn", "confident_mode_turn",
        "along_track_error", "cross_track_error", "signed_along_error",
        "predicted_to_logged", "logged_to_predicted",
        "mode_path_spread", "winning_mode_path_length",
        "reachable_distance", "logged_endpoint_distance", "widest_anchor_radius",
        "assigned_heading_error_deg", "sigma_floor_step_fraction",
    )}
    identifiers = []

    for batch, scenario_ids, track_ids in batches_with_identifiers(staged_directory, BATCH_SIZE):
        with torch.no_grad():
            (
                trajectories, heading_cosine_sine, position_log_standard_deviation,
                confidence_logits, _, selected_unit_anchors, _,
            ) = predictor.predict_with_heading(batch)
        scoreable = batch["future_mask"].any(dim=-1)
        if not scoreable.any():
            continue
        scored_trajectories = trajectories[scoreable]
        future_positions = batch["future_positions"][scoreable]
        future_mask = batch["future_mask"][scoreable]
        agent_history = batch["agent_history"][scoreable]
        pruned, _ = model.prune_modes_batched(scored_trajectories, confidence_logits[scoreable])

        all_mode_distances = metrics.mean_distance_per_mode(
            scored_trajectories, future_positions, future_mask
        )
        forward_lanes, lane_change_lanes = reachable_lane_counts(
            batch["map_chunk_lane_context"][scoreable]
        )
        columns["confidence_error"].append(
            metrics.mean_distance_per_mode(pruned, future_positions, future_mask).min(dim=1).values
        )
        columns["all_mode_error"].append(all_mode_distances.min(dim=1).values)
        columns["curve_ceiling"].append(
            best_possible_curve_error(future_positions, future_mask)
        )
        mode_turns = path_turn_against_current_heading(scored_trajectories, agent_history)
        winning_mode = all_mode_distances.argmin(dim=1)
        columns["winning_mode_turn"].append(
            mode_turns.gather(1, winning_mode.unsqueeze(-1)).squeeze(-1)
        )
        columns["widest_mode_turn"].append(mode_turns.abs().amax(dim=-1))
        columns["confident_mode_turn"].append(
            path_turn_against_current_heading(pruned[:, :1], agent_history).squeeze(-1)
        )
        winning_trajectory = scored_trajectories.gather(
            1, winning_mode[:, None, None, None].expand(-1, -1, contract.FUTURE_STEPS, 2)
        ).squeeze(1)
        along_track, cross_track, signed_along = along_and_cross_track_error(
            winning_trajectory, future_positions, batch["future_headings"][scoreable], future_mask
        )
        columns["along_track_error"].append(along_track)
        columns["cross_track_error"].append(cross_track)
        columns["signed_along_error"].append(signed_along)
        predicted_to_logged, logged_to_predicted = path_distances_both_directions(
            winning_trajectory, future_positions, future_mask
        )
        columns["predicted_to_logged"].append(predicted_to_logged)
        columns["logged_to_predicted"].append(logged_to_predicted)
        mode_steps = torch.cat(
            [scored_trajectories[..., :1, :], scored_trajectories.diff(dim=-2)], dim=-2
        )
        mode_path_lengths = mode_steps.norm(dim=-1).sum(dim=-1)
        columns["mode_path_spread"].append(
            mode_path_lengths.amax(dim=-1) - mode_path_lengths.amin(dim=-1)
        )
        reachable = model.agent_reachable_distance(agent_history)
        last_valid_step = (
            future_mask.to(torch.long) * torch.arange(future_mask.shape[1])
        ).argmax(dim=-1)
        logged_endpoint = future_positions[torch.arange(future_positions.shape[0]), last_valid_step]
        type_anchors = unit_anchors[model.predicted_type_index(agent_history)]
        columns["reachable_distance"].append(reachable)
        columns["logged_endpoint_distance"].append(logged_endpoint.norm(dim=-1))
        columns["widest_anchor_radius"].append(type_anchors.norm(dim=-1).amax(dim=-1))
        columns["winning_mode_path_length"].append(
            mode_path_lengths.gather(1, winning_mode.unsqueeze(-1)).squeeze(-1)
        )
        columns["winning_mode"].append(all_mode_distances.argmin(dim=1))
        columns["type_index"].append(model.predicted_type_index(agent_history))
        columns["speed"].append(
            agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY].norm(dim=-1)
        )
        columns["heading_change"].append(final_heading_change(
            agent_history, batch["future_headings"][scoreable], future_mask
        ))
        columns["path_length"].append(logged_path_length(future_positions, future_mask))
        columns["reachable_lanes"].append(forward_lanes)
        columns["lane_change_lanes"].append(lane_change_lanes)
        columns["signal_present"].append(
            signal_is_present(
                batch["map_chunk_signal_history"][scoreable],
                batch["map_chunk_lane_context"][scoreable],
            )
        )
        assigned_mode = loss.anchor_assigned_mode(
            selected_unit_anchors[scoreable], future_positions, future_mask,
        )
        assigned_heading = heading_cosine_sine[scoreable].gather(
            1, assigned_mode[:, None, None, None].expand(-1, -1, contract.FUTURE_STEPS, 2)
        ).squeeze(1)
        unit_heading = assigned_heading / assigned_heading.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(assigned_heading.dtype).eps
        )
        heading_dot = (unit_heading * batch["future_headings"][scoreable]).sum(dim=-1).clamp(-1.0, 1.0)
        heading_validity = future_mask.to(heading_dot.dtype)
        columns["assigned_heading_error_deg"].append(
            torch.rad2deg(
                (heading_dot.acos() * heading_validity).sum(dim=-1)
                / heading_validity.sum(dim=-1).clamp_min(1.0)
            )
        )
        assigned_log_sigma = position_log_standard_deviation[scoreable].gather(
            1, assigned_mode[:, None, None, None].expand(-1, -1, contract.FUTURE_STEPS, 2)
        ).squeeze(1)
        sigma_floor = assigned_log_sigma == model.MINIMUM_LOG_STANDARD_DEVIATION
        columns["sigma_floor_step_fraction"].append(
            (sigma_floor.any(dim=-1).to(heading_validity.dtype) * heading_validity).sum(dim=-1)
            / heading_validity.sum(dim=-1).clamp_min(1.0)
        )
        identifiers.extend(
            pair for pair, keep in zip(zip(scenario_ids, track_ids), scoreable.tolist()) if keep
        )

    measured = {name: torch.cat(values).numpy() for name, values in columns.items()}
    errors = measured["confidence_error"]
    sample_count = errors.size

    print(f"{checkpoint_path} on {staged_directory}, anchors {fitted_anchors_path}")
    print(f"{sample_count} designated targets, minADE over all 80 logged future steps,"
          f" best of the 6 modes the confidence head keeps")
    print(f"overall mean {errors.mean():.3f}  p50 {np.percentile(errors, 50):.3f}"
          f"  p90 {np.percentile(errors, 90):.3f}  max {errors.max():.3f}")

    type_labels = list(contract.PREDICTED_OBJECT_TYPES)
    print_breakdown("BY OBJECT TYPE", measured["type_index"], type_labels, errors)
    print_breakdown(
        "BY TRAFFIC SIGNAL ON THE AGENT'S OWN LANE",
        measured["signal_present"].astype(np.int64),
        ["no signal", "signal present"],
        errors,
    )
    measured["heading_change"] = np.abs(measured["heading_change"])
    for title, name, unit in (
        ("BY SPEED AT THE MOMENT OF PREDICTION", "speed", "m/s"),
        ("BY HOW MUCH THE AGENT ACTUALLY TURNED", "heading_change", "rad"),
        ("BY HOW FAR THE AGENT ACTUALLY TRAVELLED", "path_length", "m"),
        ("BY FORWARD-REACHABLE LANE CHUNKS (junction proxy)", "reachable_lanes", "chunks"),
        ("BY LANE-CHANGE-REACHABLE LANE CHUNKS", "lane_change_lanes", "chunks"),
    ):
        assignment, labels = quantile_bucket_labels(
            measured[name].astype(np.float64), BUCKET_COUNT, unit
        )
        print_breakdown(title, assignment, labels, errors)

    turn_assignment, turn_labels = quantile_bucket_labels(
        measured["heading_change"].astype(np.float64), BUCKET_COUNT, "rad"
    )
    print("\nWHERE THE TURN ERROR COMES FROM — curve ceiling is the best any 3-control-point"
          " Bernstein curve could score, fitted directly to the logged future")
    print(f"  {'bucket':26s}{'ceiling':>10s}{'best of 18':>12s}{'best of 6':>11s}"
          f"{'ceiling share':>15s}")
    for bucket_index, label in enumerate(turn_labels):
        selected = turn_assignment == bucket_index
        ceiling = measured["curve_ceiling"][selected].mean()
        best_of_all = measured["all_mode_error"][selected].mean()
        best_of_six = errors[selected].mean()
        print(f"  {label:26s}{ceiling:10.3f}{best_of_all:12.3f}{best_of_six:11.3f}"
              f"{ceiling / best_of_six:15.1%}")

    print("\nDOES THE MODEL TURN AT ALL — radians of path bend against the agent's current heading")
    print(f"  {'bucket':26s}{'logged':>9s}{'winning mode':>14s}{'top confidence':>16s}"
          f"{'widest mode':>14s}")
    for bucket_index, label in enumerate(turn_labels):
        selected = turn_assignment == bucket_index
        print(f"  {label:26s}{np.abs(measured['heading_change'][selected]).mean():9.3f}"
              f"{np.abs(measured['winning_mode_turn'][selected]).mean():14.3f}"
              f"{np.abs(measured['confident_mode_turn'][selected]).mean():16.3f}"
              f"{measured['widest_mode_turn'][selected].mean():14.3f}")

    print("\nIS THE MISS AHEAD/BEHIND OR SIDEWAYS — best mode, split in the logged agent's own frame")
    print(f"  {'bucket':26s}{'along track':>13s}{'cross track':>13s}{'along share':>13s}"
          f"{'signed along':>14s}")
    for bucket_index, label in enumerate(turn_labels):
        selected = turn_assignment == bucket_index
        along = measured["along_track_error"][selected].mean()
        cross = measured["cross_track_error"][selected].mean()
        print(f"  {label:26s}{along:13.3f}{cross:13.3f}{along / (along + cross):13.1%}"
              f"{measured['signed_along_error'][selected].mean():14.3f}")

    print("\nCAN THE MODES EXPRESS DIFFERENT TRAVEL DISTANCES — metres of path, not direction")
    print(f"  {'bucket':26s}{'logged spread':>15s}{'model spread':>14s}{'logged mean':>13s}"
          f"{'winner mean':>13s}")
    for bucket_index, label in enumerate(turn_labels):
        selected = turn_assignment == bucket_index
        logged_spread = (
            np.percentile(measured["path_length"][selected], 90)
            - np.percentile(measured["path_length"][selected], 10)
        )
        print(f"  {label:26s}{logged_spread:15.1f}"
              f"{measured['mode_path_spread'][selected].mean():14.1f}"
              f"{measured['path_length'][selected].mean():13.1f}"
              f"{measured['winning_mode_path_length'][selected].mean():13.1f}")

    print("\nDO THE ANCHORS REACH FAR ENOUGH — straight-line distance, metres")
    print(f"  {'bucket':26s}{'reach budget':>14s}{'widest anchor':>15s}{'logged endpoint':>17s}"
          f"{'endpoint / anchor':>19s}{'over anchor':>13s}")
    for bucket_index, label in enumerate(turn_labels):
        selected = turn_assignment == bucket_index
        reach = measured["reachable_distance"][selected]
        widest = measured["widest_anchor_radius"][selected]
        logged_end = measured["logged_endpoint_distance"][selected]
        print(f"  {label:26s}{reach.mean():14.1f}{widest.mean():15.1f}{logged_end.mean():17.1f}"
              f"{(logged_end / widest).mean():19.2f}{(logged_end > widest).mean():13.1%}")

    print("\nROUTE VERSUS TIMING — how far the predicted PATH is from the logged PATH, ignoring when")
    print(f"  {'bucket':26s}{'pred->logged':>14s}{'logged->pred':>14s}{'symmetric':>11s}"
          f"{'time-indexed':>14s}{'timing share':>14s}")
    for bucket_index, label in enumerate(turn_labels):
        selected = turn_assignment == bucket_index
        forward = measured["predicted_to_logged"][selected].mean()
        backward = measured["logged_to_predicted"][selected].mean()
        symmetric = 0.5 * (forward + backward)
        timed = measured["all_mode_error"][selected].mean()
        print(f"  {label:26s}{forward:14.3f}{backward:14.3f}{symmetric:11.3f}{timed:14.3f}"
              f"{1.0 - symmetric / timed:14.1%}")

    winner_counts = np.bincount(measured["winning_mode"], minlength=int(model.QUERY_COUNT))
    print("\nFAILURE SIGNATURES — the same reads that named run 25 (dead modes) and run 26 (heading ~90°, sigma floor). Not pass marks.")
    print(
        f"  assigned-mode heading error p50 {np.percentile(measured['assigned_heading_error_deg'], 50):.1f}°"
        f"  p90 {np.percentile(measured['assigned_heading_error_deg'], 90):.1f}°"
        f"  (run 24 was 0.9°; run 26 died at 89.5°)"
    )
    print(
        f"  steps at the sigma floor {100 * measured['sigma_floor_step_fraction'].mean():.1f}%"
        f"  (run 26 died at 19.8%)"
    )
    print(
        f"  modes that never win: {int((winner_counts == 0).sum())} of {winner_counts.size}"
        f"  covering 90% of wins:"
        f" {int(np.searchsorted(np.cumsum(np.sort(winner_counts)[::-1]) / sample_count, 0.9) + 1)}"
        f"  (run 24: 0 never won, 13 for 90%; run 25 died with 5 never winning)"
    )

    print("\nMODE UTILISATION — how often each of the model's modes is the closest to the truth")
    for mode_index, count in enumerate(winner_counts):
        print(f"  mode {mode_index:2d}{count:9d}{count / sample_count:9.1%}")

    print(f"\nWORST {WORST_SAMPLE_COUNT} SAMPLES")
    print(f"  {'scenario':18s}{'track':>7s}{'minADE':>9s}{'best-of-all':>12s}{'type':>12s}"
          f"{'speed':>8s}{'turn':>8s}{'travel':>9s}{'lanes':>7s}{'signal':>8s}")
    for rank in np.argsort(errors)[::-1][:WORST_SAMPLE_COUNT]:
        scenario_id, track_id = identifiers[rank]
        print(f"  {scenario_id:18s}{track_id:7d}{errors[rank]:9.3f}"
              f"{measured['all_mode_error'][rank]:10.3f}"
              f"{contract.PREDICTED_OBJECT_TYPES[measured['type_index'][rank]][5:]:>12s}"
              f"{measured['speed'][rank]:8.2f}{measured['heading_change'][rank]:8.2f}"
              f"{measured['path_length'][rank]:9.1f}{measured['reachable_lanes'][rank]:7d}"
              f"{'yes' if measured['signal_present'][rank] else 'no':>8s}")


if __name__ == "__main__":
    main()
