import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

from womd import baseline, contract, loader, loss, metrics, model, pipeline

BATCH_SIZE = 16
TURN_BUCKET_COUNT = 5
SCENE_PAGE_FIGURE_SIZE = (13.5, 12.5)
PAGE_THREE_FIGURE_SIZE = (16.5, 10.5)
PAGE_FOUR_FIGURE_SIZE = (16.5, 9.5)
SPEED_COLORMAP = "plasma"
LOGGED_COLOR = "#238b45"
ANCHOR_COLOR = "#54278f"
LANE_COLOR = "#9ecae1"
NEIGHBOUR_COLOR = "#fdae6b"
NEIGHBOUR_EDGE_COLOR = "#7f2704"
DEAD_MODE_COLOR = "#bdbdbd"
PREDICTED_COLOR = "#3182bd"


def pick_turning_sample(staged_directory):
    best = None
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
            final = sample["future_headings"][-1]
            turn = abs(float(np.arctan2(final[1], final[0])))
            travelled = float(np.linalg.norm(sample["future_positions"][-1]))
            if turn < 1.0 or travelled < 30.0:
                continue
            if best is None or turn > best[0]:
                best = (turn, sample)
        if best is not None:
            return best[1]
    raise SystemExit("no turning designated target found")


def draw_scene(axis, sample):
    map_rows = sample["map_rows"]
    lane_kind = contract.MAP_POLYLINE_KINDS.index("lane")
    is_lane = map_rows[:, contract.MAP_KIND][:, lane_kind] > 0.0
    axis.scatter(map_rows[~is_lane, 0], map_rows[~is_lane, 1], s=0.4, c="#d9d9d9")
    axis.scatter(map_rows[is_lane, 0], map_rows[is_lane, 1], s=0.6, c=LANE_COLOR)
    neighbours = sample["neighbour_history"]
    present = sample["neighbour_history_mask"][:, contract.CURRENT_STEP_INDEX]
    now = neighbours[present, contract.CURRENT_STEP_INDEX]
    axis.scatter(now[:, 0], now[:, 1], s=26, c=NEIGHBOUR_COLOR, edgecolors=NEIGHBOUR_EDGE_COLOR, linewidths=0.5)
    history = sample["agent_history"][sample["agent_history_mask"]]
    axis.plot(history[:, 0], history[:, 1], c="#252525", linewidth=2.0)
    axis.scatter([0.0], [0.0], s=70, c="#252525", marker=">")


SCENE_X_MARGIN_FRACTION = 0.35
SCENE_Y_MARGIN_FRACTION = 0.6


def frame(axis, title, half_width):
    axis.set_title(title, fontsize=10, loc="left")
    axis.set_xlim(-half_width * SCENE_X_MARGIN_FRACTION, half_width)
    axis.set_ylim(-half_width * SCENE_Y_MARGIN_FRACTION, half_width * SCENE_Y_MARGIN_FRACTION)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#bdbdbd")


def type_index_and_anchors(sample, unit_anchors):
    type_index = int(model.predicted_type_index(torch.from_numpy(sample["agent_history"])[None])[0])
    return type_index, unit_anchors[type_index]


def type_label(type_index):
    return contract.PREDICTED_OBJECT_TYPES[type_index][5:].lower()


def scene_half_width(sample, anchors):
    future = sample["future_positions"]
    reach = float(np.linalg.norm(future, axis=1).max())
    return max(reach * 1.6, float(np.linalg.norm(anchors, axis=1).max()) * 0.55)


def heading_change_abs(agent_history, future_headings, future_mask):
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
    return torch.atan2(difference.sin(), difference.cos()).abs()


def along_and_cross_track_per_step(trajectory, future_positions, future_headings):
    offset = trajectory - future_positions
    forward = future_headings
    sideways = torch.stack([-future_headings[..., 1], future_headings[..., 0]], dim=-1)
    along = (offset * forward).sum(dim=-1)
    cross = (offset * sideways).sum(dim=-1)
    return along, cross


def cumulative_distance(positions):
    steps = torch.cat([positions[..., :1, :], positions.diff(dim=-2)], dim=-2)
    return steps.norm(dim=-1).cumsum(dim=-1)


def quantile_bucket_assignment(values, bucket_count):
    edges = np.quantile(values, np.linspace(0.0, 1.0, bucket_count + 1))
    edges[-1] = np.inf
    assignment = np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, bucket_count - 1)
    return assignment, edges


def turn_bucket_assignment_and_labels(measured):
    assignment, edges = quantile_bucket_assignment(measured["heading_change_abs"], TURN_BUCKET_COUNT)
    edges_degrees = np.degrees(edges)
    labels = []
    for bucket_index in range(TURN_BUCKET_COUNT):
        upper = f"{edges_degrees[bucket_index + 1]:.0f}" if bucket_index < TURN_BUCKET_COUNT - 1 else "+"
        labels.append(f"{edges_degrees[bucket_index]:.0f}-{upper}°")
    return assignment, labels


def batches_with_scenario_context(staged_directory, batch_size):
    for batch in pipeline.track_sample_batches(staged_directory, batch_size, True):
        samples = [sample for _, _, sample in batch]
        scenario_arrays = [scenario_array for scenario_array, _, _ in batch]
        track_indices = [track_index for _, track_index, _ in batch]
        yield pipeline.collate_samples(samples), samples, scenario_arrays, track_indices


def case_bundle(sample, trajectories, pruned_trajectories, pruned_confidence_logits, row_index, model_best_of_6):
    return {
        "sample": sample,
        "raw_trajectories": trajectories[row_index].numpy(),
        "pruned_trajectories": pruned_trajectories[row_index].numpy(),
        "pruned_confidence_logits": pruned_confidence_logits[row_index].numpy(),
        "model_best_of_6": model_best_of_6,
    }


def gather(staged_directory, predictor, unit_anchors, turning_scenario_id, turning_track_id):
    columns = {name: [] for name in (
        "heading_change_abs", "model_best_of_6", "along_track_error", "cross_track_error",
        "signed_along_track_error", "reachable_distance",
        "winning_mode_18", "future_fully_valid",
        "anchor_null_best_of_6", "lane_null_best_of_6",
        "constant_velocity_best_of_6", "constant_turn_rate_best_of_6",
    )}
    logged_cumulative_rows = []
    predicted_cumulative_rows = []
    turning_prediction = {}
    hard_case = {"model_best_of_6": -math.inf}
    easy_case = {"model_best_of_6": math.inf}

    for batch, samples, scenario_arrays, track_indices in batches_with_scenario_context(
        staged_directory, BATCH_SIZE
    ):
        with torch.no_grad():
            (
                trajectories, heading_cosine_sine, position_log_standard_deviation,
                confidence_logits, predicted_speed, selected_unit_anchors, neighbour_future_positions,
            ) = predictor.predict_with_heading(batch)
        future_positions = batch["future_positions"]
        future_mask = batch["future_mask"]
        future_headings = batch["future_headings"]
        agent_history = batch["agent_history"]
        future_fully_valid = future_mask.all(dim=-1)

        all_mode_distances = metrics.mean_distance_per_mode(trajectories, future_positions, future_mask)
        winning_mode_18 = all_mode_distances.argmin(dim=1)
        winning_trajectory_18 = trajectories.gather(
            1, winning_mode_18[:, None, None, None].expand(-1, -1, contract.FUTURE_STEPS, 2)
        ).squeeze(1)

        pruned_trajectories, pruned_confidence_logits = model.prune_modes_batched(
            trajectories, confidence_logits
        )
        model_best_of_6 = metrics.mean_distance_per_mode(
            pruned_trajectories, future_positions, future_mask
        ).min(dim=1).values

        assigned_anchor_18 = loss.anchor_assigned_mode(selected_unit_anchors, future_positions, future_mask)

        along_per_step, cross_per_step = along_and_cross_track_per_step(
            winning_trajectory_18, future_positions, future_headings
        )
        validity = future_mask.to(along_per_step.dtype)
        step_count = validity.sum(dim=-1).clamp_min(1.0)
        along_track_error = (along_per_step.abs() * validity).sum(dim=-1) / step_count
        cross_track_error = (cross_per_step.abs() * validity).sum(dim=-1) / step_count
        signed_along_track_error = (along_per_step * validity).sum(dim=-1) / step_count

        constant_velocity_trajectories, constant_velocity_confidence = baseline.constant_velocity(batch)
        constant_velocity_pruned, _ = model.prune_modes_batched(
            constant_velocity_trajectories, constant_velocity_confidence
        )
        constant_velocity_best_of_6 = metrics.mean_distance_per_mode(
            constant_velocity_pruned, future_positions, future_mask
        ).min(dim=1).values

        constant_turn_rate_trajectories, constant_turn_rate_confidence = (
            baseline.constant_turn_rate_and_velocity(batch)
        )
        constant_turn_rate_pruned, _ = model.prune_modes_batched(
            constant_turn_rate_trajectories, constant_turn_rate_confidence
        )
        constant_turn_rate_best_of_6 = metrics.mean_distance_per_mode(
            constant_turn_rate_pruned, future_positions, future_mask
        ).min(dim=1).values

        anchor_null_trajectories, anchor_null_confidence = baseline.straight_lines_to_most_used_anchors(
            batch, unit_anchors
        )
        anchor_null_best_of_6 = metrics.mean_distance_per_mode(
            anchor_null_trajectories, future_positions, future_mask
        ).min(dim=1).values

        lane_null_trajectories = torch.stack([
            torch.from_numpy(baseline.follow_the_lane_predictions(scenario_array, track_index)[0])
            for scenario_array, track_index in zip(scenario_arrays, track_indices)
        ])
        lane_null_best_of_6 = metrics.mean_distance_per_mode(
            lane_null_trajectories, future_positions, future_mask
        ).min(dim=1).values

        columns["heading_change_abs"].append(
            heading_change_abs(agent_history, future_headings, future_mask)
        )
        columns["model_best_of_6"].append(model_best_of_6)
        columns["along_track_error"].append(along_track_error)
        columns["cross_track_error"].append(cross_track_error)
        columns["signed_along_track_error"].append(signed_along_track_error)
        columns["reachable_distance"].append(model.agent_reachable_distance(agent_history))
        columns["winning_mode_18"].append(winning_mode_18)
        columns["future_fully_valid"].append(future_fully_valid)
        columns["anchor_null_best_of_6"].append(anchor_null_best_of_6)
        columns["lane_null_best_of_6"].append(lane_null_best_of_6)
        columns["constant_velocity_best_of_6"].append(constant_velocity_best_of_6)
        columns["constant_turn_rate_best_of_6"].append(constant_turn_rate_best_of_6)
        logged_cumulative_rows.append(cumulative_distance(future_positions))
        predicted_cumulative_rows.append(cumulative_distance(winning_trajectory_18))

        for row_index, sample in enumerate(samples):
            is_turning_sample = (
                str(sample["scenario_id"]) == turning_scenario_id
                and int(sample["track_id"]) == turning_track_id
            )
            if is_turning_sample:
                turning_prediction["raw_trajectories"] = trajectories[row_index].numpy()
                turning_prediction["pruned_trajectories"] = pruned_trajectories[row_index].numpy()
                turning_prediction["pruned_confidence_logits"] = (
                    pruned_confidence_logits[row_index].numpy()
                )
                turning_prediction["winning_trajectory_18"] = winning_trajectory_18[row_index].numpy()
                turning_prediction["model_best_of_6"] = float(model_best_of_6[row_index])
                turning_prediction["raw_heading_cosine_sine"] = heading_cosine_sine[row_index].numpy()
                turning_prediction["raw_position_log_standard_deviation"] = (
                    position_log_standard_deviation[row_index].numpy()
                )
                turning_prediction["raw_predicted_speed"] = predicted_speed[row_index].numpy()
                turning_prediction["assigned_anchor_index"] = int(assigned_anchor_18[row_index])
                turning_prediction["neighbour_future_positions"] = neighbour_future_positions[
                    row_index, : sample["neighbour_history"].shape[0]
                ].numpy()
                continue
            if not future_fully_valid[row_index]:
                continue
            sample_best_of_6 = float(model_best_of_6[row_index])
            if sample_best_of_6 > hard_case["model_best_of_6"]:
                hard_case = case_bundle(
                    sample, trajectories, pruned_trajectories, pruned_confidence_logits,
                    row_index, sample_best_of_6,
                )
            if sample_best_of_6 < easy_case["model_best_of_6"]:
                easy_case = case_bundle(
                    sample, trajectories, pruned_trajectories, pruned_confidence_logits,
                    row_index, sample_best_of_6,
                )

    measured = {name: torch.cat(values).numpy() for name, values in columns.items()}
    measured["logged_cumulative_distance"] = torch.cat(logged_cumulative_rows).numpy()
    measured["predicted_cumulative_distance"] = torch.cat(predicted_cumulative_rows).numpy()
    assert "sample" in hard_case, "no fully-valid-future sample found for the hard case panel"
    assert "sample" in easy_case, "no fully-valid-future sample found for the easy case panel"
    return measured, turning_prediction, hard_case, easy_case


def raw_mode_index_for_pruned_trajectory(raw_trajectories, pruned_trajectory):
    return int(np.abs(raw_trajectories - pruned_trajectory[None]).sum(axis=(1, 2)).argmin())


def draw_model_view_panel(axis, sample, half_width):
    draw_scene(axis, sample)
    frame(axis, "1. What the model sees: map, neighbours, history", half_width)


def draw_scene_and_logged_future(axis, sample, title, half_width):
    future = sample["future_positions"]
    draw_scene(axis, sample)
    axis.plot(future[:, 0], future[:, 1], c=LOGGED_COLOR, linewidth=2.2, zorder=6)
    axis.scatter(future[-1:, 0], future[-1:, 1], s=50, c=LOGGED_COLOR, zorder=6)
    frame(axis, title, half_width)


def draw_anchor_panel(axis, sample, anchors, half_width):
    draw_scene(axis, sample)
    axis.scatter(anchors[:, 0], anchors[:, 1], s=45, c=ANCHOR_COLOR, marker="X", zorder=5)
    for anchor in anchors:
        axis.plot([0.0, anchor[0]], [0.0, anchor[1]], c=ANCHOR_COLOR, linewidth=0.6, alpha=0.35)
    frame(axis, f"2. {len(anchors)} anchors for this vehicle type (m)", half_width)


def draw_owning_anchor_panel(axis, sample, anchors, assigned_anchor_index, half_width):
    future = sample["future_positions"]
    draw_scene(axis, sample)
    for anchor_index, anchor in enumerate(anchors):
        if anchor_index == assigned_anchor_index:
            continue
        axis.scatter(*anchor, s=30, c=ANCHOR_COLOR, marker="X", alpha=0.25)
    owning_anchor = anchors[assigned_anchor_index]
    axis.plot([0.0, owning_anchor[0]], [0.0, owning_anchor[1]], c=ANCHOR_COLOR, linewidth=1.6)
    axis.scatter(*owning_anchor, s=90, c=ANCHOR_COLOR, marker="X", zorder=6)
    axis.plot(future[:, 0], future[:, 1], c=LOGGED_COLOR, linewidth=2.2, zorder=6)
    axis.scatter(future[-1:, 0], future[-1:, 1], s=50, c=LOGGED_COLOR, zorder=6)
    distance = float(np.linalg.norm(owning_anchor - future[-1]))
    axis.annotate(
        f"{distance:.1f} m from logged endpoint", owning_anchor, fontsize=8, color=ANCHOR_COLOR,
        xytext=(4, 6), textcoords="offset points",
    )
    frame(axis, "4. The anchor that owns this future", half_width)


def draw_all_predictions_panel(axis, sample, turning_prediction, half_width):
    future = sample["future_positions"]
    raw_trajectories = turning_prediction["raw_trajectories"]
    draw_scene(axis, sample)
    for trajectory in raw_trajectories:
        axis.plot(trajectory[:, 0], trajectory[:, 1], c=PREDICTED_COLOR, linewidth=1.0, alpha=0.55)
    axis.plot(future[:, 0], future[:, 1], c=LOGGED_COLOR, linewidth=2.4, zorder=6)
    frame(axis, f"5. All {len(raw_trajectories)} predicted futures", half_width)


def draw_kept_predictions_panel(axis, sample, prediction, half_width, title):
    future = sample["future_positions"]
    pruned_trajectories = prediction["pruned_trajectories"]
    pruned_confidence = prediction["pruned_confidence_logits"]
    weights = np.exp(pruned_confidence - pruned_confidence.max())
    weights = weights / weights.sum()
    order = np.argsort(weights)
    colormap = plt.get_cmap("Reds")

    draw_scene(axis, sample)
    for rank, mode_index in enumerate(order):
        trajectory = pruned_trajectories[mode_index]
        shade = colormap(0.35 + 0.6 * (rank + 1) / len(order))
        axis.plot(trajectory[:, 0], trajectory[:, 1], c=shade, linewidth=2.0)
        if weights[mode_index] >= 0.01:
            axis.annotate(
                f"{weights[mode_index]:.0%}", trajectory[-1], fontsize=7, color=shade,
                xytext=(3, 3), textcoords="offset points",
            )
    axis.plot(future[:, 0], future[:, 1], c=LOGGED_COLOR, linewidth=2.4, zorder=6)
    axis.annotate(
        f"best of 6 misses by {prediction['model_best_of_6']:.1f} m on average",
        (0.02, 0.03), xycoords="axes fraction", fontsize=8, color="#252525",
    )
    frame(axis, title, half_width)

    best_local_index = order[-1]
    return raw_mode_index_for_pruned_trajectory(
        prediction["raw_trajectories"], pruned_trajectories[best_local_index]
    )


UNCERTAINTY_ELLIPSE_COUNT = 4


def steps_visible_in_frame(trajectory, half_width):
    x_low, x_high = -half_width * SCENE_X_MARGIN_FRACTION, half_width
    y_low, y_high = -half_width * SCENE_Y_MARGIN_FRACTION, half_width * SCENE_Y_MARGIN_FRACTION
    visible = (
        (trajectory[:, 0] >= x_low) & (trajectory[:, 0] <= x_high)
        & (trajectory[:, 1] >= y_low) & (trajectory[:, 1] <= y_high)
    )
    return np.flatnonzero(visible)


def draw_uncertainty_panel(axis, sample, turning_prediction, best_raw_index, half_width):
    future = sample["future_positions"]
    trajectory = turning_prediction["raw_trajectories"][best_raw_index]
    heading = turning_prediction["raw_heading_cosine_sine"][best_raw_index]
    log_standard_deviation = turning_prediction["raw_position_log_standard_deviation"][best_raw_index]

    draw_scene(axis, sample)
    axis.plot(future[:, 0], future[:, 1], c=LOGGED_COLOR, linewidth=1.4, alpha=0.6)
    axis.plot(trajectory[:, 0], trajectory[:, 1], c=PREDICTED_COLOR, linewidth=2.0, zorder=5)

    visible_steps = steps_visible_in_frame(trajectory, half_width)
    if visible_steps.size == 0:
        visible_steps = np.array([0])
    sample_positions = np.linspace(0, len(visible_steps) - 1, min(UNCERTAINTY_ELLIPSE_COUNT, len(visible_steps)))
    ellipse_steps = visible_steps[sample_positions.astype(int)]
    for step in ellipse_steps:
        along_standard_deviation, cross_standard_deviation = np.exp(log_standard_deviation[step])
        forward_cosine, forward_sine = heading[step]
        ellipse = Ellipse(
            trajectory[step],
            width=2.0 * along_standard_deviation,
            height=2.0 * cross_standard_deviation,
            angle=np.degrees(np.arctan2(forward_sine, forward_cosine)),
            facecolor=PREDICTED_COLOR, alpha=0.18, edgecolor="#08519c", linewidth=1.0, zorder=4,
        )
        axis.add_patch(ellipse)
    annotated_step = ellipse_steps[-1]
    along_standard_deviation, cross_standard_deviation = np.exp(log_standard_deviation[annotated_step])
    axis.annotate(
        f"along {along_standard_deviation:.1f} m, cross {cross_standard_deviation:.1f} m",
        trajectory[annotated_step], fontsize=7, color="#08519c",
        xytext=(4, 6), textcoords="offset points",
    )
    frame(axis, "7. Predicted uncertainty along the best mode (1 SD ellipses)", half_width)


def draw_neighbour_future_panel(axis, sample, turning_prediction, half_width):
    present = sample["neighbour_history_mask"][:, contract.CURRENT_STEP_INDEX]
    predicted_futures = turning_prediction["neighbour_future_positions"]
    logged_futures = sample["neighbour_future_positions"]
    logged_valid = sample["neighbour_future_mask"]

    draw_scene(axis, sample)
    for neighbour_index in np.flatnonzero(present):
        axis.plot(
            predicted_futures[neighbour_index, :, 0], predicted_futures[neighbour_index, :, 1],
            c=PREDICTED_COLOR, linewidth=1.1, alpha=0.85,
        )
        masked_logged_future = np.where(
            logged_valid[neighbour_index, :, None], logged_futures[neighbour_index], np.nan
        )
        axis.plot(masked_logged_future[:, 0], masked_logged_future[:, 1], c=LOGGED_COLOR, linewidth=1.6)
    frame(axis, "8. Neighbour futures: model's predictions vs. what neighbours actually did", half_width)


def draw_speed_on_path_panel(figure, grid_cell, sample, turning_prediction, best_raw_index, half_width):
    inner = grid_cell.subgridspec(1, 3, width_ratios=(1.0, 1.0, 0.05), wspace=0.3)
    predicted_axis = figure.add_subplot(inner[0, 0])
    logged_axis = figure.add_subplot(inner[0, 1])
    colorbar_axis = figure.add_subplot(inner[0, 2])

    trajectory = turning_prediction["raw_trajectories"][best_raw_index]
    predicted_speed = turning_prediction["raw_predicted_speed"][best_raw_index]

    future_positions = torch.from_numpy(sample["future_positions"])[None]
    future_mask = torch.from_numpy(sample["future_mask"])[None]
    logged_speed, valid_step_pair = loss.logged_speed_per_step(future_positions, future_mask)
    logged_speed = logged_speed[0].numpy()
    valid_step = valid_step_pair[0].numpy()
    logged_future = sample["future_positions"]

    normalise = Normalize(
        min(float(predicted_speed[1:].min()), float(logged_speed[1:][valid_step[1:]].min())),
        max(float(predicted_speed[1:].max()), float(logged_speed[1:][valid_step[1:]].max())),
    )

    draw_scene(predicted_axis, sample)
    predicted_segments = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
    predicted_lines = LineCollection(predicted_segments, cmap=SPEED_COLORMAP, norm=normalise)
    predicted_lines.set_array(predicted_speed[1:])
    predicted_lines.set_linewidth(2.6)
    predicted_axis.add_collection(predicted_lines)
    frame(predicted_axis, "9. Predicted speed, best mode", half_width)

    draw_scene(logged_axis, sample)
    logged_segments = np.stack([logged_future[:-1], logged_future[1:]], axis=1)[valid_step[1:]]
    logged_lines = LineCollection(logged_segments, cmap=SPEED_COLORMAP, norm=normalise)
    logged_lines.set_array(logged_speed[1:][valid_step[1:]])
    logged_lines.set_linewidth(2.6)
    logged_axis.add_collection(logged_lines)
    frame(logged_axis, "logged speed", half_width)

    figure.colorbar(predicted_lines, cax=colorbar_axis, label="speed (m/s)")


def draw_along_cross_panel(axis, measured, turning_prediction, sample):
    predicted = turning_prediction["winning_trajectory_18"]
    logged = sample["future_positions"]
    headings = sample["future_headings"]
    valid = sample["future_mask"]
    along, cross = along_and_cross_track_per_step(
        torch.from_numpy(predicted), torch.from_numpy(logged), torch.from_numpy(headings)
    )
    magnitude = np.where(valid, np.hypot(along.numpy(), cross.numpy()), -1.0)
    step = int(np.argmax(magnitude))

    logged_point = logged[step]
    predicted_point = predicted[step]
    forward = headings[step]
    sideways = np.array([-forward[1], forward[0]])
    along_value = float(along[step])
    cross_value = float(cross[step])
    along_point = logged_point + forward * along_value
    cross_point = along_point + sideways * cross_value

    axis.plot(*zip(logged_point, along_point), c=PREDICTED_COLOR, linewidth=2.5)
    axis.plot(*zip(along_point, cross_point), c="#d94801", linewidth=2.5)
    axis.scatter(*logged_point, s=70, c=LOGGED_COLOR, zorder=5)
    axis.scatter(*predicted_point, s=70, c=PREDICTED_COLOR, marker="X", zorder=5)
    axis.annotate(
        f"along {abs(along_value):.1f} m", ((logged_point[0] + along_point[0]) / 2, (logged_point[1] + along_point[1]) / 2),
        fontsize=8, color=PREDICTED_COLOR, xytext=(4, 4), textcoords="offset points",
    )
    axis.annotate(
        f"cross {abs(cross_value):.1f} m", ((along_point[0] + cross_point[0]) / 2, (along_point[1] + cross_point[1]) / 2),
        fontsize=8, color="#d94801", xytext=(4, 4), textcoords="offset points",
    )
    overall_share = (
        measured["along_track_error"].mean()
        / (measured["along_track_error"].mean() + measured["cross_track_error"].mean())
    )
    axis.annotate(
        f"typically {overall_share:.0%} along-track", (0.02, 0.95), xycoords="axes fraction",
        fontsize=8, va="top",
    )
    margin = 4.0
    points = np.stack([logged_point, predicted_point, along_point, cross_point])
    axis.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    axis.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("10. Which way is the miss? (one real step)", fontsize=10, loc="left")
    for spine in axis.spines.values():
        spine.set_color("#bdbdbd")


def draw_turn_bucket_panel(axis, measured):
    assignment, labels = turn_bucket_assignment_and_labels(measured)
    total_error = measured["model_best_of_6"].sum()
    fair_share = 100.0 / TURN_BUCKET_COUNT

    shares, counts = [], []
    for bucket_index in range(TURN_BUCKET_COUNT):
        selected = assignment == bucket_index
        shares.append(measured["model_best_of_6"][selected].sum() / total_error * 100.0)
        counts.append(int(selected.sum()))

    positions = np.arange(TURN_BUCKET_COUNT)
    bars = axis.bar(positions, shares, color="#d94801", width=0.6)
    axis.axhline(fair_share, color="#525252", linewidth=1.0, linestyle="--")
    axis.text(-0.5, fair_share, "even split", fontsize=7, va="bottom", ha="left")
    for bar, share in zip(bars, shares):
        axis.annotate(
            f"{share:.0f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{label}\nn={count}" for label, count in zip(labels, counts)], fontsize=7
    )
    axis.set_ylabel("share of total error")
    axis.set_yticks([])
    axis.set_title("Harder turns carry more of the error", fontsize=10, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def draw_drift_panel(axis, measured):
    turn_assignment, _ = turn_bucket_assignment_and_labels(measured)
    hardest = (turn_assignment == TURN_BUCKET_COUNT - 1) & measured["future_fully_valid"]
    time_seconds = np.arange(1, contract.FUTURE_STEPS + 1) * contract.TIMESTEP_SECONDS
    logged_curve = measured["logged_cumulative_distance"][hardest].mean(axis=0)
    predicted_curve = measured["predicted_cumulative_distance"][hardest].mean(axis=0)

    axis.plot(time_seconds, logged_curve, c=LOGGED_COLOR, linewidth=2.2)
    axis.plot(time_seconds, predicted_curve, c=PREDICTED_COLOR, linewidth=2.2)
    axis.annotate("logged", (time_seconds[-1], logged_curve[-1]), fontsize=8, color=LOGGED_COLOR,
                  xytext=(-32, 4), textcoords="offset points")
    axis.annotate("predicted", (time_seconds[-1], predicted_curve[-1]), fontsize=8, color=PREDICTED_COLOR,
                  xytext=(-42, -12), textcoords="offset points")
    gap = logged_curve[-1] - predicted_curve[-1]
    axis.annotate(
        f"{gap:.1f} m behind at 8 s", (time_seconds[-1], (logged_curve[-1] + predicted_curve[-1]) / 2),
        fontsize=8, color="#252525", ha="right",
    )
    axis.set_xlabel("seconds into the future")
    axis.set_ylabel("distance travelled (m)")
    axis.set_title("The model falls behind mid-turn (hardest turns)", fontsize=10, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def draw_mode_utilisation_panel(axis, measured):
    winner_counts = np.bincount(measured["winning_mode_18"], minlength=int(model.QUERY_COUNT))
    sample_count = measured["winning_mode_18"].size
    shares = winner_counts / sample_count * 100.0
    colors = [DEAD_MODE_COLOR if count == 0 else PREDICTED_COLOR for count in winner_counts]
    axis.bar(np.arange(1, model.QUERY_COUNT + 1), shares, color=colors, width=0.7)
    dead_count = int((winner_counts == 0).sum())
    axis.annotate(
        f"{dead_count} of {model.QUERY_COUNT} modes never win", (0.02, 0.95), xycoords="axes fraction",
        fontsize=8, va="top",
    )
    axis.set_xlabel("mode number")
    axis.set_ylabel("share of predictions won")
    axis.set_xticks(np.arange(1, model.QUERY_COUNT + 1))
    axis.set_xticklabels(np.arange(1, model.QUERY_COUNT + 1), fontsize=6.5)
    axis.set_title("Which of the 18 guesses get used", fontsize=10, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def draw_controls_panel(axis, measured):
    rows = (
        ("Model, best of 6", measured["model_best_of_6"]),
        ("Constant velocity", measured["constant_velocity_best_of_6"]),
        ("Constant turn rate", measured["constant_turn_rate_best_of_6"]),
        ("Straight to an anchor", measured["anchor_null_best_of_6"]),
        ("Follow the lane", measured["lane_null_best_of_6"]),
    )
    labels = [label for label, _ in rows]
    means = [float(values.mean()) for _, values in rows]
    order = np.argsort(means)
    positions = np.arange(len(rows))
    colors = [PREDICTED_COLOR if labels[index] == "Model, best of 6" else "#969696" for index in order]

    axis.barh(positions, [means[index] for index in order], color=colors, height=0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels([labels[index] for index in order], fontsize=8)
    for position, index in zip(positions, order):
        axis.annotate(f"{means[index]:.2f} m", (means[index], position), fontsize=8,
                      xytext=(4, 0), textcoords="offset points", va="center")
    axis.set_xlabel("average miss over 8 s (m)")
    axis.set_title("How far off vs. simple guesses", fontsize=10, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def draw_signed_along_bias_panel(axis, measured):
    assignment, labels = turn_bucket_assignment_and_labels(measured)

    positions = np.arange(TURN_BUCKET_COUNT)
    bias = np.array([
        measured["signed_along_track_error"][assignment == bucket_index].mean()
        for bucket_index in range(TURN_BUCKET_COUNT)
    ])
    colors = ["#d94801" if value < 0.0 else PREDICTED_COLOR for value in bias]
    bars = axis.bar(positions, bias, color=colors, width=0.6)
    axis.axhline(0.0, color="#525252", linewidth=1.0)
    for bar, value in zip(bars, bias):
        axis.annotate(
            f"{value:.1f} m", (bar.get_x() + bar.get_width() / 2, value),
            ha="center", va="bottom" if value >= 0.0 else "top", fontsize=8,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylabel("mean signed along-track error (m)")
    axis.set_title("The falls-behind number (negative = model lags, by turn size)", fontsize=10, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def draw_path_length_distribution_panel(axis, measured):
    logged_final = measured["logged_cumulative_distance"][:, -1]
    predicted_final = measured["predicted_cumulative_distance"][:, -1]
    bins = np.linspace(0.0, max(float(logged_final.max()), float(predicted_final.max())), 40)

    axis.hist(logged_final, bins=bins, color=LOGGED_COLOR, alpha=0.55, label="logged")
    axis.hist(predicted_final, bins=bins, color=PREDICTED_COLOR, alpha=0.55, label="model, best mode")
    median_reach = float(np.median(measured["reachable_distance"]))
    axis.axvline(median_reach, color="#525252", linewidth=1.2, linestyle="--")
    axis.annotate(
        f"median reachable budget {median_reach:.0f} m", (median_reach, axis.get_ylim()[1]),
        fontsize=7, rotation=90, va="top", ha="right",
    )
    axis.set_xlabel("path length over 8 s (m)")
    axis.set_ylabel("samples")
    axis.legend(fontsize=7, frameon=False)
    axis.set_title("How far the model and the log actually travel", fontsize=10, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def draw_page_one(pdf, sample, unit_anchors, assigned_anchor_index):
    type_index, anchors = type_index_and_anchors(sample, unit_anchors)
    half_width = scene_half_width(sample, anchors)

    figure = plt.figure(figsize=SCENE_PAGE_FIGURE_SIZE)
    figure.suptitle(
        f"A turning {type_label(type_index)}, real prediction"
        f" (scenario {sample['scenario_id']} track {sample['track_id']}), same map and zoom throughout",
        fontsize=13,
    )
    grid = figure.add_gridspec(2, 2)

    draw_model_view_panel(figure.add_subplot(grid[0, 0]), sample, half_width)
    draw_anchor_panel(figure.add_subplot(grid[0, 1]), sample, anchors, half_width)
    draw_scene_and_logged_future(
        figure.add_subplot(grid[1, 0]), sample, "3. What the car actually did", half_width,
    )
    draw_owning_anchor_panel(
        figure.add_subplot(grid[1, 1]), sample, anchors, assigned_anchor_index, half_width
    )

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(figure)
    plt.close(figure)


def draw_page_two(pdf, sample, turning_prediction, half_width):
    figure = plt.figure(figsize=SCENE_PAGE_FIGURE_SIZE)
    figure.suptitle("Same vehicle, same map and zoom — every future the model considered", fontsize=13)
    grid = figure.add_gridspec(2, 2)

    draw_all_predictions_panel(figure.add_subplot(grid[0, 0]), sample, turning_prediction, half_width)
    best_raw_index = draw_kept_predictions_panel(
        figure.add_subplot(grid[0, 1]), sample, turning_prediction, half_width,
        "6. The 6 kept, shaded by confidence",
    )
    draw_uncertainty_panel(
        figure.add_subplot(grid[1, 0]), sample, turning_prediction, best_raw_index, half_width
    )
    draw_neighbour_future_panel(figure.add_subplot(grid[1, 1]), sample, turning_prediction, half_width)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(figure)
    plt.close(figure)
    return best_raw_index


def draw_page_three(
    pdf, sample, turning_prediction, best_raw_index, half_width, measured, hard_case, easy_case, unit_anchors
):
    type_index, _ = type_index_and_anchors(sample, unit_anchors)

    figure = plt.figure(figsize=PAGE_THREE_FIGURE_SIZE)
    figure.suptitle(
        f"Speed and the miss for the same {type_label(type_index)} above,"
        " a hard case and an easy case below (each its own scene, its own zoom)",
        fontsize=13,
    )
    grid = figure.add_gridspec(2, 3)

    draw_speed_on_path_panel(
        figure, grid[0, :], sample, turning_prediction, best_raw_index, half_width
    )
    draw_along_cross_panel(figure.add_subplot(grid[1, 0]), measured, turning_prediction, sample)

    hard_type_index, hard_anchors = type_index_and_anchors(hard_case["sample"], unit_anchors)
    hard_half_width = scene_half_width(hard_case["sample"], hard_anchors)
    draw_kept_predictions_panel(
        figure.add_subplot(grid[1, 1]), hard_case["sample"], hard_case, hard_half_width,
        f"11. A hard case: {type_label(hard_type_index)}"
        f" (scenario {hard_case['sample']['scenario_id']} track {hard_case['sample']['track_id']})",
    )

    easy_type_index, easy_anchors = type_index_and_anchors(easy_case["sample"], unit_anchors)
    easy_half_width = scene_half_width(easy_case["sample"], easy_anchors)
    draw_kept_predictions_panel(
        figure.add_subplot(grid[1, 2]), easy_case["sample"], easy_case, easy_half_width,
        f"12. An easy case: {type_label(easy_type_index)}"
        f" (scenario {easy_case['sample']['scenario_id']} track {easy_case['sample']['track_id']})",
    )

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(figure)
    plt.close(figure)


def draw_page_four(pdf, measured):
    figure = plt.figure(figsize=PAGE_FOUR_FIGURE_SIZE)
    figure.suptitle("Every measured number, drawn", fontsize=13)
    grid = figure.add_gridspec(2, 3)

    draw_turn_bucket_panel(figure.add_subplot(grid[0, 0]), measured)
    draw_mode_utilisation_panel(figure.add_subplot(grid[0, 1]), measured)
    draw_controls_panel(figure.add_subplot(grid[0, 2]), measured)
    draw_drift_panel(figure.add_subplot(grid[1, 0]), measured)
    draw_signed_along_bias_panel(figure.add_subplot(grid[1, 1]), measured)
    draw_path_length_distribution_panel(figure.add_subplot(grid[1, 2]), measured)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(figure)
    plt.close(figure)


def main():
    checkpoint_path = Path(sys.argv[1])
    staged_directory = Path(sys.argv[2])
    anchors_path = Path(sys.argv[3])
    output_path = Path(sys.argv[4])

    with np.load(anchors_path) as anchors_file:
        contract.check_artifact_provenance(
            anchors_file["provenance"] if "provenance" in anchors_file else None,
            anchors_path, "Refit them with fit_anchors.py.",
        )
        unit_anchors_array = anchors_file["unit_anchors"]
    unit_anchors_tensor = torch.from_numpy(unit_anchors_array)
    predictor = model.MotionPredictor(unit_anchors_tensor)
    predictor.load_state_dict(model.load_checkpoint_state(checkpoint_path)["model_state"])
    predictor.eval()

    turning_sample = pick_turning_sample(staged_directory)
    measured, turning_prediction, hard_case, easy_case = gather(
        staged_directory, predictor, unit_anchors_tensor,
        str(turning_sample["scenario_id"]), int(turning_sample["track_id"]),
    )

    _, turning_anchors = type_index_and_anchors(turning_sample, unit_anchors_array)
    half_width = scene_half_width(turning_sample, turning_anchors)

    with PdfPages(output_path) as pdf:
        draw_page_one(pdf, turning_sample, unit_anchors_array, turning_prediction["assigned_anchor_index"])
        best_raw_index = draw_page_two(pdf, turning_sample, turning_prediction, half_width)
        draw_page_three(
            pdf, turning_sample, turning_prediction, best_raw_index, half_width, measured,
            hard_case, easy_case, unit_anchors_array,
        )
        draw_page_four(pdf, measured)

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
