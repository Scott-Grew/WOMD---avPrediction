# > Training-time loader: staged scenario .npz in, per-agent samples out
# Everything on disk is in the storage frame (self-driving car at origin). A sample re-frames
# the world around one predicted agent: its history, every other agent (ragged, uncapped),
# and the map dots inside a speed-stretched crop, all rotated and shifted to that agent's view.

import numpy as np

from womd import contract, frame_ops

BASE_RADIUS_METRES = 80.0
STRETCH_GAIN = 0.5

# Crop membership: stretched-forward half-ellipse in the agent's frame. Rear half stays a
# circle of base_radius; the forward semi-axis is base_radius * forward_stretch, where the
# stretch grows with current speed - a fast car needs to see far ahead, a stopped one around.
def inside_crop(agent_frame_points, base_radius, forward_stretch):
    x = agent_frame_points[:, 0]
    y = agent_frame_points[:, 1]
    forward = (x / (base_radius * forward_stretch)) ** 2 + (y / base_radius) ** 2 <= 1.0
    rear = (x / base_radius) ** 2 + (y / base_radius) ** 2 <= 1.0
    return np.where(x > 0.0, forward, rear)

# Which tracks become samples: the three predicted object types, valid at the "now" step.
# An agent invalid at now has no defined frame to predict from. With designated_targets_only
# the set narrows further to the agents WOMD itself asks for - training's population since
# Scott's 2026-08-14 ruling; the whole eligible set stays reachable for comparison.
def eligible_track_indices(track_rows, track_valid, is_designated_target, designated_targets_only):
    now_valid = track_valid[:, contract.CURRENT_STEP_INDEX]
    predicted_type = track_rows[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE][:,
                                            :contract.NUM_OBJECT_TYPES].sum(axis=1) > 0
    selected = now_valid & predicted_type
    if designated_targets_only:
        selected = selected & is_designated_target
    return np.flatnonzero(selected)

# The sample's frame: where the predicted agent sits and faces at "now", read off its stored
# row. arctan2(sin, cos) recovers the angle from the stored pair - the recovery V6 guards.
def sample_frame(track_rows, track_index):
    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    origin = now_row[contract.AGENT_POSITION]
    heading = np.arctan2(now_row[contract.AGENT_HEADING_SINE], now_row[contract.AGENT_HEADING_COSINE])
    return origin, heading

# Re-frame agent rows into the sample's frame. Three rules for three quantities: positions
# shift then rotate, velocities rotate only, headings subtract - done through the cos/sin
# angle-difference identities so the pair never round-trips through an angle.
def track_rows_to_agent_frame(rows, origin, heading):
    reframed = rows.copy()
    reframed[..., contract.AGENT_POSITION] = frame_ops.positions_to_agent_frame(
        rows[..., contract.AGENT_POSITION], origin, heading
    )
    reframed[..., contract.AGENT_VELOCITY] = frame_ops.directions_to_agent_frame(
        rows[..., contract.AGENT_VELOCITY], heading
    )
    heading_cosine = rows[..., contract.AGENT_HEADING_COSINE]
    heading_sine = rows[..., contract.AGENT_HEADING_SINE]
    rotation_cosine, rotation_sine = np.cos(heading), np.sin(heading)
    reframed[..., contract.AGENT_HEADING_COSINE] = heading_cosine * rotation_cosine + heading_sine * rotation_sine
    reframed[..., contract.AGENT_HEADING_SINE] = heading_sine * rotation_cosine - heading_cosine * rotation_sine
    return reframed

def crop_and_reframe_map(map_rows, dot_polyline_index, dot_has_traffic_signal, origin, heading, speed):
    agent_frame_positions = frame_ops.positions_to_agent_frame(map_rows[:, contract.MAP_POSITION], origin, heading)
    crop_mask = inside_crop(agent_frame_positions, BASE_RADIUS_METRES, 1.0 + STRETCH_GAIN * speed)
    reframed = map_rows[crop_mask]
    reframed[:, contract.MAP_POSITION] = agent_frame_positions[crop_mask]
    reframed[:, contract.MAP_DIRECTION] = frame_ops.directions_to_agent_frame(reframed[:, contract.MAP_DIRECTION], heading)
    has_signal = dot_has_traffic_signal[crop_mask]
    reframed[has_signal, contract.MAP_STOP_POINT] = frame_ops.positions_to_agent_frame(reframed[has_signal, contract.MAP_STOP_POINT], origin, heading)
    surviving_polylines, compact_polyline_index = np.unique(
        dot_polyline_index[crop_mask], return_inverse=True
    )
    return reframed, compact_polyline_index

def read_scenario(scenario_path):
    with np.load(scenario_path) as scenario_file:
        scenario_array = {name: scenario_file[name] for name in scenario_file.files}
    feature_lengths = scenario_array["feature_lengths"]
    scenario_array["map_dot_polyline_index"] = np.repeat(np.arange(len(feature_lengths)), feature_lengths)
    scenario_array["map_dot_has_traffic_signal"] = scenario_array["map_rows"][:, contract.MAP_SIGNAL_STATE].any(axis=1)
    return scenario_array

def build_sample(scenario_array, track_index):
    track_rows = scenario_array["track_rows"]
    track_valid = scenario_array["track_valid"]
    origin, heading = sample_frame(track_rows, track_index)

    agent_track = track_rows_to_agent_frame(track_rows[track_index], origin, heading)
    future_positions = agent_track[contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION]

    neighbour_indices = np.flatnonzero(np.arange(len(track_rows)) != track_index)
    neighbour_history = track_rows_to_agent_frame(
        track_rows[neighbour_indices, :contract.HISTORY_STEPS], origin, heading
    )

    speed = float(np.linalg.norm(track_rows[track_index, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY]))
    agent_map, map_polyline_index = crop_and_reframe_map(
        scenario_array["map_rows"],
        scenario_array["map_dot_polyline_index"],
        scenario_array["map_dot_has_traffic_signal"],
        origin,
        heading,
        speed,
    )

    return {
        "agent_history": agent_track[:contract.HISTORY_STEPS],
        "agent_history_mask": track_valid[track_index, :contract.HISTORY_STEPS],
        "future_positions": future_positions,
        "future_mask": track_valid[track_index, contract.CURRENT_STEP_INDEX + 1:],
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": track_valid[neighbour_indices, :contract.HISTORY_STEPS],
        "map_rows": agent_map,
        "map_polyline_index": map_polyline_index.astype(np.int64),
        "frame_origin": origin,
        "frame_heading": heading,
        "scenario_id": scenario_array["scenario_id"],
        "track_id": scenario_array["track_ids"][track_index],
        "is_designated_target": scenario_array["is_designated_target"][track_index],
        "is_object_of_interest": scenario_array["is_object_of_interest"][track_index],
    }

# Only the neighbour and polyline axes are padded to a common width. The map dots stay ragged:
# every sample's dots are concatenated into one flat block, and each dot carries the global
# polyline slot (sample_index * max_polylines_in_batch + polyline index within the sample) it
# pools into, so the dot axis never pays for the batch's largest crop.
def build_batch(samples):
    batch_size = len(samples)
    max_neighbours = max(sample["neighbour_history"].shape[0] for sample in samples)
    max_polylines_in_batch = max(
        int(sample["map_polyline_index"].max()) + 1 if len(sample["map_polyline_index"]) else 0
        for sample in samples
    )

    neighbour_history = np.zeros(
        (batch_size, max_neighbours, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32
    )
    neighbour_history_mask = np.zeros((batch_size, max_neighbours, contract.HISTORY_STEPS), dtype=bool)

    for sample_index, sample in enumerate(samples):
        neighbour_count = sample["neighbour_history"].shape[0]
        neighbour_history[sample_index, :neighbour_count] = sample["neighbour_history"]
        neighbour_history_mask[sample_index, :neighbour_count] = sample["neighbour_history_mask"]

    return {
        "agent_history": np.stack([sample["agent_history"] for sample in samples]),
        "agent_history_mask": np.stack([sample["agent_history_mask"] for sample in samples]),
        "future_positions": np.stack([sample["future_positions"] for sample in samples]),
        "future_mask": np.stack([sample["future_mask"] for sample in samples]),
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": neighbour_history_mask,
        "map_rows": np.concatenate(
            [sample["map_rows"] for sample in samples], dtype=np.float32
        ),
        "map_dot_polyline_slot": np.concatenate(
            [
                sample["map_polyline_index"] + sample_index * max_polylines_in_batch
                for sample_index, sample in enumerate(samples)
            ],
            dtype=np.int64,
        ),
        "max_polylines_in_batch": np.array(max_polylines_in_batch, dtype=np.int64),
    }
