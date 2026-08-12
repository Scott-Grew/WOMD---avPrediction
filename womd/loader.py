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
# An agent invalid at now has no defined frame to predict from.
def eligible_track_indices(track_rows, track_valid):
    now_valid = track_valid[:, contract.CURRENT_STEP_INDEX]
    predicted_type = track_rows[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE][:, 
                                            :contract.NUM_OBJECT_TYPES].sum(axis=1) > 0
    return np.flatnonzero(now_valid & predicted_type)

# The sample's frame: where the predicted agent sits and faces at "now", read off its stored
# row. arctan2(sin, cos) recovers the angle from the stored pair - the recovery V6 guards.
def sample_frame(track_rows, track_index):
    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    origin = now_row[contract.AGENT_POSITION].astype(np.float64)
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


def map_rows_to_agent_frame(map_rows, origin, heading):
    reframed = map_rows.copy()
    reframed[:, contract.MAP_POSITION] = frame_ops.positions_to_agent_frame(
        map_rows[:, contract.MAP_POSITION], origin, heading
    )
    reframed[:, contract.MAP_DIRECTION] = frame_ops.directions_to_agent_frame(
        map_rows[:, contract.MAP_DIRECTION], heading
    )
    has_signal = map_rows[:, contract.MAP_SIGNAL_STATE].any(axis=1)
    reframed[has_signal, contract.MAP_STOP_POINT] = frame_ops.positions_to_agent_frame(
        map_rows[has_signal, contract.MAP_STOP_POINT], origin, heading
    )
    return reframed


def crop_map(agent_frame_map_rows, speed):
    crop_mask = inside_crop(
        agent_frame_map_rows[:, contract.MAP_POSITION],
        BASE_RADIUS_METRES,
        1.0 + STRETCH_GAIN * speed,
    )
    return agent_frame_map_rows[crop_mask]


def build_sample(scenario_file, track_index):
    track_rows = scenario_file["track_rows"]
    track_valid = scenario_file["track_valid"]
    origin, heading = sample_frame(track_rows, track_index)

    agent_track = track_rows_to_agent_frame(track_rows[track_index], origin, heading)
    future_positions = agent_track[contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION]

    neighbour_indices = np.flatnonzero(np.arange(len(track_rows)) != track_index)
    neighbour_history = track_rows_to_agent_frame(
        track_rows[neighbour_indices, :contract.HISTORY_STEPS], origin, heading
    )

    speed = float(np.linalg.norm(track_rows[track_index, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY]))
    agent_map = crop_map(map_rows_to_agent_frame(scenario_file["map_rows"], origin, heading), speed)

    return {
        "agent_history": agent_track[:contract.HISTORY_STEPS],
        "agent_history_mask": track_valid[track_index, :contract.HISTORY_STEPS],
        "future_positions": future_positions,
        "future_mask": track_valid[track_index, contract.CURRENT_STEP_INDEX + 1:],
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": track_valid[neighbour_indices, :contract.HISTORY_STEPS],
        "map_rows": agent_map,
        "frame_origin": origin,
        "frame_heading": heading,
        "scenario_id": scenario_file["scenario_id"],
        "track_id": scenario_file["track_ids"][track_index],
        "is_designated_target": scenario_file["is_designated_target"][track_index],
        "is_object_of_interest": scenario_file["is_object_of_interest"][track_index],
    }
