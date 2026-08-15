# > Training-time loader: staged scenario .npz in, per-agent samples out
# Everything on disk is in the storage frame (self-driving car at origin). A sample re-frames
# the world around one predicted agent: its history, every other agent (ragged, uncapped),
# and the map dots inside a speed-stretched crop, all rotated and shifted to that agent's view.

import heapq

import numpy as np

from womd import contract, frame_ops

BASE_RADIUS_METRES = 80.0
STRETCH_GAIN = 0.5
DRIVABLE_KIND_COLUMNS = [
    contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index(kind)
    for kind in ("lane", "crosswalk", "driveway", "speed_bump")
]

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

def lane_context_per_polyline(scenario_array, agent_position, agent_heading):
    map_rows = scenario_array["map_rows"]
    feature_lengths = scenario_array["feature_lengths"]
    feature_ids = scenario_array["feature_ids"]
    dot_polyline_index = scenario_array["map_dot_polyline_index"]
    lane_context = np.zeros((len(feature_ids), contract.LANE_CONTEXT_DIM), dtype=np.float32)

    lane_kind_column = contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index("lane")
    first_dot_of_polyline = np.cumsum(feature_lengths) - feature_lengths
    polyline_is_lane = map_rows[first_dot_of_polyline, lane_kind_column] == 1.0
    lane_dot_indices = np.flatnonzero(polyline_is_lane[dot_polyline_index])
    if len(lane_dot_indices) == 0:
        return lane_context

    lane_dot_rows = map_rows[lane_dot_indices]
    agent_offsets = lane_dot_rows[:, contract.MAP_POSITION] - agent_position
    agent_distances = np.linalg.norm(agent_offsets, axis=1)
    lane_context[:, contract.LANE_CONTEXT_AGENT_LANE_DISTANCE] = (
        agent_distances.min() / contract.DISTANCE_NORMALISER_METRES
    )
    nearest = np.flatnonzero(agent_distances == agent_distances.min())
    heading_agreement = lane_dot_rows[nearest][:, contract.MAP_DIRECTION] @ agent_heading
    agent_lane_dot = lane_dot_indices[nearest[np.argmax(heading_agreement)]]
    agent_lane_id = int(feature_ids[dot_polyline_index[agent_lane_dot]])

    polyline_row_of_lane_id = {int(feature_id): row for row, feature_id in enumerate(feature_ids)}
    arc_length_of_polyline = np.zeros(len(feature_ids))
    dot_gaps = np.linalg.norm(np.diff(map_rows[:, contract.MAP_POSITION], axis=0), axis=1)
    gap_stays_within_polyline = dot_polyline_index[1:] == dot_polyline_index[:-1]
    np.add.at(
        arc_length_of_polyline,
        dot_polyline_index[1:][gap_stays_within_polyline],
        dot_gaps[gap_stays_within_polyline],
    )

    exits_of_lane = {}
    for source_id, destination_id, _ in scenario_array["lane_connections"].tolist():
        exits_of_lane.setdefault(source_id, []).append(destination_id)
        if source_id in polyline_row_of_lane_id and destination_id not in polyline_row_of_lane_id:
            lane_context[polyline_row_of_lane_id[source_id],
                         contract.LANE_CONTEXT_HAS_EXIT_NOT_PRESENT] = 1.0
        if destination_id in polyline_row_of_lane_id and source_id not in polyline_row_of_lane_id:
            lane_context[polyline_row_of_lane_id[destination_id],
                         contract.LANE_CONTEXT_HAS_ENTRY_NOT_PRESENT] = 1.0

    graph_distance_metres = {agent_lane_id: 0.0}
    frontier = [(0.0, agent_lane_id)]
    while frontier:
        distance_metres, lane_id = heapq.heappop(frontier)
        if distance_metres > graph_distance_metres[lane_id]:
            continue
        distance_past_lane = distance_metres + arc_length_of_polyline[polyline_row_of_lane_id[lane_id]]
        for destination_id in exits_of_lane.get(lane_id, ()):
            if destination_id not in polyline_row_of_lane_id:
                continue
            if distance_past_lane < graph_distance_metres.get(destination_id, np.inf):
                graph_distance_metres[destination_id] = distance_past_lane
                heapq.heappush(frontier, (distance_past_lane, destination_id))

    for lane_id, distance_metres in graph_distance_metres.items():
        polyline_row = polyline_row_of_lane_id[lane_id]
        lane_context[polyline_row, contract.LANE_CONTEXT_REACHABLE] = 1.0
        lane_context[polyline_row, contract.LANE_CONTEXT_GRAPH_DISTANCE] = (
            distance_metres / contract.DISTANCE_NORMALISER_METRES
        )

    controlled_lanes = scenario_array["stop_sign_controlled_lanes"][:, contract.STOP_SIGN_CONTROLLED_LANE]
    for controlled_lane_id in controlled_lanes.tolist():
        if controlled_lane_id in polyline_row_of_lane_id:
            lane_context[polyline_row_of_lane_id[controlled_lane_id],
                         contract.LANE_CONTEXT_HAS_STOP_SIGN] = 1.0
    return lane_context

# Crop the map to the agent's view, reframe it, and cut the surviving dots into the chunks that
# become attention tokens: at most contract.MAP_CHUNK_DOTS consecutive dots of one polyline, so a
# token summarises a bounded stretch of road instead of a whole polyline of any length. Chunking
# follows the crop, so a polyline that survives in part is cut by its surviving dots. Chunk indices
# run 0..chunk count - 1 with no gaps, and a chunk's dots stay contiguous in the returned rows.
# The signal history is per polyline, so every chunk cut from a polyline is handed that polyline's
# history - the dots never carry it, and the model attaches it after pooling.
def crop_and_reframe_map(map_rows, dot_polyline_index, polyline_signal_histories,
                         polyline_lane_context, origin, heading, speed):
    agent_frame_positions = frame_ops.positions_to_agent_frame(map_rows[:, contract.MAP_POSITION], origin, heading)
    crop_mask = inside_crop(agent_frame_positions, BASE_RADIUS_METRES, 1.0 + STRETCH_GAIN * speed)
    reframed = map_rows[crop_mask]
    reframed[:, contract.MAP_POSITION] = agent_frame_positions[crop_mask]
    reframed[:, contract.MAP_DIRECTION] = frame_ops.directions_to_agent_frame(reframed[:, contract.MAP_DIRECTION], heading)
    has_signal = polyline_signal_histories.any(axis=(1, 2))[dot_polyline_index[crop_mask]]
    reframed[has_signal, contract.MAP_STOP_POINT] = frame_ops.positions_to_agent_frame(reframed[has_signal, contract.MAP_STOP_POINT], origin, heading)
    surviving_polylines, first_dot_position, compact_polyline_index, dots_per_polyline = np.unique(
        dot_polyline_index[crop_mask], return_index=True, return_inverse=True, return_counts=True
    )
    position_within_polyline = np.arange(len(compact_polyline_index)) - first_dot_position[compact_polyline_index]
    chunks_per_polyline = (dots_per_polyline + contract.MAP_CHUNK_DOTS - 1) // contract.MAP_CHUNK_DOTS
    first_chunk_of_polyline = np.cumsum(chunks_per_polyline) - chunks_per_polyline
    dot_chunk_index = (first_chunk_of_polyline[compact_polyline_index]
                       + position_within_polyline // contract.MAP_CHUNK_DOTS)
    chunk_polyline = np.repeat(surviving_polylines, chunks_per_polyline)
    return (reframed, dot_chunk_index, polyline_signal_histories[chunk_polyline],
            polyline_lane_context[chunk_polyline])

# feature_lengths is the ONLY thing tying a dot back to the polyline it came from: the dots are one
# flat block and the lengths cut it. A staging bug that leaves the two out of step shifts every dot
# after the first bad polyline onto a neighbour's identity, which the model reads as a plausible map
# and the loss descends against without any signature at all, so the agreement is checked on read.
def read_scenario(scenario_path):
    with np.load(scenario_path) as scenario_file:
        scenario_array = {name: scenario_file[name] for name in scenario_file.files}
    feature_lengths = scenario_array["feature_lengths"]
    map_row_count = len(scenario_array["map_rows"])
    assert feature_lengths.sum() == map_row_count, (
        f"feature_lengths sum to {feature_lengths.sum()} but {scenario_path} holds"
        f" {map_row_count} map rows"
    )
    scenario_array["map_dot_polyline_index"] = np.repeat(np.arange(len(feature_lengths)), feature_lengths)
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

    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    speed = float(np.linalg.norm(now_row[contract.AGENT_VELOCITY]))
    polyline_lane_context = lane_context_per_polyline(
        scenario_array,
        now_row[contract.AGENT_POSITION],
        now_row[contract.AGENT_HEADING_COSINE:contract.AGENT_HEADING_SINE + 1],
    )
    (agent_map, map_chunk_index, map_chunk_signal_history,
     map_chunk_lane_context) = crop_and_reframe_map(
        scenario_array["map_rows"],
        scenario_array["map_dot_polyline_index"],
        scenario_array["polyline_signal_histories"],
        polyline_lane_context,
        origin,
        heading,
        speed,
    )
    is_drivable_dot = (agent_map[:, DRIVABLE_KIND_COLUMNS] == 1.0).any(axis=1)

    return {
        "agent_history": agent_track[:contract.HISTORY_STEPS],
        "agent_history_mask": track_valid[track_index, :contract.HISTORY_STEPS],
        "future_positions": future_positions,
        "future_headings": agent_track[contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_HEADING_COSINE:contract.AGENT_HEADING_SINE + 1],
        "future_mask": track_valid[track_index, contract.CURRENT_STEP_INDEX + 1:],
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": track_valid[neighbour_indices, :contract.HISTORY_STEPS],
        "map_rows": agent_map,
        "map_chunk_index": map_chunk_index.astype(np.int64),
        "map_chunk_signal_history": map_chunk_signal_history,
        "map_chunk_lane_context": map_chunk_lane_context,
        "drivable_positions": agent_map[is_drivable_dot, contract.MAP_POSITION],
        "frame_origin": origin,
        "frame_heading": heading,
        "scenario_id": scenario_array["scenario_id"],
        "track_id": scenario_array["track_ids"][track_index],
        "is_designated_target": scenario_array["is_designated_target"][track_index],
        "is_object_of_interest": scenario_array["is_object_of_interest"][track_index],
    }

# Only the neighbour, chunk and drivable-dot axes are padded to a common width - the chunk signal histories
# pad on the chunk axis with the zeros an unsignalled chunk already carries. The map dots stay ragged:
# every sample's dots are concatenated into one flat block, and each dot carries the global
# chunk slot (sample_index * max_chunks_in_batch + chunk index within the sample) it pools
# into, so the dot axis never pays for the batch's largest crop. The slot and width keep the
# names the model reads them under.
def build_batch(samples):
    batch_size = len(samples)
    max_neighbours = max(sample["neighbour_history"].shape[0] for sample in samples)
    max_chunks_in_batch = max(
        int(sample["map_chunk_index"].max()) + 1 if len(sample["map_chunk_index"]) else 0
        for sample in samples
    )
    max_drivable_dots = max(sample["drivable_positions"].shape[0] for sample in samples)

    neighbour_history = np.zeros(
        (batch_size, max_neighbours, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32
    )
    neighbour_history_mask = np.zeros((batch_size, max_neighbours, contract.HISTORY_STEPS), dtype=bool)
    map_chunk_signal_history = np.zeros(
        (batch_size, max_chunks_in_batch, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES),
        dtype=np.float32,
    )
    map_chunk_lane_context = np.zeros(
        (batch_size, max_chunks_in_batch, contract.LANE_CONTEXT_DIM), dtype=np.float32
    )
    drivable_positions = np.zeros((batch_size, max_drivable_dots, 2), dtype=np.float32)
    drivable_mask = np.zeros((batch_size, max_drivable_dots), dtype=bool)

    for sample_index, sample in enumerate(samples):
        neighbour_count = sample["neighbour_history"].shape[0]
        neighbour_history[sample_index, :neighbour_count] = sample["neighbour_history"]
        neighbour_history_mask[sample_index, :neighbour_count] = sample["neighbour_history_mask"]
        chunk_count = sample["map_chunk_signal_history"].shape[0]
        map_chunk_signal_history[sample_index, :chunk_count] = sample["map_chunk_signal_history"]
        map_chunk_lane_context[sample_index, :chunk_count] = sample["map_chunk_lane_context"]
        drivable_count = sample["drivable_positions"].shape[0]
        drivable_positions[sample_index, :drivable_count] = sample["drivable_positions"]
        drivable_mask[sample_index, :drivable_count] = True

    return {
        "agent_history": np.stack([sample["agent_history"] for sample in samples]),
        "agent_history_mask": np.stack([sample["agent_history_mask"] for sample in samples]),
        "future_positions": np.stack([sample["future_positions"] for sample in samples]),
        "future_headings": np.stack([sample["future_headings"] for sample in samples]),
        "future_mask": np.stack([sample["future_mask"] for sample in samples]),
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": neighbour_history_mask,
        "map_rows": np.concatenate(
            [sample["map_rows"] for sample in samples], dtype=np.float32
        ),
        "map_dot_polyline_slot": np.concatenate(
            [
                sample["map_chunk_index"] + sample_index * max_chunks_in_batch
                for sample_index, sample in enumerate(samples)
            ],
            dtype=np.int64,
        ),
        "map_chunk_signal_history": map_chunk_signal_history,
        "map_chunk_lane_context": map_chunk_lane_context,
        "drivable_positions": drivable_positions,
        "drivable_mask": drivable_mask,
        "max_polylines_in_batch": np.array(max_chunks_in_batch, dtype=np.int64),
    }
