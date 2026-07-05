import numpy as np

from waymo_open_dataset.protos import scenario_pb2
from womd import contract, frame


def object_type_one_hot(object_type_name):
    encoded = np.zeros(contract.NUM_OBJECT_TYPES, dtype=np.float32)
    if object_type_name in contract.PREDICTED_OBJECT_TYPES:
        encoded[contract.PREDICTED_OBJECT_TYPES.index(object_type_name)] = 1.0
    return encoded


def track_object_type_name(track):
    return scenario_pb2.Track.ObjectType.Name(track.object_type)


def predictable_track_indices(scenario, benchmark_targets_only=False):
    if benchmark_targets_only and len(scenario.tracks_to_predict) > 0:
        return [target.track_index for target in scenario.tracks_to_predict]
    indices = []
    for track_index, track in enumerate(scenario.tracks):
        if len(track.states) <= contract.CURRENT_STEP_INDEX:
            continue
        if not track.states[contract.CURRENT_STEP_INDEX].valid:
            continue
        if track_object_type_name(track) not in contract.PREDICTED_OBJECT_TYPES:
            continue
        indices.append(track_index)
    return indices


def _agent_track_arrays(track):
    step_count = len(track.states)
    positions = np.zeros((step_count, 2), dtype=np.float64)
    headings = np.zeros(step_count, dtype=np.float64)
    velocities = np.zeros((step_count, 2), dtype=np.float64)
    extents = np.zeros((step_count, 2), dtype=np.float64)
    valid = np.zeros(step_count, dtype=bool)
    for step_index, state in enumerate(track.states):
        positions[step_index] = (state.center_x, state.center_y)
        headings[step_index] = state.heading
        velocities[step_index] = (state.velocity_x, state.velocity_y)
        extents[step_index] = (state.length, state.width)
        valid[step_index] = state.valid
    type_encoding = object_type_one_hot(track_object_type_name(track))
    return positions, headings, velocities, extents, valid, type_encoding


def _history_feature_block(track, frame_origin, frame_heading):
    positions, headings, velocities, extents, valid, type_encoding = _agent_track_arrays(track)
    history_slice = slice(0, contract.HISTORY_STEPS)
    local_positions = frame.positions_to_agent_frame(
        positions[history_slice], frame_origin, frame_heading
    )
    local_headings = frame.headings_to_agent_frame(headings[history_slice], frame_heading)
    local_velocities = frame.directions_to_agent_frame(velocities[history_slice], frame_heading)

    block = np.zeros((contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32)
    present = valid[history_slice]
    block[:, 0:2] = local_positions
    block[:, 2] = np.cos(local_headings)
    block[:, 3] = np.sin(local_headings)
    block[:, 4:6] = local_velocities
    block[:, 6:8] = extents[history_slice]
    block[:, 8:] = type_encoding
    block[~present] = 0.0
    return block, present.copy()


def _map_feature_polylines(map_feature):
    kind_to_points = {
        "lane": lambda feature: feature.lane.polyline,
        "road_line": lambda feature: feature.road_line.polyline,
        "road_edge": lambda feature: feature.road_edge.polyline,
        "crosswalk": lambda feature: feature.crosswalk.polygon,
        "speed_bump": lambda feature: feature.speed_bump.polygon,
        "driveway": lambda feature: feature.driveway.polygon,
    }
    kind = map_feature.WhichOneof("feature_data")
    if kind == "stop_sign":
        position = map_feature.stop_sign.position
        return kind, [np.array([[position.x, position.y]], dtype=np.float64)]
    if kind not in kind_to_points:
        return None, []
    points = kind_to_points[kind](map_feature)
    if len(points) == 0:
        return None, []
    coordinates = np.array([[point.x, point.y] for point in points], dtype=np.float64)
    chunks = [
        coordinates[start : start + contract.POINTS_PER_POLYLINE]
        for start in range(0, len(coordinates), contract.POINTS_PER_POLYLINE)
    ]
    return kind, chunks


def _polyline_feature_block(chunk_local, kind):
    block = np.zeros(
        (contract.POINTS_PER_POLYLINE, contract.MAP_FEATURE_DIM), dtype=np.float32
    )
    mask = np.zeros(contract.POINTS_PER_POLYLINE, dtype=bool)
    point_count = len(chunk_local)
    block[:point_count, 0:2] = chunk_local
    if point_count > 1:
        directions = np.diff(chunk_local, axis=0)
        block[: point_count - 1, 2:4] = directions
        block[point_count - 1, 2:4] = directions[-1]
    block[:point_count, 4 + contract.MAP_POLYLINE_KINDS.index(kind)] = 1.0
    mask[:point_count] = True
    return block, mask


def _map_blocks(scenario, frame_origin, frame_heading):
    candidates = []
    for map_feature in scenario.map_features:
        kind, chunks = _map_feature_polylines(map_feature)
        if kind is None:
            continue
        for chunk in chunks:
            chunk_local = frame.positions_to_agent_frame(chunk, frame_origin, frame_heading)
            distance = float(np.min(np.linalg.norm(chunk_local, axis=1)))
            candidates.append((distance, chunk_local, kind))
    candidates.sort(key=lambda entry: entry[0])

    polylines = np.zeros(
        (contract.MAX_MAP_POLYLINES, contract.POINTS_PER_POLYLINE, contract.MAP_FEATURE_DIM),
        dtype=np.float32,
    )
    masks = np.zeros((contract.MAX_MAP_POLYLINES, contract.POINTS_PER_POLYLINE), dtype=bool)
    for slot, (_, chunk_local, kind) in enumerate(candidates[: contract.MAX_MAP_POLYLINES]):
        polylines[slot], masks[slot] = _polyline_feature_block(chunk_local, kind)
    return polylines, masks


def _neighbour_blocks(scenario, target_index, frame_origin, frame_heading):
    candidates = []
    for track_index, track in enumerate(scenario.tracks):
        if track_index == target_index:
            continue
        if len(track.states) <= contract.CURRENT_STEP_INDEX:
            continue
        current_state = track.states[contract.CURRENT_STEP_INDEX]
        if not current_state.valid:
            continue
        offset = np.array([current_state.center_x, current_state.center_y]) - frame_origin
        candidates.append((float(np.linalg.norm(offset)), track_index, track))
    candidates.sort(key=lambda entry: entry[0])

    histories = np.zeros(
        (contract.MAX_NEIGHBOUR_AGENTS, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        dtype=np.float32,
    )
    masks = np.zeros((contract.MAX_NEIGHBOUR_AGENTS, contract.HISTORY_STEPS), dtype=bool)
    for slot, (_, _, track) in enumerate(candidates[: contract.MAX_NEIGHBOUR_AGENTS]):
        histories[slot], masks[slot] = _history_feature_block(
            track, frame_origin, frame_heading
        )
    return histories, masks


def build_sample(scenario, target_index):
    target_track = scenario.tracks[target_index]
    current_state = target_track.states[contract.CURRENT_STEP_INDEX]
    frame_origin = np.array([current_state.center_x, current_state.center_y], dtype=np.float64)
    frame_heading = float(current_state.heading)

    agent_history, agent_history_mask = _history_feature_block(
        target_track, frame_origin, frame_heading
    )
    neighbour_history, neighbour_history_mask = _neighbour_blocks(
        scenario, target_index, frame_origin, frame_heading
    )
    map_polylines, map_polylines_mask = _map_blocks(scenario, frame_origin, frame_heading)

    future_positions = np.zeros((contract.FUTURE_STEPS, 2), dtype=np.float32)
    future_mask = np.zeros(contract.FUTURE_STEPS, dtype=bool)
    for future_index in range(contract.FUTURE_STEPS):
        step_index = contract.CURRENT_STEP_INDEX + 1 + future_index
        if step_index >= len(target_track.states):
            break
        state = target_track.states[step_index]
        if not state.valid:
            continue
        local = frame.positions_to_agent_frame(
            np.array([[state.center_x, state.center_y]]), frame_origin, frame_heading
        )
        future_positions[future_index] = local[0]
        future_mask[future_index] = True

    return {
        "agent_history": agent_history,
        "agent_history_mask": agent_history_mask,
        "neighbour_history": neighbour_history,
        "neighbour_history_mask": neighbour_history_mask,
        "map_polylines": map_polylines,
        "map_polylines_mask": map_polylines_mask,
        "future_positions": future_positions,
        "future_mask": future_mask,
        "frame_origin": frame_origin.astype(np.float32),
        "frame_heading": np.float32(frame_heading),
    }


def build_scenario_samples(scenario, benchmark_targets_only=False, require_full_future=True):
    samples = []
    for target_index in predictable_track_indices(scenario, benchmark_targets_only):
        sample = build_sample(scenario, target_index)
        if require_full_future and not sample["future_mask"].all():
            continue
        samples.append(sample)
    return samples
