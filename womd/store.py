# > Complete Scenerio to disk pipeline
# store.py takes the senario abstraction from contrat.py and builds specifc objects from WOMD provided data
# File Architecture: Functions only call functions that are defined above it

import numpy as np

from womd import contract
from womd import frame_ops
from waymo_open_dataset.protos import map_pb2, scenario_pb2

####    ------------------------------------------------------------- < AGENTS >
# Reads two facts: where the self-driving car (sdc) is and which way it faces, at the "now" step, in world coordinates
def scenario_storage_frame(scenario):
    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_state = sdc_track.states[contract.CURRENT_STEP_INDEX]
    origin = np.array([sdc_state.center_x, sdc_state.center_y])
    heading = sdc_state.heading
    return origin, heading

# Define sdc in WOMD's world coordiantes -> Affine transform to sdc origin units
def track_to_storage_frame(track, origin, heading):
    positions = np.array([[state.center_x, state.center_y] for state in track.states])
    headings = np.array([state.heading for state in track.states])
    velocities = np.array([[state.velocity_x, state.velocity_y] for state in track.states])
    valid = np.array([state.valid for state in track.states])

    stored_positions = frame_ops.positions_to_agent_frame(positions, origin, heading)
    stored_headings = frame_ops.headings_to_agent_frame(headings, heading)
    stored_velocities = frame_ops.directions_to_agent_frame(velocities, heading)

    return stored_positions, stored_headings, stored_velocities, valid

# FLAG: Takes in dynamic object and returns its 91(timestamps) x11(features) table
def track_to_feature_rows(track, origin, heading):
    positions, headings, velocities, valid = track_to_storage_frame(track, origin, heading)
    dimensions = np.array([[state.length, state.width] for state in track.states])

    type_name = scenario_pb2.Track.ObjectType.Name(track.object_type)
    type_onehot = np.zeros(contract.NUM_OBJECT_TYPES)
    if type_name in contract.PREDICTED_OBJECT_TYPES:
        type_onehot[contract.PREDICTED_OBJECT_TYPES.index(type_name)] = 1.0
    type_rows = np.tile(type_onehot, (contract.TOTAL_STEPS, 1))

    rows = np.zeros((contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM))
    rows[:, contract.AGENT_POSITION] = positions
    rows[:, contract.AGENT_HEADING_COSINE] = np.cos(headings)
    rows[:, contract.AGENT_HEADING_SINE] = np.sin(headings)
    rows[:, contract.AGENT_VELOCITY] = velocities
    rows[:, contract.AGENT_DIMENSIONS] = dimensions
    rows[:, contract.AGENT_TYPE] = type_rows

    return rows, valid

# Creates a library of the all tracks (as tables) in the scenario: head -> table 91(timestamps) x11(features)
# Similar to vector<Matrix> in C++
# Scenatio is WOMD provided, defined in WOMD's world coordinates
def scenario_track_arrays(scenario):
    origin, heading = scenario_storage_frame(scenario)

    all_rows = []
    all_valid = []
    for track in scenario.tracks:
        rows, valid = track_to_feature_rows(track, origin, heading)
        all_rows.append(rows)
        all_valid.append(valid)

    return np.stack(all_rows), np.stack(all_valid)

####    ------------------------------------------------------------- < MAP >
MAP_POLYGON_KINDS = ("crosswalk", "speed_bump", "driveway")

# Reads shape WOMD drew -> preserve WOMD labeling, pull its points 
def map_feature_points(feature):
    kind = feature.WhichOneof("feature_data")
    if kind is None:
        return None, None
    if kind == "stop_sign":
        raw_points = [feature.stop_sign.position]
    elif kind in MAP_POLYGON_KINDS:
        corners = getattr(feature, kind).polygon
        if len(corners) == 0:
            return None, None
        raw_points = list(corners) + [corners[0]]
    else:
        raw_points = getattr(feature, kind).polyline
    if len(raw_points) == 0:
        return None, None
    points = np.array([[point.x, point.y] for point in raw_points])
    return points, contract.MAP_POLYLINE_KINDS.index(kind)

# Walk the shape from start to end -> drop a new dot every half metre -> return 2d-array.
def points_along_polyline(points, spacing):
    if len(points) < 2:
        return points
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = arc_lengths[-1]
    if total_length == 0.0:
        return points[:1]
    sample_distances = np.arange(0.0, total_length, spacing)
    sample_distances = np.append(sample_distances, total_length)
    spaced_x = np.interp(sample_distances, arc_lengths, points[:, 0])
    spaced_y = np.interp(sample_distances, arc_lengths, points[:, 1])
    return np.stack([spaced_x, spaced_y], axis=1)

# Takes evenly placed 0.5 dots applied from function above and give them a diretion -> 
def polyline_directions(points):
    if len(points) < 2:
        return np.zeros_like(points)
    steps = np.diff(points, axis=0)
    directions = steps / np.linalg.norm(steps, axis=1, keepdims=True)
    return np.concatenate([directions, directions[-1:]])

# Runs a feature through all above methods -> places them in ego/sdc coordiantes 
# -> drops those outside STAGING_CROP_RADIUS_METRES./contract.py
def map_feature_to_storage_frame(feature, origin, heading):
    raw_points, kind_index = map_feature_points(feature)
    if raw_points is None:
        return None, None, None
    spaced_points = points_along_polyline(raw_points, contract.MAP_POINT_SPACING_METRES)
    arrows = polyline_directions(spaced_points)
    stored_points = frame_ops.positions_to_agent_frame(spaced_points, origin, heading)
    stored_arrows = frame_ops.directions_to_agent_frame(arrows, heading)
    keep = np.linalg.norm(stored_points, axis=1) <= contract.STAGING_CROP_RADIUS_METRES
    return stored_points[keep], stored_arrows[keep], kind_index

# Traffic signal look-up/define: 1 second history -> now 
# Hotspot approach , 11(timestamp) x 9(state)
def scenario_traffic_signal_histories(scenario):
    histories = {}
    for step_index, dynamic_state in enumerate(scenario.dynamic_map_states[:contract.HISTORY_STEPS]):
        for lane_state in dynamic_state.lane_states:
            state_name = map_pb2.TrafficSignalLaneState.State.Name(lane_state.state)
            history = histories.setdefault(
                lane_state.lane,
                np.zeros((contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)),
            )
            history[step_index, contract.TRAFFIC_SIGNAL_STATES.index(state_name)] = 1.0
    return histories

# FLAG: # Joins everything built above into one finished N x 127 table for a single feature: geometry in every row, 
#   kind one-hot, then the detail blocks only that kind fills (lane -> signals/type/speed, line/edge -> boundary type).
# N (dots per feature) stays ragged on purpose: staging stores the world complete; the loader crops/rotates per 
#   target agent at train time, where the choice costs a flag instead of a restage. 
# One scenario map per stageing, prev one per sample. Map - about 85% sample data. Current cut 1/3 map storage from per sample.
#   Sample: one agent, at one moment, being asked the question. Scenario yields as many samples as it has agents worth predicting.
# The scenario is the unit of storage; the sample is the unit of training
#   WOMD Leaderboard asks for 8.
def map_feature_rows(feature, origin, heading, signal_histories):
    stored_points, stored_arrows, kind_index = map_feature_to_storage_frame(feature, origin, heading)
    if stored_points is None or len(stored_points) == 0:
        return None
    kind = contract.MAP_POLYLINE_KINDS[kind_index]

    rows = np.zeros((len(stored_points), contract.MAP_FEATURE_DIM))
    rows[:, contract.MAP_POSITION] = stored_points
    rows[:, contract.MAP_DIRECTION] = stored_arrows
    rows[:, contract.MAP_KIND.start + kind_index] = 1.0

    if kind == "lane":
        if feature.id in signal_histories:
            rows[:, contract.MAP_SIGNAL_STATE] = signal_histories[feature.id].reshape(-1)
        rows[:, contract.MAP_LANE_TYPE.start + feature.lane.type] = 1.0
        rows[:, contract.MAP_SPEED_LIMIT] = feature.lane.speed_limit_mph
    elif kind == "road_line":
        rows[:, contract.MAP_BOUNDARY_TYPE.start + feature.road_line.type] = 1.0
    elif kind == "road_edge":
        rows[:, contract.MAP_BOUNDARY_TYPE.start + len(contract.ROAD_LINE_TYPES) + feature.road_edge.type] = 1.0

    return rows

# Turns a whole scenario's map into one packed thing
def scenario_map_arrays(scenario):
    origin, heading = scenario_storage_frame(scenario)
    signal_histories = scenario_traffic_signal_histories(scenario)

    feature_tables = []
    for feature in scenario.map_features:
        rows = map_feature_rows(feature, origin, heading, signal_histories)
        if rows is not None:
            feature_tables.append(rows)

    if not feature_tables:
        return np.zeros((0, contract.MAP_FEATURE_DIM)), np.zeros(0, dtype=np.int64)

    map_rows = np.concatenate(feature_tables)
    feature_lengths = np.array([len(table) for table in feature_tables], dtype=np.int64)
    return map_rows, feature_lengths