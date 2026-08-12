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
    assert sdc_state.valid, f"invalid SDC at current step, scenario {scenario.scenario_id}"
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

# FLAG: Takes in dynamic object and returns its 91(timestamps) x13(features) table
def track_to_feature_rows(track, origin, heading, is_sdc):
    positions, headings, velocities, valid = track_to_storage_frame(track, origin, heading)
    dimensions = np.array([[state.length, state.width] for state in track.states])

    type_name = scenario_pb2.Track.ObjectType.Name(track.object_type)
    type_onehot = np.zeros(contract.NUM_AGENT_TYPES)
    if type_name in contract.AGENT_TYPES:
        type_onehot[contract.AGENT_TYPES.index(type_name)] = 1.0
    else:
        type_onehot[contract.AGENT_TYPES.index("TYPE_OTHER")] = 1.0
    type_rows = np.tile(type_onehot, (contract.TOTAL_STEPS, 1))

    rows = np.zeros((contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM))
    rows[:, contract.AGENT_POSITION] = positions
    rows[:, contract.AGENT_HEADING_COSINE] = np.cos(headings)
    rows[:, contract.AGENT_HEADING_SINE] = np.sin(headings)
    rows[:, contract.AGENT_VELOCITY] = velocities
    rows[:, contract.AGENT_DIMENSIONS] = dimensions
    rows[:, contract.AGENT_TYPE] = type_rows
    rows[:, contract.AGENT_IS_SDC] = 1.0 if is_sdc else 0.0

    return rows, valid

# Creates a library of the all tracks (as tables) in the scenario: head -> table 91(timestamps) x13(features)
# Similar to vector<Matrix> in C++
# Scenatio is WOMD provided, defined in WOMD's world coordinates
#
# COMPLETED HERE — the agent tensor, track_rows (num_tracks, 91, 13): one 91x13 table per track,
#   stacked in scenario.tracks order. Row = one 0.1 s step (0-9 history, 10 = now, 11-90 future).
#   All geometry in the storage frame (SDC at origin facing +x at "now").
#
#                   0  1    2      3     4  5    6  7      8 ..... 11     12
#                ┌───────┬──────┬──────┬───────┬───────┬────────────────┬─────┐
#                │  POS  │ HCOS │ HSIN │  VEL  │ DIMS  │      TYPE      │ SDC │
#                │  x  y │ cos0 │ sin0 │ vx vy │ ln wd │ veh ped cyc oth│ 0/1 │
#                ├───────┼──────┼──────┼───────┼───────┼────────────────┼─────┤
# SDC (vehicle)  │  x  y │  c   │  s   │ vx vy │ ln wd │  1   0   0   0 │  1  │
# vehicle        │  x  y │  c   │  s   │ vx vy │ ln wd │  1   0   0   0 │  0  │
# pedestrian     │  x  y │  c   │  s   │ vx vy │ ln wd │  0   1   0   0 │  0  │
# cyclist        │  x  y │  c   │  s   │ vx vy │ ln wd │  0   0   1   0 │  0  │
# TYPE_OTHER     │  x  y │  c   │  s   │ vx vy │ ln wd │  0   0   0   1 │  0  │
#                └───────┴──────┴──────┴───────┴───────┴────────────────┴─────┘
#                  ↑ one band per track kind shown once; the real axis is 91 timesteps deep
#
# COMPLETED HERE — the validity matrix, track_valid (num_tracks, 91) bool: cell [k, t] True iff
#   track k was observed at step t. Rows align 1:1 with track_rows; a False cell means that
#   timestep's feature row is zeros — zeros are padding, validity says so.
def scenario_track_arrays(scenario):
    origin, heading = scenario_storage_frame(scenario)

    all_rows = []
    all_valid = []
    for track_index, track in enumerate(scenario.tracks):
        rows, valid = track_to_feature_rows(track, origin, heading, track_index 
                                            == scenario.sdc_track_index)
        all_rows.append(rows)
        all_valid.append(valid)

    return np.stack(all_rows), np.stack(all_valid)

# Builds three lists, one entry per agent in the scenario:
#   1. What is this agent's ID number? (track_ids)
#   2. Is this one of the agents Waymo grades us on? (is_designated_target — True/False)
#   3. Did Waymo tag this agent as "doing something interesting"? (is_object_of_interest — True/False)
def scenario_track_labels(scenario):
    track_ids = np.array([track.id for track in scenario.tracks], dtype=np.int64)
    designated_indices = {required.track_index for required in scenario.tracks_to_predict}
    is_designated_target = np.array(
        [track_index in designated_indices for track_index in range(len(scenario.tracks))]
    )
    interest_ids = set(scenario.objects_of_interest)
    is_object_of_interest = np.array([track.id in interest_ids for track in scenario.tracks])
    return track_ids, is_designated_target, is_object_of_interest

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
        if len(corners) < 2:
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
    step_lengths = np.linalg.norm(steps, axis=1, keepdims=True)
    zero_step_indices = np.flatnonzero(step_lengths[:, 0] == 0.0)
    step_lengths[zero_step_indices] = 1.0
    directions = steps / step_lengths
    for zero_index in zero_step_indices:
        directions[zero_index] = directions[zero_index - 1]
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
# Also collects each signal's stop point (world frame, first sighting wins) - the spot on the
# lane where vehicles must halt when the light says stop.
def scenario_traffic_signal_histories(scenario):
    histories = {}
    stop_points = {}
    for step_index, dynamic_state in enumerate(scenario.dynamic_map_states[:contract.HISTORY_STEPS]):
        for lane_state in dynamic_state.lane_states:
            state_name = map_pb2.TrafficSignalLaneState.State.Name(lane_state.state)
            history = histories.setdefault(
                lane_state.lane,
                np.zeros((contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)),
            )
            history[step_index, contract.TRAFFIC_SIGNAL_STATES.index(state_name)] = 1.0
            stop_points.setdefault(
                lane_state.lane,
                np.array([lane_state.stop_point.x, lane_state.stop_point.y]),
            )
    return histories, stop_points

# FLAG: # Joins everything built above into one finished N x 129 table for a single feature: geometry in every row,
#   kind one-hot, then the detail blocks only that kind fills (lane -> signals/type/speed/stop point, line/edge -> boundary type).
# N (dots per feature) stays ragged on purpose: staging stores the world complete; the loader crops/rotates per 
#   predicted agent at train time, where the choice costs a flag instead of a restage.
# One scenario map per stageing, prev one per sample. Map - about 85% sample data. Current cut 1/3 map storage from per sample.
#   Sample: one agent, at one moment, being asked the question. Scenario yields as many samples as it has agents worth predicting.
# The scenario is the unit of storage; the sample is the unit of training
#   WOMD Leaderboard asks for 8.
#
# The final matrix — map_rows (N_total, 129), one contiguous block of rows per feature:
#
#                   0  1   2  3    4 .. 10   11 ................ 109   110-113   114   115 .. 123  124-126  127-128
#                ┌───────┬───────┬─────────┬─────────────────────────┬─────────┬─────┬─────────────────────┬───────┐
#                │  POS  │  DIR  │  KIND   │      SIGNAL_STATE       │LANE_TYPE│ SPD │    BOUNDARY_TYPE    │ STOP  │
#                │  x  y │ dx dy │1-hot of7│ 11 steps x 9 states     │1-hot of4│ mph │ 9 road_line│3 r_edge│ sx sy │
#                ├───────┼───────┼─────────┼─────────────────────────┼─────────┼─────┼────────────┼────────┼───────┤
# lane           │  x  y │ dx dy │  1 @ 4  │ (11x9) 1-hot history *  │  1-hot  │ mph │     0      │   0    │ sx sy │
# road_line      │  x  y │ dx dy │  1 @ 5  │            0            │    0    │  0  │ 1-hot of 9 │   0    │  0 0  │
# road_edge      │  x  y │ dx dy │  1 @ 6  │            0            │    0    │  0  │     0      │1-hot/3 │  0 0  │
# stop_sign      │  x  y │  0  0 │  1 @ 7  │            0            │    0    │  0  │     0      │   0    │  0 0  │
# crosswalk      │  x  y │ dx dy │  1 @ 8  │            0            │    0    │  0  │     0      │   0    │  0 0  │
# speed_bump     │  x  y │ dx dy │  1 @ 9  │            0            │    0    │  0  │     0      │   0    │  0 0  │
# driveway       │  x  y │ dx dy │ 1 @ 10  │            0            │    0    │  0  │     0      │   0    │  0 0  │
#                │   :   │   :   │    :    │            :            │    :    │  :  │     :      │   :    │   :   │
#                └───────┴───────┴─────────┴─────────────────────────┴─────────┴─────┴────────────┴────────┴───────┘
#                  ↑ one row = one 0.5 m dot
#
# Rows: features stay contiguous in scenario.map_features order (kinds interleave — one band per
#   kind shown once). Block i has feature_lengths[i] rows; boundaries = cumsum(feature_lengths).
# * SIGNAL [11:110] (99 of 129 cols): signalised lanes only, else zeros. Step-major:
#   [11:20]=t-10 ... [101:110]=now. Within each 9-wide step: 0 UNKNOWN, 1 ARROW_STOP,
#   2 ARROW_CAUTION, 3 ARROW_GO, 4 STOP, 5 CAUTION, 6 GO, 7 FLASHING_STOP, 8 FLASHING_CAUTION.
# KIND hot column = 4 + kind index, in MAP_POLYLINE_KINDS order. stop_sign is a single dot -> DIR 0,0.
# STOP [127:129]: signalised lanes only — the signal's stop point in the storage frame, same pair
#   on every dot of that lane; zeros everywhere else (an unsignalled lane has no stop point).
# Column widths not to scale: SIGNAL alone is 77% of the row.
def map_feature_rows(feature, origin, heading, signal_histories, signal_stop_points):
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
            rows[:, contract.MAP_STOP_POINT] = frame_ops.positions_to_agent_frame(
                signal_stop_points[feature.id], origin, heading
            )
        assert feature.lane.type < contract.NUM_LANE_TYPES
        rows[:, contract.MAP_LANE_TYPE.start + feature.lane.type] = 1.0
        rows[:, contract.MAP_SPEED_LIMIT] = feature.lane.speed_limit_mph
    elif kind == "road_line":
        assert feature.road_line.type < len(contract.ROAD_LINE_TYPES)
        rows[:, contract.MAP_BOUNDARY_TYPE.start + feature.road_line.type] = 1.0
    elif kind == "road_edge":
        assert feature.road_edge.type < len(contract.ROAD_EDGE_TYPES)
        rows[:, contract.MAP_BOUNDARY_TYPE.start + len(contract.ROAD_LINE_TYPES) + feature.road_edge.type] = 1.0

    return rows

# Turns a whole scenario's map into one packed thing
#
# COMPLETED HERE — the map matrix, map_rows (N_total, 129): every feature's row-table from
#   map_feature_rows (diagram above), stacked top to bottom in scenario.map_features order.
#   feature_lengths (F,) int64 records each block's height; cumsum recovers the boundaries.
#
#        map_rows                      feature_lengths = [L0, L1, L2, ...]
#   ┌───────────────┐
#   │  feature 0    │  rows 0 .. L0-1
#   ├───────────────┤
#   │  feature 1    │  rows L0 .. L0+L1-1
#   ├───────────────┤
#   │  feature 2    │  rows L0+L1 .. L0+L1+L2-1
#   ├───────────────┤
#   │      ...      │  (scenario 1: F=167 blocks, N_total=15,620 rows)
#   └───────────────┘
def scenario_map_arrays(scenario):
    origin, heading = scenario_storage_frame(scenario)
    signal_histories, signal_stop_points = scenario_traffic_signal_histories(scenario)

    feature_tables = []
    for feature in scenario.map_features:
        rows = map_feature_rows(feature, origin, heading, signal_histories, signal_stop_points)
        if rows is not None:
            feature_tables.append(rows)

    if not feature_tables:
        return np.zeros((0, contract.MAP_FEATURE_DIM)), np.zeros(0, dtype=np.int64)

    map_rows = np.concatenate(feature_tables)
    feature_lengths = np.array([len(table) for table in feature_tables], dtype=np.int64)
    return map_rows, feature_lengths

## Write boundary: one scenario -> one .npz on disk. Float32 cast + compression live here and only here.
# Returns the scenario's worst timestep-spacing deviation from 0.1 s. Measured on real data:
# ~3% of scenarios have one skipped frame (gap ~0.2 s) or start-of-recording jitter, and Waymo
# scores them like any other, so irregular spacing is COUNTED by the caller, never a veto.
def write_scenario(scenario, output_path):
    assert scenario.current_time_index == contract.CURRENT_STEP_INDEX, (
        f"current_time_index {scenario.current_time_index}, scenario {scenario.scenario_id}"
    )
    timestamp_gaps = np.diff(np.array(scenario.timestamps_seconds))
    worst_spacing_deviation = float(np.max(np.abs(timestamp_gaps - 0.1))) if len(timestamp_gaps) else 0.0
    track_rows, track_valid = scenario_track_arrays(scenario)
    track_ids, is_designated_target, is_object_of_interest = scenario_track_labels(scenario)
    map_rows, feature_lengths = scenario_map_arrays(scenario)
    origin, heading = scenario_storage_frame(scenario)

    np.savez_compressed(
        output_path,
        track_rows=track_rows.astype(np.float32),
        track_valid=track_valid,
        track_ids=track_ids,
        is_designated_target=is_designated_target,
        is_object_of_interest=is_object_of_interest,
        map_rows=map_rows.astype(np.float32),
        feature_lengths=feature_lengths,
        frame_origin=origin.astype(np.float32),
        frame_heading=np.float32(heading),
        scenario_id=scenario.scenario_id,
    )
    return worst_spacing_deviation