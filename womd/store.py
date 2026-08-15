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

# Reads shape WOMD drew -> preserve WOMD labeling and pull its points.
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

# Walk the shape from start to end -> read one per-point column at every dot dropped along the way.
# The walk is parameterised by arc length, so a dot lands every MAP_POINT_SPACING_METRES of travel
# and reads whatever the column held at that distance.
def polyline_arc_lengths(points):
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])

def polyline_sample_distances(arc_lengths, spacing):
    return np.append(np.arange(0.0, arc_lengths[-1], spacing), arc_lengths[-1])

def column_along_polyline(points, column_values, spacing):
    if len(points) < 2:
        return column_values
    arc_lengths = polyline_arc_lengths(points)
    if arc_lengths[-1] == 0.0:
        return column_values[:1]
    return np.interp(polyline_sample_distances(arc_lengths, spacing), arc_lengths, column_values)

# Walk the shape from start to end -> drop a new dot every MAP_POINT_SPACING_METRES -> 2d-array.
def points_along_polyline(points, spacing):
    if len(points) < 2:
        return points
    return np.stack(
        [column_along_polyline(points, points[:, 0], spacing),
         column_along_polyline(points, points[:, 1], spacing)],
        axis=1,
    )

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

def map_feature_boundary_crossing_codes(feature, raw_points, dot_count):
    crossing_codes = np.zeros((dot_count, 2))
    if feature.WhichOneof("feature_data") != "lane":
        return crossing_codes
    arc_lengths = polyline_arc_lengths(raw_points)
    sample_distances = polyline_sample_distances(arc_lengths, contract.MAP_POINT_SPACING_METRES)
    for side_index, side_name in enumerate(contract.LANE_SIDES):
        for segment in getattr(feature.lane, f"{side_name}_boundaries"):
            first_dot = np.searchsorted(sample_distances, arc_lengths[segment.lane_start_index], side="left")
            last_dot = np.searchsorted(sample_distances, arc_lengths[segment.lane_end_index], side="right")
            crossing_codes[first_dot:last_dot, side_index] = 1.0 + segment.boundary_type
    return crossing_codes

# Runs a feature through all above methods -> places them in ego/sdc coordiantes
# -> drops those outside STAGING_CROP_RADIUS_METRES./contract.py
def map_feature_to_storage_frame(feature, origin, heading):
    raw_points, kind_index = map_feature_points(feature)
    if raw_points is None:
        return None, None, None, None
    spaced_points = points_along_polyline(raw_points, contract.MAP_POINT_SPACING_METRES)
    arrows = polyline_directions(spaced_points)
    crossing_codes = map_feature_boundary_crossing_codes(feature, raw_points, len(spaced_points))
    stored_points = frame_ops.positions_to_agent_frame(spaced_points, origin, heading)
    stored_arrows = frame_ops.directions_to_agent_frame(arrows, heading)
    keep = np.linalg.norm(stored_points, axis=1) <= contract.STAGING_CROP_RADIUS_METRES
    return stored_points[keep], stored_arrows[keep], crossing_codes[keep], kind_index

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

# FLAG: # Joins everything built above into one finished N x 32 table for a single feature: geometry in every row,
#   kind one-hot, then the detail blocks only that kind fills (lane -> type/speed/stop point, line/edge -> boundary type).
# N (dots per feature) stays ragged on purpose: staging stores the world complete; the loader crops/rotates per
#   predicted agent at train time, where the choice costs a flag instead of a restage.
# One scenario map per stageing, prev one per sample. Map - about 85% sample data. Current cut 1/3 map storage from per sample.
#   Sample: one agent, at one moment, being asked the question. Scenario yields as many samples as it has agents worth predicting.
# The scenario is the unit of storage; the sample is the unit of training
#   WOMD Leaderboard asks for 8.
#
# The final matrix — map_rows (N_total, 32), one contiguous block of rows per feature:
#
#                   0  1   2  3    4 .. 10    11-14    15    16 .. 24   25-27   28-29   30-31
#                ┌───────┬───────┬─────────┬─────────┬─────┬────────────┬────────┬───────┬───────┐
#                │  POS  │  DIR  │  KIND   │LANE_TYPE│ SPD │    BOUNDARY_TYPE    │ STOP  │ CROSS │
#                │  x  y │ dx dy │1-hot of7│1-hot of4│ mph │ 9 road_line│3 r_edge│ sx sy │ lf rt │
#                ├───────┼───────┼─────────┼─────────┼─────┼────────────┼────────┼───────┼───────┤
# lane           │  x  y │ dx dy │  1 @ 4  │  1-hot  │ mph │     0      │   0    │ sx sy │ lf rt │
# road_line      │  x  y │ dx dy │  1 @ 5  │    0    │  0  │ 1-hot of 9 │   0    │  0 0  │  0 0  │
# road_edge      │  x  y │ dx dy │  1 @ 6  │    0    │  0  │     0      │1-hot/3 │  0 0  │  0 0  │
# stop_sign      │  x  y │  0  0 │  1 @ 7  │    0    │  0  │     0      │   0    │  0 0  │  0 0  │
# crosswalk      │  x  y │ dx dy │  1 @ 8  │    0    │  0  │     0      │   0    │  0 0  │  0 0  │
# speed_bump     │  x  y │ dx dy │  1 @ 9  │    0    │  0  │     0      │   0    │  0 0  │  0 0  │
# driveway       │  x  y │ dx dy │ 1 @ 10  │    0    │  0  │     0      │   0    │  0 0  │  0 0  │
#                │   :   │   :   │    :    │    :    │  :  │     :      │   :    │   :   │   :   │
#                └───────┴───────┴─────────┴─────────┴─────┴────────────┴────────┴───────┴───────┘
#                  ↑ one row = one dot, MAP_POINT_SPACING_METRES apart
#
# Rows: features stay contiguous in scenario.map_features order (kinds interleave — one band per
#   kind shown once). Block i has feature_lengths[i] rows; boundaries = cumsum(feature_lengths).
# The traffic-signal history is NOT here: it is constant along a lane, so it lives once per
#   polyline in the (F, 11, 9) array scenario_map_arrays returns, not 99 repeated columns per dot.
# KIND hot column = 4 + kind index, in MAP_POLYLINE_KINDS order. stop_sign is a single dot -> DIR 0,0.
# STOP [28:30]: signalised lanes only — the signal's stop point in the storage frame, same pair
#   on every dot of that lane; zeros everywhere else (an unsignalled lane has no stop point). It
#   stays per dot because it is geometry the dot encoder reads against that dot's own position.
# CROSS [30:32]: lanes only — the BoundarySegment covering this dot on its left and on its right,
#   as a CODE not a one-hot, since a dot has at most one boundary per side. 0 = no segment covers
#   this dot on that side; a recorded type is 1 + its index into BOUNDARY_TYPES, keeping "absent"
#   distinct from BOUNDARY_TYPES[0] = TYPE_UNKNOWN. lane_start_index / lane_end_index name RAW
#   polyline points, so they are read as arc lengths and matched against each resampled dot's own
#   sample distance, never scaled. Overlapping segments: proto order, last one wins.
def map_feature_rows(feature, origin, heading, signal_histories, signal_stop_points):
    stored_points, stored_arrows, crossing_codes, kind_index = map_feature_to_storage_frame(
        feature, origin, heading
    )
    if stored_points is None or len(stored_points) == 0:
        return None
    kind = contract.MAP_POLYLINE_KINDS[kind_index]

    rows = np.zeros((len(stored_points), contract.MAP_FEATURE_DIM))
    rows[:, contract.MAP_POSITION] = stored_points
    rows[:, contract.MAP_DIRECTION] = stored_arrows
    rows[:, contract.MAP_KIND.start + kind_index] = 1.0
    rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING] = crossing_codes[:, 0]
    rows[:, contract.MAP_RIGHT_BOUNDARY_CROSSING] = crossing_codes[:, 1]

    if kind == "lane":
        if feature.id in signal_histories:
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

# The one 11 x 9 signal history this polyline lives under, or zeros when it lives under none.
# Only lanes carry signals; a lane WOMD never lit in the history second is unsignalled like any
# crosswalk. One history per polyline, never per dot - it is the same light for the whole lane.
def map_feature_signal_history(feature, signal_histories):
    if feature.WhichOneof("feature_data") == "lane" and feature.id in signal_histories:
        return signal_histories[feature.id]
    return np.zeros((contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES))

# map.proto LaneCenter.interpolating = 3: True when WOMD interpolated this lane centre between two
# other lanes instead of surveying it, so the geometry is inferred rather than observed. False for
# every kind that is not a lane, which is exactly what it means - only lane centres are interpolated.
def map_feature_is_interpolating(feature):
    return feature.WhichOneof("feature_data") == "lane" and feature.lane.interpolating

# Turns a whole scenario's map into one packed thing
#
# COMPLETED HERE — the map matrix, map_rows (N_total, 32): every feature's row-table from
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
#   │      ...      │  (first scenario of validation.tfrecord-00000-of-00150:
#   │               │   F=167 blocks, N_total=7,924 rows)
#   └───────────────┘
#
# COMPLETED HERE — the signal tensor, polyline_signal_histories (F, 11, 9): row i is polyline i's
#   traffic-signal history, index-for-index with feature_lengths, zeros for every unsignalled
#   polyline. Step-major within a row: [0]=t-10 ... [10]=now; within a step: 0 UNKNOWN,
#   1 ARROW_STOP, 2 ARROW_CAUTION, 3 ARROW_GO, 4 STOP, 5 CAUTION, 6 GO, 7 FLASHING_STOP,
#   8 FLASHING_CAUTION. The loader hands each chunk token the row of the polyline it was cut from.
#
# Two more arrays ride alongside, both index-for-index with feature_lengths:
#
#   ┌───────────────────────────────────────────┐
#   │ feature_ids (F,)                 int64    │
#   │   WOMD MapFeature.id — the join key every  │
#   │   lane-graph table below resolves against  │
#   │ feature_is_interpolating (F,)     bool     │
#   │   LaneCenter.interpolating, False non-lane │
#   └───────────────────────────────────────────┘
#
# feature_ids is what makes the crop survivable downstream: a feature that produced no surviving
#   dots is dropped from every one of these axes, so row i of feature_lengths is NOT map feature i
#   of the scenario, and only the id says which lane a row actually is.
def scenario_map_arrays(scenario):
    origin, heading = scenario_storage_frame(scenario)
    signal_histories, signal_stop_points = scenario_traffic_signal_histories(scenario)

    feature_tables = []
    feature_ids = []
    feature_is_interpolating = []
    feature_signal_histories = []
    for feature in scenario.map_features:
        rows = map_feature_rows(feature, origin, heading, signal_histories, signal_stop_points)
        if rows is None:
            continue
        feature_tables.append(rows)
        feature_ids.append(feature.id)
        feature_is_interpolating.append(map_feature_is_interpolating(feature))
        feature_signal_histories.append(map_feature_signal_history(feature, signal_histories))

    if not feature_tables:
        return (
            np.zeros((0, contract.MAP_FEATURE_DIM)),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)),
            np.zeros(0, dtype=bool),
        )

    map_rows = np.concatenate(feature_tables)
    feature_lengths = np.array([len(table) for table in feature_tables], dtype=np.int64)
    return (
        map_rows,
        feature_lengths,
        np.array(feature_ids, dtype=np.int64),
        np.stack(feature_signal_histories),
        np.array(feature_is_interpolating, dtype=bool),
    )

####    ------------------------------------------------------------- < LANE GRAPH >
# Every lane's raw polyline, world frame, keyed by WOMD MapFeature.id. Built once per scenario
# because a neighbour relation resolves indices into the OTHER lane's polyline, not its own, and a
# neighbour is any lane in the scenario - including one the staging crop is about to delete.
def lane_raw_polylines(scenario):
    polylines = {}
    for feature in scenario.map_features:
        if feature.WhichOneof("feature_data") == "lane" and len(feature.lane.polyline) > 0:
            polylines[feature.id] = np.array([[point.x, point.y] for point in feature.lane.polyline])
    return polylines

# One list of BoundarySegments -> its identity rows and its two world-frame endpoints per row.
# WOMD puts these in two places and they are not the same fact: LaneCenter.left_boundaries /
# right_boundaries describe what runs along a whole side of the lane, while LaneNeighbor.boundaries
# describe what runs along the stretch shared with one named neighbour. shared_neighbour_lane_id
# separates them. Both index THIS lane's polyline, so both resolve against the same points.
def boundary_segment_rows(lane_id, shared_neighbour_lane_id, side_index, segments, lane_points):
    identity_rows = []
    endpoint_rows = []
    for segment in segments:
        assert 0 <= segment.lane_start_index < len(lane_points), (
            f"boundary start {segment.lane_start_index} off lane {lane_id}"
        )
        assert 0 <= segment.lane_end_index < len(lane_points), (
            f"boundary end {segment.lane_end_index} off lane {lane_id}"
        )
        assert segment.boundary_type < len(contract.ROAD_LINE_TYPES)
        identity_rows.append([lane_id, shared_neighbour_lane_id, segment.boundary_feature_id,
                              side_index, segment.boundary_type])
        endpoint_rows.append(np.concatenate([lane_points[segment.lane_start_index],
                                             lane_points[segment.lane_end_index]]))
    return identity_rows, endpoint_rows

# Rows of packed world (x, y) pairs -> the same rows in the storage frame. Flattened to one point
# list, transformed once, folded back, so relation geometry goes through the identical affine
# transform the map dots do rather than a second hand-written copy of it.
def world_point_rows_to_storage_frame(packed_rows, row_width, origin, heading):
    packed_points = np.array(packed_rows, dtype=np.float64).reshape(-1, 2)
    reframed = frame_ops.positions_to_agent_frame(packed_points, origin, heading)
    return reframed.reshape(-1, row_width)

# Turns the lane relations WOMD ships into flat tables — the part of the map that says what a
# driver is ALLOWED to do, which no arrangement of dots can express.
#
# Two decisions govern every table here, and both exist because of the crops.
#
#   IDS, NOT ROW NUMBERS. A relation names its lanes by WOMD MapFeature.id. Staging already drops
#     features that produce no surviving dots, and the loader crops again per agent, so a row
#     number into feature_lengths is true only at the crop that computed it. An id is resolvable at
#     any crop, against feature_ids, and an id that resolves against nothing is the honest record
#     of a lane the crop cut - the edge still says a lane is out there.
#
#   METRES, NOT POLYLINE INDICES. self_start_index, neighbor_end_index, lane_start_index and the
#     rest all index the RAW polyline. points_along_polyline resamples that polyline to
#     MAP_POINT_SPACING_METRES and the crop then deletes whatever falls outside
#     STAGING_CROP_RADIUS_METRES, so proto index 37 is not row 37 of map_rows and no rewriting
#     would keep it one through the loader's second crop. Each index is therefore RESOLVED here:
#     read the raw point it names, put it in the storage frame, store the metres. A metre
#     coordinate does not move when other dots are deleted, so the extent of a relation still means
#     the same thing however hard anything downstream crops. These endpoints are NOT themselves
#     cropped to STAGING_CROP_RADIUS_METRES: clipping them would silently shorten the stretch over
#     which a lane change is legal.
#
# COMPLETED HERE — six tables, one row per relation, ids kept int64 and geometry float:
#
#  lane_connections (E, 3) int64                     entry_lanes = 9 / exit_lanes = 10
#  ┌─────────┬─────────┬──────┐
#  │ SOURCE  │  DEST   │ KIND │   both lists become source -> destination rows; KIND keeps which
#  │ lane id │ lane id │ 0/1  │   list said so (0 entry, 1 exit) because either can name an edge
#  └─────────┴─────────┴──────┘   the other omits
#
#  lane_neighbour_ids (N, 3) int64          lane_neighbour_bounds (N, 8) float
#  ┌─────────┬─────────┬──────┐             ┌───────────┬───────────┬────────────┬────────────┐
#  │  LANE   │  OTHER  │ SIDE │             │ SELF_START│  SELF_END │OTHER_START │ OTHER_END  │
#  │ lane id │ lane id │ 0/1  │             │   x  y    │   x  y    │   x  y     │   x  y     │
#  └─────────┴─────────┴──────┘             └───────────┴───────────┴────────────┴────────────┘
#    SIDE: 0 left, 1 right                    where on THIS lane the two run alongside, and where
#                                             on the OTHER lane a change into it lands
#
#  lane_boundary_ids (B, 5) int64                              lane_boundary_bounds (B, 4) float
#  ┌─────────┬──────────┬──────────┬──────┬──────┐             ┌───────────┬───────────┐
#  │  LANE   │  SHARED  │ BOUNDARY │ SIDE │ TYPE │             │   START   │    END    │
#  │ lane id │ lane id  │ feat. id │ 0/1  │ line │             │   x  y    │   x  y    │
#  │         │  or -1   │          │      │ type │             └───────────┴───────────┘
#  └─────────┴──────────┴──────────┴──────┴──────┘               the stretch of THIS lane's
#    SHARED = -1: the segment runs along the whole side          polyline the segment covers
#      (left_boundaries = 13 / right_boundaries = 14)
#    SHARED = a lane id: the segment runs along the stretch shared with that neighbour
#      (LaneNeighbor.boundaries = 6) - a different fact, not a duplicate of the row above
#    TYPE indexes ROAD_LINE_TYPES; the proto writes TYPE_UNKNOWN when the boundary is a road edge
#
#  stop_sign_controlled_lanes (S, 2) int64            StopSign.lane = 1
#  ┌──────────┬─────────┐
#  │ STOP SIGN│  LANE   │   one row per controlled lane. Without it the sign is an isolated dot and
#  │ feat. id │ lane id │   nothing on disk says which approach it governs
#  └──────────┴─────────┘
#
# Only features that survived staging get rows: the graph annotates the map that is in the file.
# The lanes those rows POINT AT need not have survived, and often have not.
def scenario_lane_graph_arrays(scenario, origin, heading, stored_feature_ids):
    raw_polylines = lane_raw_polylines(scenario)

    connection_rows = []
    neighbour_identity_rows = []
    neighbour_extent_rows = []
    boundary_identity_rows = []
    boundary_endpoint_rows = []
    stop_sign_rows = []

    for feature in scenario.map_features:
        if feature.id not in stored_feature_ids:
            continue
        kind = feature.WhichOneof("feature_data")
        if kind == "stop_sign":
            for controlled_lane_id in feature.stop_sign.lane:
                stop_sign_rows.append([feature.id, controlled_lane_id])
            continue
        if kind != "lane":
            continue

        lane_points = raw_polylines[feature.id]
        for entry_lane_id in feature.lane.entry_lanes:
            connection_rows.append(
                [entry_lane_id, feature.id, contract.LANE_CONNECTION_KINDS.index("entry")]
            )
        for exit_lane_id in feature.lane.exit_lanes:
            connection_rows.append(
                [feature.id, exit_lane_id, contract.LANE_CONNECTION_KINDS.index("exit")]
            )

        for side_index, side_name in enumerate(contract.LANE_SIDES):
            side_identities, side_endpoints = boundary_segment_rows(
                feature.id,
                contract.NO_SHARED_NEIGHBOUR_LANE,
                side_index,
                getattr(feature.lane, f"{side_name}_boundaries"),
                lane_points,
            )
            boundary_identity_rows.extend(side_identities)
            boundary_endpoint_rows.extend(side_endpoints)

            for neighbour in getattr(feature.lane, f"{side_name}_neighbors"):
                assert neighbour.feature_id in raw_polylines, (
                    f"neighbour lane {neighbour.feature_id} of lane {feature.id}"
                    f" absent from scenario {scenario.scenario_id}"
                )
                neighbour_points = raw_polylines[neighbour.feature_id]
                assert 0 <= neighbour.self_start_index < len(lane_points)
                assert 0 <= neighbour.self_end_index < len(lane_points)
                assert 0 <= neighbour.neighbor_start_index < len(neighbour_points)
                assert 0 <= neighbour.neighbor_end_index < len(neighbour_points)
                neighbour_identity_rows.append([feature.id, neighbour.feature_id, side_index])
                neighbour_extent_rows.append(np.concatenate([
                    lane_points[neighbour.self_start_index],
                    lane_points[neighbour.self_end_index],
                    neighbour_points[neighbour.neighbor_start_index],
                    neighbour_points[neighbour.neighbor_end_index],
                ]))
                shared_identities, shared_endpoints = boundary_segment_rows(
                    feature.id, neighbour.feature_id, side_index, neighbour.boundaries, lane_points
                )
                boundary_identity_rows.extend(shared_identities)
                boundary_endpoint_rows.extend(shared_endpoints)

    return (
        np.array(connection_rows, dtype=np.int64).reshape(-1, contract.LANE_CONNECTION_WIDTH),
        np.array(neighbour_identity_rows, dtype=np.int64).reshape(-1, contract.LANE_NEIGHBOUR_ID_WIDTH),
        world_point_rows_to_storage_frame(
            neighbour_extent_rows, contract.LANE_NEIGHBOUR_BOUND_WIDTH, origin, heading
        ),
        np.array(boundary_identity_rows, dtype=np.int64).reshape(-1, contract.LANE_BOUNDARY_ID_WIDTH),
        world_point_rows_to_storage_frame(
            boundary_endpoint_rows, contract.LANE_BOUNDARY_BOUND_WIDTH, origin, heading
        ),
        np.array(stop_sign_rows, dtype=np.int64).reshape(-1, contract.STOP_SIGN_LANE_WIDTH),
    )

## Write boundary: one scenario -> one .npz on disk. Float32 cast + compression live here and only here.
# Returns the scenario's worst timestep-spacing deviation from 0.1 s. Measured on real data:
# ~3% of scenarios have one skipped frame (gap ~0.2 s) or start-of-recording jitter, and Waymo
# scores them like any other, so irregular spacing is COUNTED by the caller, never a veto.
# The write lands on a .partial sibling and is renamed into place, so a staging session killed
# part-way through leaves no truncated file wearing a legitimate scenario name among the rest.
# savez_compressed is handed an open stream and never the partial PATH, because given a path it
# appends .npz to anything not already ending in one and the file would land at <id>.npz.partial.npz.
def write_scenario(scenario, output_path):
    assert scenario.current_time_index == contract.CURRENT_STEP_INDEX, (
        f"current_time_index {scenario.current_time_index}, scenario {scenario.scenario_id}"
    )
    timestamp_gaps = np.diff(np.array(scenario.timestamps_seconds))
    worst_spacing_deviation = float(np.max(np.abs(timestamp_gaps - 0.1))) if len(timestamp_gaps) else 0.0
    track_rows, track_valid = scenario_track_arrays(scenario)
    track_ids, is_designated_target, is_object_of_interest = scenario_track_labels(scenario)
    (map_rows, feature_lengths, feature_ids, polyline_signal_histories,
     feature_is_interpolating) = scenario_map_arrays(scenario)
    origin, heading = scenario_storage_frame(scenario)
    (lane_connections, lane_neighbour_ids, lane_neighbour_bounds, lane_boundary_ids,
     lane_boundary_bounds, stop_sign_controlled_lanes) = scenario_lane_graph_arrays(
        scenario, origin, heading, set(feature_ids.tolist())
    )

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    with open(partial_path, "wb") as partial_file:
        np.savez_compressed(
            partial_file,
            track_rows=track_rows.astype(np.float32),
            track_valid=track_valid,
            track_ids=track_ids,
            is_designated_target=is_designated_target,
            is_object_of_interest=is_object_of_interest,
            map_rows=map_rows.astype(np.float32),
            feature_lengths=feature_lengths,
            feature_ids=feature_ids,
            feature_is_interpolating=feature_is_interpolating,
            polyline_signal_histories=polyline_signal_histories.astype(np.float32),
            lane_connections=lane_connections,
            lane_neighbour_ids=lane_neighbour_ids,
            lane_neighbour_bounds=lane_neighbour_bounds.astype(np.float32),
            lane_boundary_ids=lane_boundary_ids,
            lane_boundary_bounds=lane_boundary_bounds.astype(np.float32),
            stop_sign_controlled_lanes=stop_sign_controlled_lanes,
            frame_origin=origin.astype(np.float32),
            frame_heading=np.float32(heading),
            scenario_id=scenario.scenario_id,
        )
    partial_path.replace(output_path)
    return worst_spacing_deviation