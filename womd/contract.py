# contract.py is the the shared constants and array shapes every other file reads from; 
# one source of truth for feature layout, dataset facts, and staging parameters.
# > general abstract senario features / senario abstraction

# > GLOBAL DATA FACTS. DERIVED FROM WOMD, FIRST PRINCIPLES DERIVATION, OR DESIGN CHOICES.
HISTORY_STEPS = 11                              # 1 Second: history include "Now", so +1
FUTURE_STEPS = 80                               # 8 Seconds: does not include "Now"
CURRENT_STEP_INDEX = 10                         # "Now". Current pivot: before is input, after is target
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS
TIMESTEP_SECONDS = 0.1                          # WOMD logs at 10 Hz. store.py's scenario_timestep_deviation measures every staged scenario's logged gap against this number rather than trusting it.
FUTURE_HORIZON_SECONDS = FUTURE_STEPS * TIMESTEP_SECONDS                        # The 8 s WOMD asks to be predicted, divided out of the step count instead of typed in.
MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED = 3.56                           # STATUS §LIMITS' longitudinal acceleration figure of record, 0.5 s window, derived from this project's own logged ground truth (data/eval_local, 4,842 samples, moving steps at or above 0.5 m/s, 7.5% spread across 4 disjoint shard groups) - not invented and not cited from a paper. Scott 2026-08-15 ruled explicitly AGAINST the textbook alternatives after comparing all three over an 8 s horizon: tyre friction on dry asphalt at mu = 0.8 gives a 251 m allowance and friction-limited at 1 g gives 314 m, both ABOVE the 158-241 m endpoint spray they exist to catch, while the measured limit gives 114 m and still clears every logged future with a third to spare.
PREDICTED_OBJECT_TYPES = ("TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST")    # WOMD object labeling schema
NUM_OBJECT_TYPES = len(PREDICTED_OBJECT_TYPES)
AGENT_TYPES = PREDICTED_OBJECT_TYPES + ("TYPE_OTHER",)
NUM_AGENT_TYPES = len(AGENT_TYPES)
STAGING_CROP_RADIUS_METRES = 400.0              # Scott 2026-08-14. measure_staging.py on validation.tfrecord-00000-of-00150, 287 scenarios: designated targets lose 3.0% of wanted map dots at 250 m (worst target 46.6%), 0.1% at 400 m; storage +4.2% vs 250 m because Waymo's own map ends at a median of 192.7 m and 99.9% of all dots within 500 m already sit inside 400 m.
MAP_POINT_SPACING_METRES = 1.0                  # Scott 2026-08-14, was 0.5. Halves staged bytes, per-sample cost and read bandwidth. Geometry cost, measure_staging.py over 3,564,503 dropped dots: p50 0.0000 m, p99 0.138 m, max 0.495 m. Model cost measured directly on the polyline tokens the attention actually reads, 4,054 tokens over 25 scenarios: cosine similarity p50 1.00000 / p5 0.99965, relative change p50 0.0005 / p95 0.0266; uncorrelated with polyline length (r = 0.007); lanes 0.0003, road lines 0.0001, stop signs 0.0000, and the change concentrates on polygon kinds (speed bump 0.031, driveway 0.016, crosswalk 0.015) whose corners are what alternate-dot dropping costs.
MAP_CHUNK_DOTS = 20                             # Scott 2026-08-14: a map token is at most 20 consecutive dots of one polyline, not one token per whole polyline. Measured over four DISJOINT 40-scenario blocks of ../data/staged (400 m, 1.0 m spacing, ALL designated targets): tokens/sample 189->450, 209->488, 215->520, 234->528, i.e. a stable 2.25-2.41x, and every token capped at 20 dots = 20 m against p90 75 m unchunked. Attention measured 6.1% of total epoch time on a Kaggle T4 (profile_step.py) so the token growth is affordable. An earlier 178->426 pair recorded here came from sampling only the FIRST target per scenario and does not reproduce; it is superseded by the four blocks above.
NUM_PREDICTED_MODES = 6                         # Mode = possible future. We predict X futures. WOMD caps submissions at 6. Prune from X.
STAGING_CODE_VERSION = "2026-08-15-b"           # Bumped by hand on every code-dataset upload. kaggle_preflight compares working copy vs mount.

# > SUBMISSION FORMAT. DICTATED BY waymo_open_dataset/protos/motion_submission.proto, READ 2026-08-14.
SUBMISSION_STEPS = 16                           # motion_submission.proto, read 2026-08-14: "these fields must be exactly length 16 - 8 seconds with 2 steps per second".
SUBMISSION_FIRST_SCENARIO_STEP = 15             # motion_submission.proto, read 2026-08-14: "the first entry in each of these fields must correspond to time step 15 in the scenario NOT step 10 or 11".
SUBMISSION_STEP_STRIDE = FUTURE_STEPS // SUBMISSION_STEPS                       # The submission covers the same 8 s the log does, at 2 Hz against the log's 10 Hz.
SUBMISSION_FIRST_FUTURE_INDEX = SUBMISSION_FIRST_SCENARIO_STEP - (CURRENT_STEP_INDEX + 1)   # future_positions[0] is scenario step CURRENT_STEP_INDEX + 1, so the proto's step 15 sits four rows in - NOT row zero.
SUBMISSION_FUTURE_INDICES = tuple(range(SUBMISSION_FIRST_FUTURE_INDEX, FUTURE_STEPS, SUBMISSION_STEP_STRIDE))
assert len(SUBMISSION_FUTURE_INDICES) == SUBMISSION_STEPS
assert SUBMISSION_FUTURE_INDICES[-1] == FUTURE_STEPS - 1

# > FEATURE SCALES. STANDARD DEVIATIONS MEASURED ON THIS PROJECT'S OWN STAGED DATA, NOT CHOSEN.
DISTANCE_NORMALISER_METRES = 67.9               # measure_scales.py on ../data/staged, 287 scenarios / 1,278 designated-target samples, 2026-08-14: standard deviation of map position, which is 20,002,407 of the values the model reads against 96,164 output values. Every distance-like quantity divides by this one number so geometry is preserved.
VELOCITY_NORMALISER_METRES_PER_SECOND = 5.2     # measure_scales.py on ../data/staged, 287 scenarios / 1,278 designated-target samples, 2026-08-14: standard deviation of agent and neighbour velocity.
DIMENSION_NORMALISER_METRES = 1.6               # measure_scales.py on ../data/staged, 287 scenarios / 1,278 designated-target samples, 2026-08-14: standard deviation of agent and neighbour dimensions.
SPEED_LIMIT_NORMALISER_MILES_PER_HOUR = 12.4    # measure_scales.py on ../data/staged, 287 scenarios / 1,278 designated-target samples, 2026-08-14: standard deviation of map speed limit over the rows where it applies. The column is map.proto LaneCenter.speed_limit_mph = 4 and is stored unconverted, so the divisor is in miles per hour - it was measured on that same column, so the scale is right and only the former metres-per-second name was wrong. Heading cosine/sine and map direction arrows are unit-scale already (measured max abs 1.000) and are not divided.

AGENT_POSITION = slice(0, 2)
AGENT_HEADING_COSINE = 2
AGENT_HEADING_SINE = 3
AGENT_VELOCITY = slice(4, 6)
AGENT_DIMENSIONS = slice(6, 8)
AGENT_TYPE = slice(8, 8 + NUM_AGENT_TYPES)
AGENT_IS_SDC = AGENT_TYPE.stop
AGENT_FEATURE_DIM = AGENT_IS_SDC + 1


PREDICTED_AGENT_ARRAY_SPEC = {
    "agent_history": (HISTORY_STEPS, AGENT_FEATURE_DIM),
    "agent_history_mask": (HISTORY_STEPS,),
    "future_positions": (FUTURE_STEPS, 2),
    "future_headings": (FUTURE_STEPS, 2),    # Cosine and sine, the pair the track rows already store, not an angle.
    "future_mask": (FUTURE_STEPS,),
    "frame_origin": (2,),
    "frame_heading": (),
    "scenario_id": (),
    "track_id": (),
    "is_designated_target": (),    # For reporting final 8 WOMD requested Agent-Per-Scenario. Prune from ALL.
    "is_object_of_interest": (),
}

TRAFFIC_SIGNAL_STATES = (
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_ARROW_STOP",
    "LANE_STATE_ARROW_CAUTION",
    "LANE_STATE_ARROW_GO",
    "LANE_STATE_STOP",
    "LANE_STATE_CAUTION",
    "LANE_STATE_GO",
    "LANE_STATE_FLASHING_STOP",
    "LANE_STATE_FLASHING_CAUTION",
)
NUM_TRAFFIC_SIGNAL_STATES = len(TRAFFIC_SIGNAL_STATES)

MAP_POLYLINE_KINDS = (
    "lane",
    "road_line",
    "road_edge",
    "stop_sign",
    "crosswalk",
    "speed_bump",
    "driveway",
)
NUM_MAP_POLYLINE_KINDS = len(MAP_POLYLINE_KINDS)

MAP_POSITION = slice(0, 2)
MAP_DIRECTION = slice(2, 4)
MAP_KIND = slice(4, 4 + NUM_MAP_POLYLINE_KINDS)
MAP_FEATURE_DIM = MAP_KIND.stop
POLYLINE_SIGNAL_DIM = HISTORY_STEPS * NUM_TRAFFIC_SIGNAL_STATES   # Scott 2026-08-14: a traffic-signal history belongs to a LANE, so it is stored once per polyline and attached after pooling, at the chunk token; it used to be 99 of the 129 per-dot columns, repeated identically on every dot of the lane (median ~20-26 dots per polyline). Measured cost of the repeat: measure_compression.py on 287 scenarios of ../data/staged, 2026-08-14, map_rows = 81.2% of compressed and 95.0% of uncompressed bytes; profile_step.py on a Kaggle T4, 2026-08-14, .npz decompression = 28.8% of total pipeline time, more than the model's forward plus backward.

LANE_TYPES = (
    "TYPE_UNDEFINED",
    "TYPE_FREEWAY",
    "TYPE_SURFACE_STREET",
    "TYPE_BIKE_LANE",
)
NUM_LANE_TYPES = len(LANE_TYPES)

MAP_LANE_TYPE = slice(MAP_KIND.stop, MAP_KIND.stop + NUM_LANE_TYPES)
MAP_SPEED_LIMIT = MAP_LANE_TYPE.stop
MAP_FEATURE_DIM = MAP_SPEED_LIMIT + 1

# MAP_KIND already tells us whether LaneType/boundary-type apply to this point 
# → if not, zero is enough, no separate 'not applicable' flag needed.
ROAD_LINE_TYPES = (
    "TYPE_UNKNOWN",
    "TYPE_BROKEN_SINGLE_WHITE",
    "TYPE_SOLID_SINGLE_WHITE",
    "TYPE_SOLID_DOUBLE_WHITE",
    "TYPE_BROKEN_SINGLE_YELLOW",
    "TYPE_BROKEN_DOUBLE_YELLOW",
    "TYPE_SOLID_SINGLE_YELLOW",
    "TYPE_SOLID_DOUBLE_YELLOW",
    "TYPE_PASSING_DOUBLE_YELLOW",
)

ROAD_EDGE_TYPES = (
    "TYPE_UNKNOWN",
    "TYPE_ROAD_EDGE_BOUNDARY",
    "TYPE_ROAD_EDGE_MEDIAN",
)
BOUNDARY_TYPES = ROAD_LINE_TYPES + ROAD_EDGE_TYPES
NUM_BOUNDARY_TYPES = len(BOUNDARY_TYPES)

MAP_BOUNDARY_TYPE = slice(MAP_SPEED_LIMIT + 1, MAP_SPEED_LIMIT + 1 + NUM_BOUNDARY_TYPES)
MAP_STOP_POINT = slice(MAP_BOUNDARY_TYPE.stop, MAP_BOUNDARY_TYPE.stop + 2)
MAP_FEATURE_DIM = MAP_STOP_POINT.stop

# > LANE GRAPH. MIRRORS proto/waymo_open_dataset/protos/map.proto, READ 2026-08-14.
# Every relation below is a flat table of WOMD MapFeature ids plus storage-frame geometry. It holds
# ids and not row numbers because both crops - STAGING_CROP_RADIUS_METRES here and the loader's own
# per-agent crop later - delete features, so a row number is only true at the crop that produced it
# while an id resolves against whatever survived. It holds metres and not polyline indices because
# staging resamples every polyline to MAP_POINT_SPACING_METRES and then crops it, so the proto's
# index into the raw polyline names no row on disk; each index is resolved to the point it names.
LANE_CONNECTION_KINDS = ("entry", "exit")       # map.proto LaneCenter.entry_lanes = 9 and exit_lanes = 10. A row always reads source -> destination; the kind records which of the two repeated fields asserted it, because they are separate fields and one can name an edge the other omits.
LANE_CONNECTION_SOURCE = 0                      # map.proto MapFeature.id = 1 of the lane traffic comes FROM.
LANE_CONNECTION_DESTINATION = 1                 # map.proto MapFeature.id = 1 of the lane traffic goes TO.
LANE_CONNECTION_KIND = 2                        # Index into LANE_CONNECTION_KINDS.
LANE_CONNECTION_WIDTH = LANE_CONNECTION_KIND + 1

LANE_SIDES = ("left", "right")                  # map.proto LaneCenter.left_neighbors = 11 / right_neighbors = 12 and left_boundaries = 13 / right_boundaries = 14. The names are the proto's own field prefixes.

LANE_NEIGHBOUR_LANE = 0                         # map.proto MapFeature.id = 1 of the lane that owns the LaneNeighbor entry.
LANE_NEIGHBOUR_OTHER_LANE = 1                   # map.proto LaneNeighbor.feature_id = 1.
LANE_NEIGHBOUR_SIDE = 2                         # Index into LANE_SIDES.
LANE_NEIGHBOUR_ID_WIDTH = LANE_NEIGHBOUR_SIDE + 1

LANE_NEIGHBOUR_SELF_START = slice(0, 2)         # map.proto LaneNeighbor.self_start_index = 2, resolved to the raw polyline point it names.
LANE_NEIGHBOUR_SELF_END = slice(2, 4)           # map.proto LaneNeighbor.self_end_index = 3, resolved the same way.
LANE_NEIGHBOUR_OTHER_START = slice(4, 6)        # map.proto LaneNeighbor.neighbor_start_index = 4, resolved against the NEIGHBOUR's raw polyline.
LANE_NEIGHBOUR_OTHER_END = slice(6, 8)          # map.proto LaneNeighbor.neighbor_end_index = 5, resolved the same way.
LANE_NEIGHBOUR_BOUND_WIDTH = LANE_NEIGHBOUR_OTHER_END.stop

NO_SHARED_NEIGHBOUR_LANE = -1                   # map.proto MapFeature.id = 1 is a WOMD-assigned identifier and is never negative, so -1 cannot collide with one. It marks a boundary that runs along a whole side of the lane (LaneCenter.left_boundaries / right_boundaries) rather than along the stretch shared with one named neighbour (LaneNeighbor.boundaries = 6).
LANE_BOUNDARY_LANE = 0                          # map.proto MapFeature.id = 1 of the lane whose polyline the segment indexes.
LANE_BOUNDARY_SHARED_NEIGHBOUR_LANE = 1         # map.proto LaneNeighbor.feature_id = 1 when the segment came from LaneNeighbor.boundaries = 6, else NO_SHARED_NEIGHBOUR_LANE.
LANE_BOUNDARY_FEATURE = 2                       # map.proto BoundarySegment.boundary_feature_id = 3, the RoadLine or RoadEdge feature running alongside.
LANE_BOUNDARY_SIDE = 3                          # Index into LANE_SIDES.
LANE_BOUNDARY_TYPE = 4                          # map.proto BoundarySegment.boundary_type = 4, a RoadLine.RoadLineType, so an index into ROAD_LINE_TYPES; the proto sets it to TYPE_UNKNOWN when the boundary is a road edge.
LANE_BOUNDARY_ID_WIDTH = LANE_BOUNDARY_TYPE + 1

LANE_BOUNDARY_START = slice(0, 2)               # map.proto BoundarySegment.lane_start_index = 1, resolved to the raw polyline point it names.
LANE_BOUNDARY_END = slice(2, 4)                 # map.proto BoundarySegment.lane_end_index = 2, resolved the same way.
LANE_BOUNDARY_BOUND_WIDTH = LANE_BOUNDARY_END.stop

STOP_SIGN_FEATURE = 0                           # map.proto MapFeature.id = 1 of the stop sign.
STOP_SIGN_CONTROLLED_LANE = 1                   # map.proto StopSign.lane = 1, one row per controlled lane.
STOP_SIGN_LANE_WIDTH = STOP_SIGN_CONTROLLED_LANE + 1