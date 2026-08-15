# contract.py is the the shared constants and array shapes every other file reads from; 
# one source of truth for feature layout, dataset facts, and staging parameters.
# > general abstract senario features / senario abstraction

# > GLOBAL DATA FACTS. DERIVED FROM WOMD, FIRST PRINCIPLES DERIVATION, OR DESIGN CHOICES.
HISTORY_STEPS = 11                              # 1 Second: history include "Now", so +1
FUTURE_STEPS = 80                               # 8 Seconds: does not include "Now"
CURRENT_STEP_INDEX = 10                         # "Now". Current pivot: before is input, after is target
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS
PREDICTED_OBJECT_TYPES = ("TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST")    # WOMD object labeling schema
NUM_OBJECT_TYPES = len(PREDICTED_OBJECT_TYPES)
AGENT_TYPES = PREDICTED_OBJECT_TYPES + ("TYPE_OTHER",)
NUM_AGENT_TYPES = len(AGENT_TYPES)
STAGING_CROP_RADIUS_METRES = 400.0              # Scott 2026-08-14. measure_staging.py on validation.tfrecord-00000-of-00150, 287 scenarios: designated targets lose 3.0% of wanted map dots at 250 m (worst target 46.6%), 0.1% at 400 m; storage +4.2% vs 250 m because Waymo's own map ends at a median of 192.7 m and 99.9% of all dots within 500 m already sit inside 400 m.
MAP_POINT_SPACING_METRES = 1.0                  # Scott 2026-08-14, was 0.5. Halves staged bytes, per-sample cost and read bandwidth. Geometry cost, measure_staging.py over 3,564,503 dropped dots: p50 0.0000 m, p99 0.138 m, max 0.495 m. Model cost measured directly on the polyline tokens the attention actually reads, 4,054 tokens over 25 scenarios: cosine similarity p50 1.00000 / p5 0.99965, relative change p50 0.0005 / p95 0.0266; uncorrelated with polyline length (r = 0.007); lanes 0.0003, road lines 0.0001, stop signs 0.0000, and the change concentrates on polygon kinds (speed bump 0.031, driveway 0.016, crosswalk 0.015) whose corners are what alternate-dot dropping costs.
NUM_PREDICTED_MODES = 6                         # Mode = possible future. We predict X futures. WOMD caps submissions at 6. Prune from X.
STAGING_CODE_VERSION = "2026-08-14-h"           # Bumped by hand on every code-dataset upload. kaggle_preflight compares working copy vs mount.

# > FEATURE SCALES. STANDARD DEVIATIONS MEASURED ON THIS PROJECT'S OWN STAGED DATA, NOT CHOSEN.
DISTANCE_NORMALISER_METRES = 67.9               # measure_scales.py on ../data/staged, 287 scenarios / 1,278 designated-target samples, 2026-08-14: standard deviation of map position, which is 20,002,407 of the values the model reads against 96,164 output values. Every distance-like quantity divides by this one number so geometry is preserved.
VELOCITY_NORMALISER_METRES_PER_SECOND = 5.2     # measure_scales.py on ../data/staged, 287 scenarios / 1,278 designated-target samples, 2026-08-14: standard deviation of agent and neighbour velocity.
DIMENSION_NORMALISER_METRES = 1.6               # measure_scales.py on ../data/staged, 287 scenarios / 1,278 designated-target samples, 2026-08-14: standard deviation of agent and neighbour dimensions.
SPEED_LIMIT_NORMALISER_METRES_PER_SECOND = 12.4 # measure_scales.py on ../data/staged, 287 scenarios / 1,278 designated-target samples, 2026-08-14: standard deviation of map speed limit over the rows where it applies. Heading cosine/sine and map direction arrows are unit-scale already (measured max abs 1.000) and are not divided.

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
MAP_SIGNAL_STATE = slice(MAP_KIND.stop, MAP_KIND.stop + NUM_TRAFFIC_SIGNAL_STATES * HISTORY_STEPS)
MAP_FEATURE_DIM = MAP_SIGNAL_STATE.stop

LANE_TYPES = (
    "TYPE_UNDEFINED",
    "TYPE_FREEWAY",
    "TYPE_SURFACE_STREET",
    "TYPE_BIKE_LANE",
)
NUM_LANE_TYPES = len(LANE_TYPES)

MAP_LANE_TYPE = slice(MAP_SIGNAL_STATE.stop, MAP_SIGNAL_STATE.stop + NUM_LANE_TYPES)
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