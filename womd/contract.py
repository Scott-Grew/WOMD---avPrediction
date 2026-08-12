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
STAGING_CROP_RADIUS_METRES = 250.0              # VERIFY: Staging time constant: max travel scenario 133m, store.py disk writing governer.
MAP_POINT_SPACING_METRES = 0.5
NUM_PREDICTED_MODES = 6                         # Mode = possible future. We predict X futures. WOMD caps submissions at 6. Prune from X.

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
MAP_FEATURE_DIM = MAP_BOUNDARY_TYPE.stop