import numpy as np

from waymo_open_dataset.protos import map_pb2, scenario_pb2
from womd import contract

STEP_SECONDS = 0.1
LANE_HALF_WIDTH = 1.8
APPROACH_LENGTH = 80.0
TURN_RADIUS = 12.0
LANE_POINT_SPACING = 2.0

MANOEUVRES = ("straight", "left", "right")


def _approach_rotation(approach_index):
    angle = approach_index * (np.pi / 2.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def _straight_path():
    distances = np.arange(-APPROACH_LENGTH, APPROACH_LENGTH, LANE_POINT_SPACING)
    return np.stack([distances, np.full_like(distances, -LANE_HALF_WIDTH)], axis=1)


def _turn_path(turn_sign):
    entry_x = np.arange(-APPROACH_LENGTH, -TURN_RADIUS, LANE_POINT_SPACING)
    entry = np.stack([entry_x, np.full_like(entry_x, -LANE_HALF_WIDTH)], axis=1)

    angles = np.linspace(0.0, np.pi / 2.0, 24)
    centre = np.array([-TURN_RADIUS, -LANE_HALF_WIDTH + turn_sign * TURN_RADIUS])
    arc = np.stack(
        [
            centre[0] + TURN_RADIUS * np.sin(angles),
            centre[1] - turn_sign * TURN_RADIUS * np.cos(angles),
        ],
        axis=1,
    )

    exit_distances = np.arange(LANE_POINT_SPACING, APPROACH_LENGTH, LANE_POINT_SPACING)
    exit_leg = np.stack(
        [
            np.full_like(exit_distances, arc[-1, 0]),
            arc[-1, 1] + turn_sign * exit_distances,
        ],
        axis=1,
    )
    return np.concatenate([entry, arc, exit_leg], axis=0)


def manoeuvre_path(manoeuvre):
    if manoeuvre == "straight":
        return _straight_path()
    return _turn_path(1.0 if manoeuvre == "left" else -1.0)


def resample_at_constant_speed(path, speed, step_count, start_distance):
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    travelled = start_distance + speed * STEP_SECONDS * np.arange(step_count)
    travelled = np.clip(travelled, 0.0, cumulative[-1])
    sampled_x = np.interp(travelled, cumulative, path[:, 0])
    sampled_y = np.interp(travelled, cumulative, path[:, 1])
    positions = np.stack([sampled_x, sampled_y], axis=1)
    deltas = np.gradient(positions, axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    velocities = deltas / STEP_SECONDS
    return positions, headings, velocities


def _fill_track(track, positions, headings, velocities, object_type, extent):
    track.object_type = object_type
    for step_index in range(len(positions)):
        state = track.states.add()
        state.center_x = float(positions[step_index, 0])
        state.center_y = float(positions[step_index, 1])
        state.center_z = 0.0
        state.length = float(extent[0])
        state.width = float(extent[1])
        state.height = 1.6
        state.heading = float(headings[step_index])
        state.velocity_x = float(velocities[step_index, 0])
        state.velocity_y = float(velocities[step_index, 1])
        state.valid = True


def _add_lane_feature(scenario, feature_id, points):
    map_feature = scenario.map_features.add()
    map_feature.id = feature_id
    map_feature.lane.type = map_pb2.LaneCenter.TYPE_SURFACE_STREET
    for point in points:
        polyline_point = map_feature.lane.polyline.add()
        polyline_point.x = float(point[0])
        polyline_point.y = float(point[1])
        polyline_point.z = 0.0


def build_intersection_scenario(scenario_id, agent_specifications, random_generator):
    scenario = scenario_pb2.Scenario()
    scenario.scenario_id = scenario_id
    scenario.current_time_index = contract.CURRENT_STEP_INDEX
    for step_index in range(contract.TOTAL_STEPS):
        scenario.timestamps_seconds.append(step_index * STEP_SECONDS)

    feature_id = 1
    for approach_index in range(4):
        rotation = _approach_rotation(approach_index)
        for manoeuvre in MANOEUVRES:
            _add_lane_feature(scenario, feature_id, manoeuvre_path(manoeuvre) @ rotation.T)
            feature_id += 1

    for approach_index, manoeuvre, speed, start_distance in agent_specifications:
        rotation = _approach_rotation(approach_index)
        path = manoeuvre_path(manoeuvre) @ rotation.T
        positions, headings, velocities = resample_at_constant_speed(
            path, speed, contract.TOTAL_STEPS, start_distance
        )
        jitter = random_generator.normal(0.0, 0.05, size=positions.shape)
        _fill_track(
            scenario.tracks.add(),
            positions + jitter,
            headings,
            velocities,
            scenario_pb2.Track.TYPE_VEHICLE,
            (4.6, 2.0),
        )

    for track_index in range(len(scenario.tracks)):
        required = scenario.tracks_to_predict.add()
        required.track_index = track_index
        required.difficulty = 1
    scenario.sdc_track_index = 0
    return scenario


def random_scenario(scenario_index, random_generator, agent_count=6):
    specifications = []
    for _ in range(agent_count):
        specifications.append(
            (
                int(random_generator.integers(0, 4)),
                MANOEUVRES[int(random_generator.integers(0, len(MANOEUVRES)))],
                float(random_generator.uniform(6.0, 14.0)),
                float(random_generator.uniform(0.0, 30.0)),
            )
        )
    return build_intersection_scenario(f"synthetic-{scenario_index:06d}", specifications, random_generator)
