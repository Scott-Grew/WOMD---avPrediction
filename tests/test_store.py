import numpy as np

from womd import contract, store
from waymo_open_dataset.protos import map_pb2

WORLD_ORIGIN = np.zeros(2)
WORLD_HEADING = 0.0


def polyline_feature(kind, points, feature_id=1):
    feature = map_pb2.MapFeature()
    feature.id = feature_id
    for x, y in points:
        getattr(feature, kind).polyline.add(x=x, y=y)
    return feature


def polygon_feature(kind, corners, feature_id=1):
    feature = map_pb2.MapFeature()
    feature.id = feature_id
    for x, y in corners:
        getattr(feature, kind).polygon.add(x=x, y=y)
    return feature


def test_map_rows_layout_for_all_seven_kinds():
    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    lane = polyline_feature("lane", [(0.0, 0.0), (10.0, 0.0)], feature_id=7)
    lane.lane.type = 2
    lane.lane.speed_limit_mph = 45.0
    road_line = polyline_feature("road_line", [(0.0, 1.0), (10.0, 1.0)])
    road_line.road_line.type = 2
    road_edge = polyline_feature("road_edge", [(0.0, 2.0), (10.0, 2.0)])
    road_edge.road_edge.type = 1
    stop_sign = map_pb2.MapFeature()
    stop_sign.id = 2
    stop_sign.stop_sign.position.x = 5.0
    stop_sign.stop_sign.position.y = 5.0
    features = [
        lane,
        road_line,
        road_edge,
        stop_sign,
        polygon_feature("crosswalk", square),
        polygon_feature("speed_bump", square),
        polygon_feature("driveway", square),
    ]

    lane_signal_history = np.zeros((contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES))
    lane_signal_history[:, 6] = 1.0
    signal_histories = {7: lane_signal_history}

    for kind_index, feature in enumerate(features):
        rows = store.map_feature_rows(feature, WORLD_ORIGIN, WORLD_HEADING, signal_histories)
        kind = contract.MAP_POLYLINE_KINDS[kind_index]

        assert rows is not None, kind
        assert rows.shape[1] == contract.MAP_FEATURE_DIM
        assert np.all(np.isfinite(rows))
        kind_block = rows[:, contract.MAP_KIND]
        assert np.all(kind_block.sum(axis=1) == 1.0)
        assert np.all(kind_block[:, kind_index] == 1.0)

        detail_block = rows[:, contract.MAP_KIND.stop:]
        if kind == "lane":
            assert np.all(rows[:, contract.MAP_SIGNAL_STATE] == lane_signal_history.reshape(-1))
            assert np.all(rows[:, contract.MAP_LANE_TYPE.start + 2] == 1.0)
            assert np.all(rows[:, contract.MAP_SPEED_LIMIT] == 45.0)
            assert np.all(rows[:, contract.MAP_BOUNDARY_TYPE] == 0.0)
        elif kind == "road_line":
            hot_columns = np.flatnonzero(detail_block.any(axis=0)) + contract.MAP_KIND.stop
            assert list(hot_columns) == [contract.MAP_BOUNDARY_TYPE.start + 2]
        elif kind == "road_edge":
            hot_columns = np.flatnonzero(detail_block.any(axis=0)) + contract.MAP_KIND.stop
            assert list(hot_columns) == [
                contract.MAP_BOUNDARY_TYPE.start + len(contract.ROAD_LINE_TYPES) + 1
            ]
        else:
            assert np.all(detail_block == 0.0)
            if kind == "stop_sign":
                assert len(rows) == 1
                assert np.all(rows[:, contract.MAP_DIRECTION] == 0.0)


def test_fill_bridges_measured_worst_case_shapes():
    endpoint_pair_edge = polyline_feature("road_edge", [(0.0, 0.0), (159.8, 0.0)])
    edge_points, edge_arrows, _ = store.map_feature_to_storage_frame(
        endpoint_pair_edge, WORLD_ORIGIN, WORLD_HEADING
    )
    assert len(edge_points) == 321
    edge_gaps = np.linalg.norm(np.diff(edge_points, axis=0), axis=1)
    assert np.all(edge_gaps <= contract.MAP_POINT_SPACING_METRES + 1e-9)
    assert np.allclose(np.linalg.norm(edge_arrows, axis=1), 1.0)

    corners_only_crosswalk = polygon_feature(
        "crosswalk", [(0.0, 0.0), (62.0, 0.0), (62.0, 62.0), (0.0, 62.0)]
    )
    ring_points, _, _ = store.map_feature_to_storage_frame(
        corners_only_crosswalk, WORLD_ORIGIN, WORLD_HEADING
    )
    ring_gaps = np.linalg.norm(np.diff(ring_points, axis=0), axis=1)
    assert np.all(ring_gaps <= contract.MAP_POINT_SPACING_METRES + 1e-9)
    assert np.allclose(ring_points[0], ring_points[-1])


def test_crop_keeps_polyline_crossing_the_boundary():
    crossing_edge = polyline_feature("road_edge", [(-300.0, 1.0), (300.0, 1.0)])
    stored_points, stored_arrows, _ = store.map_feature_to_storage_frame(
        crossing_edge, WORLD_ORIGIN, WORLD_HEADING
    )
    assert len(stored_points) > 900
    assert len(stored_points) == len(stored_arrows)
    assert np.all(
        np.linalg.norm(stored_points, axis=1) <= contract.STAGING_CROP_RADIUS_METRES
    )


def test_feature_row_layout_is_pinned():
    assert contract.MAP_FEATURE_DIM == 127
    assert contract.AGENT_FEATURE_DIM == 13
