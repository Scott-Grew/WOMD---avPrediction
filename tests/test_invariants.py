import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import fit_anchors
import measure_sensitivity
import submit
import train
from waymo_open_dataset.protos import scenario_pb2
from womd import baseline, contract, frame_ops, loader, loss, metrics, model, pipeline, store, tfrecord

STAGED_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "staged"


def test_crc32c_matches_known_answer():
    assert tfrecord.crc32c(b"123456789") == 0xE3069283


def test_tfrecord_round_trips_payloads():
    payloads = [b"", b"a", b"scenario-bytes" * 37]
    buffer = io.BytesIO()
    for payload in payloads:
        tfrecord.write_record(buffer, payload)
    buffer.seek(0)
    assert list(tfrecord.read_records(buffer)) == payloads


def test_tfrecord_rejects_corrupt_payload():
    buffer = io.BytesIO()
    tfrecord.write_record(buffer, b"intact-payload")
    corrupted = bytearray(buffer.getvalue())
    corrupted[14] ^= 0xFF
    with pytest.raises(tfrecord.CorruptRecordError):
        list(tfrecord.read_records(io.BytesIO(bytes(corrupted))))


def test_v1_agent_frame_transform_inverts():
    random_generator = np.random.default_rng(3)
    world_positions = random_generator.uniform(-120.0, 120.0, size=(64, 2))
    origin = np.array([13.5, -42.25])
    heading = 0.9137

    local = frame_ops.positions_to_agent_frame(world_positions, origin, heading)
    recovered = frame_ops.positions_to_world_frame(local, origin, heading)
    assert np.allclose(recovered, world_positions, atol=1e-9)


def test_v1_heading_wrap_stays_in_range():
    angles = np.array([-7.0, -np.pi, 0.0, np.pi, 7.0, 100.0])
    wrapped = frame_ops.wrap_to_pi(angles)
    assert (wrapped >= -np.pi).all() and (wrapped < np.pi).all()


def test_v6_storage_then_agent_frame_matches_direct_world_to_agent():
    track = scenario_pb2.Track()
    track.object_type = scenario_pb2.Track.TYPE_VEHICLE
    for step_index in range(contract.TOTAL_STEPS):
        turn_angle = 0.03 * step_index
        state = track.states.add()
        state.center_x = 40.0 + 1.5 * step_index * np.cos(turn_angle)
        state.center_y = -25.0 + 1.5 * step_index * np.sin(turn_angle)
        state.heading = turn_angle + 0.4
        state.velocity_x = 15.0 * np.cos(turn_angle + 0.4)
        state.velocity_y = 15.0 * np.sin(turn_angle + 0.4)
        state.length = 4.5
        state.width = 2.0
        state.valid = True

    sdc_origin = np.array([-12.0, 31.0])
    sdc_heading = -1.2
    stored_rows, stored_valid = store.track_to_feature_rows(track, sdc_origin, sdc_heading, False)
    track_rows = stored_rows.astype(np.float32)[np.newaxis]

    origin, heading = loader.sample_frame(track_rows, 0)
    two_step = loader.track_rows_to_agent_frame(track_rows[0], origin, heading)

    world_positions = np.array([[state.center_x, state.center_y] for state in track.states])
    world_headings = np.array([state.heading for state in track.states])
    world_velocities = np.array([[state.velocity_x, state.velocity_y] for state in track.states])
    now_state = track.states[contract.CURRENT_STEP_INDEX]
    now_position = np.array([now_state.center_x, now_state.center_y])

    direct_positions = frame_ops.positions_to_agent_frame(world_positions, now_position, now_state.heading)
    direct_headings = world_headings - now_state.heading
    direct_velocities = frame_ops.directions_to_agent_frame(world_velocities, now_state.heading)

    assert np.allclose(two_step[:, contract.AGENT_POSITION], direct_positions, atol=1e-3)
    assert np.allclose(two_step[:, contract.AGENT_HEADING_COSINE], np.cos(direct_headings), atol=1e-3)
    assert np.allclose(two_step[:, contract.AGENT_HEADING_SINE], np.sin(direct_headings), atol=1e-3)
    assert np.allclose(two_step[:, contract.AGENT_VELOCITY], direct_velocities, atol=1e-3)


def test_v_permuting_neighbours_and_map_dots_leaves_trajectories_unchanged():
    torch.manual_seed(11)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    batch = {
        "agent_history": torch.randn(2, 11, 13),
        "agent_history_mask": torch.ones(2, 11, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 6, 11, 13),
        "neighbour_history_mask": torch.ones(2, 6, 11, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            2, 6, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": torch.randn(80, contract.MAP_FEATURE_DIM),
        "map_dot_polyline_slot": torch.arange(80) // 8,
        "map_chunk_signal_history": torch.zeros(
            2, 5, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(2, 5, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(5),
    }
    batch["neighbour_history_mask"][:, 5] = False
    batch["agent_signal_history"][0, :, 4] = 1.0
    batch["neighbour_signal_history"][:, 2, :, 6] = 1.0
    batch["map_chunk_signal_history"][0, 1:3, :, 6] = 1.0
    batch["map_chunk_lane_context"][0, 1:3, contract.LANE_CONTEXT_REACHABLE] = 1.0
    batch["map_rows"][:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (80, 2), dtype=torch.float32
    )

    neighbour_order = torch.randperm(6)
    map_order = torch.randperm(80)
    permuted = dict(batch)
    permuted["neighbour_history"] = batch["neighbour_history"][:, neighbour_order]
    permuted["neighbour_history_mask"] = batch["neighbour_history_mask"][:, neighbour_order]
    permuted["neighbour_signal_history"] = batch["neighbour_signal_history"][:, neighbour_order]
    permuted["map_rows"] = batch["map_rows"][map_order]
    permuted["map_dot_polyline_slot"] = batch["map_dot_polyline_slot"][map_order]

    with torch.no_grad():
        base_trajectories, base_logits = predictor(batch)
        permuted_trajectories, permuted_logits = predictor(permuted)

    assert torch.allclose(base_trajectories, permuted_trajectories, atol=1e-4)
    assert torch.allclose(base_logits, permuted_logits, atol=1e-5)


def test_polyline_pooling_isolates_groups_and_leaves_empty_slots_absent():
    torch.manual_seed(5)
    dot_embeddings = torch.randn(9, 8)
    dot_polyline_slot = torch.tensor([0, 0, 0, 1, 1, 1, 3, 3, 4])

    tokens, present = model.pool_dots_to_polyline_tokens(dot_embeddings, dot_polyline_slot, 2, 3)
    assert tokens.shape == (2, 3, 8)
    assert present.tolist() == [[True, True, False], [True, True, False]]
    assert torch.all(tokens[:, 2] == 0.0)

    poisoned = dot_embeddings.clone()
    poisoned[:3] = 999.0
    poisoned_tokens, _ = model.pool_dots_to_polyline_tokens(poisoned, dot_polyline_slot, 2, 3)
    assert torch.equal(poisoned_tokens[0, 1:], tokens[0, 1:])
    assert torch.equal(poisoned_tokens[1], tokens[1])
    assert not torch.equal(poisoned_tokens[0, 0], tokens[0, 0])


def lane_polyline_rows(first_dot_x, dot_count):
    rows = np.zeros((dot_count, contract.MAP_FEATURE_DIM), dtype=np.float32)
    rows[:, contract.MAP_POSITION] = np.stack(
        [first_dot_x + np.arange(dot_count, dtype=np.float64), np.zeros(dot_count)], axis=1
    )
    rows[:, contract.MAP_DIRECTION] = np.array([1.0, 0.0])
    rows[:, contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index("lane")] = 1.0
    return rows


def test_lane_context_walks_connections_forward_only_and_charges_the_lane_left_behind():
    feature_lengths = np.array([11, 11, 11], dtype=np.int64)
    scenario_array = {
        "map_rows": np.concatenate(
            [lane_polyline_rows(0.0, 11), lane_polyline_rows(20.0, 11), lane_polyline_rows(-30.0, 11)]
        ),
        "feature_lengths": feature_lengths,
        "feature_ids": np.array([101, 102, 103], dtype=np.int64),
        "map_dot_polyline_index": np.repeat(np.arange(3), feature_lengths),
        "lane_connections": np.array([[101, 102, 1], [103, 101, 1]], dtype=np.int64),
        "lane_neighbour_ids": np.zeros((0, contract.LANE_NEIGHBOUR_ID_WIDTH), dtype=np.int64),
        "stop_sign_controlled_lanes": np.zeros((0, contract.STOP_SIGN_LANE_WIDTH), dtype=np.int64),
    }

    lane_context = loader.lane_context_per_polyline(
        loader.lane_graph_of_scenario(scenario_array), np.array([5.0, 2.0]), np.array([1.0, 0.0])
    )
    own_lane, one_hop_downstream, upstream_only = lane_context

    assert np.all(
        lane_context[:, contract.LANE_CONTEXT_AGENT_LANE_DISTANCE]
        == pytest.approx(2.0 / contract.DISTANCE_NORMALISER_METRES, rel=1e-6)
    )
    assert own_lane[contract.LANE_CONTEXT_REACHABLE] == 1.0
    assert own_lane[contract.LANE_CONTEXT_GRAPH_DISTANCE] == 0.0
    assert one_hop_downstream[contract.LANE_CONTEXT_REACHABLE] == 1.0
    assert one_hop_downstream[contract.LANE_CONTEXT_GRAPH_DISTANCE] == pytest.approx(
        10.0 / contract.DISTANCE_NORMALISER_METRES, rel=1e-6
    )
    assert upstream_only[contract.LANE_CONTEXT_REACHABLE] == 0.0
    assert upstream_only[contract.LANE_CONTEXT_GRAPH_DISTANCE] == 0.0


def test_a_lane_reached_only_sideways_is_flagged_a_lane_change_at_the_distance_it_changed_from():
    parallel_lane_rows = lane_polyline_rows(20.0, 11)
    parallel_lane_rows[:, contract.MAP_POSITION.start + 1] = 3.5
    feature_lengths = np.array([11, 11, 11], dtype=np.int64)
    scenario_array = {
        "map_rows": np.concatenate(
            [lane_polyline_rows(0.0, 11), lane_polyline_rows(20.0, 11), parallel_lane_rows]
        ),
        "feature_lengths": feature_lengths,
        "feature_ids": np.array([101, 102, 103], dtype=np.int64),
        "map_dot_polyline_index": np.repeat(np.arange(3), feature_lengths),
        "lane_connections": np.array([[101, 102, 1]], dtype=np.int64),
        "lane_neighbour_ids": np.array(
            [[102, 103, contract.LANE_SIDES.index("left")]], dtype=np.int64
        ),
        "stop_sign_controlled_lanes": np.zeros((0, contract.STOP_SIGN_LANE_WIDTH), dtype=np.int64),
    }

    own_lane, one_hop_downstream, sideways_only = loader.lane_context_per_polyline(
        loader.lane_graph_of_scenario(scenario_array), np.array([5.0, 0.0]), np.array([1.0, 0.0])
    )

    assert one_hop_downstream[contract.LANE_CONTEXT_REACHABLE] == 1.0
    assert one_hop_downstream[contract.LANE_CONTEXT_REACHABLE_BY_LANE_CHANGE] == 0.0
    assert sideways_only[contract.LANE_CONTEXT_REACHABLE] == 0.0
    assert sideways_only[contract.LANE_CONTEXT_REACHABLE_BY_LANE_CHANGE] == 1.0
    assert own_lane[contract.LANE_CONTEXT_REACHABLE_BY_LANE_CHANGE] == 0.0
    assert sideways_only[contract.LANE_CONTEXT_GRAPH_DISTANCE] == pytest.approx(
        one_hop_downstream[contract.LANE_CONTEXT_GRAPH_DISTANCE], rel=1e-6
    )
    assert sideways_only[contract.LANE_CONTEXT_GRAPH_DISTANCE] == pytest.approx(
        10.0 / contract.DISTANCE_NORMALISER_METRES, rel=1e-6
    )


def test_neighbour_future_loss_scores_only_the_neighbour_steps_that_were_logged():
    torch.manual_seed(23)
    logged_positions = torch.randn(2, 3, contract.FUTURE_STEPS, 2)
    neighbour_future_mask = torch.ones(2, 3, contract.FUTURE_STEPS, dtype=torch.bool)
    neighbour_future_mask[:, 2] = False

    neighbour_readable = torch.ones(2, 3, dtype=torch.bool)
    predicted_positions = logged_positions.clone()
    exact = loss.neighbour_future_loss(
        predicted_positions, logged_positions, neighbour_future_mask, neighbour_readable
    )
    assert torch.isfinite(exact) and float(exact) == 0.0

    predicted_positions[:, 2] = 1e6
    assert float(
        loss.neighbour_future_loss(
            predicted_positions, logged_positions, neighbour_future_mask, neighbour_readable
        )
    ) == 0.0

    predicted_positions[:, 0] = logged_positions[:, 0] + torch.tensor([3.0, 4.0])
    scored = loss.neighbour_future_loss(
        predicted_positions, logged_positions, neighbour_future_mask, neighbour_readable
    )
    assert torch.isfinite(scored)
    assert float(scored) == pytest.approx(2.5, rel=1e-5)

    unreadable = neighbour_readable.clone()
    unreadable[:, 0] = False
    assert float(
        loss.neighbour_future_loss(
            predicted_positions, logged_positions, neighbour_future_mask, unreadable
        )
    ) == 0.0


def test_padded_chunk_slot_stays_exactly_zero_through_both_map_projections():
    torch.manual_seed(19)
    encoder = model.SceneEncoder()
    map_rows = torch.randn(24, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (24, 2), dtype=torch.float32
    )
    signal_history = torch.zeros(2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES)
    lane_context = torch.zeros(2, 2, contract.LANE_CONTEXT_DIM)
    signal_history[0, 0, :, 6] = 1.0
    lane_context[0, 0, contract.LANE_CONTEXT_REACHABLE] = 1.0

    with torch.no_grad():
        map_tokens, map_present = model.pool_dots_to_polyline_tokens(
            encoder.map_encoder(map_rows), torch.arange(24) // 12, 2, 2
        )
        map_tokens = (
            map_tokens
            + encoder.signal_projection(signal_history.flatten(start_dim=-2))
            + encoder.lane_context_projection(lane_context)
        )

    assert map_present.tolist() == [[True, True], [False, False]]
    assert torch.all(map_tokens[1] == 0.0)
    assert torch.any(map_tokens[0] != 0.0)


def test_v4_null_baselines_reproduce_the_motion_each_one_assumes():
    step_offsets = np.arange(-contract.CURRENT_STEP_INDEX, contract.FUTURE_STEPS + 1)
    elapsed = step_offsets * baseline.TIMESTEP_SECONDS
    speeds = np.array([8.0, 0.0])
    yaw_rates = np.array([0.3, 0.0])

    turned = yaw_rates[:, None] * elapsed[None, :]
    radii = np.where(yaw_rates == 0.0, 0.0, speeds / np.where(yaw_rates == 0.0, 1.0, yaw_rates))
    arc_positions = np.stack(
        [radii[:, None] * np.sin(turned), radii[:, None] * (1.0 - np.cos(turned))], axis=-1
    )

    agent_track = np.zeros((2, len(step_offsets), contract.AGENT_FEATURE_DIM), dtype=np.float32)
    agent_track[..., contract.AGENT_POSITION] = arc_positions
    agent_track[..., contract.AGENT_HEADING_COSINE] = np.cos(turned)
    agent_track[..., contract.AGENT_HEADING_SINE] = np.sin(turned)
    agent_track[..., contract.AGENT_VELOCITY] = speeds[:, None, None] * np.stack(
        [np.cos(turned), np.sin(turned)], axis=-1
    )

    batch = {
        "agent_history": torch.from_numpy(agent_track[:, :contract.HISTORY_STEPS]),
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
    }
    logged_future = torch.from_numpy(arc_positions[:, contract.HISTORY_STEPS:].astype(np.float32))

    turning_trajectories, turning_logits = baseline.constant_turn_rate_and_velocity(batch)
    straight_trajectories, _ = baseline.constant_velocity(batch)
    assert turning_trajectories.shape == (2, 1, contract.FUTURE_STEPS, 2)
    assert turning_logits.shape == (2, 1)

    assert torch.allclose(turning_trajectories[:, 0], logged_future, atol=1e-3)
    assert (straight_trajectories[0, 0, -1] - logged_future[0, -1]).norm() > 1.0
    assert torch.all(turning_trajectories[1] == 0.0)
    assert torch.all(straight_trajectories[1] == 0.0)

    pruned, _ = model.prune_modes_batched(turning_trajectories, turning_logits)
    assert torch.equal(pruned, turning_trajectories.expand(-1, contract.NUM_PREDICTED_MODES, -1, -1))


def test_v4_anchor_null_drives_the_reachable_distance_to_each_most_used_anchor_in_a_straight_line():
    unit_anchors = model.unit_anchor_offsets_per_type() * torch.tensor(
        [1.0, 0.6, 0.3]
    )[:, None, None]
    driven_types = torch.tensor([0, contract.NUM_OBJECT_TYPES - 1])
    current_speeds = torch.tensor([8.0, 0.0])
    agent_history = torch.zeros(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.stack(
        [current_speeds, torch.zeros(2)], dim=-1
    )
    agent_history[
        torch.arange(2), contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE.start + driven_types
    ] = 1.0
    batch = {"agent_history": agent_history}

    trajectories, confidence_logits = baseline.straight_lines_to_most_used_anchors(
        batch, unit_anchors
    )
    reachable_distance_metres = model.agent_reachable_distance(agent_history)

    assert trajectories.shape == (2, contract.NUM_PREDICTED_MODES, contract.FUTURE_STEPS, 2)
    assert confidence_logits.shape == (2, contract.NUM_PREDICTED_MODES)
    assert torch.allclose(
        trajectories[:, :, -1],
        reachable_distance_metres[:, None, None]
        * unit_anchors[driven_types][:, : contract.NUM_PREDICTED_MODES],
        atol=1e-4,
    )

    steps = torch.cat([trajectories[..., :1, :], trajectories.diff(dim=-2)], dim=-2)
    assert torch.allclose(steps, steps[..., :1, :].expand_as(steps), atol=1e-5)


def test_v4_lane_null_drives_the_centreline_at_the_logged_speed_and_forks_into_separate_routes():
    turning_lane = lane_polyline_rows(0.0, 21)
    turning_lane[:, contract.MAP_POSITION] = np.stack([np.full(21, 11.0), np.arange(21.0)], axis=1)
    turning_lane[:, contract.MAP_DIRECTION] = np.array([0.0, 1.0])
    feature_lengths = np.array([11, 20, 21], dtype=np.int64)

    speed = 10.0
    track_rows = np.zeros((1, contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32)
    track_rows[0, :, contract.AGENT_POSITION] = np.array([2.0, 0.0])
    track_rows[0, :, contract.AGENT_HEADING_COSINE] = 1.0
    track_rows[0, :, contract.AGENT_VELOCITY] = np.array([speed, 0.0])
    scenario_array = {
        "map_rows": np.concatenate(
            [lane_polyline_rows(0.0, 11), lane_polyline_rows(11.0, 20), turning_lane]
        ),
        "feature_lengths": feature_lengths,
        "feature_ids": np.array([101, 102, 103], dtype=np.int64),
        "map_dot_polyline_index": np.repeat(np.arange(3), feature_lengths),
        "lane_connections": np.array([[101, 102, 1], [101, 103, 1]], dtype=np.int64),
        "track_rows": track_rows,
    }

    trajectories, followed_a_lane = baseline.follow_the_lane_predictions(scenario_array, 0)
    step_metres = speed * baseline.TIMESTEP_SECONDS
    driven_metres = step_metres * np.arange(1, contract.FUTURE_STEPS + 1)

    assert followed_a_lane
    assert trajectories.shape == (contract.NUM_PREDICTED_MODES, contract.FUTURE_STEPS, 2)
    assert np.allclose(
        trajectories[0], np.stack([driven_metres, np.zeros(contract.FUTURE_STEPS)], axis=1), atol=1e-4
    )
    assert np.allclose(
        np.linalg.norm(np.diff(trajectories[:2], axis=1), axis=-1), step_metres, atol=1e-4
    )
    assert np.allclose(trajectories[1, -1], np.array([9.0, 71.0]), atol=1e-4)
    assert np.array_equal(trajectories[2], trajectories[0])


def test_anchor_fitting_recovers_tight_well_separated_clusters_and_leaves_no_anchor_empty():
    random_generator = np.random.default_rng(17)
    cluster_centres = model.unit_anchor_offsets() + torch.tensor(
        random_generator.uniform(-0.08, 0.08, (model.QUERY_COUNT, 2)), dtype=torch.float32
    )
    endpoints = (
        cluster_centres[:, None, :]
        + torch.tensor(
            random_generator.normal(0.0, 0.005, (model.QUERY_COUNT, 50, 2)), dtype=torch.float32
        )
    ).reshape(-1, 2)

    fitted, assignment, _, _, stopped_by_convergence = fit_anchors.fit_unit_anchors(endpoints)
    counts = fit_anchors.endpoints_per_centre(assignment, model.QUERY_COUNT)

    assert stopped_by_convergence
    assert fitted.shape == (model.QUERY_COUNT, 2)
    assert torch.allclose(fitted, cluster_centres, atol=0.01)
    assert int(counts.min()) > 0


def test_batched_pruning_walk_matches_the_single_sample_walk():
    torch.manual_seed(7)
    trajectories = model.PRUNE_DISTANCE_METRES * torch.randn(
        2, model.QUERY_COUNT, contract.FUTURE_STEPS, 2
    )
    confidence_logits = torch.randn(2, model.QUERY_COUNT)
    trajectories[1] = trajectories[1, :1]

    batched_trajectories, batched_logits, kept_counts = model.prune_modes_batched_with_kept_count(
        trajectories, confidence_logits
    )
    assert kept_counts[0] > 1 and kept_counts[0] <= contract.NUM_PREDICTED_MODES
    assert kept_counts[1] == 1
    for sample_index in range(len(kept_counts)):
        walked_trajectories, walked_logits = model.prune_modes(
            trajectories[sample_index], confidence_logits[sample_index]
        )
        assert torch.equal(walked_trajectories, batched_trajectories[sample_index])
        assert torch.equal(walked_logits, batched_logits[sample_index])


def total_turning_degrees(polyline):
    steps = polyline.diff(dim=-2)
    leading, trailing = steps[..., :-1, :], steps[..., 1:, :]
    cross = leading[..., 0] * trailing[..., 1] - leading[..., 1] * trailing[..., 0]
    return torch.atan2(cross.abs(), (leading * trailing).sum(dim=-1)).rad2deg().sum(dim=-1)


def test_a_trajectory_cannot_outrun_or_outturn_the_control_polygon_it_is_drawn_through():
    torch.manual_seed(23)
    assert model.TRAJECTORY_CONTROL_POINTS < contract.FUTURE_STEPS

    curve_basis = model.bernstein_curve_basis(
        model.TRAJECTORY_CONTROL_POINTS, contract.FUTURE_STEPS
    )
    control_points = 20.0 * torch.randn(512, model.TRAJECTORY_CONTROL_POINTS, 2)
    curve = torch.matmul(curve_basis[:, 1:], control_points)
    walked = torch.cat([torch.zeros(512, 1, 2), curve], dim=1)
    control_polygon = torch.cat([torch.zeros(512, 1, 2), control_points], dim=1)

    assert curve_basis.shape == (contract.FUTURE_STEPS, model.TRAJECTORY_CONTROL_POINTS + 1)
    assert torch.allclose(curve[:, -1], control_points[:, -1], atol=1e-4)
    assert (
        walked.diff(dim=1).norm(dim=-1).sum(dim=-1)
        <= control_polygon.diff(dim=1).norm(dim=-1).sum(dim=-1)
    ).all()
    assert (total_turning_degrees(walked) <= total_turning_degrees(control_polygon)).all()


def test_every_emitted_trajectory_stays_in_the_curve_family_after_the_anchor_ramp_is_added():
    torch.manual_seed(31)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type()).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.mul_(50.0)

    agent_history = torch.randn(3, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.tensor(
        [[0.0, 0.0], [12.0, 0.0], [0.0, -25.0]]
    )
    map_rows = torch.randn(30, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = 0.0
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(3, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(3, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(3, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            3, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            3, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.arange(30) // 10,
        "map_chunk_signal_history": torch.zeros(
            3, 1, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(3, 1, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(1),
    }

    with torch.no_grad():
        trajectories, _ = predictor(batch)
    flattened = trajectories.permute(2, 0, 1, 3).reshape(contract.FUTURE_STEPS, -1)
    curve_basis = predictor.mode_decoder.curve_basis
    projected = curve_basis @ torch.linalg.lstsq(curve_basis, flattened).solution

    independent_positions = trajectories.std() * torch.randn_like(flattened)
    independent_residual = (
        independent_positions - curve_basis @ torch.linalg.lstsq(
            curve_basis, independent_positions
        ).solution
    ).norm()

    assert trajectories.abs().max() > 1.0
    assert (projected - flattened).norm() < 1e-3 * flattened.norm()
    assert independent_residual > 0.1 * independent_positions.norm()


def test_swapping_two_anchors_swaps_the_modes_that_carry_them():
    torch.manual_seed(29)
    unit_anchors = model.unit_anchor_offsets_per_type()
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.queries.zero_()

    agent_history = torch.randn(1, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.tensor([10.0, 0.0])
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE] = 0.0
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE.start] = 1.0
    map_rows = torch.randn(12, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = 0.0
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(1, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(1, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(1, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            1, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            1, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.zeros(12, dtype=torch.long),
        "map_chunk_signal_history": torch.zeros(
            1, 1, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(1, 1, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(1),
    }

    forward_mode = 0
    backward_mode = model.ANCHOR_DISTANCE_COUNT * (model.ANCHOR_DIRECTION_COUNT // 2)
    swapped_order = torch.arange(model.QUERY_COUNT)
    swapped_order[[forward_mode, backward_mode]] = swapped_order[[backward_mode, forward_mode]]
    with torch.no_grad():
        endpoints = predictor(batch)[0][0, :, -1]
        predictor.mode_decoder.anchor_offsets.copy_(unit_anchors[:, swapped_order])
        swapped_endpoints = predictor(batch)[0][0, :, -1]

    assert (endpoints[forward_mode] - endpoints[backward_mode]).norm() > 1.0
    assert torch.allclose(swapped_endpoints[forward_mode], endpoints[backward_mode], atol=1e-4)
    assert torch.allclose(swapped_endpoints[backward_mode], endpoints[forward_mode], atol=1e-4)
    untouched_modes = [
        mode for mode in range(model.QUERY_COUNT) if mode not in (forward_mode, backward_mode)
    ]
    assert torch.allclose(swapped_endpoints[untouched_modes], endpoints[untouched_modes], atol=1e-4)


def test_a_mode_endpoint_carries_its_anchor_at_the_distance_the_loss_assigns_by():
    torch.manual_seed(37)
    unit_anchors = model.unit_anchor_offsets_per_type() * torch.tensor(
        [1.0, 0.6, 0.3]
    )[:, None, None]
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.zero_()
        predictor.mode_decoder.trajectory_head[-1].bias.zero_()

    driven_types = torch.tensor([0, contract.NUM_OBJECT_TYPES - 1])
    agent_history = torch.randn(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.tensor(
        [[7.0, 0.0], [0.0, 0.0]]
    )
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE] = 0.0
    agent_history[
        torch.arange(2), contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE.start + driven_types
    ] = 1.0
    map_rows = torch.randn(20, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = 0.0
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(2, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.arange(20) // 10,
        "map_chunk_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(2, 2, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(2),
    }

    reachable_distance_metres = model.agent_reachable_distance(agent_history)
    with torch.no_grad():
        trajectories, _ = predictor(batch)
    endpoints = trajectories[:, :, -1]

    assert torch.allclose(
        endpoints,
        unit_anchors[driven_types] * reachable_distance_metres[:, None, None],
        atol=1e-4,
    )

    ramp = torch.arange(1, contract.FUTURE_STEPS + 1) / contract.FUTURE_STEPS
    logged_futures = endpoints[0][:, None, :] * ramp[None, :, None]
    assigned_mode = loss.anchor_assigned_mode(
        predictor.unit_anchors[driven_types[0]].expand(model.QUERY_COUNT, -1, -1),
        reachable_distance_metres[:1].expand(model.QUERY_COUNT),
        logged_futures,
        torch.ones(model.QUERY_COUNT, contract.FUTURE_STEPS, dtype=torch.bool),
    )

    assert torch.equal(assigned_mode, torch.arange(model.QUERY_COUNT))


def test_two_agents_alike_but_for_their_object_type_get_their_own_types_anchor_set():
    torch.manual_seed(71)
    geometric_fan = model.unit_anchor_offsets()
    vehicle_index = contract.PREDICTED_OBJECT_TYPES.index("TYPE_VEHICLE")
    pedestrian_index = contract.PREDICTED_OBJECT_TYPES.index("TYPE_PEDESTRIAN")
    unit_anchors = model.unit_anchor_offsets_per_type()
    unit_anchors[pedestrian_index] = 0.6 * geometric_fan.flip(0)
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.zero_()
        predictor.mode_decoder.trajectory_head[-1].bias.zero_()

    agent_history = torch.randn(
        1, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM
    ).repeat(2, 1, 1)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.tensor([9.0, 0.0])
    agent_history[:, :, contract.AGENT_TYPE] = 0.0
    agent_history[0, :, contract.AGENT_TYPE.start + vehicle_index] = 1.0
    agent_history[1, :, contract.AGENT_TYPE.start + pedestrian_index] = 1.0
    map_rows = torch.randn(10, contract.MAP_FEATURE_DIM).repeat(2, 1)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = 0.0
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(
            1, 2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM
        ).repeat(2, 1, 1, 1),
        "neighbour_history_mask": torch.ones(2, 2, contract.HISTORY_STEPS, dtype=torch.bool),
        "agent_signal_history": torch.zeros(
            2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "neighbour_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_rows": map_rows,
        "map_dot_polyline_slot": torch.repeat_interleave(torch.tensor([0, 2]), 10),
        "map_chunk_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.zeros(2, 2, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(2),
    }

    with torch.no_grad():
        (
            trajectories, _, _, _, reachable_distance_metres, selected_unit_anchors, _,
        ) = predictor.predict_with_heading(batch)
    endpoints = trajectories[:, :, -1]

    assert torch.allclose(selected_unit_anchors[0], unit_anchors[vehicle_index], atol=1e-6)
    assert torch.allclose(selected_unit_anchors[1], unit_anchors[pedestrian_index], atol=1e-6)
    assert torch.allclose(
        endpoints[0], unit_anchors[vehicle_index] * reachable_distance_metres[0], atol=1e-4
    )
    assert torch.allclose(
        endpoints[1], unit_anchors[pedestrian_index] * reachable_distance_metres[1], atol=1e-4
    )
    assert float((endpoints[0] - endpoints[1]).norm(dim=-1).min()) > 1.0

    shared_target_mode = 3
    logged_endpoint = (
        unit_anchors[vehicle_index, shared_target_mode] * reachable_distance_metres[0]
    )
    ramp = torch.arange(1, contract.FUTURE_STEPS + 1) / contract.FUTURE_STEPS
    future_positions = (logged_endpoint[None, :] * ramp[:, None]).expand(2, -1, -1)
    assigned_mode = loss.anchor_assigned_mode(
        selected_unit_anchors, reachable_distance_metres, future_positions,
        torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool),
    )

    assert int(assigned_mode[0]) == shared_target_mode
    assert int(assigned_mode[1]) != shared_target_mode
    assert int(assigned_mode[1]) == int(
        (unit_anchors[pedestrian_index] * reachable_distance_metres[1] - logged_endpoint)
        .norm(dim=-1).argmin()
    )


def test_opposed_logged_endpoints_are_assigned_to_opposed_anchor_territories():
    unit_anchors = model.unit_anchor_offsets()
    reachable_distance_metres = torch.tensor([40.0, 40.0])
    future_positions = torch.zeros(2, contract.FUTURE_STEPS, 2)
    future_positions[0, :, 0] = torch.linspace(0.5, 40.0, contract.FUTURE_STEPS)
    future_positions[1, :, 0] = torch.linspace(-0.5, -40.0, contract.FUTURE_STEPS)
    future_mask = torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool)

    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors.expand(2, -1, -1), reachable_distance_metres, future_positions, future_mask
    )

    assert assigned_mode[0] != assigned_mode[1]
    assert torch.allclose(unit_anchors[assigned_mode[0]], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(unit_anchors[assigned_mode[1]], torch.tensor([-1.0, 0.0]), atol=1e-6)


def test_assignment_reads_the_last_valid_future_step_and_not_the_padded_tail():
    unit_anchors = model.unit_anchor_offsets()
    reachable_distance_metres = torch.tensor([40.0])
    future_positions = torch.zeros(1, contract.FUTURE_STEPS, 2)
    future_positions[0, :10, 0] = 40.0
    future_mask = torch.zeros(1, contract.FUTURE_STEPS, dtype=torch.bool)
    future_mask[0, :10] = True

    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors.expand(1, -1, -1), reachable_distance_metres, future_positions, future_mask
    )
    padded_tail_mode = loss.anchor_assigned_mode(
        unit_anchors.expand(1, -1, -1), reachable_distance_metres, future_positions,
        torch.ones_like(future_mask),
    )

    assert torch.allclose(unit_anchors[assigned_mode[0]], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert padded_tail_mode[0] != assigned_mode[0]
    assert float(unit_anchors[padded_tail_mode[0]].norm()) == pytest.approx(
        1.0 / model.ANCHOR_DISTANCE_COUNT
    )


def test_one_training_step_runs_the_whole_path_over_staged_scenarios(tmp_path):
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:2]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")

    torch.manual_seed(0)
    predictor = model.MotionPredictor(model.unit_anchor_offsets_per_type())
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    batch = next(iter(pipeline.batches(scenario_paths, 0, 2, 0, 0, True)))

    (
        trajectories, heading_cosine_sine, position_log_standard_deviation, confidence_logits,
        reachable_distance_metres, selected_unit_anchors, neighbour_future_positions,
    ) = predictor.predict_with_heading(batch)
    assert neighbour_future_positions.shape == (
        batch["neighbour_future_positions"].shape[:2] + (contract.FUTURE_STEPS, 2)
    )
    assert selected_unit_anchors.shape == (
        batch["agent_history"].shape[0], model.QUERY_COUNT, 2
    )
    components = loss.prediction_loss(
        trajectories, heading_cosine_sine, position_log_standard_deviation, confidence_logits,
        batch["future_positions"], batch["future_headings"], batch["future_mask"],
        selected_unit_anchors, reachable_distance_metres, 1.0,
    )
    neighbour_future = loss.neighbour_future_loss(
        neighbour_future_positions,
        batch["neighbour_future_positions"],
        batch["neighbour_future_mask"],
        batch["neighbour_history_mask"].any(dim=-1),
    )
    assert torch.stack(components).isfinite().all()
    assert torch.isfinite(neighbour_future)
    (components[0] + neighbour_future).backward()
    assert all(parameter.grad is not None for parameter in predictor.parameters())
    assert any(parameter.grad.abs().sum() > 0.0 for parameter in predictor.parameters())
    optimizer.step()

    accumulator = metrics.MetricAccumulator()
    accumulator.update(
        trajectories.detach(), confidence_logits.detach(),
        batch["future_positions"], batch["future_mask"],
    )
    assert all(math.isfinite(value) for value in accumulator.results().values())

    checkpoint_path = tmp_path / "predictor.pt"
    torch.save(predictor.state_dict(), checkpoint_path)
    reloaded_state = torch.load(checkpoint_path)
    assert all(
        torch.equal(reloaded_state[name], parameter)
        for name, parameter in predictor.state_dict().items()
    )


def test_submission_world_frame_returns_the_logged_future_to_its_logged_place():
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:1]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")

    scenario_array = loader.read_scenario(scenario_paths[0])
    track_index = int(loader.eligible_track_indices(
        scenario_array["track_rows"], scenario_array["track_valid"],
        scenario_array["is_designated_target"], True,
    )[0])
    sample = loader.build_sample(scenario_array, track_index)
    agent_frame_future = sample["future_positions"]

    world_future = submit.agent_frame_to_world_frame(agent_frame_future, sample, scenario_array)
    world_coordinate_resolution = np.finfo(np.float32).eps * np.abs(world_future).max()

    back_to_storage_frame = frame_ops.positions_to_agent_frame(
        world_future, scenario_array["frame_origin"], scenario_array["frame_heading"]
    )
    back_to_agent_frame = frame_ops.positions_to_agent_frame(
        back_to_storage_frame, sample["frame_origin"], sample["frame_heading"]
    )
    assert np.allclose(back_to_agent_frame, agent_frame_future, atol=world_coordinate_resolution)

    logged_world_future = frame_ops.positions_to_world_frame(
        scenario_array["track_rows"][
            track_index, contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION
        ],
        scenario_array["frame_origin"], scenario_array["frame_heading"],
    )
    logged = scenario_array["track_valid"][track_index, contract.CURRENT_STEP_INDEX + 1:]
    assert logged.any()
    assert np.allclose(
        world_future[logged], logged_world_future[logged], atol=world_coordinate_resolution
    )


def two_lane_signal_scenario(track_count, signalled_lane_history):
    signalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane[:, contract.MAP_POSITION.start + 1] = 40.0
    feature_lengths = np.array([11, 11], dtype=np.int64)
    polyline_signal_histories = np.zeros(
        (2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES), dtype=np.float32
    )
    polyline_signal_histories[0] = signalled_lane_history

    track_rows = np.zeros(
        (track_count, contract.TOTAL_STEPS, contract.AGENT_FEATURE_DIM), dtype=np.float32
    )
    track_valid = np.ones((track_count, contract.TOTAL_STEPS), dtype=bool)
    track_rows[:, :, contract.AGENT_HEADING_COSINE] = 1.0
    track_rows[0, :, contract.AGENT_POSITION] = np.array([5.0, 0.0])
    track_rows[1, :, contract.AGENT_POSITION] = np.array([5.0, 40.0])
    if track_count > 2:
        track_rows[2, :, contract.AGENT_POSITION] = np.array([7.0, 0.0])
        track_valid[2, contract.CURRENT_STEP_INDEX] = False

    scenario_array = {
        "map_rows": np.concatenate([signalled_lane, unsignalled_lane]),
        "feature_lengths": feature_lengths,
        "feature_ids": np.array([101, 102], dtype=np.int64),
        "lane_connections": np.zeros((0, contract.LANE_CONNECTION_WIDTH), dtype=np.int64),
        "lane_neighbour_ids": np.zeros((0, contract.LANE_NEIGHBOUR_ID_WIDTH), dtype=np.int64),
        "stop_sign_controlled_lanes": np.zeros((0, contract.STOP_SIGN_LANE_WIDTH), dtype=np.int64),
        "polyline_signal_histories": polyline_signal_histories,
        "track_rows": track_rows,
        "track_valid": track_valid,
        "scenario_id": np.array("two-lane"),
        "track_ids": np.arange(track_count),
        "is_designated_target": np.ones(track_count, dtype=bool),
        "is_object_of_interest": np.zeros(track_count, dtype=bool),
    }
    return loader.with_derived_arrays(scenario_array)


def test_each_agent_carries_the_signal_history_of_the_lane_it_is_assigned_to():
    signalled_lane_history = np.zeros(
        (contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES), dtype=np.float32
    )
    signalled_lane_history[:8, contract.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_STOP")] = 1.0
    signalled_lane_history[8:, contract.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_GO")] = 1.0

    three_agent_sample = loader.build_sample(
        two_lane_signal_scenario(3, signalled_lane_history), 0
    )
    two_agent_sample = loader.build_sample(two_lane_signal_scenario(2, signalled_lane_history), 0)

    assert np.array_equal(three_agent_sample["agent_signal_history"], signalled_lane_history)
    assert np.all(three_agent_sample["neighbour_signal_history"][0] == 0.0)
    assert np.all(three_agent_sample["neighbour_signal_history"][1] == 0.0)

    batch = loader.build_batch([three_agent_sample, two_agent_sample])
    assert batch["agent_signal_history"].shape == (
        2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
    )
    assert batch["neighbour_signal_history"].shape == (
        2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
    )
    assert np.all(batch["neighbour_signal_history"][1, 1] == 0.0)

    torch.manual_seed(47)
    encoder = model.SceneEncoder()
    with torch.no_grad():
        projected_agent = encoder.signal_projection(
            torch.from_numpy(batch["agent_signal_history"]).flatten(start_dim=-2)
        )
        projected_neighbours = encoder.signal_projection(
            torch.from_numpy(batch["neighbour_signal_history"]).flatten(start_dim=-2)
        )

    assert torch.any(projected_agent != 0.0)
    assert torch.all(projected_neighbours[1, 1] == 0.0)

    torch_batch = {name: torch.from_numpy(array) for name, array in batch.items()}
    silenced_batch = dict(torch_batch)
    silenced_batch["agent_signal_history"] = torch.zeros_like(
        torch_batch["agent_signal_history"]
    )
    with torch.no_grad():
        tokens, _ = encoder(torch_batch)
        silenced_tokens, _ = encoder(silenced_batch)

    assert not torch.allclose(tokens[0, 0], silenced_tokens[0, 0])


def test_the_heading_rides_one_bernstein_curve_of_its_own_control_pairs():
    torch.manual_seed(43)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    with torch.no_grad():
        decoder.trajectory_head[-1].weight.mul_(20.0)
    tokens = torch.randn(2, 9, model.HIDDEN_DIM)
    token_present = torch.ones(2, 9, dtype=torch.bool)

    with torch.no_grad():
        _, heading_cosine_sine, _, _, _ = decoder(
            tokens, token_present, torch.tensor([30.0, 70.0]), torch.zeros(2, dtype=torch.long)
        )

    time_fractions = (
        torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float32) / contract.FUTURE_STEPS
    )
    degree = model.TRAJECTORY_CONTROL_POINTS
    independent_basis = torch.stack(
        [
            math.comb(degree, index)
            * time_fractions ** index
            * (1.0 - time_fractions) ** (degree - index)
            for index in range(1, degree + 1)
        ],
        dim=1,
    )
    assert heading_cosine_sine.shape == (2, model.QUERY_COUNT, contract.FUTURE_STEPS, 2)

    flattened = heading_cosine_sine.permute(2, 0, 1, 3).reshape(contract.FUTURE_STEPS, -1)
    fitted_control_pairs = torch.linalg.lstsq(independent_basis, flattened).solution
    reconstructed = independent_basis @ fitted_control_pairs
    assert flattened.abs().max() > 0.1
    assert (reconstructed - flattened).norm() < 1e-4 * flattened.norm()

    control_polygon = torch.cat(
        [
            torch.zeros(2, model.QUERY_COUNT, 1, 2),
            fitted_control_pairs.view(degree, 2, model.QUERY_COUNT, 2).permute(1, 2, 0, 3),
        ],
        dim=2,
    )
    walked_heading = torch.cat(
        [torch.zeros(2, model.QUERY_COUNT, 1, 2), heading_cosine_sine], dim=2
    )
    assert (
        total_turning_degrees(walked_heading) <= total_turning_degrees(control_polygon) + 1e-2
    ).all()

    smooth_control_angles = torch.tensor([0.05, 0.20, 0.30])
    smooth_curve = independent_basis @ torch.stack(
        [smooth_control_angles.cos(), smooth_control_angles.sin()], dim=-1
    )
    step_angle_changes = torch.atan2(smooth_curve[:, 1], smooth_curve[:, 0]).diff().abs()
    assert step_angle_changes.max() <= smooth_control_angles.max() - smooth_control_angles.min()


def test_the_step_uncertainty_rides_one_bernstein_curve_and_never_leaves_the_clamped_range():
    torch.manual_seed(59)
    decoder = model.ModeDecoder(model.unit_anchor_offsets_per_type()).eval()
    tokens = torch.randn(2, 7, model.HIDDEN_DIM)
    token_present = torch.ones(2, 7, dtype=torch.bool)
    log_standard_deviation_start = model.POSITION_CONTROL_VALUES + model.HEADING_CONTROL_VALUES
    moderate_control_pairs = torch.tensor([[-1.0, 0.5], [0.4, -0.2], [1.2, 0.9]])

    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        decoder.trajectory_head[-1].bias.zero_()
        decoder.trajectory_head[-1].bias[log_standard_deviation_start:] = (
            moderate_control_pairs.flatten()
        )
        _, _, moderate_log_standard_deviation, _, _ = decoder(
            tokens, token_present, torch.tensor([30.0, 70.0]), torch.zeros(2, dtype=torch.long)
        )

    time_fractions = (
        torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float32) / contract.FUTURE_STEPS
    )
    degree = model.TRAJECTORY_CONTROL_POINTS
    independent_basis = torch.stack(
        [
            math.comb(degree, index)
            * time_fractions ** index
            * (1.0 - time_fractions) ** (degree - index)
            for index in range(1, degree + 1)
        ],
        dim=1,
    )

    assert moderate_log_standard_deviation.shape == (
        2, model.QUERY_COUNT, contract.FUTURE_STEPS, 2
    )
    assert torch.allclose(
        moderate_log_standard_deviation, independent_basis @ moderate_control_pairs, atol=1e-5
    )

    with torch.no_grad():
        decoder.trajectory_head[-1].bias[log_standard_deviation_start:] = torch.tensor(
            [-1e3, 1e3, -1e3, 1e3, -1e3, 1e3]
        )
        _, _, extreme_log_standard_deviation, _, _ = decoder(
            tokens, token_present, torch.tensor([30.0, 70.0]), torch.zeros(2, dtype=torch.long)
        )
    extreme_standard_deviation = extreme_log_standard_deviation.exp()
    floor_metres = math.exp(model.MINIMUM_LOG_STANDARD_DEVIATION)
    ceiling_metres = math.exp(model.MAXIMUM_LOG_STANDARD_DEVIATION)

    assert (extreme_standard_deviation >= floor_metres).all()
    assert (extreme_standard_deviation <= ceiling_metres).all()
    assert float(extreme_standard_deviation.min()) == pytest.approx(floor_metres, rel=1e-6)
    assert float(extreme_standard_deviation.max()) == pytest.approx(ceiling_metres, rel=1e-6)
    assert floor_metres == pytest.approx(0.2, abs=1e-3)


def test_the_regression_term_is_the_gaussian_negative_log_likelihood_at_the_stated_uncertainty():
    torch.manual_seed(53)
    sample_count = 4
    future_positions = torch.randn(sample_count, contract.FUTURE_STEPS, 2)
    future_headings = torch.nn.functional.normalize(
        torch.randn(sample_count, contract.FUTURE_STEPS, 2), dim=-1
    )
    future_mask = torch.ones(sample_count, contract.FUTURE_STEPS, dtype=torch.bool)
    heading_cosine_sine = future_headings.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
    confidence_logits = torch.zeros(sample_count, model.QUERY_COUNT)
    unit_anchors = model.unit_anchor_offsets().expand(sample_count, -1, -1)
    reachable_distance_metres = torch.full((sample_count,), 40.0)
    displacement = torch.tensor([0.3, -0.4])

    def regression_at(log_standard_deviation, error):
        trajectories = (
            future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1) + error
        )
        _, regression, _, _ = loss.prediction_loss(
            trajectories, heading_cosine_sine,
            torch.full_like(trajectories, log_standard_deviation), confidence_logits,
            future_positions, future_headings, future_mask,
            unit_anchors, reachable_distance_metres, 1.0,
        )
        return float(regression)

    floor = model.MINIMUM_LOG_STANDARD_DEVIATION
    floor_standard_deviation = math.exp(floor)
    squared_distance_form = (
        2.0 * floor
        + 2.0 * loss.HALF_LOG_TWO_PI
        + 0.5 * float((displacement ** 2).sum()) / floor_standard_deviation ** 2
    )

    assert regression_at(floor, displacement) == pytest.approx(squared_distance_form, rel=1e-5)
    for log_standard_deviation in (floor, 0.0, model.MAXIMUM_LOG_STANDARD_DEVIATION):
        assert regression_at(log_standard_deviation, torch.zeros(2)) < regression_at(
            log_standard_deviation, displacement
        )


def test_every_feature_column_the_contract_declares_has_a_sensitivity_perturbation():
    measure_sensitivity.assert_selectors_cover_every_column(
        measure_sensitivity.AGENT_FEATURE_SELECTORS, contract.AGENT_FEATURE_DIM, "agent"
    )
    measure_sensitivity.assert_selectors_cover_every_column(
        measure_sensitivity.MAP_FEATURE_SELECTORS, contract.MAP_FEATURE_DIM, "map"
    )
    measure_sensitivity.assert_selectors_cover_every_column(
        measure_sensitivity.LANE_CONTEXT_SELECTORS, contract.LANE_CONTEXT_DIM, "lane context"
    )
