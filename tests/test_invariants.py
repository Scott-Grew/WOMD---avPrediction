import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch

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
    predictor = model.MotionPredictor().eval()
    batch = {
        "agent_history": torch.randn(2, 11, 13),
        "agent_history_mask": torch.ones(2, 11, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 6, 11, 13),
        "neighbour_history_mask": torch.ones(2, 6, 11, dtype=torch.bool),
        "map_rows": torch.randn(80, contract.MAP_FEATURE_DIM),
        "map_dot_polyline_slot": torch.arange(80) // 8,
        "map_chunk_signal_history": torch.zeros(
            2, 5, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "max_polylines_in_batch": torch.tensor(5),
    }
    batch["neighbour_history_mask"][:, 5] = False
    batch["map_chunk_signal_history"][0, 1:3, :, 6] = 1.0

    neighbour_order = torch.randperm(6)
    map_order = torch.randperm(80)
    permuted = dict(batch)
    permuted["neighbour_history"] = batch["neighbour_history"][:, neighbour_order]
    permuted["neighbour_history_mask"] = batch["neighbour_history_mask"][:, neighbour_order]
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


def test_no_predicted_position_escapes_the_agent_own_reachable_distance():
    torch.manual_seed(13)
    predictor = model.MotionPredictor().eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.mul_(1000.0)
        predictor.mode_decoder.trajectory_head[-1].bias.mul_(1000.0)

    current_speeds = torch.tensor([0.0, 30.0])
    agent_history = torch.randn(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM)
    agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY] = torch.stack(
        [current_speeds, torch.zeros(2)], dim=-1
    )
    batch = {
        "agent_history": agent_history,
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 3, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(2, 3, contract.HISTORY_STEPS, dtype=torch.bool),
        "map_rows": torch.randn(40, contract.MAP_FEATURE_DIM),
        "map_dot_polyline_slot": torch.arange(40) // 10,
        "map_chunk_signal_history": torch.zeros(
            2, 2, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "max_polylines_in_batch": torch.tensor(2),
    }

    reachable = (
        current_speeds * contract.FUTURE_HORIZON_SECONDS
        + 0.5 * contract.MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED * contract.FUTURE_HORIZON_SECONDS ** 2
    )
    with torch.no_grad():
        trajectories, _ = predictor(batch)
    distances = trajectories.norm(dim=-1)

    float32_rounding_headroom = 1.0 + 8 * torch.finfo(distances.dtype).eps
    assert (distances <= reachable[:, None, None] * float32_rounding_headroom).all()
    assert (distances.amax(dim=(1, 2)) > 0.99 * reachable).all()


def test_one_training_step_runs_the_whole_path_over_staged_scenarios(tmp_path):
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:2]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")

    torch.manual_seed(0)
    predictor = model.MotionPredictor()
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    batch = next(iter(pipeline.batches(scenario_paths, 0, 2, 0, 0, True)))

    trajectories, heading_cosine_sine, confidence_logits = predictor.predict_with_heading(batch)
    components = loss.prediction_loss(
        trajectories, heading_cosine_sine, confidence_logits,
        batch["future_positions"], batch["future_headings"], batch["future_mask"], 1.0,
    )
    assert torch.stack(components).isfinite().all()
    components[0].backward()
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
