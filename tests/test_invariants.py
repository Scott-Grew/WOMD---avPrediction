import io

import numpy as np
import pytest

from waymo_open_dataset.protos import scenario_pb2
from womd import contract, frame_ops, loader, store, tfrecord


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
