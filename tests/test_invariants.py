import io

import numpy as np
import pytest
import torch

from womd import (
    baseline,
    contract,
    dataset,
    features,
    frame,
    kinematics,
    loss,
    metrics,
    model,
    report,
    synthetic,
    tfrecord,
)


@pytest.fixture(scope="module")
def synthetic_scenario():
    return synthetic.random_scenario(0, np.random.default_rng(7))


@pytest.fixture(scope="module")
def synthetic_batch(synthetic_scenario):
    samples = features.build_scenario_samples(synthetic_scenario)
    tensors = [
        {
            name: torch.from_numpy(np.ascontiguousarray(np.asarray(sample[name])))
            if np.asarray(sample[name]).dtype == np.bool_
            else torch.from_numpy(np.ascontiguousarray(np.asarray(sample[name], dtype=np.float32)))
            for name in contract.SAMPLE_ARRAY_SPEC
        }
        for sample in samples
    ]
    return dataset.samples_to_batch(tensors)


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


def test_v1_agent_frame_transform_inverts(synthetic_scenario):
    rng = np.random.default_rng(3)
    world_positions = rng.uniform(-120.0, 120.0, size=(64, 2))
    origin = np.array([13.5, -42.25])
    heading = 0.9137

    local = frame.positions_to_agent_frame(world_positions, origin, heading)
    recovered = frame.positions_to_world_frame(local, origin, heading)
    assert np.allclose(recovered, world_positions, atol=1e-9)


def test_v1_target_agent_sits_at_frame_origin(synthetic_scenario):
    sample = features.build_sample(synthetic_scenario, 0)
    current_row = sample["agent_history"][contract.CURRENT_STEP_INDEX]
    assert np.allclose(current_row[0:2], 0.0, atol=1e-4)
    assert np.isclose(current_row[2], 1.0, atol=1e-4)
    assert np.isclose(current_row[3], 0.0, atol=1e-4)


def test_v2_masked_future_steps_do_not_change_loss(synthetic_batch):
    torch.manual_seed(0)
    predictor = model.MotionPredictor()
    predictor.eval()
    with torch.no_grad():
        trajectories, mode_logits = predictor(synthetic_batch)

    future_positions = synthetic_batch["future_positions"].clone()
    future_mask = synthetic_batch["future_mask"].clone()
    future_mask[:, -20:] = False

    reference, _ = loss.best_of_many_loss(
        trajectories, mode_logits, future_positions, future_mask
    )

    perturbed = future_positions.clone()
    perturbed[:, -20:] += 1000.0
    perturbed_loss, _ = loss.best_of_many_loss(
        trajectories, mode_logits, perturbed, future_mask
    )
    assert torch.allclose(reference, perturbed_loss, atol=1e-5)


def test_v3_futures_map_back_to_logged_world_positions(synthetic_scenario):
    target_index = 0
    sample = features.build_sample(synthetic_scenario, target_index)
    world = frame.positions_to_world_frame(
        sample["future_positions"], sample["frame_origin"], sample["frame_heading"]
    )
    states = synthetic_scenario.tracks[target_index].states
    for future_index in range(contract.FUTURE_STEPS):
        if not sample["future_mask"][future_index]:
            continue
        logged = states[contract.CURRENT_STEP_INDEX + 1 + future_index]
        assert np.allclose(world[future_index], (logged.center_x, logged.center_y), atol=1e-3)


def test_v4_baseline_and_model_share_the_evaluation_path(synthetic_batch):
    torch.manual_seed(0)
    predictor = model.MotionPredictor()
    predictor.eval()
    with torch.no_grad():
        trajectories, _ = predictor(synthetic_batch)
    baseline_trajectories = baseline.constant_velocity_predictions(synthetic_batch)

    model_results = metrics.evaluate_predictions(
        trajectories, synthetic_batch["future_positions"], synthetic_batch["future_mask"]
    )
    baseline_results = metrics.evaluate_predictions(
        baseline_trajectories, synthetic_batch["future_positions"], synthetic_batch["future_mask"]
    )
    assert model_results.keys() == baseline_results.keys()
    assert all(np.isfinite(value) for value in model_results.values())
    assert all(np.isfinite(value) for value in baseline_results.values())


def test_model_forward_produces_finite_gradients(synthetic_batch):
    torch.manual_seed(0)
    predictor = model.MotionPredictor()
    trajectories, mode_logits = predictor(synthetic_batch)
    assert trajectories.shape[1:] == (contract.NUM_PREDICTED_MODES, contract.FUTURE_STEPS, 2)

    total, components = loss.best_of_many_loss(
        trajectories,
        mode_logits,
        synthetic_batch["future_positions"],
        synthetic_batch["future_mask"],
    )
    total.backward()

    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in components.values())
    gradients = [
        parameter.grad for parameter in predictor.parameters() if parameter.grad is not None
    ]
    assert len(gradients) > 0
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_empty_map_and_neighbours_stay_finite():
    torch.manual_seed(0)
    empty = {
        name: torch.zeros((2,) + shape, dtype=torch.float32)
        for name, shape in contract.SAMPLE_ARRAY_SPEC.items()
    }
    for mask_name in (
        "agent_history_mask",
        "neighbour_history_mask",
        "map_polylines_mask",
        "future_mask",
    ):
        empty[mask_name] = torch.zeros_like(empty[mask_name], dtype=torch.bool)
    empty["agent_history_mask"][:, contract.CURRENT_STEP_INDEX] = True
    empty["future_mask"][:, :] = True

    predictor = model.MotionPredictor()
    trajectories, mode_logits = predictor(empty)
    total, _ = loss.best_of_many_loss(
        trajectories, mode_logits, empty["future_positions"], empty["future_mask"]
    )
    total.backward()
    assert torch.isfinite(total)
    gradients = [
        parameter.grad for parameter in predictor.parameters() if parameter.grad is not None
    ]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_v5_every_mode_receives_gradient_when_all_modes_are_trained(synthetic_batch):
    torch.manual_seed(0)
    predictor = model.MotionPredictor()
    trajectories, mode_logits = predictor(synthetic_batch)
    trajectories.retain_grad()

    total, _ = loss.best_of_many_loss(
        trajectories,
        mode_logits,
        synthetic_batch["future_positions"],
        synthetic_batch["future_mask"],
        trained_mode_count=contract.NUM_PREDICTED_MODES,
    )
    total.backward()

    per_mode_gradient = trajectories.grad.abs().sum(dim=(0, 2, 3))
    assert (per_mode_gradient > 0).all()


def test_winner_take_all_leaves_losing_modes_without_gradient(synthetic_batch):
    torch.manual_seed(0)
    predictor = model.MotionPredictor()
    trajectories, mode_logits = predictor(synthetic_batch)
    trajectories.retain_grad()

    total, _ = loss.best_of_many_loss(
        trajectories,
        mode_logits,
        synthetic_batch["future_positions"],
        synthetic_batch["future_mask"],
        trained_mode_count=1,
    )
    total.backward()

    per_mode_gradient = trajectories.grad.abs().sum(dim=(0, 2, 3))
    assert (per_mode_gradient == 0).any()


def test_shard_local_sampler_visits_every_sample_exactly_once(tmp_path, synthetic_scenario):
    samples = features.build_scenario_samples(synthetic_scenario)
    for shard_index in range(3):
        dataset.write_shard(tmp_path / f"shard-{shard_index:06d}.npz", samples)

    loaded = dataset.ShardedSampleDataset(tmp_path)
    sampler = dataset.ShardLocalShuffleSampler(loaded, seed=11)
    sampler.set_epoch(0)
    first_epoch = list(iter(sampler))
    repeated_first_epoch = list(iter(sampler))
    sampler.set_epoch(1)
    second_epoch = list(iter(sampler))

    assert sorted(first_epoch) == list(range(len(loaded)))
    assert sorted(second_epoch) == list(range(len(loaded)))
    assert repeated_first_epoch == first_epoch
    assert second_epoch != first_epoch


def test_shard_round_trips_through_dataset(tmp_path, synthetic_scenario):
    samples = features.build_scenario_samples(synthetic_scenario)
    written = dataset.write_shard(tmp_path / "shard-000000.npz", samples)
    assert written == len(samples)

    loaded = dataset.ShardedSampleDataset(tmp_path)
    assert len(loaded) == len(samples)
    first = loaded[0]
    assert first["agent_history"].shape == contract.SAMPLE_ARRAY_SPEC["agent_history"]
    assert np.allclose(
        first["future_positions"].numpy(), samples[0]["future_positions"], atol=1e-6
    )


def test_minimum_average_displacement_respects_gaps_in_the_future_mask():
    steps = contract.FUTURE_STEPS
    future_positions = torch.zeros(1, steps, 2)
    future_positions[0, :, 0] = torch.arange(steps, dtype=torch.float32)
    future_mask = torch.zeros(1, steps, dtype=torch.bool)
    future_mask[0, ::2] = True

    trajectories = future_positions.unsqueeze(1).clone()
    trajectories[0, 0, ~future_mask[0], 0] += 1000.0

    error = metrics.minimum_average_displacement(
        trajectories, future_positions, future_mask, steps
    )
    assert torch.allclose(error, torch.zeros(1), atol=1e-6)


def test_per_sample_minimum_average_displacement_equals_a_constant_offset():
    offset = 2.5
    steps = contract.FUTURE_STEPS
    future_positions = torch.randn(7, steps, 2)
    future_mask = torch.ones(7, steps, dtype=torch.bool)
    trajectories = future_positions.unsqueeze(1).repeat(1, contract.NUM_PREDICTED_MODES, 1, 1)
    trajectories[..., 0] += offset

    per_sample = metrics.per_sample_metrics(trajectories, future_positions, future_mask)
    assert torch.allclose(
        per_sample[f"minADE@{steps / 10:g}s"], torch.full((7,), offset), atol=1e-5
    )


def test_report_slices_partition_every_sample(synthetic_batch):
    labels = report.slice_labels(synthetic_batch)
    sample_count = synthetic_batch["future_positions"].shape[0]
    for name, names in report.SLICE_NAMES.items():
        counts = [int((labels[name] == value).sum()) for value in range(len(names))]
        assert sum(counts) == sample_count


def test_tail_statistics_match_the_analytic_order_statistics():
    values = torch.arange(100, dtype=torch.float32)
    median, p90, p95, p99, maximum = report.tail_statistics(values)
    assert np.isclose(median, 49.5, atol=1e-4)
    assert np.isclose(p90, 89.1, atol=1e-4)
    assert np.isclose(p95, 94.05, atol=1e-4)
    assert np.isclose(p99, 98.01, atol=1e-4)
    assert np.isclose(maximum, 99.0, atol=1e-6)


def test_whole_batch_costs_at_least_one_sample(synthetic_batch):
    torch.manual_seed(0)
    predictor = model.MotionPredictor()
    measurements = report.measure_latency(predictor, synthetic_batch, repeats=3)
    assert measurements["batched_total_ms"] >= measurements["single_sample_ms"]


def test_kinematics_match_circular_arc_geometry():
    radius, speed = 100.0, 10.0
    angular_rate = speed / radius
    elapsed = torch.arange(contract.FUTURE_STEPS) * kinematics.STEP_SECONDS
    arc = torch.stack(
        [radius * torch.sin(angular_rate * elapsed), radius * (1 - torch.cos(angular_rate * elapsed))],
        dim=-1,
    ).unsqueeze(0).unsqueeze(0)

    quantities, moving = kinematics.motion_quantities(arc, window_steps=1)
    assert np.isclose(quantities["lateral"][moving].mean().item(), speed**2 / radius, atol=1e-3)
    assert np.isclose(quantities["yaw_rate"][moving].mean().item(), angular_rate, atol=1e-3)


def test_constant_velocity_predictions_are_never_implausible(synthetic_batch):
    rates = kinematics.violation_rates(baseline.constant_velocity_predictions(synthetic_batch))
    for name, values in rates.items():
        assert values.max().item() == 0.0, name
