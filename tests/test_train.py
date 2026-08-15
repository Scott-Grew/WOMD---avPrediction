import torch

import train
from womd import contract, loss
from womd.model import MotionPredictor, unit_anchor_offsets


def build_synthetic_map_rows(dot_count):
    map_rows = torch.randn(dot_count, contract.MAP_FEATURE_DIM)
    map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:] = torch.randint(
        0, contract.NUM_BOUNDARY_CROSSING_CODES, (dot_count, 2)
    ).float()
    return map_rows


def test_cosine_schedule_starts_at_the_learning_rate_and_decays_to_zero_over_the_run():
    total_steps = 200
    rates = [train.cosine_learning_rate(step, total_steps) for step in range(total_steps + 1)]

    assert rates[0] == train.LEARNING_RATE
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))
    assert rates[-2] < 1e-3 * train.LEARNING_RATE
    assert rates[-1] < 1e-12


def test_resuming_retrains_the_interrupted_epoch_and_never_skips_a_completed_one():
    predictor = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(predictor.parameters())
    interrupted = train.checkpoint_state(predictor, optimizer, 0, 2, 137)
    finished = train.checkpoint_state(predictor, optimizer, 0, 3, None)

    assert list(train.epochs_left_to_train(interrupted["completed_epochs"], 5)) == [2, 3, 4]
    assert list(train.epochs_left_to_train(finished["completed_epochs"], 5)) == [3, 4]
    assert list(train.epochs_left_to_train(0, 1)) == [0]
    assert list(
        train.epochs_left_to_train(
            train.checkpoint_state(predictor, optimizer, 0, 0, 137)["completed_epochs"], 1
        )
    ) == [0]
    assert list(
        train.epochs_left_to_train(
            train.checkpoint_state(predictor, optimizer, 0, 1, None)["completed_epochs"], 1
        )
    ) == []


def test_one_training_step_runs_forward_loss_backward_and_optimizer_step():
    torch.manual_seed(0)
    predictor = MotionPredictor(unit_anchor_offsets())
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    batch = {
        "agent_history": torch.randn(2, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "agent_history_mask": torch.ones(2, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_history": torch.randn(2, 3, contract.HISTORY_STEPS, contract.AGENT_FEATURE_DIM),
        "neighbour_history_mask": torch.ones(2, 3, contract.HISTORY_STEPS, dtype=torch.bool),
        "neighbour_future_positions": torch.randn(2, 3, contract.FUTURE_STEPS, 2),
        "neighbour_future_mask": torch.ones(2, 3, contract.FUTURE_STEPS, dtype=torch.bool),
        "map_rows": build_synthetic_map_rows(20),
        "map_dot_polyline_slot": torch.arange(20) // 5,
        "map_chunk_signal_history": torch.zeros(
            2, 4, contract.HISTORY_STEPS, contract.NUM_TRAFFIC_SIGNAL_STATES
        ),
        "map_chunk_lane_context": torch.randn(2, 4, contract.LANE_CONTEXT_DIM),
        "max_polylines_in_batch": torch.tensor(4),
        "future_positions": torch.randn(2, contract.FUTURE_STEPS, 2),
        "future_headings": torch.nn.functional.normalize(torch.randn(2, contract.FUTURE_STEPS, 2), dim=-1),
        "future_mask": torch.ones(2, contract.FUTURE_STEPS, dtype=torch.bool),
    }

    (
        trajectories, heading_cosine_sine, confidence_logits, reachable_distance_metres,
        neighbour_future_positions,
    ) = predictor.predict_with_heading(batch)
    total, _, _, _ = loss.prediction_loss(
        trajectories, heading_cosine_sine, confidence_logits,
        batch["future_positions"], batch["future_headings"], batch["future_mask"],
        predictor.unit_anchors, reachable_distance_metres, 1.0,
    )
    total = total + loss.neighbour_future_loss(
        neighbour_future_positions,
        batch["neighbour_future_positions"],
        batch["neighbour_future_mask"],
    )
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    assert torch.isfinite(total)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0.0
        for parameter in predictor.parameters()
    )
