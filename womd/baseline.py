import torch

from womd import contract

STRAIGHT_YAW_RATE_LIMIT = 1e-3


def constant_velocity_predictions(batch, future_steps=contract.FUTURE_STEPS):
    history = batch["agent_history"]
    current_velocity = history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY]
    elapsed = torch.arange(
        1, future_steps + 1, device=history.device, dtype=history.dtype
    ) * 0.1
    displacement = current_velocity.unsqueeze(1) * elapsed.unsqueeze(-1)
    return displacement.unsqueeze(1)


def constant_turn_rate_predictions(
    batch,
    future_steps=contract.FUTURE_STEPS,
    straight_yaw_rate_limit=STRAIGHT_YAW_RATE_LIMIT,
):
    history = batch["agent_history"]
    history_mask = batch["agent_history_mask"]
    current_index = contract.CURRENT_STEP_INDEX

    speed = torch.linalg.vector_norm(
        history[:, current_index, contract.AGENT_VELOCITY], dim=-1
    )
    headings = torch.atan2(
        history[:, :, contract.AGENT_HEADING_SINE],
        history[:, :, contract.AGENT_HEADING_COSINE],
    )
    raw_heading_change = headings[:, current_index] - headings[:, 0]
    heading_change = (raw_heading_change + torch.pi) % (2.0 * torch.pi) - torch.pi
    both_steps_valid = history_mask[:, 0] & history_mask[:, current_index]
    yaw_rate = torch.where(
        both_steps_valid, heading_change / (0.1 * current_index), torch.zeros_like(speed)
    )

    elapsed = torch.arange(
        1, future_steps + 1, device=history.device, dtype=history.dtype
    ) * 0.1
    yaw_rate = yaw_rate.unsqueeze(1)
    speed = speed.unsqueeze(1)
    elapsed = elapsed.unsqueeze(0)

    turning = yaw_rate.abs() > straight_yaw_rate_limit
    divisible_yaw_rate = torch.where(turning, yaw_rate, torch.ones_like(yaw_rate))
    turn_radius = speed / divisible_yaw_rate
    swept_angle = divisible_yaw_rate * elapsed

    forward = torch.where(turning, turn_radius * torch.sin(swept_angle), speed * elapsed)
    lateral = torch.where(
        turning, turn_radius * (1.0 - torch.cos(swept_angle)), torch.zeros_like(forward)
    )
    return torch.stack([forward, lateral], dim=-1).unsqueeze(1)
