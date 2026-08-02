import torch

from womd import contract

def constant_velocity_predictions(batch, future_steps=contract.FUTURE_STEPS):
    history = batch["agent_history"]
    current_velocity = history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY]
    elapsed = torch.arange(
        1, future_steps + 1, device=history.device, dtype=history.dtype
    ) * 0.1
    displacement = current_velocity.unsqueeze(1) * elapsed.unsqueeze(-1)
    return displacement.unsqueeze(1)
