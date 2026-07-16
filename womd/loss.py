import torch
import torch.nn.functional as functional

MODE_CLASSIFICATION_WEIGHT = 1.0


def per_mode_displacement_error(trajectories, future_positions, future_mask):
    distances = torch.linalg.vector_norm(
        trajectories - future_positions.unsqueeze(1), dim=-1
    )
    weights = future_mask.unsqueeze(1).to(distances.dtype)
    valid_step_count = weights.sum(dim=-1).clamp(min=1.0)
    return (distances * weights).sum(dim=-1) / valid_step_count


def best_of_many_loss(
    trajectories,
    mode_logits,
    future_positions,
    future_mask,
    classification_weight=MODE_CLASSIFICATION_WEIGHT,
    trained_mode_count=1,
):
    displacement = per_mode_displacement_error(trajectories, future_positions, future_mask)
    best_mode_index = displacement.argmin(dim=1)

    trained_mode_count = max(1, min(trained_mode_count, trajectories.shape[1]))
    trained_mode_indices = displacement.topk(
        trained_mode_count, dim=1, largest=False
    ).indices

    batch_index = torch.arange(trajectories.shape[0], device=trajectories.device)
    trained_trajectories = trajectories[batch_index.unsqueeze(1), trained_mode_indices]

    elementwise = functional.smooth_l1_loss(
        trained_trajectories,
        future_positions.unsqueeze(1).expand_as(trained_trajectories),
        reduction="none",
    ).sum(dim=-1)
    weights = future_mask.unsqueeze(1).to(elementwise.dtype)
    regression = (elementwise * weights).sum() / (
        weights.sum() * trained_mode_count
    ).clamp(min=1.0)

    classification = functional.cross_entropy(mode_logits, best_mode_index)
    total = regression + classification_weight * classification
    return total, {
        "regression": regression.detach(),
        "classification": classification.detach(),
        "best_mode_displacement": displacement.gather(
            1, best_mode_index.unsqueeze(1)
        ).mean().detach(),
    }
