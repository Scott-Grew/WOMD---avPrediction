# > Training loss: best-of-64 winner-take-all, in metres
# Regression IS minADE's definition - mean Euclidean distance over valid steps, best mode only -
# so training optimises the exact quantity the leaderboard grades. Zero free parameters in the
# distance; the one knob is the classification weight (§14 sweep, provisional 1.0).

import torch

MODE_CLASSIFICATION_WEIGHT = 1.0


def prediction_loss(trajectories, confidence_logits, future_positions, future_mask):
    step_distances = (trajectories - future_positions.unsqueeze(1)).norm(dim=-1)
    validity = future_mask.unsqueeze(1).to(step_distances.dtype)
    valid_step_counts = validity.sum(dim=-1)
    mode_mean_distance = (step_distances * validity).sum(dim=-1) / valid_step_counts.clamp_min(1.0)

    scoreable = valid_step_counts[:, 0] > 0
    if not scoreable.any():
        zero = trajectories.sum() * 0.0
        return zero, zero, zero

    mode_mean_distance = mode_mean_distance[scoreable]
    best_mode = mode_mean_distance.argmin(dim=1)
    regression = mode_mean_distance.gather(1, best_mode[:, None]).mean()
    classification = torch.nn.functional.cross_entropy(confidence_logits[scoreable], best_mode)
    return regression + MODE_CLASSIFICATION_WEIGHT * classification, regression, classification
