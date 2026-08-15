# > Training loss: best-of-64 winner-take-all, in metres
# Regression IS minADE's definition - mean Euclidean distance over valid steps, best mode only -
# so training optimises the exact quantity the leaderboard grades. Zero free parameters in the
# distance; the one knob is the classification weight (§14 sweep, provisional 1.0).
# The heading term is auxiliary and scores the SAME mode the position term won with - letting it
# pick its own winner would let the two terms pull toward different futures. It is the Euclidean
# distance between the predicted heading cosine/sine pair and the logged one: the same form as the
# position term so the two magnitudes are comparable, monotone in absolute angle error, and with no
# angle anywhere there is no wraparound to handle and no free parameter to choose.
# A sample with no valid future step is dropped by MULTIPLYING by the scoreable mask, never by
# indexing with it: indexing has to know how many rows survive, which reads a device tensor on the
# host and stalls the launch queue in front of backward. Every masked quantity is finite before the
# multiply - the mode means divide by a clamped count - so a dropped sample contributes an exact
# zero, and clamping the scoreable count the same way leaves a batch with nothing to score
# returning the same zeros the early return used to.

import torch

MODE_CLASSIFICATION_WEIGHT = 1.0


def prediction_loss(
    trajectories, heading_cosine_sine, confidence_logits,
    future_positions, future_headings, future_mask, heading_loss_weight,
):
    step_distances = (trajectories - future_positions.unsqueeze(1)).norm(dim=-1)
    step_heading_distances = (heading_cosine_sine - future_headings.unsqueeze(1)).norm(dim=-1)
    validity = future_mask.unsqueeze(1).to(step_distances.dtype)
    valid_step_counts = validity.sum(dim=-1)
    mode_mean_distance = (step_distances * validity).sum(dim=-1) / valid_step_counts.clamp_min(1.0)
    mode_mean_heading_distance = (
        (step_heading_distances * validity).sum(dim=-1) / valid_step_counts.clamp_min(1.0)
    )

    scoreable = (valid_step_counts[:, 0] > 0).to(step_distances.dtype)
    scoreable_count = scoreable.sum().clamp_min(1.0)

    best_mode = mode_mean_distance.argmin(dim=1)
    regression = (
        mode_mean_distance.gather(1, best_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    heading = (
        mode_mean_heading_distance.gather(1, best_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    classification = (
        torch.nn.functional.cross_entropy(confidence_logits, best_mode, reduction="none") * scoreable
    ).sum() / scoreable_count
    total = (
        regression
        + MODE_CLASSIFICATION_WEIGHT * classification
        + heading_loss_weight * heading
    )
    return total, regression, classification, heading
