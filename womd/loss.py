# > Training loss: best-of-64 winner-take-all, in metres
# Regression IS minADE's definition - mean Euclidean distance over valid steps, best mode only -
# so training optimises the exact quantity the leaderboard grades. Zero free parameters in the
# distance; the one knob is the classification weight (§14 sweep, provisional 1.0).
# The heading term is auxiliary and scores the SAME mode the position term won with - letting it
# pick its own winner would let the two terms pull toward different futures. It is the Euclidean
# distance between the predicted heading cosine/sine pair and the logged one: the same form as the
# position term so the two magnitudes are comparable, monotone in absolute angle error, and with no
# angle anywhere there is no wraparound to handle and no free parameter to choose.
# That monotonicity holds only for a prediction ON the unit circle, and the head's cosine/sine pair
# is free to be any 2-vector, so the loss normalises it first. A shrunken pair beats a committed one
# otherwise - the exact zero scores 1.0 against every logged heading while a unit vector a right
# angle out scores 1.414 - and the un-normalised term's minimiser is the conditional MEAN of the
# logged unit vectors rather than a unit vector at the right angle. The target is a unit pair by
# construction, so normalising removes a degree of freedom the head never had rather than adding a
# parameter, and flooring the divisor at the dtype's own eps keeps the exactly-zero pair finite
# without anybody choosing a number. The head's raw pair is left untouched: train.py's median
# heading norm is the diagnostic that watches it, and normalising upstream would pin that to 1.0.
# A sample with no valid future step is dropped by MULTIPLYING by the scoreable mask, never by
# indexing with it: indexing has to know how many rows survive, which reads a device tensor on the
# host and stalls the launch queue in front of backward. Every masked quantity is finite before the
# multiply - the mode means divide by a clamped count - so a dropped sample contributes an exact
# zero, and clamping the scoreable count the same way leaves a batch with nothing to score
# returning the same zeros the early return used to.

import torch

MODE_CLASSIFICATION_WEIGHT = 1.0


def nearest_drivable_dot_index(query_positions, drivable_positions, drivable_mask):
    separations = torch.cdist(query_positions, drivable_positions)
    separations.masked_fill_(~drivable_mask.unsqueeze(1), float("inf"))
    return separations.argmin(dim=-1)


def gather_drivable_dots(drivable_positions, dot_indices):
    return drivable_positions.gather(1, dot_indices.unsqueeze(-1).expand(-1, -1, 2))


def offroad_excess_beyond_logged(
    trajectories, future_positions, future_mask, drivable_positions, drivable_mask,
):
    batch_size, mode_count = trajectories.shape[:2]
    if drivable_positions.shape[1] == 0:
        return trajectories.new_zeros(batch_size)

    validity = future_mask.to(trajectories.dtype)
    valid_step_counts = validity.sum(dim=-1).clamp_min(1.0)
    with torch.no_grad():
        logged_dots = gather_drivable_dots(
            drivable_positions,
            nearest_drivable_dot_index(future_positions, drivable_positions, drivable_mask),
        )
        logged_distances = (future_positions - logged_dots).norm(dim=-1)

    mode_mean_excess = []
    for mode_index in range(mode_count):
        mode_positions = trajectories[:, mode_index]
        with torch.no_grad():
            dot_indices = nearest_drivable_dot_index(
                mode_positions, drivable_positions, drivable_mask
            )
        nearest_dots = gather_drivable_dots(drivable_positions, dot_indices)
        excess = (
            (mode_positions - nearest_dots).norm(dim=-1) - logged_distances
        ).relu()
        mode_mean_excess.append((excess * validity).sum(dim=-1) / valid_step_counts)

    sample_has_drivable_dot = drivable_mask.any(dim=-1).to(trajectories.dtype)
    return torch.stack(mode_mean_excess, dim=1).mean(dim=1) * sample_has_drivable_dot


def prediction_loss(
    trajectories, heading_cosine_sine, confidence_logits,
    future_positions, future_headings, future_mask,
    drivable_positions, drivable_mask, heading_loss_weight, offroad_loss_weight,
):
    step_distances = (trajectories - future_positions.unsqueeze(1)).norm(dim=-1)
    unit_heading_cosine_sine = heading_cosine_sine / heading_cosine_sine.norm(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(heading_cosine_sine.dtype).eps)
    step_heading_distances = (unit_heading_cosine_sine - future_headings.unsqueeze(1)).norm(dim=-1)
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
    offroad = (
        offroad_excess_beyond_logged(
            trajectories, future_positions, future_mask, drivable_positions, drivable_mask
        ) * scoreable
    ).sum() / scoreable_count
    total = (
        regression
        + MODE_CLASSIFICATION_WEIGHT * classification
        + heading_loss_weight * heading
        + offroad_loss_weight * offroad
    )
    return total, regression, classification, heading, offroad
