# > Training loss: one scored mode per anchor territory, in metres
# The scored mode is the one whose ANCHOR endpoint sits nearest the logged endpoint, not the one
# whose own prediction landed closest, so no mode can be starved of gradient by never happening to
# win. The anchors are learned, so the assignment is read from their CURRENT value but detached: the
# choice stays a hard non-differentiable one and an anchor moves through the regression term of the
# mode it was assigned to, never by making itself easier to select. The distance itself is unchanged and still has zero free parameters -
# mean Euclidean distance over valid steps, one mode, which is minADE's definition - so training
# still optimises the quantity the leaderboard grades; the one knob is the classification weight
# (§14 sweep, provisional 1.0).
# The logged endpoint is the last VALID future position, never future_positions[:, -1]: a track can
# stop early, and reading the padded tail would assign every early-ending sample to whichever anchor
# sits nearest the origin.
# The heading term is auxiliary and scores the SAME mode the position term does - letting it pick
# its own winner would let the two terms pull toward different futures. It is the Euclidean
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


def anchor_assigned_mode(unit_anchors, reachable_distance_metres, future_positions, future_mask):
    validity = future_mask.to(future_positions.dtype)
    step_positions = torch.arange(
        future_mask.shape[-1], device=future_mask.device, dtype=future_positions.dtype
    )
    last_valid_step = (validity * step_positions).argmax(dim=-1)
    logged_endpoint = future_positions.gather(
        1, last_valid_step[:, None, None].expand(-1, -1, 2)
    ).squeeze(1)
    anchor_endpoints = unit_anchors.detach().unsqueeze(0) * reachable_distance_metres[:, None, None]
    return (anchor_endpoints - logged_endpoint.unsqueeze(1)).norm(dim=-1).argmin(dim=1)


def prediction_loss(
    trajectories, heading_cosine_sine, confidence_logits,
    future_positions, future_headings, future_mask,
    unit_anchors, reachable_distance_metres, heading_loss_weight,
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

    assigned_mode = anchor_assigned_mode(
        unit_anchors, reachable_distance_metres, future_positions, future_mask
    )
    regression = (
        mode_mean_distance.gather(1, assigned_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    heading = (
        mode_mean_heading_distance.gather(1, assigned_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    classification = (
        torch.nn.functional.cross_entropy(confidence_logits, assigned_mode, reduction="none")
        * scoreable
    ).sum() / scoreable_count
    total = (
        regression
        + MODE_CLASSIFICATION_WEIGHT * classification
        + heading_loss_weight * heading
    )
    return total, regression, classification, heading
