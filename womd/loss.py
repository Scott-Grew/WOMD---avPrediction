# > Training loss: one scored mode per anchor territory
# The scored mode is the one whose ANCHOR endpoint sits nearest the logged endpoint, not the one
# whose own prediction landed closest, so no mode can be starved of gradient by never happening to
# win. The anchors are learned, so the assignment is read from their CURRENT value but detached: the
# choice stays a hard non-differentiable one and an anchor moves through the regression term of the
# mode it was assigned to, never by making itself easier to select. There is one anchor set per
# predicted object type and the decoder is what selects a sample's; the assignment is handed that
# same per-sample (batch, modes, 2) tensor rather than re-selecting from the table, so the mode the
# loss scores and the mode the decoder aimed cannot come from different sets. The regression term is the
# negative log-likelihood of the logged future under that mode's per-step axis-aligned Gaussians,
# averaged over the valid steps: a mode reports how sure it is and pays for being sure and wrong,
# which mean Euclidean distance cannot express. It is no longer in metres, so it is not the
# leaderboard's quantity and metrics.py is what still watches that.
# The heading term is auxiliary and scores the SAME mode the position term does - letting it pick
# its own winner would let the two terms pull toward different futures. It is the Euclidean
# distance between the predicted heading cosine/sine pair and the logged one: monotone in absolute
# angle error, and with no angle anywhere there is no wraparound to handle and no free parameter to
# choose. Its weight is measured against the regression term rather than assumed comparable to it.
# That monotonicity holds only for a prediction ON the unit circle, and the head's cosine/sine pair
# is free to be any 2-vector, so the loss normalises it first. A shrunken pair beats a committed one
# otherwise - the exact zero scores 1.0 against every logged heading while a unit vector a right
# angle out scores 1.414 - and the un-normalised term's minimiser is the conditional MEAN of the
# logged unit vectors rather than a unit vector at the right angle. The target is a unit pair by
# construction, so normalising removes a degree of freedom the head never had rather than adding a
# parameter, and flooring the divisor at the dtype's own eps keeps the exactly-zero pair finite
# without anybody choosing a number. The head's raw pair is left untouched: train.py's median
# heading norm is the diagnostic that watches it, and normalising upstream would pin that to 1.0.
# The speed term scores the SAME assigned mode again, on the speed of the trajectory the model
# actually EMITS: both sides are the distance between consecutive step positions over the timestep,
# the logged positions on one side and the emitted curve-plus-anchor positions on the other, so the
# two sides are the same construction and the anchor's motion cannot hide from the term.
# The logged endpoint is the last VALID future position, never future_positions[:, -1]: a track can
# stop early, and reading the padded tail would assign every early-ending sample to whichever anchor
# sits nearest the origin.
# A sample with no valid future step is dropped by MULTIPLYING by the scoreable mask, never by
# indexing with it: indexing has to know how many rows survive, which reads a device tensor on the
# host and stalls the launch queue in front of backward. Every masked quantity is finite before the
# multiply - the mode means divide by a clamped count - so a dropped sample contributes an exact
# zero, and clamping the scoreable count the same way leaves a batch with nothing to score
# returning the same zeros the early return used to.

import math

import torch

from womd import contract

HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def anchor_assigned_mode(selected_unit_anchors, future_positions, future_mask):
    validity = future_mask.to(future_positions.dtype)
    step_positions = torch.arange(
        future_mask.shape[-1], device=future_mask.device, dtype=future_positions.dtype
    )
    last_valid_step = (validity * step_positions).argmax(dim=-1)
    logged_endpoint = future_positions.gather(
        1, last_valid_step[:, None, None].expand(-1, -1, 2)
    ).squeeze(1)
    anchor_endpoints = selected_unit_anchors.detach()
    return (anchor_endpoints - logged_endpoint.unsqueeze(1)).norm(dim=-1).argmin(dim=1)


def logged_speed_per_step(future_positions, future_mask):
    now_position = torch.zeros_like(future_positions[:, :1])
    step_positions = torch.cat([now_position, future_positions], dim=1)
    step_distances = (step_positions[:, 1:] - step_positions[:, :-1]).norm(dim=-1)
    now_valid = torch.ones_like(future_mask[:, :1])
    step_validity = torch.cat([now_valid, future_mask], dim=1)
    valid_step_pair = step_validity[:, 1:] & step_validity[:, :-1]
    return step_distances / contract.TIMESTEP_SECONDS, valid_step_pair


def gaussian_negative_log_likelihood(
    trajectories, position_log_standard_deviation, future_positions
):
    standardised_error = (
        trajectories - future_positions.unsqueeze(1)
    ) / position_log_standard_deviation.exp()
    return (
        position_log_standard_deviation + 0.5 * standardised_error ** 2 + HALF_LOG_TWO_PI
    ).sum(dim=-1)


def prediction_loss(
    trajectories, heading_cosine_sine, position_log_standard_deviation, confidence_logits,
    predicted_speed,
    future_positions, future_headings, future_mask,
    selected_unit_anchors,
    heading_loss_weight, classification_loss_weight, speed_loss_weight,
):
    step_negative_log_likelihoods = gaussian_negative_log_likelihood(
        trajectories, position_log_standard_deviation, future_positions
    )
    unit_heading_cosine_sine = heading_cosine_sine / heading_cosine_sine.norm(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(heading_cosine_sine.dtype).eps)
    step_heading_distances = (unit_heading_cosine_sine - future_headings.unsqueeze(1)).norm(dim=-1)
    validity = future_mask.unsqueeze(1).to(step_negative_log_likelihoods.dtype)
    valid_step_counts = validity.sum(dim=-1)
    mode_mean_negative_log_likelihood = (
        (step_negative_log_likelihoods * validity).sum(dim=-1) / valid_step_counts.clamp_min(1.0)
    )
    mode_mean_heading_distance = (
        (step_heading_distances * validity).sum(dim=-1) / valid_step_counts.clamp_min(1.0)
    )

    scoreable = (valid_step_counts[:, 0] > 0).to(step_negative_log_likelihoods.dtype)
    scoreable_count = scoreable.sum().clamp_min(1.0)

    assigned_mode = anchor_assigned_mode(selected_unit_anchors, future_positions, future_mask)
    regression = (
        mode_mean_negative_log_likelihood.gather(1, assigned_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    heading = (
        mode_mean_heading_distance.gather(1, assigned_mode[:, None]).squeeze(1) * scoreable
    ).sum() / scoreable_count
    assigned_mode_one_hot = torch.nn.functional.one_hot(
        assigned_mode, confidence_logits.shape[1]
    ).to(confidence_logits.dtype)
    classification = (
        (
            torch.nn.functional.binary_cross_entropy_with_logits(
                confidence_logits, assigned_mode_one_hot, reduction="none"
            )
        ).sum(dim=1)
        * scoreable
    ).sum() / scoreable_count

    assigned_mode_speed = predicted_speed.gather(
        1, assigned_mode[:, None, None].expand(-1, -1, contract.FUTURE_STEPS)
    ).squeeze(1)
    logged_speed, valid_speed_step = logged_speed_per_step(future_positions, future_mask)
    valid_speed_step = valid_speed_step.to(assigned_mode_speed.dtype)
    speed = (
        (assigned_mode_speed - logged_speed).abs() * valid_speed_step
    ).sum() / valid_speed_step.sum().clamp_min(1.0)

    total = (
        regression
        + heading_loss_weight * heading
        + classification_loss_weight * classification
        + speed_loss_weight * speed
    )
    return total, regression, heading, classification, speed


def neighbour_future_loss(
    neighbour_future_positions, logged_positions, neighbour_future_mask, neighbour_readable
):
    step_distances = (neighbour_future_positions - logged_positions).norm(dim=-1)
    validity = (
        neighbour_future_mask.to(step_distances.dtype)
        * neighbour_readable.to(step_distances.dtype).unsqueeze(-1)
    )
    return (step_distances * validity).sum() / validity.sum().clamp_min(1.0)
