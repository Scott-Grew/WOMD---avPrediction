# > Null baselines: the future the logged state already implies, with nothing learned
# Both take the batch MotionPredictor.forward takes and return what it returns - trajectories in
# metres in the predicted agent's own frame, plus confidence logits - so metrics.MetricAccumulator
# scores baseline and model through one path (V4). Neither has a free parameter: constant velocity
# assumes the logged velocity persists, CTRV assumes the logged yaw rate persists with it, and
# there is nothing in either to tune toward a flattering number.

import math

import torch

from womd import contract

# WOMD asks for 8 s of future and contract.FUTURE_STEPS covers exactly that, so a step is 0.1 s -
# the dataset's 10 Hz logging rate, divided out of the contract rather than typed in.
FUTURE_HORIZON_SECONDS = 8.0
TIMESTEP_SECONDS = FUTURE_HORIZON_SECONDS / contract.FUTURE_STEPS


def future_elapsed_seconds(device, dtype):
    return TIMESTEP_SECONDS * torch.arange(1, contract.FUTURE_STEPS + 1, device=device, dtype=dtype)


# The "now" row is real for every sample by construction: loader.eligible_track_indices only makes
# a sample out of a track that is valid at contract.CURRENT_STEP_INDEX. Heading comes back out of
# the stored cosine/sine pair the way loader.sample_frame recovers it.
def current_state(batch):
    now_row = batch["agent_history"][:, contract.CURRENT_STEP_INDEX]
    heading = torch.atan2(now_row[:, contract.AGENT_HEADING_SINE], now_row[:, contract.AGENT_HEADING_COSINE])
    return now_row[:, contract.AGENT_POSITION], heading, now_row[:, contract.AGENT_VELOCITY]


# A null has exactly one hypothesis, so it emits one mode rather than six copies of one. The
# min-over-modes metric reads the same number either way, but one mode says structurally that the
# six-mode budget went unspent, and six copies could later drift into six near-duplicates that
# quietly lower the minimum. model.prune_modes_batched, which MetricAccumulator runs first, fans
# the single mode out to contract.NUM_PREDICTED_MODES identical slots.
def as_single_mode(trajectories):
    confidence_logits = torch.zeros(
        trajectories.shape[0], 1, device=trajectories.device, dtype=trajectories.dtype
    )
    return trajectories.unsqueeze(1), confidence_logits


# > Null 1: the velocity logged at "now" persists for the whole 8 s, so the agent runs a straight
# line off its current position. No other history row is read - that is the whole assumption.
def constant_velocity(batch):
    position, _, velocity = current_state(batch)
    elapsed_seconds = future_elapsed_seconds(position.device, position.dtype)
    return as_single_mode(position[:, None, :] + velocity[:, None, :] * elapsed_seconds[None, :, None])


# Total heading change from the earliest valid history step to "now", over the time between them:
# all the evidence the sample carries and no window left to choose. The change comes out of the
# stored cosine/sine pairs through the angle-difference identities, so it arrives already wrapped
# into (-pi, pi] without an angle round trip. An agent whose only valid step is "now" compares that
# row against itself, leaving the change exactly zero - the no-turning answer, which is the one the
# absence of a second observation supports. Clamping the span to a single step only divides that
# zero by something finite.
def observed_yaw_rate(batch):
    agent_history = batch["agent_history"]
    step_indices = torch.arange(contract.HISTORY_STEPS, device=agent_history.device)
    valid_step_indices = torch.where(
        batch["agent_history_mask"], step_indices, torch.full_like(step_indices, contract.HISTORY_STEPS)
    )
    earliest_valid_step = valid_step_indices.min(dim=1).values
    earliest_row = agent_history.gather(
        1, earliest_valid_step[:, None, None].expand(-1, 1, contract.AGENT_FEATURE_DIM)
    ).squeeze(1)

    now_row = agent_history[:, contract.CURRENT_STEP_INDEX]
    change_sine = (
        now_row[:, contract.AGENT_HEADING_SINE] * earliest_row[:, contract.AGENT_HEADING_COSINE]
        - now_row[:, contract.AGENT_HEADING_COSINE] * earliest_row[:, contract.AGENT_HEADING_SINE]
    )
    change_cosine = (
        now_row[:, contract.AGENT_HEADING_COSINE] * earliest_row[:, contract.AGENT_HEADING_COSINE]
        + now_row[:, contract.AGENT_HEADING_SINE] * earliest_row[:, contract.AGENT_HEADING_SINE]
    )
    observed_seconds = (contract.CURRENT_STEP_INDEX - earliest_valid_step) * TIMESTEP_SECONDS
    return torch.atan2(change_sine, change_cosine) / observed_seconds.clamp(min=TIMESTEP_SECONDS)


# > Null 2: the observed yaw rate persists as well, so the agent traces a constant-radius arc at
# constant speed along its heading. Speed is the logged velocity projected onto the heading, which
# is CTRV's own state - it keeps a reversing agent going backwards and drops sideslip the model has
# no state for. Integrating in chord form rather than the textbook v/yaw_rate form removes the
# straight-line singularity outright: the arc of length speed * t has a chord of that length times
# sin(half_turn)/half_turn, laid along the heading rotated by half the turn, and torch.sinc is
# exactly 1 at zero turn, so a non-turning agent falls out as constant velocity with no threshold
# to pick. A stationary agent has zero speed, so every step of the arc is its current position.
def constant_turn_rate_and_velocity(batch):
    position, heading, velocity = current_state(batch)
    heading_direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    forward_speed = (velocity * heading_direction).sum(dim=-1)

    elapsed_seconds = future_elapsed_seconds(position.device, position.dtype)
    turn_angle = observed_yaw_rate(batch)[:, None] * elapsed_seconds[None, :]
    chord_to_arc_ratio = torch.sinc(turn_angle / (2.0 * math.pi))
    chord_length = forward_speed[:, None] * elapsed_seconds[None, :] * chord_to_arc_ratio
    chord_direction = heading[:, None] + 0.5 * turn_angle

    displacement = torch.stack(
        [chord_length * torch.cos(chord_direction), chord_length * torch.sin(chord_direction)], dim=-1
    )
    return as_single_mode(position[:, None, :] + displacement)
