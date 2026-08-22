# > The model: one agent's view of the scene in, six possible 8 s futures out
# Every encoder outputs HIDDEN_DIM-wide embeddings; attention runs at that width.

import hashlib

import torch
from torch import nn

from womd import contract

HIDDEN_DIM = 128


# One divisor per feature column, applied before each encoder's first Linear. Every distance-like
# column shares contract.DISTANCE_NORMALISER_METRES so geometry survives; a column with no
# measured scale - heading cosine/sine, direction arrows, one-hots, the SDC flag - keeps a divisor
# of 1 and passes through untouched. The signal history never reaches a divisor: it is one-hot and
# it enters at the agent, neighbour and chunk tokens, past every per-dot and per-step encoder.
def agent_feature_divisors():
    divisors = torch.ones(contract.AGENT_FEATURE_DIM)
    divisors[contract.AGENT_POSITION] = contract.DISTANCE_NORMALISER_METRES
    divisors[contract.AGENT_VELOCITY] = contract.VELOCITY_NORMALISER_METRES_PER_SECOND
    divisors[contract.AGENT_DIMENSIONS] = contract.DIMENSION_NORMALISER_METRES
    return divisors


def map_feature_divisors():
    divisors = torch.ones(contract.MAP_FEATURE_DIM)
    divisors[contract.MAP_POSITION] = contract.DISTANCE_NORMALISER_METRES
    divisors[contract.MAP_SPEED_LIMIT] = contract.SPEED_LIMIT_NORMALISER_MILES_PER_HOUR
    divisors[contract.MAP_STOP_POINT] = contract.DISTANCE_NORMALISER_METRES
    return divisors


# Flatten the 11 history rows into one ordered vector (position-in-vector IS time, §28 flatten
# ruling) plus the 11 validity flags, then MLP down to one embedding. Invalid rows are zeroed
# FIRST: the loader's re-framing turns stored zero-rows into non-zero garbage positions, so the
# mask is what says "missing", not the zeros.
class AgentHistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_width = contract.HISTORY_STEPS * (contract.AGENT_FEATURE_DIM + 1)
        self.register_buffer("feature_divisors", agent_feature_divisors())
        self.network = nn.Sequential(
            nn.Linear(input_width, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, agent_history, agent_history_mask):
        validity = agent_history_mask.unsqueeze(-1).to(agent_history.dtype)
        masked_history = agent_history / self.feature_divisors * validity
        flattened = torch.cat([masked_history, validity], dim=-1).flatten(start_dim=-2)
        return self.network(flattened)


# One embedding per 0.5 m map dot, over the batch's flat ragged block of dots - every row is
# a real dot, so there is no padding to keep unread.
class MapDotEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("feature_divisors", map_feature_divisors())
        self.register_buffer(
            "crossing_code_rows", torch.eye(contract.NUM_BOUNDARY_CROSSING_CODES)
        )
        input_width = (contract.MAP_FEATURE_DIM - 2) + 2 * contract.NUM_BOUNDARY_CROSSING_CODES
        self.network = nn.Sequential(
            nn.Linear(input_width, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, map_rows):
        scaled = map_rows / self.feature_divisors
        crossing_codes = map_rows[:, contract.MAP_LEFT_BOUNDARY_CROSSING:].long()
        crossing_one_hot = self.crossing_code_rows[crossing_codes].flatten(start_dim=-2)
        return self.network(torch.cat(
            [scaled[:, :contract.MAP_LEFT_BOUNDARY_CROSSING], crossing_one_hot.to(scaled.dtype)],
            dim=-1,
        ))


# One token per POLYLINE, not per dot (§34 token-count recomputation, 2026-08-14: measured
# p50 9,998 dots/sample makes per-dot attention ~3,800x the estimated cost). Every dot still
# passes through the encoder; a max over each polyline's dot embeddings packages them as one
# token. The batch's dots arrive flat and ragged, each carrying its global polyline slot, so
# the max is one scatter over (batch_size * max_polylines) slots; a slot no dot landed in
# comes out absent and zeroed.
def pool_dots_to_polyline_tokens(dot_embeddings, dot_polyline_slot, batch_size, max_polylines):
    hidden_width = dot_embeddings.shape[-1]
    tokens = torch.full(
        (batch_size * max_polylines, hidden_width), float("-inf"),
        dtype=dot_embeddings.dtype, device=dot_embeddings.device,
    )
    tokens = tokens.scatter_reduce(
        0, dot_polyline_slot.unsqueeze(-1).expand(-1, hidden_width),
        dot_embeddings, reduce="amax", include_self=True,
    )
    polyline_present = tokens[:, 0] > float("-inf")
    tokens = torch.where(polyline_present.unsqueeze(-1), tokens, torch.zeros_like(tokens))
    return (
        tokens.view(batch_size, max_polylines, hidden_width),
        polyline_present.view(batch_size, max_polylines),
    )


ATTENTION_HEAD_COUNT = 4
SCENE_ATTENTION_ROUNDS = 6
FEEDFORWARD_DIM = 4 * HIDDEN_DIM


# Structure is ours (projections, heads, masking); the inner score-softmax-weight step is
# torch's fused kernel. key_present True = this token may be read; padded slots stay unread.
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert HIDDEN_DIM % ATTENTION_HEAD_COUNT == 0
        self.query_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.key_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.value_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def split_heads(self, projected):
        batch_size, token_count, _ = projected.shape
        per_head = HIDDEN_DIM // ATTENTION_HEAD_COUNT
        return projected.view(batch_size, token_count, ATTENTION_HEAD_COUNT, per_head).transpose(1, 2)

    def forward(self, query_tokens, key_value_tokens, key_present):
        queries = self.split_heads(self.query_projection(query_tokens))
        keys = self.split_heads(self.key_projection(key_value_tokens))
        values = self.split_heads(self.value_projection(key_value_tokens))
        readable = key_present[:, None, None, :]
        attended = nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=readable)
        merged = attended.transpose(1, 2).flatten(start_dim=-2)
        return self.output_projection(merged)


# One self-attention round, pre-norm: normalise, attend, add back; normalise, feedforward, add
# back. Padded tokens produce garbage outputs but nothing downstream reads an absent token.
class SceneAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(HIDDEN_DIM)
        self.attention = MultiHeadAttention()
        self.feedforward_norm = nn.LayerNorm(HIDDEN_DIM)
        self.feedforward = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(FEEDFORWARD_DIM, HIDDEN_DIM),
        )

    def forward(self, tokens, token_present):
        normed = self.attention_norm(tokens)
        tokens = tokens + self.attention(normed, normed, token_present)
        return tokens + self.feedforward(self.feedforward_norm(tokens))


# The whole scene as one token sequence: [predicted agent | neighbours | map dots], agent first.
# Predicted agent and neighbours get separate encoder weights - different roles. A neighbour is
# present if it has at least one valid snapshot; the agent is present by construction.
# The chunk's traffic-signal history joins its map token here, after pooling, because the light is
# the same on every dot of the lane the chunk was cut from. It is projected and ADDED rather than
# concatenated: a projection carrying no bias sends the all-zero history of an unsignalled chunk to
# an exact zero, so an absent slot's token stays the exact zero the pooling contract promises,
# which a concatenation's bias term would lift off zero on every padded slot.
class SceneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent_encoder = AgentHistoryEncoder()
        self.neighbour_encoder = AgentHistoryEncoder()
        self.map_encoder = MapDotEncoder()
        self.signal_projection = nn.Linear(contract.POLYLINE_SIGNAL_DIM, HIDDEN_DIM, bias=False)
        self.lane_context_projection = nn.Linear(contract.LANE_CONTEXT_DIM, HIDDEN_DIM, bias=False)
        self.layers = nn.ModuleList(SceneAttentionLayer() for _ in range(SCENE_ATTENTION_ROUNDS))

    def forward(self, batch):
        agent_token = (
            self.agent_encoder(batch["agent_history"], batch["agent_history_mask"])
            + self.signal_projection(batch["agent_signal_history"].flatten(start_dim=-2))
        ).unsqueeze(1)
        neighbour_tokens = self.neighbour_encoder(
            batch["neighbour_history"], batch["neighbour_history_mask"]
        ) + self.signal_projection(batch["neighbour_signal_history"].flatten(start_dim=-2))
        map_tokens, map_present = pool_dots_to_polyline_tokens(
            self.map_encoder(batch["map_rows"]), batch["map_dot_polyline_slot"],
            agent_token.shape[0], int(batch["max_polylines_in_batch"]),
        )
        map_tokens = (
            map_tokens
            + self.signal_projection(batch["map_chunk_signal_history"].flatten(start_dim=-2))
            + self.lane_context_projection(batch["map_chunk_lane_context"])
        )

        tokens = torch.cat([agent_token, neighbour_tokens, map_tokens], dim=1)
        agent_present = torch.ones(agent_token.shape[:2], dtype=torch.bool, device=tokens.device)
        neighbour_present = batch["neighbour_history_mask"].any(dim=-1)
        token_present = torch.cat([agent_present, neighbour_present, map_present], dim=1)

        for layer in self.layers:
            tokens = layer(tokens, token_present)
        return tokens, token_present


QUERY_COUNT = 25
PRUNE_DISTANCE_METRES = 2.5
ANCHOR_DIRECTION_COUNT = 5
ANCHOR_DISTANCE_COUNT = 5
assert ANCHOR_DIRECTION_COUNT * ANCHOR_DISTANCE_COUNT == QUERY_COUNT
DECODER_ROUNDS = 3
TRAJECTORY_CONTROL_POINTS = 3
POSITION_CONTROL_VALUES = 2 * TRAJECTORY_CONTROL_POINTS
HEADING_CONTROL_VALUES = 2 * TRAJECTORY_CONTROL_POINTS
LOG_STANDARD_DEVIATION_CONTROL_VALUES = 2 * TRAJECTORY_CONTROL_POINTS
MINIMUM_LOG_STANDARD_DEVIATION = -1.609
MAXIMUM_LOG_STANDARD_DEVIATION = 5.0


def bernstein_curve_basis(control_point_count, step_count):
    time_fraction = torch.arange(1, step_count + 1, dtype=torch.float32) / step_count
    control_indices = torch.arange(control_point_count + 1, dtype=torch.float32)
    log_binomial_coefficients = (
        torch.lgamma(torch.tensor(control_point_count + 1.0))
        - torch.lgamma(control_indices + 1.0)
        - torch.lgamma(control_point_count - control_indices + 1.0)
    )
    return (
        log_binomial_coefficients.exp()
        * time_fraction[:, None] ** control_indices
        * (1.0 - time_fraction[:, None]) ** (control_point_count - control_indices)
    )



def unit_anchor_offsets():
    direction_indices = torch.arange(ANCHOR_DIRECTION_COUNT).repeat_interleave(ANCHOR_DISTANCE_COUNT)
    distance_indices = torch.arange(ANCHOR_DISTANCE_COUNT).repeat(ANCHOR_DIRECTION_COUNT)
    angles = 2 * torch.pi * direction_indices / ANCHOR_DIRECTION_COUNT
    fractions = (distance_indices + 1) / ANCHOR_DISTANCE_COUNT
    return fractions.unsqueeze(-1) * torch.stack([angles.cos(), angles.sin()], dim=-1)


def unit_anchor_offsets_per_type():
    return unit_anchor_offsets().repeat(contract.NUM_OBJECT_TYPES, 1, 1)


def predicted_type_index(agent_history):
    type_one_hot = agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_TYPE]
    return type_one_hot[:, : contract.NUM_OBJECT_TYPES].argmax(dim=-1)


# The furthest this agent could physically get inside the prediction horizon, one distance per
# sample: its own logged speed at "now" carried for the whole horizon, plus what the project's
# measured acceleration limit could add on top of that. Every term is the agent's own or measured
# from logged ground truth, so there is nothing here to tune. The "now" row is real for every
# sample by construction - loader.eligible_track_indices only makes a sample out of a track valid
# at contract.CURRENT_STEP_INDEX - so the speed is never read off a padded step, and the loader
# rotates velocity into the agent's frame, which leaves its length untouched.
def agent_reachable_distance(agent_history):
    current_speed = agent_history[:, contract.CURRENT_STEP_INDEX, contract.AGENT_VELOCITY].norm(dim=-1)
    return (
        current_speed * contract.FUTURE_HORIZON_SECONDS
        + 0.5 * contract.MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED * contract.FUTURE_HORIZON_SECONDS ** 2
    )


class ModeDecoder(nn.Module):
    def __init__(self, initial_unit_anchors):
        super().__init__()
        assert initial_unit_anchors.shape == (contract.NUM_OBJECT_TYPES, QUERY_COUNT, 2), (
            f"initial_unit_anchors has shape {tuple(initial_unit_anchors.shape)}, but the decoder"
            f" holds one {QUERY_COUNT}-anchor set per predicted object type and needs"
            f" ({contract.NUM_OBJECT_TYPES}, {QUERY_COUNT}, 2)."
            f" Re-run fit_anchors.py to produce a per-type anchor file"
        )
        self.queries = nn.Parameter(torch.randn(QUERY_COUNT, HIDDEN_DIM) * 0.02)
        self.anchor_offsets = nn.Parameter(initial_unit_anchors.detach().clone())
        self.anchor_projection = nn.Linear(2, HIDDEN_DIM)
        self.scene_norm = nn.LayerNorm(HIDDEN_DIM)
        self.round_norms = nn.ModuleList(nn.LayerNorm(HIDDEN_DIM) for _ in range(DECODER_ROUNDS))
        self.round_attention = nn.ModuleList(MultiHeadAttention() for _ in range(DECODER_ROUNDS))
        self.draft_projection = nn.Linear(2, HIDDEN_DIM, bias=False)
        self.trajectory_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(
                FEEDFORWARD_DIM,
                POSITION_CONTROL_VALUES
                + HEADING_CONTROL_VALUES
                + LOG_STANDARD_DEVIATION_CONTROL_VALUES,
            ),
        )
        self.confidence_head = nn.Linear(HIDDEN_DIM, 1)
        self.register_buffer(
            "curve_basis",
            bernstein_curve_basis(TRAJECTORY_CONTROL_POINTS, contract.FUTURE_STEPS)[:, 1:],
            persistent=False,
        )
        self.register_buffer(
            "anchor_ramp",
            torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float32)
            / contract.FUTURE_STEPS,
            persistent=False,
        )

    @property
    def unit_anchors(self):
        return self.anchor_offsets

    def draft_endpoint(self, queries, unit_anchors, batch_size):
        control_points = self.trajectory_head(queries)[..., :POSITION_CONTROL_VALUES].view(
            batch_size, QUERY_COUNT, TRAJECTORY_CONTROL_POINTS, 2
        )
        return torch.matmul(self.curve_basis[-1], control_points) + unit_anchors

    def decode_from_anchors(self, normed_tokens, token_present, unit_anchors, batch_size):
        queries = self.queries + self.anchor_projection(
            unit_anchors / contract.DISTANCE_NORMALISER_METRES
        )
        for round_index, (round_norm, round_attention) in enumerate(
            zip(self.round_norms, self.round_attention)
        ):
            queries = queries + round_attention(round_norm(queries), normed_tokens, token_present)
            if round_index + 1 < DECODER_ROUNDS:
                queries = queries + self.draft_projection(
                    self.draft_endpoint(queries, unit_anchors, batch_size)
                    / contract.DISTANCE_NORMALISER_METRES
                )
        head_output = self.trajectory_head(queries)
        control_points = head_output[..., :POSITION_CONTROL_VALUES].view(
            batch_size, QUERY_COUNT, TRAJECTORY_CONTROL_POINTS, 2
        )
        heading_control_points = head_output[
            ..., POSITION_CONTROL_VALUES:POSITION_CONTROL_VALUES + HEADING_CONTROL_VALUES
        ].view(batch_size, QUERY_COUNT, TRAJECTORY_CONTROL_POINTS, 2)
        log_standard_deviation_control_points = head_output[
            ..., POSITION_CONTROL_VALUES + HEADING_CONTROL_VALUES:
        ].view(batch_size, QUERY_COUNT, TRAJECTORY_CONTROL_POINTS, 2)
        heading_cosine_sine = torch.matmul(self.curve_basis, heading_control_points)
        position_log_standard_deviation = torch.matmul(
            self.curve_basis, log_standard_deviation_control_points
        ).clamp(MINIMUM_LOG_STANDARD_DEVIATION, MAXIMUM_LOG_STANDARD_DEVIATION)
        confidence_logits = self.confidence_head(queries).squeeze(-1)
        anchored_position = (
            torch.matmul(self.curve_basis, control_points)
            + unit_anchors[:, :, None, :] * self.anchor_ramp[None, None, :, None]
        )
        emitted_step_positions = torch.cat(
            [torch.zeros_like(anchored_position[..., :1, :]), anchored_position], dim=-2
        )
        predicted_speed = emitted_step_positions.diff(dim=-2).norm(
            dim=-1
        ) / contract.TIMESTEP_SECONDS
        return (
            anchored_position, heading_cosine_sine, position_log_standard_deviation,
            confidence_logits, predicted_speed,
        )

    def forward(self, tokens, token_present, predicted_type_index):
        batch_size = tokens.shape[0]
        normed_tokens = self.scene_norm(tokens)
        selected_unit_anchors = self.unit_anchors[predicted_type_index]
        (
            anchored_position, heading_cosine_sine, position_log_standard_deviation,
            confidence_logits, predicted_speed,
        ) = self.decode_from_anchors(normed_tokens, token_present, selected_unit_anchors, batch_size)
        return (
            anchored_position, heading_cosine_sine, position_log_standard_deviation,
            confidence_logits, predicted_speed, selected_unit_anchors,
        )


# Confidence-ordered endpoint pruning (MTR's reduction): walk modes by descending confidence,
# keep one whose 8 s endpoint sits at least PRUNE_DISTANCE_METRES from every kept endpoint.
# Fewer than 6 survivors -> backfill with the most confident dropped modes; always exactly 6 out.
def prune_modes(trajectories, confidence_logits):
    kept_indices = []
    dropped_indices = []
    endpoints = trajectories[:, -1]
    for mode_index in torch.argsort(confidence_logits, descending=True).tolist():
        if kept_indices and bool(
            (torch.cdist(endpoints[mode_index][None], endpoints[kept_indices]) < PRUNE_DISTANCE_METRES).any()
        ):
            dropped_indices.append(mode_index)
            continue
        kept_indices.append(mode_index)
        if len(kept_indices) == contract.NUM_PREDICTED_MODES:
            break
    kept_indices.extend(dropped_indices[: contract.NUM_PREDICTED_MODES - len(kept_indices)])
    kept = torch.tensor(kept_indices, device=trajectories.device)
    return trajectories[kept], confidence_logits[kept]


# The same walk run for every sample at once, plus the count of modes each sample kept BEFORE
# backfill - the mode-collapse detector, since a sample that keeps one mode emits six trajectories
# of which five are duplicates. The walk stays sequential because the rule is greedy, but each step
# is one batched comparison over the whole batch. It stops as soon as every sample has filled its
# NUM_PREDICTED_MODES slots: from there on still_walking is false everywhere, so every remaining
# step writes nothing. Reading that condition on the host is the walk's one device synchronisation
# per step, and it buys the up-to-58 steps the early stop skips.
def prune_modes_batched_with_kept_count(trajectories, confidence_logits):
    batch_size, mode_count = confidence_logits.shape
    device = trajectories.device
    endpoints = trajectories[:, :, -1]
    confidence_order = torch.argsort(confidence_logits, dim=-1, descending=True)
    slot_positions = torch.arange(contract.NUM_PREDICTED_MODES, device=device)

    kept_indices = torch.zeros(batch_size, contract.NUM_PREDICTED_MODES, dtype=torch.long, device=device)
    kept_endpoints = torch.zeros_like(endpoints[:, : contract.NUM_PREDICTED_MODES])
    kept_slot_filled = torch.zeros(batch_size, contract.NUM_PREDICTED_MODES, dtype=torch.bool, device=device)
    kept_count = torch.zeros(batch_size, dtype=torch.long, device=device)
    dropped_indices = torch.zeros_like(kept_indices)
    dropped_count = torch.zeros_like(kept_count)

    for walk_position in range(mode_count):
        candidate_index = confidence_order[:, walk_position]
        candidate_endpoint = endpoints.gather(1, candidate_index[:, None, None].expand(-1, -1, 2))
        separations = torch.cdist(candidate_endpoint, kept_endpoints).squeeze(1)
        too_close = ((separations < PRUNE_DISTANCE_METRES) & kept_slot_filled).any(dim=-1)
        still_walking = kept_count < contract.NUM_PREDICTED_MODES

        keeps = still_walking & ~too_close
        keep_slot = keeps[:, None] & (slot_positions[None, :] == kept_count[:, None])
        kept_indices = torch.where(keep_slot, candidate_index[:, None], kept_indices)
        kept_endpoints = torch.where(keep_slot[:, :, None], candidate_endpoint, kept_endpoints)
        kept_slot_filled = kept_slot_filled | keep_slot
        kept_count = kept_count + keeps.long()

        drops = still_walking & too_close
        drop_slot = drops[:, None] & (slot_positions[None, :] == dropped_count[:, None])
        dropped_indices = torch.where(drop_slot, candidate_index[:, None], dropped_indices)
        dropped_count = dropped_count + drops.long()

        if bool((kept_count == contract.NUM_PREDICTED_MODES).all()):
            break

    backfill_positions = (slot_positions[None, :] - kept_count[:, None]).clamp(min=0)
    final_indices = torch.where(
        slot_positions[None, :] < kept_count[:, None],
        kept_indices,
        dropped_indices.gather(1, backfill_positions),
    )
    trajectory_selector = final_indices[:, :, None, None].expand(-1, -1, *trajectories.shape[2:])
    return (
        trajectories.gather(1, trajectory_selector),
        confidence_logits.gather(1, final_indices),
        kept_count,
    )


# The pruned pair on its own, for every caller that grades trajectories rather than watching the
# walk: the metric accumulator, the null baselines and the submission path.
def prune_modes_batched(trajectories, confidence_logits):
    kept_trajectories, kept_confidence_logits, _ = prune_modes_batched_with_kept_count(
        trajectories, confidence_logits
    )
    return kept_trajectories, kept_confidence_logits


class NeighbourFutureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_norm = nn.LayerNorm(HIDDEN_DIM)
        self.network = nn.Sequential(
            nn.Linear(HIDDEN_DIM, FEEDFORWARD_DIM),
            nn.ReLU(),
            nn.Linear(FEEDFORWARD_DIM, 2 * contract.FUTURE_STEPS),
        )

    def forward(self, neighbour_tokens):
        positions = self.network(self.token_norm(neighbour_tokens))
        return positions.view(*neighbour_tokens.shape[:2], contract.FUTURE_STEPS, 2)


class MotionPredictor(nn.Module):
    def __init__(self, initial_unit_anchors):
        super().__init__()
        self.scene_encoder = SceneEncoder()
        self.mode_decoder = ModeDecoder(initial_unit_anchors)
        self.neighbour_future_head = NeighbourFutureHead()
        self.neighbour_future_feedback_projection = nn.Linear(
            2 * contract.FUTURE_STEPS, HIDDEN_DIM, bias=False
        )

    @property
    def unit_anchors(self):
        return self.mode_decoder.unit_anchors

    def encode_scene_and_modes(self, batch):
        tokens, token_present = self.scene_encoder(batch)
        neighbour_count = batch["neighbour_history"].shape[1]
        neighbour_tokens = tokens[:, 1:1 + neighbour_count]
        neighbour_present = token_present[:, 1:1 + neighbour_count]
        neighbour_future_positions = self.neighbour_future_head(neighbour_tokens)
        neighbour_feedback = self.neighbour_future_feedback_projection(
            (neighbour_future_positions / contract.DISTANCE_NORMALISER_METRES).flatten(start_dim=-2)
        )
        neighbour_feedback = (
            neighbour_feedback * neighbour_present.unsqueeze(-1).to(neighbour_feedback.dtype)
        )
        tokens = torch.cat(
            [tokens[:, :1], neighbour_tokens + neighbour_feedback, tokens[:, 1 + neighbour_count:]],
            dim=1,
        )
        (
            trajectories, heading_cosine_sine, position_log_standard_deviation, confidence_logits,
            predicted_speed, selected_unit_anchors,
        ) = self.mode_decoder(
            tokens, token_present, predicted_type_index(batch["agent_history"]),
        )
        return (tokens, trajectories, heading_cosine_sine, position_log_standard_deviation,
                confidence_logits, predicted_speed, selected_unit_anchors, neighbour_future_positions)

    def predict_with_heading(self, batch):
        (
            _, trajectories, heading_cosine_sine, position_log_standard_deviation,
            confidence_logits, predicted_speed, selected_unit_anchors, neighbour_future_positions,
        ) = self.encode_scene_and_modes(batch)
        return (trajectories, heading_cosine_sine, position_log_standard_deviation,
                confidence_logits, predicted_speed, selected_unit_anchors, neighbour_future_positions)

    def forward(self, batch):
        _, trajectories, _, _, confidence_logits, _, _, _ = self.encode_scene_and_modes(batch)
        return trajectories, confidence_logits


def parameter_fingerprint(model_state):
    parameter_description = ",".join(
        f"{name}:{tuple(tensor.shape)}" for name, tensor in sorted(model_state.items())
    )
    return hashlib.sha256(parameter_description.encode()).hexdigest()


def load_checkpoint_state(checkpoint_path, map_location="cpu", allow_version_mismatch=False):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_code_version = checkpoint.get("code_version")
    if not allow_version_mismatch:
        assert checkpoint_code_version == contract.STAGING_CODE_VERSION, (
            f"{checkpoint_path} was saved under code_version {checkpoint_code_version!r}, but the"
            f" working tree is at contract.STAGING_CODE_VERSION {contract.STAGING_CODE_VERSION!r}."
            f" Loading it with strict=True would silently accept a byte-compatible but"
            f" architecturally different state dict. Pass allow_version_mismatch=True to load it"
            f" anyway."
        )
    stamped_fingerprint = checkpoint.get("parameter_fingerprint")
    if stamped_fingerprint is not None:
        recomputed_fingerprint = parameter_fingerprint(checkpoint["model_state"])
        assert stamped_fingerprint == recomputed_fingerprint, (
            f"{checkpoint_path} carries parameter_fingerprint {stamped_fingerprint} but its"
            f" model_state hashes to {recomputed_fingerprint}: the state dict was altered after"
            f" the stamp was written."
        )
    return checkpoint
