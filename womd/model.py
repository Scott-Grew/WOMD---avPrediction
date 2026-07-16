import torch
import torch.nn as nn

from womd import contract

HIDDEN_DIM = 128
ATTENTION_HEADS = 4
ATTENTION_LAYERS = 3
FEEDFORWARD_DIM = 512
DROPOUT_RATE = 0.1


class PolylineEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.point_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, points, point_mask):
        encoded = self.point_network(points)
        expanded_mask = point_mask.unsqueeze(-1)
        encoded = encoded.masked_fill(~expanded_mask, torch.finfo(encoded.dtype).min)
        pooled = encoded.max(dim=-2).values
        return torch.where(point_mask.any(dim=-1, keepdim=True), pooled, torch.zeros_like(pooled))


class MotionPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        attention_heads=ATTENTION_HEADS,
        attention_layers=ATTENTION_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout_rate=DROPOUT_RATE,
        num_modes=contract.NUM_PREDICTED_MODES,
        future_steps=contract.FUTURE_STEPS,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps

        self.agent_encoder = PolylineEncoder(contract.AGENT_FEATURE_DIM, hidden_dim)
        self.map_encoder = PolylineEncoder(contract.MAP_FEATURE_DIM, hidden_dim)
        self.target_token_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.neighbour_token_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.map_token_bias = nn.Parameter(torch.zeros(hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.scene_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=attention_layers, enable_nested_tensor=False
        )

        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, num_modes * future_steps * 2),
        )
        self.mode_head = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, num_modes),
        )

    def forward(self, batch):
        target_token = self.agent_encoder(
            batch["agent_history"], batch["agent_history_mask"]
        ) + self.target_token_bias
        neighbour_tokens = self.agent_encoder(
            batch["neighbour_history"], batch["neighbour_history_mask"]
        ) + self.neighbour_token_bias
        map_tokens = self.map_encoder(
            batch["map_polylines"], batch["map_polylines_mask"]
        ) + self.map_token_bias

        tokens = torch.cat([target_token.unsqueeze(1), neighbour_tokens, map_tokens], dim=1)

        batch_size = tokens.shape[0]
        target_present = torch.ones(
            batch_size, 1, dtype=torch.bool, device=tokens.device
        )
        token_present = torch.cat(
            [
                target_present,
                batch["neighbour_history_mask"].any(dim=-1),
                batch["map_polylines_mask"].any(dim=-1),
            ],
            dim=1,
        )

        encoded = self.scene_encoder(tokens, src_key_padding_mask=~token_present)
        scene_summary = encoded[:, 0]
        trajectories = self.trajectory_head(scene_summary).view(
            batch_size, self.num_modes, self.future_steps, 2
        )
        mode_logits = self.mode_head(scene_summary)
        return trajectories, mode_logits
