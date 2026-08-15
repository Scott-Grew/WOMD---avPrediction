# > Checkpoint measurement: the comparable read on a trained model, so two runs can be set beside
# each other without a throwaway script in between. Reports the three numbers that separate what the
# model KNOWS from what it SUBMITS - the best of all its modes, the best of the six the confidence
# head chooses, and the best of six drawn at random - plus how far apart the modes spread and how
# much of each agent's own reachable budget the predictions spend.
# The random draw is the control the confidence number is read against: a confidence head that
# cannot beat it has learned nothing about which mode to trust, whatever the loss curve says.
# Every distance is masked to the steps the logged track actually recorded.
# Run: python3 measure_checkpoint.py checkpoint.pt ../data/staged

import sys
from pathlib import Path

import numpy as np
import torch

from womd import contract, metrics, model, pipeline

SAMPLE_LIMIT = 512
RANDOM_DRAW_SEED = 0


def mode_mean_distances(trajectories, future_positions, future_mask):
    step_distances = (trajectories - future_positions.unsqueeze(1)).norm(dim=-1)
    validity = future_mask.unsqueeze(1).to(step_distances.dtype)
    return (step_distances * validity).sum(dim=-1) / validity.sum(dim=-1).clamp_min(1.0)


def main():
    checkpoint_path = Path(sys.argv[1])
    staged_directory = Path(sys.argv[2])
    predictor = model.MotionPredictor()
    predictor.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state"])
    predictor.eval()
    generator = torch.Generator().manual_seed(RANDOM_DRAW_SEED)

    all_modes, by_confidence, at_random, widest_gap, budget_use = [], [], [], [], []
    sample_count = 0
    for batch in pipeline.batches(sorted(staged_directory.glob("*.npz")), 0, 16, 0, 0, True):
        with torch.no_grad():
            trajectories, confidence_logits = predictor(batch)
        scoreable = batch["future_mask"].any(dim=-1)
        if not scoreable.any():
            continue
        distances = mode_mean_distances(
            trajectories, batch["future_positions"], batch["future_mask"]
        )[scoreable]
        pruned, pruned_logits = model.prune_modes_batched(
            trajectories[scoreable], confidence_logits[scoreable]
        )
        pruned_distances = mode_mean_distances(
            pruned, batch["future_positions"][scoreable], batch["future_mask"][scoreable]
        )
        draw = torch.stack([
            torch.randperm(distances.shape[1], generator=generator)[: contract.NUM_PREDICTED_MODES]
            for _ in range(distances.shape[0])
        ])

        endpoints = trajectories[scoreable][:, :, -1]
        reachable = model.agent_reachable_distance(batch["agent_history"][scoreable])
        steps = torch.cat([
            trajectories[scoreable][..., :1, :], trajectories[scoreable].diff(dim=-2)
        ], dim=-2)

        all_modes.append(distances.min(dim=1).values)
        by_confidence.append(pruned_distances.min(dim=1).values)
        at_random.append(distances.gather(1, draw).min(dim=1).values)
        widest_gap.append(torch.cdist(endpoints, endpoints).amax(dim=(1, 2)))
        budget_use.append(steps.norm(dim=-1).sum(dim=-1).amax(dim=1) / reachable)
        sample_count += int(scoreable.sum())
        if sample_count >= SAMPLE_LIMIT:
            break

    joined = {name: torch.cat(values).numpy() for name, values in (
        ("best of ALL modes", all_modes), ("best of 6 by CONFIDENCE", by_confidence),
        ("best of 6 at RANDOM", at_random), ("widest gap between modes", widest_gap),
        ("path / reachable budget", budget_use),
    )}
    print(f"{checkpoint_path} on {staged_directory}: {sample_count} designated targets")
    print(f"{'quantity':28s}{'mean':>10s}{'p50':>10s}{'p90':>10s}{'max':>10s}")
    for name, values in joined.items():
        print(f"{name:28s}{values.mean():10.3f}{np.percentile(values, 50):10.3f}"
              f"{np.percentile(values, 90):10.3f}{values.max():10.3f}")


if __name__ == "__main__":
    main()
