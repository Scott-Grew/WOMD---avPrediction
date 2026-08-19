# > Pure-torch training monitor: minADE / minFDE over the PRUNED 6 modes, and the walk's own health
# Monitor only - every reported number comes from Waymo's scorer in the container. Counts
# follow B5's lesson: minADE averages over samples with any valid future step, minFDE over
# samples whose 8 s endpoint is present - never over batch size.
# Every sample is selected by MULTIPLYING by its mask rather than indexing with it: indexing has to
# know how many rows survive, which reads a device tensor on the host and stalls the queue, and the
# monitor runs on every batch. A sample with no valid future step averages 0/0, so its denominator
# is clamped to 1 first - it then contributes an exact zero to a sum it is masked out of anyway,
# and a sample with any valid step divides by its own untouched count.
# The pruning walk's kept-mode count comes back with the trajectories: a sample that kept fewer than
# NUM_PREDICTED_MODES modes had the rest backfilled with duplicates, so a backfill rate near 1 means
# the six emitted futures are one future repeated.

import torch

from womd import contract
from womd.model import prune_modes_batched_with_kept_count


def mean_distance_per_mode(trajectories, future_positions, future_mask):
    step_distances = (trajectories - future_positions.unsqueeze(1)).norm(dim=-1)
    validity = future_mask.unsqueeze(1).to(step_distances.dtype)
    return (step_distances * validity).sum(dim=-1) / validity.sum(dim=-1).clamp_min(1.0)


class MetricAccumulator:
    def __init__(self):
        self.ade_sum = 0.0
        self.ade_count = 0
        self.fde_sum = 0.0
        self.fde_count = 0
        self.kept_mode_sum = 0
        self.backfilled_sample_count = 0
        self.sample_count = 0

    def update(self, trajectories, confidence_logits, future_positions, future_mask):
        kept_trajectories, _, kept_mode_count = prune_modes_batched_with_kept_count(
            trajectories, confidence_logits
        )
        distances = (kept_trajectories - future_positions.unsqueeze(1)).norm(dim=-1)
        valid_steps = future_mask.unsqueeze(1)
        summed = torch.where(valid_steps, distances, torch.zeros_like(distances)).sum(dim=-1)
        average_distances = summed / future_mask.sum(dim=-1, keepdim=True).clamp_min(1)

        has_any_valid_step = future_mask.any(dim=-1)
        self.ade_sum += (average_distances.min(dim=-1).values * has_any_valid_step).sum()
        self.ade_count += has_any_valid_step.sum()

        final_step_valid = future_mask[:, -1]
        self.fde_sum += (distances[:, :, -1].min(dim=-1).values * final_step_valid).sum()
        self.fde_count += final_step_valid.sum()

        self.kept_mode_sum += kept_mode_count.sum()
        self.backfilled_sample_count += (kept_mode_count < contract.NUM_PREDICTED_MODES).sum()
        self.sample_count += confidence_logits.shape[0]

    def results(self):
        return {
            "min_ade": float(self.ade_sum / self.ade_count) if self.ade_count else float("nan"),
            "min_fde": float(self.fde_sum / self.fde_count) if self.fde_count else float("nan"),
            "mean_kept_modes": (
                float(self.kept_mode_sum / self.sample_count) if self.sample_count else float("nan")
            ),
            "backfill_rate": (
                float(self.backfilled_sample_count / self.sample_count)
                if self.sample_count else float("nan")
            ),
        }
