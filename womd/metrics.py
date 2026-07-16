import torch

from womd import contract


def _horizon_slice(tensor, horizon_steps):
    return tensor[..., :horizon_steps, :]


def minimum_average_displacement(trajectories, future_positions, future_mask, horizon_steps):
    distances = torch.linalg.vector_norm(
        _horizon_slice(trajectories, horizon_steps)
        - _horizon_slice(future_positions, horizon_steps).unsqueeze(1),
        dim=-1,
    )
    weights = future_mask[..., :horizon_steps].unsqueeze(1).to(distances.dtype)
    per_mode = (distances * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)
    return per_mode.min(dim=1).values


def minimum_final_displacement(trajectories, future_positions, future_mask, horizon_steps):
    final_index = horizon_steps - 1
    distances = torch.linalg.vector_norm(
        trajectories[:, :, final_index] - future_positions[:, final_index].unsqueeze(1), dim=-1
    )
    present = future_mask[:, final_index]
    minimum = distances.min(dim=1).values
    return minimum, present


def evaluate_predictions(
    trajectories,
    future_positions,
    future_mask,
    horizons=contract.EVALUATED_HORIZON_STEPS,
    miss_threshold=contract.MISS_RATE_THRESHOLD_METRES,
):
    results = {}
    for horizon_steps in horizons:
        seconds = horizon_steps / 10.0
        average = minimum_average_displacement(
            trajectories, future_positions, future_mask, horizon_steps
        )
        final, final_present = minimum_final_displacement(
            trajectories, future_positions, future_mask, horizon_steps
        )
        present_count = final_present.sum().clamp(min=1)
        results[f"minADE@{seconds:g}s"] = average.mean().item()
        results[f"minFDE@{seconds:g}s"] = (
            final * final_present
        ).sum().item() / present_count.item()
        results[f"missRate@{seconds:g}s"] = (
            ((final > miss_threshold) & final_present).sum().item() / present_count.item()
        )
    return results


class MetricAccumulator:
    def __init__(self):
        self.totals = {}
        self.sample_count = 0

    def update(self, batch_results, batch_size):
        for name, value in batch_results.items():
            self.totals[name] = self.totals.get(name, 0.0) + value * batch_size
        self.sample_count += batch_size

    def summary(self):
        if self.sample_count == 0:
            return {}
        return {name: total / self.sample_count for name, total in self.totals.items()}
