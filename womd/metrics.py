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


ENDPOINT_PRESENT_PREFIX = "endpointPresent"


def per_sample_metrics(
    trajectories,
    future_positions,
    future_mask,
    horizons=contract.EVALUATED_HORIZON_STEPS,
    miss_threshold=contract.MISS_RATE_THRESHOLD_METRES,
):
    per_sample = {}
    for horizon_steps in horizons:
        seconds = horizon_steps / 10.0
        average = minimum_average_displacement(
            trajectories, future_positions, future_mask, horizon_steps
        )
        final, final_present = minimum_final_displacement(
            trajectories, future_positions, future_mask, horizon_steps
        )
        per_sample[f"minADE@{seconds:g}s"] = average
        per_sample[f"minFDE@{seconds:g}s"] = final
        per_sample[f"missRate@{seconds:g}s"] = (final > miss_threshold).to(final.dtype)
        per_sample[f"{ENDPOINT_PRESENT_PREFIX}@{seconds:g}s"] = final_present
    return per_sample


def endpoint_presence_key(name):
    if "@" not in name:
        return None
    return f"{ENDPOINT_PRESENT_PREFIX}@{name.split('@')[1]}"


def reduce_per_sample(per_sample):
    results = {}
    for name, values in per_sample.items():
        if name.startswith(ENDPOINT_PRESENT_PREFIX):
            continue
        presence_key = endpoint_presence_key(name)
        if name.startswith("minADE") or presence_key not in per_sample:
            results[name] = values.mean().item()
            continue
        present = per_sample[presence_key]
        present_count = present.sum().clamp(min=1)
        results[name] = (values * present).sum().item() / present_count.item()
    return results


def evaluate_predictions(
    trajectories,
    future_positions,
    future_mask,
    horizons=contract.EVALUATED_HORIZON_STEPS,
    miss_threshold=contract.MISS_RATE_THRESHOLD_METRES,
):
    return reduce_per_sample(
        per_sample_metrics(
            trajectories, future_positions, future_mask, horizons, miss_threshold
        )
    )


def breaches(pinned, measured, relative_tolerance):
    found = {}
    for name, pinned_value in pinned.items():
        if name not in measured:
            found[name] = (pinned_value, None)
            continue
        allowed = abs(pinned_value) * relative_tolerance
        if abs(measured[name] - pinned_value) > allowed:
            found[name] = (pinned_value, measured[name])
    return found


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
