import time

import torch

from womd import contract, metrics

STATIONARY_PATH_METRES = 1.0
TURNING_HEADING_RADIANS = 0.5236

HEADLINE_HORIZON = f"@{contract.EVALUATED_HORIZON_STEPS[-1] / 10:g}s"

MANOEUVRE_NAMES = ("stationary", "straight", "turning")
STATIONARY, STRAIGHT, TURNING = range(len(MANOEUVRE_NAMES))


def manoeuvre_labels(
    future_positions,
    future_mask,
    stationary_path_metres=STATIONARY_PATH_METRES,
    turning_heading_radians=TURNING_HEADING_RADIANS,
):
    steps = future_positions.roll(1, dims=1)
    steps[:, 0] = 0.0
    displacements = future_positions - steps
    valid_steps = future_mask.clone()
    valid_steps[:, 0] = False
    path_length = (
        torch.linalg.vector_norm(displacements, dim=-1) * valid_steps
    ).sum(dim=-1)

    first_valid = future_mask.float().argmax(dim=1)
    last_valid = future_mask.shape[1] - 1 - future_mask.flip(1).float().argmax(dim=1)
    sample_index = torch.arange(future_positions.shape[0], device=future_positions.device)
    start = future_positions[sample_index, first_valid]
    end = future_positions[sample_index, last_valid]
    heading_change = torch.atan2(end[:, 1] - start[:, 1], end[:, 0] - start[:, 0]).abs()

    labels = torch.full_like(path_length, STRAIGHT, dtype=torch.long)
    labels[heading_change > turning_heading_radians] = TURNING
    labels[path_length < stationary_path_metres] = STATIONARY
    return labels


def slice_labels(batch):
    current_row = batch["agent_history"][:, contract.CURRENT_STEP_INDEX]
    return {
        "agent_type": current_row[:, contract.AGENT_TYPE].argmax(dim=-1),
        "manoeuvre": manoeuvre_labels(batch["future_positions"], batch["future_mask"]),
    }


SLICE_NAMES = {
    "agent_type": contract.PREDICTED_OBJECT_TYPES,
    "manoeuvre": MANOEUVRE_NAMES,
}


LATENCY_WARMUP_REPEATS = 2


def _time_forward(predictor, batch, repeats):
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP_REPEATS):
            predictor(batch)
        started = time.perf_counter()
        for _ in range(repeats):
            predictor(batch)
    return (time.perf_counter() - started) / repeats


def measure_latency(predictor, batch, repeats=10):
    predictor.eval()
    batch_size = batch["future_positions"].shape[0]
    single = {name: values[:1] for name, values in batch.items()}
    batched_seconds = _time_forward(predictor, batch, repeats)
    single_seconds = _time_forward(predictor, single, repeats)
    return {
        "batch_size": float(batch_size),
        "batched_total_ms": batched_seconds * 1000.0,
        "batched_per_sample_ms": batched_seconds * 1000.0 / batch_size,
        "single_sample_ms": single_seconds * 1000.0,
    }


def render_latency(measurements):
    return "\n".join(
        [
            "inference latency",
            f"{'batch size':<26}{measurements['batch_size']:>10.0f}",
            f"{'batched, whole batch':<26}{measurements['batched_total_ms']:>10.2f} ms",
            f"{'batched, per sample':<26}{measurements['batched_per_sample_ms']:>10.3f} ms",
            f"{'single sample':<26}{measurements['single_sample_ms']:>10.3f} ms",
        ]
    )


def _reduce(per_sample, selected):
    return metrics.reduce_per_sample(
        {name: values[selected] for name, values in per_sample.items()}
    )


TAIL_QUANTILES = (0.5, 0.9, 0.95, 0.99)
TAIL_HEADINGS = ("median", "p90", "p95", "p99", "max")


def tail_statistics(values):
    quantiles = torch.quantile(
        values, torch.tensor(TAIL_QUANTILES, dtype=values.dtype)
    ).tolist()
    return quantiles + [values.max().item()]


def _tail_row(label, count, values):
    cells = "".join(f"{statistic:>10.3f}" for statistic in tail_statistics(values))
    return f"{label:<14}{count:>7}{cells}"


def _row(label, count, model_results, baseline_results, columns):
    cells = "".join(
        f"{model_results[column]:>10.3f}{baseline_results[column]:>10.3f}"
        for column in columns
    )
    return f"{label:<14}{count:>7}{cells}"


def render(model_per_sample, baseline_per_sample, labels=None, columns=None):
    columns = columns or [
        name
        for name in model_per_sample
        if name.endswith(HEADLINE_HORIZON)
        and not name.startswith(metrics.ENDPOINT_PRESENT_PREFIX)
    ]
    sample_count = next(iter(model_per_sample.values())).shape[0]
    everything = torch.ones(sample_count, dtype=torch.bool)

    header = f"{'slice':<14}{'n':>7}" + "".join(
        f"{column:>20}" for column in columns
    )
    subheader = " " * 21 + "".join(f"{'model':>10}{'const-vel':>10}" for _ in columns)
    lines = [header, subheader]
    lines.append(
        _row(
            "all",
            sample_count,
            _reduce(model_per_sample, everything),
            _reduce(baseline_per_sample, everything),
            columns,
        )
    )
    if labels is None:
        return "\n".join(lines)

    selections = [("all", everything)]
    for slice_name, names in SLICE_NAMES.items():
        lines.append("")
        for value, name in enumerate(names):
            selected = labels[slice_name] == value
            count = int(selected.sum())
            if count == 0:
                continue
            selections.append((name, selected))
            lines.append(
                _row(
                    name,
                    count,
                    _reduce(model_per_sample, selected),
                    _reduce(baseline_per_sample, selected),
                    columns,
                )
            )

    headline = f"minADE{HEADLINE_HORIZON}"
    for source_name, per_sample in (
        ("model", model_per_sample),
        ("const-vel", baseline_per_sample),
    ):
        lines.append("")
        lines.append(f"{headline} distribution — {source_name}")
        lines.append(
            f"{'slice':<14}{'n':>7}" + "".join(f"{name:>10}" for name in TAIL_HEADINGS)
        )
        for name, selected in selections:
            lines.append(_tail_row(name, int(selected.sum()), per_sample[headline][selected]))
    return "\n".join(lines)
