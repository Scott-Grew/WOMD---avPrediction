import torch

from womd import baseline, contract, kinematics, metrics, offroad, report


def _move_batch(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def _concatenate(collected):
    return {name: torch.cat(values) for name, values in collected.items()}


LOGGED_SOURCE = "logged"
OFFROAD_SOURCES = ("model", "const-vel", LOGGED_SOURCE)


def _offroad_rates(batch, labels, trajectory_sources):
    selected = report.offroad_selection(labels)
    sample_count = selected.shape[0]
    rates = {name: torch.zeros(sample_count) for name in trajectory_sources}
    rates["covered"] = torch.zeros(sample_count, dtype=torch.bool)
    if not bool(selected.any()):
        return rates

    polylines = batch["map_polylines"][selected]
    polylines_mask = batch["map_polylines_mask"][selected]
    for name, trajectories in trajectory_sources.items():
        rate, covered = offroad.rate_and_coverage(
            trajectories[selected],
            polylines,
            polylines_mask,
            contract.OFFROAD_HORIZON_STEPS,
            contract.OFFROAD_DISTANCE_LIMIT_METRES,
        )
        rates[name][selected] = rate.to(rates[name].dtype)
        if name == LOGGED_SOURCE:
            rates["covered"][selected] = covered
    return rates


@torch.no_grad()
def collect_per_sample(predictor, dataloader, device):
    predictor.eval()
    model_batches, baseline_batches, label_batches, offroad_batches = {}, {}, {}, {}

    for batch in dataloader:
        batch = _move_batch(batch, device)
        trajectories, _ = predictor(batch)
        baseline_trajectories = baseline.constant_velocity_predictions(batch)
        model_values = metrics.per_sample_metrics(
            trajectories, batch["future_positions"], batch["future_mask"]
        )
        model_values.update(kinematics.violation_rates(trajectories))
        baseline_values = metrics.per_sample_metrics(
            baseline_trajectories, batch["future_positions"], batch["future_mask"]
        )
        baseline_values.update(kinematics.violation_rates(baseline_trajectories))
        labels = report.slice_labels(batch)
        offroad_values = _offroad_rates(
            batch,
            labels,
            {
                "model": trajectories,
                "const-vel": baseline_trajectories,
                LOGGED_SOURCE: batch["future_positions"].unsqueeze(1),
            },
        )
        for collected, produced in (
            (model_batches, model_values),
            (baseline_batches, baseline_values),
            (label_batches, labels),
            (offroad_batches, offroad_values),
        ):
            for name, values in produced.items():
                collected.setdefault(name, []).append(values.cpu())

    return (
        _concatenate(model_batches),
        _concatenate(baseline_batches),
        _concatenate(label_batches),
        _concatenate(offroad_batches),
    )


def evaluate_dataloader(predictor, dataloader, device):
    model_per_sample, baseline_per_sample, _, _ = collect_per_sample(
        predictor, dataloader, device
    )
    return (
        metrics.reduce_per_sample(model_per_sample),
        metrics.reduce_per_sample(baseline_per_sample),
    )


def render_dataloader_report(predictor, dataloader, device, sliced=True):
    model_per_sample, baseline_per_sample, labels, offroad_rates = collect_per_sample(
        predictor, dataloader, device
    )
    rendered = report.render(
        model_per_sample, baseline_per_sample, labels if sliced else None
    )
    return "\n\n".join([rendered, report.render_offroad(offroad_rates)])
