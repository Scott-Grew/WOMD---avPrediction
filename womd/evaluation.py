import torch

from womd import baseline, metrics, report


def _move_batch(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def _concatenate(collected):
    return {name: torch.cat(values) for name, values in collected.items()}


@torch.no_grad()
def collect_per_sample(predictor, dataloader, device):
    predictor.eval()
    model_batches, baseline_batches, label_batches = {}, {}, {}

    for batch in dataloader:
        batch = _move_batch(batch, device)
        trajectories, _ = predictor(batch)
        model_values = metrics.per_sample_metrics(
            trajectories, batch["future_positions"], batch["future_mask"]
        )
        baseline_values = metrics.per_sample_metrics(
            baseline.constant_velocity_predictions(batch),
            batch["future_positions"],
            batch["future_mask"],
        )
        labels = report.slice_labels(batch)
        for collected, produced in (
            (model_batches, model_values),
            (baseline_batches, baseline_values),
            (label_batches, labels),
        ):
            for name, values in produced.items():
                collected.setdefault(name, []).append(values.cpu())

    return (
        _concatenate(model_batches),
        _concatenate(baseline_batches),
        _concatenate(label_batches),
    )


def evaluate_dataloader(predictor, dataloader, device):
    model_per_sample, baseline_per_sample, _ = collect_per_sample(
        predictor, dataloader, device
    )
    return (
        metrics.reduce_per_sample(model_per_sample),
        metrics.reduce_per_sample(baseline_per_sample),
    )


def render_dataloader_report(predictor, dataloader, device, sliced=True):
    model_per_sample, baseline_per_sample, labels = collect_per_sample(
        predictor, dataloader, device
    )
    return report.render(
        model_per_sample, baseline_per_sample, labels if sliced else None
    )
