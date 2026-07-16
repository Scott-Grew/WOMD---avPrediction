import torch

from womd import baseline, metrics


def _move_batch(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


@torch.no_grad()
def evaluate_dataloader(predictor, dataloader, device):
    predictor.eval()
    model_accumulator = metrics.MetricAccumulator()
    baseline_accumulator = metrics.MetricAccumulator()

    for batch in dataloader:
        batch = _move_batch(batch, device)
        batch_size = batch["future_positions"].shape[0]
        trajectories, _ = predictor(batch)
        model_accumulator.update(
            metrics.evaluate_predictions(
                trajectories, batch["future_positions"], batch["future_mask"]
            ),
            batch_size,
        )
        baseline_accumulator.update(
            metrics.evaluate_predictions(
                baseline.constant_velocity_predictions(batch),
                batch["future_positions"],
                batch["future_mask"],
            ),
            batch_size,
        )
    return model_accumulator.summary(), baseline_accumulator.summary()


def format_comparison(model_results, baseline_results):
    lines = [f"{'metric':<18}{'model':>12}{'constant-vel':>16}{'delta':>12}"]
    for name in model_results:
        model_value = model_results[name]
        baseline_value = baseline_results[name]
        lines.append(
            f"{name:<18}{model_value:>12.3f}{baseline_value:>16.3f}"
            f"{baseline_value - model_value:>12.3f}"
        )
    return "\n".join(lines)
