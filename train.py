# > Training entry point: recipe v1 - plain AdamW, constant rate, no schedule, no clipping
# The epoch loop computes the loss components and the minADE/minFDE monitor ONLY (§36:
# kinematics and off-road are finish-line diagnostics, never inside the epoch). Weight decay
# skips biases, LayerNorms and the 64 learned queries - parameters whose job is to become
# non-zero, which decay would fight (the 2026-08-06 optimizer incident, applied forward).
# LEARNING_RATE 1e-3 and WEIGHT_DECAY 0.01 are Scott's provisional values, 2026-08-13,
# falsifiable from the first training curve.

import argparse
import time
from pathlib import Path

import torch

from womd import loss, metrics, pipeline
from womd.model import MotionPredictor

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
LOG_EVERY_BATCHES = 20

GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)


def parameter_groups(predictor):
    decayed = []
    undecayed = []
    for name, parameter in predictor.named_parameters():
        if parameter.ndim >= 2 and not name.endswith("queries"):
            decayed.append(parameter)
        else:
            undecayed.append(parameter)
    return [
        {"params": decayed, "weight_decay": WEIGHT_DECAY},
        {"params": undecayed, "weight_decay": 0.0},
    ]


def train_epoch(predictor, optimizer, batches, device, gradient_scaler):
    accumulator = metrics.MetricAccumulator()
    loss_sums = {"total": 0.0, "regression": 0.0, "classification": 0.0}
    seconds = {"data_wait": 0.0, "step": 0.0, "monitor": 0.0}
    batch_count = 0
    sample_count = 0
    polyline_slots = 0
    wait_start = time.perf_counter()
    for batch in batches:
        seconds["data_wait"] += time.perf_counter() - wait_start

        step_start = time.perf_counter()
        batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
        with torch.amp.autocast(device_type=device.type, enabled=gradient_scaler.is_enabled()):
            trajectories, confidence_logits = predictor(batch)
            total, regression, classification = loss.prediction_loss(
                trajectories, confidence_logits, batch["future_positions"], batch["future_mask"]
            )
        optimizer.zero_grad()
        gradient_scaler.scale(total).backward()
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        loss_sums["total"] += float(total.detach())
        loss_sums["regression"] += float(regression.detach())
        loss_sums["classification"] += float(classification.detach())
        seconds["step"] += time.perf_counter() - step_start

        monitor_start = time.perf_counter()
        with torch.no_grad():
            accumulator.update(
                trajectories.detach().float(), confidence_logits.detach().float(),
                batch["future_positions"], batch["future_mask"],
            )
        seconds["monitor"] += time.perf_counter() - monitor_start

        batch_count += 1
        sample_count += batch["agent_history"].shape[0]
        polyline_slots += int(batch["max_polylines_in_batch"])
        if batch_count % LOG_EVERY_BATCHES == 0:
            elapsed = sum(seconds.values())
            monitor = accumulator.results()
            peak_gigabytes = (
                torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            )
            print(
                f"  batch {batch_count} | loss {loss_sums['total'] / batch_count:.4f} "
                f"(reg {loss_sums['regression'] / batch_count:.4f}) | "
                f"minADE {monitor['min_ade']:.3f} minFDE {monitor['min_fde']:.3f} | "
                f"{sample_count / elapsed:.1f} samples/s | "
                f"wait {100 * seconds['data_wait'] / elapsed:.0f}% "
                f"step {100 * seconds['step'] / elapsed:.0f}% "
                f"monitor {100 * seconds['monitor'] / elapsed:.0f}% | "
                f"polylines/sample {polyline_slots / batch_count:.0f} | "
                f"peak {peak_gigabytes:.1f} GB",
                flush=True,
            )
        wait_start = time.perf_counter()
    averages = {name: value / max(batch_count, 1) for name, value in loss_sums.items()}
    return averages, accumulator.results(), seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--all-eligible-agents", action="store_true")
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = MotionPredictor().to(device)
    optimizer = torch.optim.AdamW(parameter_groups(predictor), lr=LEARNING_RATE)
    gradient_scaler = GradScaler(enabled=arguments.mixed_precision and device.type == "cuda")
    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))
    assert scenario_paths, f"no .npz scenarios in {arguments.staged_directory}"

    for epoch_index in range(arguments.epochs):
        batches = pipeline.batches(
            scenario_paths, arguments.workers, arguments.batch_size,
            arguments.prefetch, arguments.seed + epoch_index,
            not arguments.all_eligible_agents,
        )
        averages, monitor, seconds = train_epoch(
            predictor, optimizer, batches, device, gradient_scaler
        )
        print(
            f"epoch {epoch_index + 1}/{arguments.epochs} | "
            f"loss {averages['total']:.4f} (reg {averages['regression']:.4f}"
            f" + cls {averages['classification']:.4f}) | "
            f"minADE {monitor['min_ade']:.4f} | minFDE {monitor['min_fde']:.4f} | "
            f"data_wait {seconds['data_wait']:.0f} s · step {seconds['step']:.0f} s"
            f" · monitor {seconds['monitor']:.0f} s",
            flush=True,
        )
        partial_path = arguments.checkpoint_path.with_suffix(arguments.checkpoint_path.suffix + ".partial")
        torch.save(
            {
                "model_state": predictor.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch_index": epoch_index,
                "seed": arguments.seed,
            },
            partial_path,
        )
        partial_path.replace(arguments.checkpoint_path)


if __name__ == "__main__":
    main()
