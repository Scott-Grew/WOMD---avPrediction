# > Training entry point: recipe v1 - plain AdamW, constant rate, no schedule, no clipping
# The epoch loop computes the loss components and the minADE/minFDE monitor ONLY. Weight decay
# skips biases, LayerNorms and the 64 learned queries - parameters whose job is to become
# non-zero, which decay would fight (the 2026-08-06 optimizer incident, applied forward).
# LEARNING_RATE 1e-3 and WEIGHT_DECAY 0.01 are Scott's provisional values, 2026-08-13,
# falsifiable from the first training curve. --heading-loss-weight and --offroad-loss-weight carry
# no default: an auxiliary term's weight is set per run and never inherited from the code.
# ade_80step / fde_80step are NOT the leaderboard's minADE / minFDE and are named so nobody can
# read them as such: they average over all 80 valid future steps at 10 Hz, while a WOMD submission
# is 16 steps at 2 Hz scored by Waymo's own referee, which lives in the container and does not
# exist yet. The computation is the monitor's, unchanged; only the printed names say what it is.
# Every heartbeat number is printed twice, cumulative from epoch start and over the batches since
# the last heartbeat: at tens of thousands of batches a total collapse in the last tenth of an
# epoch moves the cumulative mean by well under a percent, which reads exactly like convergence.
# The counters beside them exist because a run can rot without moving any mean at all - a
# non-finite total, or a GradScaler skipping the step because the gradient was not finite, both
# leave the printed loss curve looking healthy. They are COUNTED and printed, never thresholded.

import argparse
import math
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


def save_checkpoint(checkpoint_path, previous_checkpoint_path, checkpoint_state):
    partial_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
    torch.save(checkpoint_state, partial_path)
    if checkpoint_path.exists():
        checkpoint_path.replace(previous_checkpoint_path)
    partial_path.replace(checkpoint_path)


def train_epoch(
    predictor, optimizer, batches, device, gradient_scaler, heading_loss_weight,
    offroad_loss_weight, checkpoint_path, previous_checkpoint_path, checkpoint_every_seconds,
    epoch_index, seed,
):
    accumulator = metrics.MetricAccumulator()
    window_accumulator = metrics.MetricAccumulator()
    loss_sums = {
        "total": 0.0, "regression": 0.0, "classification": 0.0, "heading": 0.0, "offroad": 0.0,
    }
    window_loss_sums = dict.fromkeys(loss_sums, 0.0)
    seconds = {"data_wait": 0.0, "step": 0.0, "monitor": 0.0}
    batch_count = 0
    sample_count = 0
    polyline_slots = 0
    window_peak_tokens = 0
    non_finite_total_count = 0
    gradient_scaler_skip_count = 0
    wait_start = time.perf_counter()
    checkpoint_wait_start = time.perf_counter()
    for batch in batches:
        seconds["data_wait"] += time.perf_counter() - wait_start

        step_start = time.perf_counter()
        batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
        with torch.amp.autocast(device_type=device.type, enabled=gradient_scaler.is_enabled()):
            trajectories, heading_cosine_sine, confidence_logits = predictor.predict_with_heading(batch)
            total, regression, classification, heading, offroad = loss.prediction_loss(
                trajectories, heading_cosine_sine, confidence_logits,
                batch["future_positions"], batch["future_headings"], batch["future_mask"],
                batch["drivable_positions"], batch["drivable_mask"],
                heading_loss_weight, offroad_loss_weight,
            )
        optimizer.zero_grad()
        gradient_scaler.scale(total).backward()
        gradient_scaler.step(optimizer)
        scale_before_update = gradient_scaler.get_scale()
        gradient_scaler.update()
        gradient_scaler_skip_count += int(gradient_scaler.get_scale() < scale_before_update)
        component_values = {
            "total": float(total.detach()),
            "regression": float(regression.detach()),
            "classification": float(classification.detach()),
            "heading": float(heading.detach()),
            "offroad": float(offroad.detach()),
        }
        for name, value in component_values.items():
            loss_sums[name] += value
            window_loss_sums[name] += value
        non_finite_total_count += int(not math.isfinite(component_values["total"]))
        seconds["step"] += time.perf_counter() - step_start

        monitor_start = time.perf_counter()
        with torch.no_grad():
            accumulator.update(
                trajectories.detach().float(), confidence_logits.detach().float(),
                batch["future_positions"], batch["future_mask"],
            )
            window_accumulator.update(
                trajectories.detach().float(), confidence_logits.detach().float(),
                batch["future_positions"], batch["future_mask"],
            )
        seconds["monitor"] += time.perf_counter() - monitor_start

        batch_count += 1
        sample_count += batch["agent_history"].shape[0]
        chunk_slots = int(batch["max_polylines_in_batch"])
        polyline_slots += chunk_slots
        window_peak_tokens = max(
            window_peak_tokens, 1 + batch["neighbour_history"].shape[1] + chunk_slots
        )
        if batch_count % LOG_EVERY_BATCHES == 0:
            elapsed = sum(seconds.values())
            monitor = accumulator.results()
            window_monitor = window_accumulator.results()
            median_heading_norm = float(
                heading_cosine_sine.detach().float().norm(dim=-1).median()
            )
            peak_gigabytes = (
                torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            )
            print(
                f"  batch {batch_count} | loss {loss_sums['total'] / batch_count:.4f} "
                f"(window {window_loss_sums['total'] / LOG_EVERY_BATCHES:.4f}) "
                f"reg {loss_sums['regression'] / batch_count:.4f} "
                f"(window {window_loss_sums['regression'] / LOG_EVERY_BATCHES:.4f}) | "
                f"ade_80step {monitor['min_ade']:.3f} (window {window_monitor['min_ade']:.3f}) "
                f"fde_80step {monitor['min_fde']:.3f} (window {window_monitor['min_fde']:.3f}) | "
                f"kept modes {window_monitor['mean_kept_modes']:.2f} "
                f"backfilled {100 * window_monitor['backfill_rate']:.0f}% | "
                f"hdg/reg {heading_loss_weight * window_loss_sums['heading'] / window_loss_sums['regression']:.4f} "
                f"off/reg {offroad_loss_weight * window_loss_sums['offroad'] / window_loss_sums['regression']:.4f} "
                f"hdg norm {median_heading_norm:.4f} | "
                f"non-finite {non_finite_total_count} skipped steps {gradient_scaler_skip_count} | "
                f"{sample_count / elapsed:.1f} samples/s | "
                f"wait {100 * seconds['data_wait'] / elapsed:.0f}% "
                f"step {100 * seconds['step'] / elapsed:.0f}% "
                f"monitor {100 * seconds['monitor'] / elapsed:.0f}% | "
                f"polylines/sample {polyline_slots / batch_count:.0f} "
                f"peak tokens {window_peak_tokens} | "
                f"peak {peak_gigabytes:.1f} GB",
                flush=True,
            )
            window_accumulator = metrics.MetricAccumulator()
            window_loss_sums = dict.fromkeys(loss_sums, 0.0)
            window_peak_tokens = 0
        if time.perf_counter() - checkpoint_wait_start >= checkpoint_every_seconds:
            if math.isfinite(component_values["total"]):
                save_checkpoint(
                    checkpoint_path, previous_checkpoint_path,
                    {
                        "model_state": predictor.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch_index": epoch_index,
                        "batch_index": batch_count,
                        "seed": seed,
                    },
                )
            else:
                print(
                    f"batch {batch_count} total loss {component_values['total']},"
                    f" checkpoint {checkpoint_path} left as it was",
                    flush=True,
                )
            checkpoint_wait_start = time.perf_counter()
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
    parser.add_argument("--heading-loss-weight", type=float, required=True)
    parser.add_argument("--offroad-loss-weight", type=float, required=True)
    parser.add_argument("--checkpoint-every-seconds", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
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
    previous_checkpoint_path = arguments.checkpoint_path.with_suffix(
        arguments.checkpoint_path.suffix + ".previous"
    )

    start_epoch_index = 0
    if arguments.resume and arguments.checkpoint_path.exists():
        checkpoint = torch.load(arguments.checkpoint_path, map_location=device)
        predictor.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch_index = checkpoint["epoch_index"] + 1

    for epoch_index in range(start_epoch_index, arguments.epochs):
        batches = pipeline.batches(
            scenario_paths, arguments.workers, arguments.batch_size,
            arguments.prefetch, arguments.seed + epoch_index,
            not arguments.all_eligible_agents,
        )
        averages, monitor, seconds = train_epoch(
            predictor, optimizer, batches, device, gradient_scaler, arguments.heading_loss_weight,
            arguments.offroad_loss_weight, arguments.checkpoint_path, previous_checkpoint_path,
            arguments.checkpoint_every_seconds, epoch_index, arguments.seed,
        )
        print(
            f"epoch {epoch_index + 1}/{arguments.epochs} | "
            f"loss {averages['total']:.4f} (reg {averages['regression']:.4f}"
            f" + cls {averages['classification']:.4f}"
            f" + hdg {averages['heading']:.4f}"
            f" + off {averages['offroad']:.4f}) | "
            f"ade_80step {monitor['min_ade']:.4f} | fde_80step {monitor['min_fde']:.4f} | "
            f"kept modes {monitor['mean_kept_modes']:.2f}"
            f" | backfilled {100 * monitor['backfill_rate']:.0f}% | "
            f"data_wait {seconds['data_wait']:.0f} s · step {seconds['step']:.0f} s"
            f" · monitor {seconds['monitor']:.0f} s",
            flush=True,
        )
        if not math.isfinite(averages["total"]):
            print(
                f"epoch {epoch_index + 1} mean total loss {averages['total']},"
                f" checkpoint {arguments.checkpoint_path} left as it was",
                flush=True,
            )
            continue
        save_checkpoint(
            arguments.checkpoint_path, previous_checkpoint_path,
            {
                "model_state": predictor.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch_index": epoch_index,
                "batch_index": None,
                "seed": arguments.seed,
            },
        )


if __name__ == "__main__":
    main()
