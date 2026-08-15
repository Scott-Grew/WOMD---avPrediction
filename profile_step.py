# > Per-component profile of one training iteration: where the loader time and the device step
# time actually go, at the granularity train.py's three-bucket heartbeat cannot reach. The loader
# phase builds one real batch of designated-target samples through womd.loader and times
# read_scenario, build_sample and collation; the public loader stages inside build_sample are
# then re-run in isolation on the same inputs, OUTSIDE the batch wall so they do not inflate it,
# which is how their cost becomes visible without editing loader.py. The device phase runs the
# real training iteration - host-to-device copy, forward, loss, backward, optimizer, minADE/minFDE
# monitor - and splits the forward with forward hooks on the model's own submodules plus a timing
# shim around womd.model.pool_dots_to_polyline_tokens, so model.py is untouched and the forward
# being measured is the forward that trains. Learning rate, weight decay, parameter groups and the
# gradient scaler are imported from train.py rather than restated.
# Every bracket calls torch.cuda.synchronize() before reading the clock when the device is cuda,
# because the queue is asynchronous and unsynchronised device numbers are fiction; serialising
# this way slightly inflates the measured total against a real unsynchronised step. On cpu no cuda
# call is made and no cuda number is reported.
# The first iteration is discarded - it pays CUDA context creation, cuDNN autotuning and allocator
# growth - and the number discarded is printed with the results.
# Percent is of the measured total, which is the loader wall plus the device iteration wall; those
# two rows sum to 100 and every other row is nested inside one of them. At each level the children
# plus an explicit unattributed row account for the parent, so nothing is hidden in a remainder.
# Run: python3 profile_step.py ../data/staged 4 20

import argparse
import collections
import functools
import itertools
import time
from pathlib import Path

import numpy as np
import torch

import train
from womd import contract, loader, loss, metrics, model, pipeline
from womd.model import MotionPredictor

DISCARDED_WARMUP_ITERATIONS = 1
POOLING_COMPONENT = "polyline pooling"
LOADER_WALL = "loader batch wall"
DEVICE_WALL = "device iteration wall"
LOADER_COMPONENTS = ("read_scenario", "build_sample", "collate (build_batch + from_numpy)")
DEVICE_COMPONENTS = (
    "host to device copy", "forward", "loss", "backward", "optimizer step", "monitor update"
)
BUILD_SAMPLE_STAGES = (
    "sample_frame", "agent track reframe", "neighbour history reframe",
    "lane_context_per_polyline", "crop_and_reframe_map"
)


class ComponentTimer:
    def __init__(self, device):
        self.device = device
        self.seconds = collections.defaultdict(float)
        self.started_at = {}

    def synchronise(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def start(self, component):
        self.synchronise()
        self.started_at[component] = time.perf_counter()

    def stop(self, component):
        self.synchronise()
        self.seconds[component] += time.perf_counter() - self.started_at[component]

    def reset(self):
        self.seconds.clear()


def begin_module_timing(timer, component, module, module_inputs):
    timer.start(component)


def end_module_timing(timer, component, module, module_inputs, module_outputs):
    timer.stop(component)


def install_module_timers(timer, named_modules):
    for component, module in named_modules:
        module.register_forward_pre_hook(functools.partial(begin_module_timing, timer, component))
        module.register_forward_hook(functools.partial(end_module_timing, timer, component))


def install_pooling_timer(timer):
    untimed_pooling = model.pool_dots_to_polyline_tokens

    def timed_pooling(*pooling_arguments):
        timer.start(POOLING_COMPONENT)
        pooled = untimed_pooling(*pooling_arguments)
        timer.stop(POOLING_COMPONENT)
        return pooled

    model.pool_dots_to_polyline_tokens = timed_pooling


def forward_submodules(predictor):
    scene_encoder = predictor.scene_encoder
    decoder = predictor.mode_decoder
    named_modules = [
        ("agent history encoder", scene_encoder.agent_encoder),
        ("neighbour history encoder", scene_encoder.neighbour_encoder),
        ("map dot encoder", scene_encoder.map_encoder),
    ]
    for round_index, attention_layer in enumerate(scene_encoder.layers):
        named_modules.append((f"scene attention round {round_index + 1}", attention_layer))
    named_modules.extend(
        [
            ("decoder scene norm", decoder.scene_norm),
            ("decoder cross attention", decoder.cross_attention),
            ("decoder trajectory head", decoder.trajectory_head),
            ("decoder confidence head", decoder.confidence_head),
        ]
    )
    return named_modules


def batch_stream(timer, scenario_paths, batch_size):
    samples = []
    sample_sources = []
    for scenario_path in itertools.cycle(scenario_paths):
        timer.start("read_scenario")
        scenario_arrays = loader.read_scenario(scenario_path)
        timer.stop("read_scenario")
        for track_index in loader.eligible_track_indices(
            scenario_arrays["track_rows"],
            scenario_arrays["track_valid"],
            scenario_arrays["is_designated_target"],
            True,
        ):
            timer.start("build_sample")
            samples.append(loader.build_sample(scenario_arrays, int(track_index)))
            timer.stop("build_sample")
            sample_sources.append((scenario_arrays, int(track_index)))
            if len(samples) < batch_size:
                continue
            timer.start("collate (build_batch + from_numpy)")
            collated = pipeline.collate_samples(samples)
            timer.stop("collate (build_batch + from_numpy)")
            yield collated, sample_sources
            samples = []
            sample_sources = []


def time_build_sample_stages(timer, scenario_arrays, track_index):
    track_rows = scenario_arrays["track_rows"]

    timer.start("sample_frame")
    origin, heading = loader.sample_frame(track_rows, track_index)
    timer.stop("sample_frame")

    timer.start("agent track reframe")
    loader.track_rows_to_agent_frame(track_rows[track_index], origin, heading)
    timer.stop("agent track reframe")

    neighbour_indices = np.flatnonzero(np.arange(len(track_rows)) != track_index)
    timer.start("neighbour history reframe")
    loader.track_rows_to_agent_frame(
        track_rows[neighbour_indices, :contract.HISTORY_STEPS], origin, heading
    )
    timer.stop("neighbour history reframe")

    now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
    speed = float(np.linalg.norm(now_row[contract.AGENT_VELOCITY]))
    timer.start("lane_context_per_polyline")
    polyline_lane_context = loader.lane_context_per_polyline(
        scenario_arrays,
        now_row[contract.AGENT_POSITION],
        now_row[contract.AGENT_HEADING_COSINE:contract.AGENT_HEADING_SINE + 1],
    )
    timer.stop("lane_context_per_polyline")

    timer.start("crop_and_reframe_map")
    loader.crop_and_reframe_map(
        scenario_arrays["map_rows"],
        scenario_arrays["map_dot_polyline_index"],
        scenario_arrays["polyline_signal_histories"],
        polyline_lane_context,
        origin,
        heading,
        speed,
    )
    timer.stop("crop_and_reframe_map")


def run_device_iteration(timer, predictor, optimizer, gradient_scaler, accumulator, batch, device):
    timer.start(DEVICE_WALL)

    timer.start("host to device copy")
    batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
    timer.stop("host to device copy")

    with torch.amp.autocast(device_type=device.type, enabled=gradient_scaler.is_enabled()):
        timer.start("forward")
        (
            trajectories, heading_cosine_sine, confidence_logits, reachable_distance_metres
        ) = predictor.predict_with_heading(batch)
        timer.stop("forward")

        timer.start("loss")
        total, _, _, _ = loss.prediction_loss(
            trajectories, heading_cosine_sine, confidence_logits,
            batch["future_positions"], batch["future_headings"], batch["future_mask"],
            predictor.unit_anchors, reachable_distance_metres, 1.0
        )
        timer.stop("loss")

    timer.start("optimizer step")
    optimizer.zero_grad()
    timer.stop("optimizer step")

    timer.start("backward")
    gradient_scaler.scale(total).backward()
    timer.stop("backward")

    timer.start("optimizer step")
    gradient_scaler.step(optimizer)
    gradient_scaler.update()
    timer.stop("optimizer step")

    timer.start("monitor update")
    with torch.no_grad():
        accumulator.update(
            trajectories.detach().float(),
            confidence_logits.detach().float(),
            batch["future_positions"],
            batch["future_mask"],
        )
    timer.stop("monitor update")

    timer.stop(DEVICE_WALL)


def print_component(label, seconds, iterations, measured_total, depth):
    print(
        f"{'  ' * depth + label:<40}{seconds:>10.3f}{1000.0 * seconds / iterations:>12.2f}"
        f"{100.0 * seconds / measured_total:>10.1f}"
    )


def report(timer, forward_components, batch_counts, iteration_walls, arguments, device):
    iterations = len(iteration_walls)
    measured_total = timer.seconds[LOADER_WALL] + timer.seconds[DEVICE_WALL]

    print(
        f"device {device.type} | mixed precision "
        f"{'on' if arguments.mixed_precision and device.type == 'cuda' else 'off'} | "
        f"batch size {arguments.batch_size} | measured iterations {iterations} | "
        f"warmup iterations discarded {DISCARDED_WARMUP_ITERATIONS}"
    )
    if device.type == "cuda":
        print(
            "every device bracket is wrapped in torch.cuda.synchronize(): the numbers are real, "
            "and serialising the queue inflates the total slightly against an unsynchronised step"
        )
    else:
        print("device is cpu: no cuda synchronisation performed, no cuda number reported")
    print(
        f"per batch: samples {batch_counts['samples'] / iterations:.1f} | map dots "
        f"{batch_counts['map dots'] / iterations:.0f} | polyline tokens "
        f"{batch_counts['polyline tokens'] / iterations:.0f}"
    )
    if device.type == "cuda":
        print(f"peak torch.cuda.max_memory_allocated {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    per_iteration_walls = np.diff(iteration_walls, prepend=0.0)
    print(
        f"measured wall per iteration: min {1000 * per_iteration_walls.min():.1f} ms | median "
        f"{1000 * np.median(per_iteration_walls):.1f} ms | max {1000 * per_iteration_walls.max():.1f} ms"
    )
    print()

    print(f"{'component':<40}{'total s':>10}{'ms/batch':>12}{'percent':>10}")
    print_component(LOADER_WALL, timer.seconds[LOADER_WALL], iterations, measured_total, 0)
    for component in LOADER_COMPONENTS:
        print_component(component, timer.seconds[component], iterations, measured_total, 1)
    loader_unattributed = timer.seconds[LOADER_WALL] - sum(
        timer.seconds[component] for component in LOADER_COMPONENTS
    )
    print_component("loader unattributed", loader_unattributed, iterations, measured_total, 1)

    print_component(DEVICE_WALL, timer.seconds[DEVICE_WALL], iterations, measured_total, 0)
    for component in DEVICE_COMPONENTS:
        print_component(component, timer.seconds[component], iterations, measured_total, 1)
        if component != "forward":
            continue
        for forward_component in forward_components:
            print_component(
                forward_component, timer.seconds[forward_component], iterations, measured_total, 2
            )
        attention_rounds = [
            name for name in forward_components if name.startswith("scene attention round")
        ]
        print_component(
            "scene attention rounds total (of the above)",
            sum(timer.seconds[name] for name in attention_rounds),
            iterations,
            measured_total,
            2,
        )
        forward_unattributed = timer.seconds["forward"] - sum(
            timer.seconds[name] for name in forward_components
        )
        print_component("forward unattributed", forward_unattributed, iterations, measured_total, 2)
    device_unattributed = timer.seconds[DEVICE_WALL] - sum(
        timer.seconds[component] for component in DEVICE_COMPONENTS
    )
    print_component("device unattributed", device_unattributed, iterations, measured_total, 1)
    print_component("measured total", measured_total, iterations, measured_total, 0)
    print()

    samples = batch_counts["samples"]
    build_sample_seconds = timer.seconds["build_sample"]
    print(
        f"build_sample stages re-run in isolation on the same inputs, outside the walls above - "
        f"{samples} samples"
    )
    print(f"{'stage':<40}{'total s':>10}{'ms/sample':>12}{'pct build_sample':>18}")
    for stage in BUILD_SAMPLE_STAGES:
        print(
            f"{stage:<40}{timer.seconds[stage]:>10.3f}"
            f"{1000.0 * timer.seconds[stage] / samples:>12.2f}"
            f"{100.0 * timer.seconds[stage] / build_sample_seconds:>18.1f}"
        )
    isolated_total = sum(timer.seconds[stage] for stage in BUILD_SAMPLE_STAGES)
    print(
        f"{'isolated stages total':<40}{isolated_total:>10.3f}"
        f"{1000.0 * isolated_total / samples:>12.2f}"
        f"{100.0 * isolated_total / build_sample_seconds:>18.1f}"
    )
    print(
        f"{'build_sample':<40}{build_sample_seconds:>10.3f}"
        f"{1000.0 * build_sample_seconds / samples:>12.2f}{100.0:>18.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("repeats", type=int)
    parser.add_argument("--mixed-precision", action="store_true")
    arguments = parser.parse_args()

    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))
    assert scenario_paths, f"no .npz scenarios in {arguments.staged_directory}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = MotionPredictor(model.unit_anchor_offsets()).to(device)
    optimizer = torch.optim.AdamW(train.parameter_groups(predictor), lr=train.LEARNING_RATE)
    gradient_scaler = train.GradScaler(
        enabled=arguments.mixed_precision and device.type == "cuda"
    )
    accumulator = metrics.MetricAccumulator()

    timer = ComponentTimer(device)
    named_modules = forward_submodules(predictor)
    install_module_timers(timer, named_modules)
    install_pooling_timer(timer)
    forward_components = [component for component, _ in named_modules]
    forward_components.insert(forward_components.index("map dot encoder") + 1, POOLING_COMPONENT)

    stream = batch_stream(timer, scenario_paths, arguments.batch_size)
    batch_counts = {"samples": 0, "map dots": 0, "polyline tokens": 0}
    iteration_walls = []

    for iteration_index in range(DISCARDED_WARMUP_ITERATIONS + arguments.repeats):
        timer.start(LOADER_WALL)
        batch, sample_sources = next(stream)
        timer.stop(LOADER_WALL)

        for scenario_arrays, track_index in sample_sources:
            time_build_sample_stages(timer, scenario_arrays, track_index)

        run_device_iteration(
            timer, predictor, optimizer, gradient_scaler, accumulator, batch, device
        )

        batch_counts["samples"] += batch["agent_history"].shape[0]
        batch_counts["map dots"] += batch["map_rows"].shape[0]
        batch_counts["polyline tokens"] += (
            batch["agent_history"].shape[0] * int(batch["max_polylines_in_batch"])
        )
        iteration_walls.append(timer.seconds[LOADER_WALL] + timer.seconds[DEVICE_WALL])

        if iteration_index + 1 == DISCARDED_WARMUP_ITERATIONS:
            timer.reset()
            batch_counts = dict.fromkeys(batch_counts, 0)
            iteration_walls.clear()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

    report(timer, forward_components, batch_counts, iteration_walls, arguments, device)


if __name__ == "__main__":
    main()
