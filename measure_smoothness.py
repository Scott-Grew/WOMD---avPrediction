# > Path-SHAPE measurement: the model's endpoints can be right while its route to them is wrong, and
# no distance-to-the-logged-track metric can tell the difference. Three questions, one script.
# 1. What shape is real driving? Path length against the straight line to its own endpoint - "wander"
#    - plus the per-step turn angle, over the logged futures.
# 2. What shape does the model draw? The same statistics over its predictions, so every line is a
#    ratio against the logged data and nothing here is a threshold somebody chose.
#    Wander is reported at STRIDES, which is what separates the two failures it could be: decimating
#    the trajectory and re-measuring leaves genuine curvature alone and deletes step-to-step jitter,
#    so a wander that collapses as the stride grows was noise and one that does not was a route.
# 3. What can a restricted trajectory representation still express? Least-squares reconstruction of
#    the logged futures themselves, per control-parameter count, which is the error floor any such
#    representation would impose. Bernstein, monomial and clamped-B-spline forms of one degree span
#    the same polynomials, so the polynomial rows stand for all three; the kinematic rows integrate a
#    speed and a yaw-rate profile instead, and are fitted by descent because that family is not
#    linear in its coefficients.
# Only tracks whose whole 8 s future is logged are measured: a partial track's path length is missing
# the segments it did not record, which reads as an agent that drove less than it did.
# Run: python3 measure_smoothness.py ../data/staged [checkpoint.pt ../data/training_anchors.npz]

import sys
from pathlib import Path

import numpy as np
import torch

from womd import contract, loader, model, pipeline

STRIDES = (1, 2, 4, 8, 16)
POLYNOMIAL_DEGREES = (2, 3, 4, 5, 6, 8, 12)
KINEMATIC_DEGREES = (3, 5, 8, 12)
KINEMATIC_FIT_ITERATIONS = 6000


def polyline_from_origin(positions):
    origin = torch.zeros(*positions.shape[:-2], 1, 2, dtype=positions.dtype)
    return torch.cat([origin, positions], dim=-2)


def path_length_at_stride(polyline, stride):
    kept = polyline[..., ::stride, :]
    if (polyline.shape[-2] - 1) % stride != 0:
        kept = torch.cat([kept, polyline[..., -1:, :]], dim=-2)
    return kept.diff(dim=-2).norm(dim=-1).sum(dim=-1)


def turn_angles_degrees(polyline):
    steps = polyline.diff(dim=-2)
    leading, trailing = steps[..., :-1, :], steps[..., 1:, :]
    cross = leading[..., 0] * trailing[..., 1] - leading[..., 1] * trailing[..., 0]
    return torch.atan2(cross.abs(), (leading * trailing).sum(dim=-1)).rad2deg()


def shape_statistics(polyline):
    endpoint_distance = polyline[..., -1, :].norm(dim=-1)
    statistics = {
        "path length (m)": path_length_at_stride(polyline, 1),
        "endpoint distance (m)": endpoint_distance,
    }
    for stride in STRIDES:
        statistics[f"wander at stride {stride}"] = (
            path_length_at_stride(polyline, stride) / endpoint_distance.clamp_min(1e-6)
        )
    return statistics


def report(title, rows):
    print(f"\n{title}")
    print(f"{'quantity':30s}{'mean':>10s}{'p50':>10s}{'p90':>10s}{'max':>10s}")
    for name, values in rows.items():
        values = np.asarray(values)
        print(f"{name:30s}{values.mean():10.3f}{np.percentile(values, 50):10.3f}"
              f"{np.percentile(values, 90):10.3f}{values.max():10.3f}")


def complete_logged_futures(staged_directory):
    futures = []
    for scenario_path in sorted(Path(staged_directory).glob("*.npz")):
        scenario_array = loader.read_scenario(scenario_path)
        track_rows = scenario_array["track_rows"]
        track_valid = scenario_array["track_valid"]
        for track_index in loader.eligible_track_indices(
            track_rows, track_valid, scenario_array["is_designated_target"], True
        ):
            if not track_valid[track_index, contract.CURRENT_STEP_INDEX + 1:].all():
                continue
            origin, heading = loader.sample_frame(track_rows, track_index)
            agent_track = loader.track_rows_to_agent_frame(track_rows[track_index], origin, heading)
            futures.append(
                agent_track[contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION]
            )
    return torch.tensor(np.stack(futures), dtype=torch.float64)


def mean_step_error(reconstructed, logged):
    return (reconstructed - logged).norm(dim=-1).mean(dim=-1)


def polynomial_reconstruction(logged, degree):
    time_fraction = (
        torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float64) / contract.FUTURE_STEPS
    )
    basis = torch.stack([time_fraction ** power for power in range(1, degree + 1)], dim=-1)
    targets = logged.permute(1, 0, 2).reshape(contract.FUTURE_STEPS, -1)
    fitted = basis @ torch.linalg.lstsq(basis, targets).solution
    return fitted.reshape(contract.FUTURE_STEPS, -1, 2).permute(1, 0, 2)


def integrate_speed_and_yaw_rate(speed_coefficients, yaw_rate_coefficients, basis):
    heading = torch.cumsum(basis @ yaw_rate_coefficients.T, dim=0) * contract.TIMESTEP_SECONDS
    step_lengths = (basis @ speed_coefficients.T) * contract.TIMESTEP_SECONDS
    steps = torch.stack([heading.cos(), heading.sin()], dim=-1) * step_lengths.unsqueeze(-1)
    return torch.cumsum(steps, dim=0).permute(1, 0, 2)


def kinematic_reconstruction(logged, degree):
    time_fraction = (
        torch.arange(1, contract.FUTURE_STEPS + 1, dtype=torch.float64) / contract.FUTURE_STEPS
    )
    basis = torch.stack([time_fraction ** power for power in range(degree + 1)], dim=-1)
    steps = torch.cat([logged[:, :1], logged.diff(dim=1)], dim=1)
    logged_direction = torch.atan2(steps[..., 1], steps[..., 0])
    unwrapped = torch.from_numpy(np.unwrap(logged_direction.numpy(), axis=-1))
    logged_yaw_rate = torch.cat(
        [unwrapped[:, :1], unwrapped.diff(dim=1)], dim=1
    ) / contract.TIMESTEP_SECONDS
    speed_coefficients = torch.linalg.lstsq(
        basis, (steps.norm(dim=-1) / contract.TIMESTEP_SECONDS).T
    ).solution.T.clone().requires_grad_(True)
    yaw_rate_coefficients = torch.linalg.lstsq(
        basis, logged_yaw_rate.T
    ).solution.T.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([speed_coefficients, yaw_rate_coefficients], lr=0.2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, KINEMATIC_FIT_ITERATIONS)
    for _ in range(KINEMATIC_FIT_ITERATIONS):
        optimizer.zero_grad()
        reconstructed = integrate_speed_and_yaw_rate(
            speed_coefficients, yaw_rate_coefficients, basis
        )
        mean_step_error(reconstructed, logged).mean().backward()
        optimizer.step()
        schedule.step()
    with torch.no_grad():
        return integrate_speed_and_yaw_rate(speed_coefficients, yaw_rate_coefficients, basis)


def report_reconstruction(logged):
    print(f"\nleast-squares reconstruction of the same {logged.shape[0]} logged futures")
    print("error = mean Euclidean distance over the 80 steps, metres, the form minADE takes")
    print(f"{'representation':32s}{'params':>7s}{'mean':>10s}{'p50':>10s}{'p90':>10s}{'max':>10s}")
    families = [
        (f"polynomial degree {degree}", 2 * degree, polynomial_reconstruction(logged, degree))
        for degree in POLYNOMIAL_DEGREES
    ] + [
        (f"kinematic speed+yaw degree {degree}", 2 * (degree + 1),
         kinematic_reconstruction(logged, degree))
        for degree in KINEMATIC_DEGREES
    ] + [
        ("free positions or displacements", 2 * contract.FUTURE_STEPS, logged)
    ]
    for name, parameter_count, reconstructed in families:
        errors = mean_step_error(reconstructed, logged).numpy()
        print(f"{name:32s}{parameter_count:7d}{errors.mean():10.3f}"
              f"{np.percentile(errors, 50):10.3f}{np.percentile(errors, 90):10.3f}"
              f"{errors.max():10.3f}")


def predicted_shape(checkpoint_path, anchors_path, staged_directory):
    with np.load(anchors_path) as anchors_file:
        unit_anchors = torch.from_numpy(anchors_file["unit_anchors"])
    predictor = model.MotionPredictor(unit_anchors)
    predictor.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state"])
    predictor.eval()

    rows, turns, budget_use = [], [], []
    for batch in pipeline.batches(sorted(staged_directory.glob("*.npz")), 0, 16, 0, 0, True):
        complete = batch["future_mask"].all(dim=-1)
        if not complete.any():
            continue
        with torch.no_grad():
            trajectories, confidence_logits = predictor(batch)
            pruned, _ = model.prune_modes_batched(
                trajectories[complete], confidence_logits[complete]
            )
        logged = batch["future_positions"][complete]
        nearest_mode = (pruned - logged.unsqueeze(1)).norm(dim=-1).mean(dim=-1).argmin(dim=-1)
        scored = polyline_from_origin(pruned).gather(
            1, nearest_mode[:, None, None, None].expand(-1, -1, contract.FUTURE_STEPS + 1, 2)
        ).squeeze(1)
        rows.append(shape_statistics(scored))
        turns.append(turn_angles_degrees(scored).flatten())
        budget_use.append(
            path_length_at_stride(polyline_from_origin(pruned), 1).amax(dim=1)
            / model.agent_reachable_distance(batch["agent_history"][complete])
        )
    joined = {name: torch.cat([row[name] for row in rows]).numpy() for name in rows[0]}
    return joined, torch.cat(turns).numpy(), torch.cat(budget_use).numpy()


def main():
    staged_directory = Path(sys.argv[1])
    logged = complete_logged_futures(staged_directory)
    logged_polyline = polyline_from_origin(logged)
    logged_turns = turn_angles_degrees(logged_polyline).flatten().numpy()
    print(f"{staged_directory}: {logged.shape[0]} designated targets with a complete 8 s track")
    report("LOGGED futures", {name: values.numpy() for name, values in
                              shape_statistics(logged_polyline).items()})

    turn_rows = {"logged": logged_turns}
    if len(sys.argv) > 2:
        predicted, predicted_turns, budget_use = predicted_shape(
            Path(sys.argv[2]), Path(sys.argv[3]), staged_directory
        )
        report(f"PREDICTED best of the 6 emitted, {sys.argv[2]}", predicted)
        turn_rows["predicted"] = predicted_turns
        print(f"\npath / reachable budget over all 6 emitted modes: "
              f"p50 {np.percentile(budget_use, 50):.3f} p90 {np.percentile(budget_use, 90):.3f}"
              f" max {budget_use.max():.3f}")

    print(f"\n{'per-step turn angle (deg)':30s}{'mean':>10s}{'p50':>10s}{'p90':>10s}{'p99':>10s}")
    for name, turns in turn_rows.items():
        print(f"{name:30s}{turns.mean():10.3f}{np.percentile(turns, 50):10.3f}"
              f"{np.percentile(turns, 90):10.3f}{np.percentile(turns, 99):10.3f}")

    report_reconstruction(logged)


if __name__ == "__main__":
    main()
