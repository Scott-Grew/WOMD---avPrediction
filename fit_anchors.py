import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from womd import loader, model

MAXIMUM_ITERATIONS = 100


def budget_fraction_endpoints(scenario_paths):
    agent_histories = []
    logged_endpoints = []
    for scenario_path in scenario_paths:
        scenario_array = loader.read_scenario(scenario_path)
        eligible = loader.eligible_track_indices(
            scenario_array["track_rows"],
            scenario_array["track_valid"],
            scenario_array["is_designated_target"],
            True,
        )
        for track_index in eligible:
            sample = loader.build_sample(scenario_array, int(track_index))
            valid_future_steps = np.flatnonzero(sample["future_mask"])
            if valid_future_steps.size == 0:
                continue
            agent_histories.append(sample["agent_history"])
            logged_endpoints.append(sample["future_positions"][valid_future_steps[-1]])

    agent_history = torch.from_numpy(np.stack(agent_histories))
    endpoints = torch.from_numpy(np.stack(logged_endpoints))
    return endpoints / model.agent_reachable_distance(agent_history)[:, None]


def endpoints_per_centre(assignment, centre_count):
    counts = torch.zeros(centre_count, dtype=torch.long)
    return counts.index_add_(0, assignment, torch.ones_like(assignment))


def move_centres_to_assigned_means(endpoints, assignment, centres):
    totals = torch.zeros_like(centres).index_add_(0, assignment, endpoints)
    counts = endpoints_per_centre(assignment, centres.shape[0]).to(endpoints.dtype)
    means = totals / counts.clamp_min(1.0).unsqueeze(-1)
    return torch.where(counts.unsqueeze(-1) > 0.0, means, centres), counts


def reseed_empty_centres(endpoints, assignment, centres, counts):
    empty_centres = (counts == 0.0).nonzero(as_tuple=True)[0]
    if empty_centres.numel() == 0:
        return centres
    own_centre_distances = (endpoints - centres[assignment]).norm(dim=-1)
    furthest = torch.argsort(own_centre_distances, descending=True, stable=True)
    reseeded = centres.clone()
    reseeded[empty_centres] = endpoints[furthest[: empty_centres.numel()]]
    return reseeded


def fit_unit_anchors(endpoints):
    centres = model.unit_anchor_offsets()
    initial_assignment = torch.cdist(endpoints, centres).argmin(dim=1)
    assignment = initial_assignment
    for iteration_count in range(1, MAXIMUM_ITERATIONS + 1):
        centres, counts = move_centres_to_assigned_means(endpoints, assignment, centres)
        centres = reseed_empty_centres(endpoints, assignment, centres, counts)
        next_assignment = torch.cdist(endpoints, centres).argmin(dim=1)
        if torch.equal(next_assignment, assignment):
            return centres, next_assignment, initial_assignment, iteration_count, True
        assignment = next_assignment
    return centres, assignment, initial_assignment, MAXIMUM_ITERATIONS, False


def minimum_pairwise_distance(anchors):
    separations = torch.cdist(anchors, anchors)
    separations.fill_diagonal_(float("inf"))
    return float(separations.min())


def main():
    staged_directory = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    scenario_paths = sorted(staged_directory.glob("*.npz"))

    start_seconds = time.perf_counter()
    endpoints = budget_fraction_endpoints(scenario_paths)
    (
        centres, assignment, initial_assignment, iteration_count, stopped_by_convergence
    ) = fit_unit_anchors(endpoints)
    elapsed_seconds = time.perf_counter() - start_seconds

    sample_count = endpoints.shape[0]
    fitted_counts = endpoints_per_centre(assignment, model.QUERY_COUNT)
    share_order = torch.argsort(fitted_counts, descending=True, stable=True)
    fitted_anchors = centres[share_order]
    fitted_counts = fitted_counts[share_order]
    np.savez(output_path, unit_anchors=fitted_anchors.numpy().astype(np.float32))

    geometric_fan = model.unit_anchor_offsets()
    fan_counts = endpoints_per_centre(initial_assignment, model.QUERY_COUNT)

    print(f"scenarios {len(scenario_paths)} | samples {sample_count} | {elapsed_seconds:.1f} s")
    print(
        f"k-means ran {iteration_count} iterations, stopped by "
        f"{'an unchanged assignment' if stopped_by_convergence else f'the {MAXIMUM_ITERATIONS} iteration cap'}"
    )
    print(f"wrote {output_path}, most-used anchor first")
    print()

    print(f"{'anchor':>7}{'x':>10}{'y':>10}{'distance':>10}{'angle deg':>11}{'share':>9}")
    for anchor_index in range(model.QUERY_COUNT):
        offset_x, offset_y = fitted_anchors[anchor_index].tolist()
        print(
            f"{anchor_index:>7}{offset_x:>10.3f}{offset_y:>10.3f}"
            f"{math.hypot(offset_x, offset_y):>10.3f}"
            f"{math.degrees(math.atan2(offset_y, offset_x)):>11.1f}"
            f"{int(fitted_counts[anchor_index]) / sample_count:>9.1%}"
        )
    print()

    print(f"{'':<28}{'geometric fan':>16}{'fitted':>12}")
    print(
        f"{'anchors with an endpoint':<28}{int((fan_counts > 0).sum()):>16}"
        f"{int((fitted_counts > 0).sum()):>12}"
    )
    print(
        f"{'largest single share':<28}{int(fan_counts.max()) / sample_count:>16.1%}"
        f"{int(fitted_counts.max()) / sample_count:>12.1%}"
    )
    print(
        f"{'minimum pairwise distance':<28}{minimum_pairwise_distance(geometric_fan):>16.3f}"
        f"{minimum_pairwise_distance(fitted_anchors):>12.3f}"
    )


if __name__ == "__main__":
    main()
