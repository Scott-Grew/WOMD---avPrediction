import torch

from womd import contract

DRIVABLE_KINDS = ("lane", "driveway")
DRIVABLE_KIND_INDICES = tuple(
    contract.MAP_POLYLINE_KINDS.index(kind) for kind in DRIVABLE_KINDS
)


def drivable_point_mask(map_polylines, map_polylines_mask):
    kinds = map_polylines[..., contract.MAP_KIND].argmax(dim=-1)
    drivable = torch.zeros_like(map_polylines_mask)
    for kind_index in DRIVABLE_KIND_INDICES:
        drivable |= kinds == kind_index
    return map_polylines_mask & drivable


def crop_radius(map_polylines, map_polylines_mask):
    radius = torch.linalg.vector_norm(map_polylines[..., contract.MAP_POSITION], dim=-1)
    radius = radius.masked_fill(~map_polylines_mask, float("inf"))
    polyline_nearest = radius.min(dim=-1).values
    populated = map_polylines_mask.any(dim=-1)
    return polyline_nearest.masked_fill(~populated, -float("inf")).max(dim=-1).values


def distances_to_drivable(positions, map_polylines, map_polylines_mask):
    batch_size = positions.shape[0]
    queries = positions.reshape(batch_size, -1, 2)
    points = map_polylines[..., contract.MAP_POSITION].flatten(1, 2)
    drivable = drivable_point_mask(map_polylines, map_polylines_mask).flatten(1, 2)
    distances = torch.cdist(queries, points).masked_fill(
        ~drivable.unsqueeze(1), float("inf")
    )
    return distances.min(dim=-1).values.reshape(positions.shape[:-1])


def covered_samples(positions, map_polylines, map_polylines_mask):
    radius = crop_radius(map_polylines, map_polylines_mask)
    step_radius = torch.linalg.vector_norm(positions, dim=-1).reshape(
        positions.shape[0], -1
    )
    return (step_radius <= radius.unsqueeze(1)).all(dim=1) & torch.isfinite(radius)


def rate_and_coverage(
    trajectories,
    map_polylines,
    map_polylines_mask,
    horizon_steps,
    distance_limit_metres,
):
    horizon = trajectories[..., :horizon_steps, :]
    distances = distances_to_drivable(horizon, map_polylines, map_polylines_mask)
    exceeded = distances > distance_limit_metres
    rate = exceeded.flatten(start_dim=1).to(trajectories.dtype).mean(dim=1)
    reachable = torch.isfinite(distances).flatten(start_dim=1).any(dim=1)
    covered = covered_samples(horizon, map_polylines, map_polylines_mask) & reachable
    return rate, covered
