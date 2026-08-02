import torch

STEP_SECONDS = 0.1
SMOOTHING_WINDOW_STEPS = 5
MOVING_SPEED_METRES_PER_SECOND = 1.0

LONGITUDINAL_ACCELERATION_LIMIT = 3.71
LATERAL_ACCELERATION_LIMIT = 2.84
YAW_RATE_LIMIT = 0.51

DEFAULT_LIMITS = {
    "longitudinal": LONGITUDINAL_ACCELERATION_LIMIT,
    "lateral": LATERAL_ACCELERATION_LIMIT,
    "yaw_rate": YAW_RATE_LIMIT,
}


def _smooth(positions, window_steps):
    if window_steps <= 1:
        return positions
    leading_shape = positions.shape[:-2]
    steps, coordinates = positions.shape[-2:]
    flattened = positions.reshape(-1, steps, coordinates).transpose(-1, -2)
    padding = window_steps // 2
    padded = torch.nn.functional.pad(flattened, (padding, padding), mode="replicate")
    kernel = torch.full(
        (coordinates, 1, window_steps), 1.0 / window_steps, dtype=positions.dtype
    )
    smoothed = torch.nn.functional.conv1d(padded, kernel, groups=coordinates)
    return smoothed.transpose(-1, -2).reshape(*leading_shape, steps, coordinates)


def motion_quantities(positions, window_steps=SMOOTHING_WINDOW_STEPS):
    smoothed = _smooth(positions, window_steps)
    velocity = torch.diff(smoothed, dim=-2) / STEP_SECONDS
    acceleration = torch.diff(velocity, dim=-2) / STEP_SECONDS

    speed = torch.linalg.vector_norm(velocity, dim=-1)
    direction = velocity / speed.clamp(min=1e-6).unsqueeze(-1)
    longitudinal = (acceleration * direction[..., 1:, :]).sum(dim=-1)
    lateral = torch.linalg.vector_norm(
        acceleration - longitudinal.unsqueeze(-1) * direction[..., 1:, :], dim=-1
    )

    heading = torch.atan2(velocity[..., 1], velocity[..., 0])
    heading_change = torch.diff(heading, dim=-1)
    wrapped_change = torch.atan2(torch.sin(heading_change), torch.cos(heading_change))
    yaw_rate = wrapped_change / STEP_SECONDS
    moving = speed[..., 1:] > MOVING_SPEED_METRES_PER_SECOND
    edge_steps = window_steps // 2 + 1
    if edge_steps > 0:
        moving[..., :edge_steps] = False
        moving[..., -edge_steps:] = False
    return {
        "longitudinal": longitudinal.abs(),
        "lateral": lateral,
        "yaw_rate": yaw_rate.abs(),
    }, moving


def violation_rates(
    trajectories, window_steps=SMOOTHING_WINDOW_STEPS, limits=None
):
    limits = limits or DEFAULT_LIMITS
    quantities, moving = motion_quantities(trajectories, window_steps)
    considered = moving.flatten(start_dim=1).sum(dim=1).clamp(min=1)
    rates = {}
    for name, limit in limits.items():
        exceeded = (quantities[name] > limit) & moving
        rates[f"implausible_{name}"] = (
            exceeded.flatten(start_dim=1).sum(dim=1) / considered
        ).to(trajectories.dtype)
    return rates
