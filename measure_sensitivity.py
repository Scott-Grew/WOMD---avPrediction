import sys
from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import torch

from womd import contract, pipeline
from womd.model import MotionPredictor, load_checkpoint_state

BATCH_SIZE = 32
FIXED_SEED = 0
CONTROL_NOISE_METRES = 1e-6
CONSTANT_NOTE = "could not perturb - already constant"

AGENT_FEATURE_SELECTORS = (
    ("AGENT_POSITION", contract.AGENT_POSITION),
    ("AGENT_HEADING_COSINE_SINE", slice(contract.AGENT_HEADING_COSINE, contract.AGENT_HEADING_SINE + 1)),
    ("AGENT_VELOCITY", contract.AGENT_VELOCITY),
    ("AGENT_DIMENSIONS", contract.AGENT_DIMENSIONS),
    ("AGENT_TYPE", contract.AGENT_TYPE),
    ("AGENT_IS_SDC", contract.AGENT_IS_SDC),
)

MAP_FEATURE_SELECTORS = (
    ("MAP_POSITION", contract.MAP_POSITION),
    ("MAP_DIRECTION", contract.MAP_DIRECTION),
    ("MAP_KIND", contract.MAP_KIND),
    ("MAP_LANE_TYPE", contract.MAP_LANE_TYPE),
    ("MAP_SPEED_LIMIT", contract.MAP_SPEED_LIMIT),
    ("MAP_BOUNDARY_TYPE", contract.MAP_BOUNDARY_TYPE),
    ("MAP_STOP_POINT", contract.MAP_STOP_POINT),
    ("MAP_LEFT_BOUNDARY_CROSSING", contract.MAP_LEFT_BOUNDARY_CROSSING),
    ("MAP_RIGHT_BOUNDARY_CROSSING", contract.MAP_RIGHT_BOUNDARY_CROSSING),
)

LANE_CONTEXT_SELECTORS = tuple(sorted(
    (
        (name, column)
        for name, column in vars(contract).items()
        if name.startswith("LANE_CONTEXT_") and name != "LANE_CONTEXT_DIM"
    ),
    key=lambda named_column: named_column[1],
))

PERTURBATIONS_SKIPPED_BY_DESIGN = (
    "map_dot_polyline_slot, zero: the column is a chunk index, so zero names chunk 0 rather than"
    " absence and would pool every dot of the batch into one token instead of deleting anything.",
    "max_polylines_in_batch, both kinds: a padded width the loader reports, not a value read off a"
    " row; changing it reshapes the token grid rather than perturbing a feature.",
)

INPUT_KEYS_WITHOUT_A_PERTURBATION = frozenset({"max_polylines_in_batch"})


def selector_columns(selector):
    if isinstance(selector, slice):
        return tuple(range(selector.start, selector.stop))
    return (selector,)


def assert_selectors_cover_every_column(selectors, feature_dim, layout_name):
    covered = sorted(column for _, selector in selectors for column in selector_columns(selector))
    assert covered == list(range(feature_dim)), (
        f"{layout_name} selectors cover columns {covered}, not every column of 0..{feature_dim - 1}"
    )


def unit_view(tensor, feature_dim_count):
    assert tensor.is_contiguous()
    if feature_dim_count == 0:
        return tensor.reshape(-1, 1)
    return tensor.reshape(-1, int(np.prod(tensor.shape[tensor.dim() - feature_dim_count:])))


def count_modified_units(original_units, modified_units):
    return int((original_units != modified_units).any(dim=-1).sum())


PerturbationTarget = namedtuple("PerturbationTarget", "tensor_key feature_dim_count live_units")


class KeyRecordingBatch(dict):
    def __init__(self, batch):
        super().__init__(batch)
        self.keys_read = set()

    def __getitem__(self, key):
        self.keys_read.add(key)
        return super().__getitem__(key)


def assert_every_key_the_forward_pass_reads_is_perturbed(keys_read, targets):
    covered = set(targets) | set(INPUT_KEYS_WITHOUT_A_PERTURBATION)
    assert keys_read == covered, (
        f"the forward pass reads {sorted(keys_read)} but the table accounts for {sorted(covered)}"
    )


def present_chunk_units(batch):
    batch_size, chunks_per_sample = batch["map_chunk_lane_context"].shape[:2]
    present = torch.zeros(batch_size * chunks_per_sample, dtype=torch.bool)
    present[batch["map_dot_polyline_slot"]] = True
    return present


def perturbation_targets(batch):
    present_chunks = present_chunk_units(batch)
    every_map_dot = torch.ones(batch["map_rows"].shape[0], dtype=torch.bool)
    return {
        "agent_history": PerturbationTarget(
            "agent_history", 1, batch["agent_history_mask"].reshape(-1)
        ),
        "agent_history_mask": PerturbationTarget(
            "agent_history_mask", 1, torch.ones(batch["agent_history_mask"].shape[0], dtype=torch.bool)
        ),
        "neighbour_history": PerturbationTarget(
            "neighbour_history", 1, batch["neighbour_history_mask"].reshape(-1)
        ),
        "neighbour_history_mask": PerturbationTarget(
            "neighbour_history_mask", 1, batch["neighbour_history_mask"].any(dim=-1).reshape(-1)
        ),
        "map_rows": PerturbationTarget("map_rows", 1, every_map_dot),
        "map_dot_polyline_slot": PerturbationTarget("map_dot_polyline_slot", 0, every_map_dot),
        "map_chunk_signal_history": PerturbationTarget("map_chunk_signal_history", 2, present_chunks),
        "map_chunk_lane_context": PerturbationTarget("map_chunk_lane_context", 1, present_chunks),
    }


def apply_column_perturbation(batch, target, columns, perturbation_kind, random_generator):
    original_units = unit_view(batch[target.tensor_key], target.feature_dim_count)
    live_indices = torch.nonzero(target.live_units, as_tuple=False).squeeze(-1)
    assert len(live_indices) > 0, f"{target.tensor_key} has no live rows to perturb"
    column_indices = torch.tensor(columns, dtype=torch.long)
    live_values = original_units[live_indices.unsqueeze(-1), column_indices.unsqueeze(0)]

    if perturbation_kind == "zero":
        if not bool(live_values.any()):
            return None
        replacement = torch.zeros_like(live_values)
    else:
        if bool((live_values == live_values[0]).all()):
            return None
        replacement = live_values[torch.randperm(len(live_indices), generator=random_generator)]

    modified_tensor = batch[target.tensor_key].clone()
    modified_units = unit_view(modified_tensor, target.feature_dim_count)
    modified_units[live_indices.unsqueeze(-1), column_indices.unsqueeze(0)] = replacement
    modified_batch = dict(batch)
    modified_batch[target.tensor_key] = modified_tensor
    return modified_batch, target.tensor_key, count_modified_units(original_units, modified_units)


def perturb_control_noise(batch):
    modified_batch = dict(batch)
    modified_batch["map_rows"] = batch["map_rows"] + CONTROL_NOISE_METRES
    return modified_batch, "map_rows", count_modified_units(
        batch["map_rows"], modified_batch["map_rows"]
    )


def perturb_agent_speed(batch):
    modified_agent_history = batch["agent_history"].clone()
    modified_agent_history[..., contract.AGENT_VELOCITY] += 1.0
    modified_batch = dict(batch)
    modified_batch["agent_history"] = modified_agent_history
    return modified_batch, "agent_history", count_modified_units(
        unit_view(batch["agent_history"], 1), unit_view(modified_agent_history, 1)
    )


def perturb_force_signal_state(batch, signal_state_name):
    signal_history = batch["map_chunk_signal_history"]
    signalled_chunks = torch.nonzero(
        signal_history.any(dim=-1).any(dim=-1).reshape(-1), as_tuple=False
    ).squeeze(-1)
    forced_one_hot = torch.zeros(contract.NUM_TRAFFIC_SIGNAL_STATES, dtype=signal_history.dtype)
    forced_one_hot[contract.TRAFFIC_SIGNAL_STATES.index(signal_state_name)] = 1.0
    modified_signal_history = signal_history.clone()
    modified_units = unit_view(modified_signal_history, 2)
    modified_units[signalled_chunks] = forced_one_hot.repeat(contract.HISTORY_STEPS)
    modified_batch = dict(batch)
    modified_batch["map_chunk_signal_history"] = modified_signal_history
    return modified_batch, "map_chunk_signal_history", count_modified_units(
        unit_view(signal_history, 2), modified_units
    )


def build_perturbations(batch, targets, random_generator):
    perturbations = []
    for tensor_key, selectors in (
        ("agent_history", AGENT_FEATURE_SELECTORS),
        ("neighbour_history", AGENT_FEATURE_SELECTORS),
        ("map_rows", MAP_FEATURE_SELECTORS),
        ("map_chunk_lane_context", LANE_CONTEXT_SELECTORS),
    ):
        for feature_name, selector in selectors:
            for perturbation_kind in ("zero", "shuffle"):
                perturbations.append((
                    f"{tensor_key}.{feature_name}",
                    perturbation_kind,
                    partial(apply_column_perturbation, batch, targets[tensor_key],
                            selector_columns(selector), perturbation_kind, random_generator),
                ))

    for tensor_key, feature_width in (
        ("agent_history_mask", contract.HISTORY_STEPS),
        ("neighbour_history_mask", contract.HISTORY_STEPS),
        ("map_chunk_signal_history", contract.POLYLINE_SIGNAL_DIM),
    ):
        for perturbation_kind in ("zero", "shuffle"):
            perturbations.append((
                tensor_key,
                perturbation_kind,
                partial(apply_column_perturbation, batch, targets[tensor_key],
                        tuple(range(feature_width)), perturbation_kind, random_generator),
            ))

    perturbations.append((
        "map_rows.EVERY COLUMN (delete the map)", "zero",
        partial(apply_column_perturbation, batch, targets["map_rows"],
                tuple(range(contract.MAP_FEATURE_DIM)), "zero", random_generator),
    ))
    perturbations.append((
        "map_dot_polyline_slot (which chunk each dot pools into)", "shuffle",
        partial(apply_column_perturbation, batch, targets["map_dot_polyline_slot"],
                (0,), "shuffle", random_generator),
    ))
    perturbations.append(
        ("agent_history.AGENT_VELOCITY", "+1 m/s", partial(perturb_agent_speed, batch))
    )
    for signal_state_name in ("LANE_STATE_STOP", "LANE_STATE_GO"):
        perturbations.append((
            "map_chunk_signal_history (signalled chunks)", f"force {signal_state_name}",
            partial(perturb_force_signal_state, batch, signal_state_name),
        ))
    return perturbations


def mean_metres_moved(base_trajectories, perturbed_trajectories):
    return float((perturbed_trajectories - base_trajectories).norm(dim=-1).mean())


def run_perturbation(predictor, batch, base_trajectories, perturbation):
    label, perturbation_kind, perturbation_function = perturbation
    outcome = perturbation_function()
    if outcome is None:
        return None
    modified_batch, tensor_key, units_modified = outcome
    assert tensor_key in batch, f"{label} writes to {tensor_key}, which is not a key of the batch"
    assert not torch.equal(batch[tensor_key], modified_batch[tensor_key]), (
        f"{label} / {perturbation_kind} left {tensor_key} bit-identical"
    )
    assert units_modified > 0, f"{label} / {perturbation_kind} modified no rows of {tensor_key}"
    with torch.no_grad():
        perturbed_trajectories, _ = predictor(modified_batch)
    return label, perturbation_kind, units_modified, mean_metres_moved(
        base_trajectories, perturbed_trajectories
    )


def print_row(label, perturbation_kind, units_modified, metres_moved):
    print(f"{label:<62}{perturbation_kind:>22}{units_modified:>16,}{metres_moved:>20.8f}")


def main():
    arguments = sys.argv[1:]
    assert len(arguments) in (2, 3), (
        "usage: measure_sensitivity.py <staged directory> <anchors .npz> [checkpoint .pt]"
    )
    staged_directory = Path(arguments[0])
    anchors_path = Path(arguments[1])
    checkpoint_path = Path(arguments[2]) if len(arguments) == 3 else None

    assert_selectors_cover_every_column(
        AGENT_FEATURE_SELECTORS, contract.AGENT_FEATURE_DIM, "agent"
    )
    assert_selectors_cover_every_column(MAP_FEATURE_SELECTORS, contract.MAP_FEATURE_DIM, "map")
    assert_selectors_cover_every_column(
        LANE_CONTEXT_SELECTORS, contract.LANE_CONTEXT_DIM, "lane context"
    )

    torch.manual_seed(FIXED_SEED)
    with np.load(anchors_path) as anchors_file:
        contract.check_artifact_provenance(
            anchors_file["provenance"] if "provenance" in anchors_file else None,
            anchors_path, "Refit them with fit_anchors.py.",
        )
        unit_anchors = torch.from_numpy(anchors_file["unit_anchors"]).float()
    predictor = MotionPredictor(unit_anchors)
    if checkpoint_path is None:
        print("UNTRAINED WORKING-TREE MODEL: the numbers below are meaningless and this run only"
              " proves every perturbation still builds and still moves its tensor.")
    else:
        predictor.load_state_dict(load_checkpoint_state(checkpoint_path)["model_state"])
    predictor.eval()

    scenario_paths = sorted(staged_directory.glob("*.npz"))
    assert scenario_paths, f"no .npz scenarios in {staged_directory}"
    batch = next(iter(pipeline.batches(scenario_paths, 0, BATCH_SIZE, 0, FIXED_SEED, True)))
    random_generator = torch.Generator().manual_seed(FIXED_SEED)

    recording_batch = KeyRecordingBatch(batch)
    with torch.no_grad():
        base_trajectories, _ = predictor(recording_batch)
    targets = perturbation_targets(batch)
    assert_every_key_the_forward_pass_reads_is_perturbed(recording_batch.keys_read, targets)

    print(
        f"batch: {batch['agent_history'].shape[0]} samples,"
        f" {batch['map_rows'].shape[0]:,} map dots,"
        f" {int(batch['max_polylines_in_batch'])} chunk slots per sample,"
        f" {batch['neighbour_history'].shape[1]} neighbour slots per sample"
    )
    print(
        "base mean absolute predicted position:"
        f" {float(base_trajectories.norm(dim=-1).mean()):.6f} m"
    )
    print(
        "batch keys the forward pass never reads, so never perturbed:"
        f" {sorted(set(batch) - recording_batch.keys_read)}"
    )
    print()
    print(f"{'feature':<62}{'perturbation':>22}{'rows modified':>16}{'mean metres moved':>20}")

    control = run_perturbation(
        predictor, batch, base_trajectories,
        ("map_rows.EVERY COLUMN (noise floor control)", f"+{CONTROL_NOISE_METRES:g}",
         partial(perturb_control_noise, batch)),
    )
    print_row(*control)
    print()

    results = []
    unperturbable = []
    for perturbation in build_perturbations(batch, targets, random_generator):
        outcome = run_perturbation(predictor, batch, base_trajectories, perturbation)
        if outcome is None:
            unperturbable.append((perturbation[0], perturbation[1]))
            continue
        results.append(outcome)

    for outcome in sorted(results, key=lambda result: result[3], reverse=True):
        print_row(*outcome)

    print()
    print(f"{CONSTANT_NOTE}:")
    for label, perturbation_kind in unperturbable:
        print(f"  {label} / {perturbation_kind}")
    print()
    print("skipped by design:")
    for reason in PERTURBATIONS_SKIPPED_BY_DESIGN:
        print(f"  {reason}")


if __name__ == "__main__":
    main()
