# > Lane-graph reachability measurement: how much of the map each walk reaches, and how much of the
# logged 8 s path it covers. Forward-only is the walk over lane_connections alone; lane-change adds
# the staged lane_neighbour_ids edges at zero road cost. Coverage counts a logged future step as
# covered when the LANE its nearest lane-centre dot belongs to carries the flag, which is the same
# nearest-dot rule the loader uses to place the agent on a lane, with no threshold to choose.

import argparse
from pathlib import Path

import numpy as np

from womd import contract, loader


def lane_polyline_mask(scenario_array):
    feature_lengths = scenario_array["feature_lengths"]
    lane_kind_column = contract.MAP_KIND.start + contract.MAP_POLYLINE_KINDS.index("lane")
    first_dot_of_polyline = np.cumsum(feature_lengths) - feature_lengths
    return scenario_array["map_rows"][first_dot_of_polyline, lane_kind_column] == 1.0


def nearest_lane_polyline(future_positions, lane_dot_positions, lane_dot_polyline):
    offsets = future_positions[:, None, :] - lane_dot_positions[None, :, :]
    return lane_dot_polyline[np.argmin(np.einsum("ijk,ijk->ij", offsets, offsets), axis=1)]


def neighbour_symmetry(scenario_array):
    neighbour_rows = scenario_array["lane_neighbour_ids"]
    staged_pairs = {
        (int(lane_id), int(other_lane_id))
        for lane_id, other_lane_id, _ in neighbour_rows.tolist()
    }
    reversed_present = sum(
        (other_lane_id, lane_id) in staged_pairs for lane_id, other_lane_id in staged_pairs
    )
    return len(staged_pairs), reversed_present


def measure(scenario_paths):
    lane_totals = {"forward": 0, "lane_change": 0, "lanes": 0, "polylines": 0}
    dot_totals = {"forward": 0, "lane_change": 0, "dots": 0}
    step_totals = {"forward": 0, "lane_change": 0, "steps": 0}
    per_sample_uncovered = []
    neighbour_pairs = 0
    neighbour_pairs_reversed = 0
    sample_count = 0

    for scenario_path in scenario_paths:
        scenario_array = loader.read_scenario(scenario_path)
        pairs, reversed_present = neighbour_symmetry(scenario_array)
        neighbour_pairs += pairs
        neighbour_pairs_reversed += reversed_present

        polyline_is_lane = lane_polyline_mask(scenario_array)
        dot_polyline_index = scenario_array["map_dot_polyline_index"]
        lane_dot_indices = np.flatnonzero(polyline_is_lane[dot_polyline_index])
        if len(lane_dot_indices) == 0:
            continue
        lane_dot_positions = scenario_array["map_rows"][lane_dot_indices][:, contract.MAP_POSITION]
        lane_dot_polyline = dot_polyline_index[lane_dot_indices]
        dots_per_polyline = np.bincount(lane_dot_polyline, minlength=len(polyline_is_lane))

        track_rows = scenario_array["track_rows"]
        track_valid = scenario_array["track_valid"]
        for track_index in loader.eligible_track_indices(
            track_rows, track_valid, scenario_array["is_designated_target"], True
        ):
            now_row = track_rows[track_index, contract.CURRENT_STEP_INDEX]
            lane_context = loader.lane_context_per_polyline(
                scenario_array["lane_graph"],
                now_row[contract.AGENT_POSITION],
                now_row[contract.AGENT_HEADING_COSINE:contract.AGENT_HEADING_SINE + 1],
            )
            forward = lane_context[:, contract.LANE_CONTEXT_REACHABLE] == 1.0
            lane_change = forward | (
                lane_context[:, contract.LANE_CONTEXT_REACHABLE_BY_LANE_CHANGE] == 1.0
            )

            lane_totals["lanes"] += int(polyline_is_lane.sum())
            lane_totals["polylines"] += len(polyline_is_lane)
            lane_totals["forward"] += int(forward.sum())
            lane_totals["lane_change"] += int(lane_change.sum())
            dot_totals["dots"] += len(lane_dot_indices)
            dot_totals["forward"] += int(dots_per_polyline[forward].sum())
            dot_totals["lane_change"] += int(dots_per_polyline[lane_change].sum())

            future_valid = track_valid[track_index, contract.CURRENT_STEP_INDEX + 1:]
            future_positions = track_rows[
                track_index, contract.CURRENT_STEP_INDEX + 1:, contract.AGENT_POSITION
            ][future_valid]
            if len(future_positions) == 0:
                continue
            nearest_polyline = nearest_lane_polyline(
                future_positions, lane_dot_positions, lane_dot_polyline
            )
            step_totals["steps"] += len(future_positions)
            step_totals["forward"] += int(forward[nearest_polyline].sum())
            step_totals["lane_change"] += int(lane_change[nearest_polyline].sum())
            per_sample_uncovered.append(
                1.0 - float(lane_change[nearest_polyline].mean())
            )
            sample_count += 1

    return (lane_totals, dot_totals, step_totals, np.array(per_sample_uncovered),
            neighbour_pairs, neighbour_pairs_reversed, sample_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("--scenarios", type=int, required=True)
    arguments = parser.parse_args()

    scenario_paths = sorted(arguments.staged_directory.glob("*.npz"))[: arguments.scenarios]
    assert scenario_paths, f"no .npz scenarios in {arguments.staged_directory}"
    (lane_totals, dot_totals, step_totals, per_sample_uncovered,
     neighbour_pairs, neighbour_pairs_reversed, sample_count) = measure(scenario_paths)

    print(
        f"{len(scenario_paths)} scenarios of {arguments.staged_directory},"
        f" {sample_count} designated-target samples"
    )
    print(
        f"map reached, share of lane polylines: forward-only"
        f" {lane_totals['forward'] / lane_totals['lanes']:.1%},"
        f" including lane changes {lane_totals['lane_change'] / lane_totals['lanes']:.1%}"
    )
    print(
        f"map reached, share of ALL map polylines: forward-only"
        f" {lane_totals['forward'] / lane_totals['polylines']:.1%},"
        f" including lane changes {lane_totals['lane_change'] / lane_totals['polylines']:.1%}"
    )
    print(
        f"map reached, share of lane centre dots: forward-only"
        f" {dot_totals['forward'] / dot_totals['dots']:.1%},"
        f" including lane changes {dot_totals['lane_change'] / dot_totals['dots']:.1%}"
    )
    print(
        f"logged 8 s path on a reached lane: forward-only"
        f" {step_totals['forward'] / step_totals['steps']:.1%},"
        f" including lane changes {step_totals['lane_change'] / step_totals['steps']:.1%}"
        f" over {step_totals['steps']} valid future steps"
    )
    print(
        f"per-sample share of the logged path OFF the lane-change set:"
        f" p50 {np.percentile(per_sample_uncovered, 50):.1%},"
        f" p90 {np.percentile(per_sample_uncovered, 90):.1%},"
        f" worst tenth mean {per_sample_uncovered[per_sample_uncovered >= np.percentile(per_sample_uncovered, 90)].mean():.1%}"
    )
    print(
        f"staged neighbour relations: {neighbour_pairs} ordered pairs,"
        f" {neighbour_pairs_reversed / max(neighbour_pairs, 1):.1%} of them also staged reversed"
    )


if __name__ == "__main__":
    main()
