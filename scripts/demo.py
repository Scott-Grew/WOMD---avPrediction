import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from womd import baseline, contract, dataset, model

MAP_KIND_STYLE = {
    "lane": ("#9aa0a6", 0.8, "-"),
    "road_line": ("#c8b900", 1.0, "--"),
    "road_edge": ("#5f6368", 1.4, "-"),
    "crosswalk": ("#4a9ad4", 1.2, "-"),
    "speed_bump": ("#d47f4a", 1.2, "-"),
    "driveway": ("#8ab4a0", 1.0, "-"),
    "stop_sign": ("#d93025", 2.0, "-"),
}


def draw_map(axes, polylines, polylines_mask):
    for polyline, mask in zip(polylines, polylines_mask):
        if not mask.any():
            continue
        points = polyline[mask][:, 0:2]
        kind_index = int(np.argmax(polyline[mask][0, 4:]))
        colour, width, style = MAP_KIND_STYLE[contract.MAP_POLYLINE_KINDS[kind_index]]
        axes.plot(
            points[:, 0], points[:, 1], color=colour, linewidth=width, linestyle=style, zorder=1
        )


def draw_neighbours(axes, neighbour_history, neighbour_mask):
    for history, mask in zip(neighbour_history, neighbour_mask):
        if not mask.any():
            continue
        track = history[mask][:, 0:2]
        axes.plot(track[:, 0], track[:, 1], color="#5f6368", linewidth=1.2, alpha=0.7, zorder=2)
        axes.scatter(track[-1, 0], track[-1, 1], s=28, color="#5f6368", zorder=3)


def render_sample(sample, trajectories, mode_logits, baseline_trajectory, output_path):
    figure, axes = plt.subplots(figsize=(9, 9))

    draw_map(axes, sample["map_polylines"].numpy(), sample["map_polylines_mask"].numpy())
    draw_neighbours(
        axes, sample["neighbour_history"].numpy(), sample["neighbour_history_mask"].numpy()
    )

    history_mask = sample["agent_history_mask"].numpy()
    history = sample["agent_history"].numpy()[history_mask][:, 0:2]
    axes.plot(history[:, 0], history[:, 1], color="#202124", linewidth=2.6, zorder=5, label="history")
    axes.scatter(0.0, 0.0, s=90, color="#202124", zorder=6)

    future_mask = sample["future_mask"].numpy()
    truth = sample["future_positions"].numpy()[future_mask]
    axes.plot(
        truth[:, 0], truth[:, 1], color="#1e8e3e", linewidth=3.0, zorder=5, label="ground truth"
    )

    axes.plot(
        baseline_trajectory[:, 0],
        baseline_trajectory[:, 1],
        color="#9aa0a6",
        linewidth=1.8,
        linestyle=":",
        zorder=4,
        label="constant velocity",
    )

    probabilities = torch.softmax(mode_logits, dim=-1).numpy()
    ordering = np.argsort(-probabilities)
    for rank, mode_index in enumerate(ordering):
        trajectory = trajectories[mode_index].numpy()
        axes.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#d93025",
            linewidth=2.4 if rank == 0 else 1.3,
            alpha=1.0 if rank == 0 else 0.55,
            zorder=4,
            label="predicted modes" if rank == 0 else None,
        )
        axes.annotate(
            f"{probabilities[mode_index]:.2f}",
            trajectory[-1],
            fontsize=8,
            color="#d93025",
            zorder=7,
        )

    axes.set_aspect("equal")
    axes.set_xlabel("metres ahead of agent")
    axes.set_ylabel("metres left of agent")
    axes.set_title("WOMD motion prediction — agent-centric frame")
    axes.legend(loc="upper left", fontsize=9)
    axes.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="demo.png")
    arguments = parser.parse_args()

    samples = dataset.ShardedSampleDataset(arguments.data)
    sample = samples[arguments.index]
    batch = dataset.samples_to_batch([sample])

    predictor = model.MotionPredictor()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu")
    predictor.load_state_dict(checkpoint["model_state"])
    predictor.eval()
    with torch.no_grad():
        trajectories, mode_logits = predictor(batch)
        baseline_trajectory = baseline.constant_velocity_predictions(batch)[0, 0]

    render_sample(
        sample, trajectories[0], mode_logits[0], baseline_trajectory.numpy(), arguments.output
    )
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
