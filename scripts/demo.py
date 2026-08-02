import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from womd import baseline, contract, dataset, metrics, model

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
PREDICTION = "#2a78d6"
GROUND_TRUTH = "#eb6834"
CONSTANT_VELOCITY = "#1baf7a"

MAP_KIND_STYLE = {
    "lane": ("#e2e1dc", 0.7, "solid"),
    "road_line": ("#d2d1cb", 0.9, "solid"),
    "road_edge": ("#a8a7a0", 1.4, "solid"),
    "crosswalk": ("#cfcec8", 1.0, (0, (2, 3))),
    "speed_bump": ("#cfcec8", 1.0, "solid"),
    "driveway": ("#e6e5e0", 0.8, "solid"),
    "stop_sign": ("#a8a7a0", 0.0, "solid"),
}

VIEW_PADDING_METRES = 12.0
MINIMUM_VIEW_METRES = 45.0


def draw_map(axes, polylines, polylines_mask):
    for polyline, mask in zip(polylines, polylines_mask):
        if not mask.any():
            continue
        points = polyline[mask][:, contract.MAP_POSITION]
        kind_index = int(np.argmax(polyline[mask][0, contract.MAP_KIND]))
        kind = contract.MAP_POLYLINE_KINDS[kind_index]
        colour, width, style = MAP_KIND_STYLE[kind]
        if kind == "stop_sign":
            axes.scatter(points[:, 0], points[:, 1], s=14, color=colour, marker="o", zorder=1)
            continue
        axes.plot(
            points[:, 0], points[:, 1], color=colour, linewidth=width, linestyle=style, zorder=1
        )


def draw_neighbours(axes, neighbour_history, neighbour_mask):
    for history, mask in zip(neighbour_history, neighbour_mask):
        if not mask.any():
            continue
        track = history[mask][:, contract.AGENT_POSITION]
        axes.plot(track[:, 0], track[:, 1], color="#b4b3ac", linewidth=1.0, zorder=2)
        axes.scatter(
            track[-1, 0],
            track[-1, 1],
            s=26,
            color="#8d8c85",
            edgecolors=SURFACE,
            linewidths=1.2,
            zorder=3,
        )


def view_window(history, truth, trajectories, baseline_trajectory):
    points = np.concatenate(
        [history, truth, trajectories.reshape(-1, 2), baseline_trajectory, np.zeros((1, 2))]
    )
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    extent = float(np.abs(points - centre).max()) + VIEW_PADDING_METRES
    extent = max(extent, MINIMUM_VIEW_METRES / 2.0)
    return centre, extent


def render_scene(axes, sample, trajectories, mode_logits, baseline_trajectory, show_legend):
    draw_map(axes, sample["map_polylines"].numpy(), sample["map_polylines_mask"].numpy())
    draw_neighbours(
        axes, sample["neighbour_history"].numpy(), sample["neighbour_history_mask"].numpy()
    )

    history = sample["agent_history"].numpy()[sample["agent_history_mask"].numpy()][
        :, contract.AGENT_POSITION
    ]
    truth = sample["future_positions"].numpy()[sample["future_mask"].numpy()]
    probabilities = torch.softmax(mode_logits, dim=-1).numpy()
    ordering = np.argsort(-probabilities)

    axes.plot(history[:, 0], history[:, 1], color=INK_PRIMARY, linewidth=2.2, zorder=6)

    for rank, mode_index in enumerate(ordering):
        trajectory = trajectories[mode_index].numpy()
        probability = float(probabilities[mode_index])
        axes.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=PREDICTION,
            linewidth=1.0 + 3.2 * probability,
            alpha=0.35 + 0.65 * probability,
            solid_capstyle="round",
            zorder=4,
            label="predicted modes" if rank == 0 else None,
        )
        if rank < 3:
            axes.annotate(
                f"{probability:.2f}",
                trajectory[-1],
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7.5,
                color=INK_SECONDARY,
                zorder=8,
            )

    axes.plot(
        baseline_trajectory[:, 0],
        baseline_trajectory[:, 1],
        color=CONSTANT_VELOCITY,
        linewidth=1.8,
        linestyle=(0, (5, 4)),
        zorder=5,
        label="constant velocity",
    )
    axes.annotate(
        "constant velocity",
        baseline_trajectory[-1],
        textcoords="offset points",
        xytext=(5, -9),
        fontsize=7.5,
        color=INK_SECONDARY,
        zorder=8,
    )

    axes.plot(
        truth[:, 0],
        truth[:, 1],
        color=GROUND_TRUTH,
        linewidth=2.8,
        solid_capstyle="round",
        zorder=7,
        label="ground truth",
    )
    axes.scatter(
        0.0, 0.0, s=70, color=INK_PRIMARY, edgecolors=SURFACE, linewidths=1.6, zorder=9
    )

    future_positions = sample["future_positions"].unsqueeze(0)
    future_mask = sample["future_mask"].unsqueeze(0)
    best_error = float(
        metrics.minimum_average_displacement(
            trajectories.unsqueeze(0), future_positions, future_mask, contract.FUTURE_STEPS
        )[0]
    )
    baseline_error = float(
        metrics.minimum_average_displacement(
            torch.from_numpy(baseline_trajectory).unsqueeze(0).unsqueeze(0),
            future_positions,
            future_mask,
            contract.FUTURE_STEPS,
        )[0]
    )
    axes.annotate(
        f"minADE {best_error:.2f} m   ·   baseline {baseline_error:.2f} m",
        xy=(0.5, 0.012),
        xycoords="axes fraction",
        ha="center",
        fontsize=8.5,
        color=INK_SECONDARY,
    )

    centre, extent = view_window(
        history, truth, trajectories.numpy(), baseline_trajectory
    )
    axes.set_xlim(centre[0] - extent, centre[0] + extent)
    axes.set_ylim(centre[1] - extent, centre[1] + extent)
    axes.set_aspect("equal")
    axes.set_facecolor(SURFACE)
    axes.tick_params(labelsize=8, colors=INK_SECONDARY, length=0)
    axes.grid(color="#e7e6e1", linewidth=0.6)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_edgecolor("#e0dfda")
    if show_legend:
        legend = axes.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.94)
        legend.get_frame().set_edgecolor("#e0dfda")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)


def predict(predictor, samples, index):
    sample = samples[index]
    batch = dataset.samples_to_batch([sample])
    with torch.no_grad():
        trajectories, mode_logits = predictor(batch)
        baseline_trajectory = baseline.constant_velocity_predictions(batch)[0, 0]
    return sample, trajectories[0], mode_logits[0], baseline_trajectory.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--indices")
    parser.add_argument("--titles")
    parser.add_argument("--output", default="demo.png")
    arguments = parser.parse_args()

    samples = dataset.ShardedSampleDataset(arguments.data)
    predictor = model.MotionPredictor()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu")
    predictor.load_state_dict(checkpoint["model_state"])
    predictor.eval()

    indices = (
        [int(value) for value in arguments.indices.split(",")]
        if arguments.indices
        else [arguments.index]
    )
    titles = arguments.titles.split(",") if arguments.titles else [None] * len(indices)

    columns = min(len(indices), 2)
    rows = (len(indices) + columns - 1) // columns
    figure, axes_grid = plt.subplots(
        rows, columns, figsize=(6.4 * columns, 6.4 * rows), squeeze=False
    )
    figure.patch.set_facecolor(SURFACE)

    for position, index in enumerate(indices):
        axes = axes_grid[position // columns][position % columns]
        sample, trajectories, mode_logits, baseline_trajectory = predict(
            predictor, samples, index
        )
        render_scene(
            axes, sample, trajectories, mode_logits, baseline_trajectory, position == 0
        )
        if titles[position]:
            axes.set_title(titles[position], fontsize=11, color=INK_PRIMARY, pad=9)
        if position // columns == rows - 1:
            axes.set_xlabel("metres ahead of agent", fontsize=8.5, color=INK_SECONDARY)
        if position % columns == 0:
            axes.set_ylabel("metres left of agent", fontsize=8.5, color=INK_SECONDARY)

    for position in range(len(indices), rows * columns):
        axes_grid[position // columns][position % columns].axis("off")

    figure.suptitle(
        "WOMD motion prediction — six futures per agent, agent-centric frame",
        fontsize=12.5,
        color=INK_PRIMARY,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(arguments.output, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
