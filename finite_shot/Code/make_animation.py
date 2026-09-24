from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from plot_style import apply_style, style_axis  # noqa: E402


BETAS = (0.2, 0.6, 1.0, 1.6)
COLORS = {1_000: "#C84A17", 100_000: "#111111"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate the first finite-shot learning steps.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data")
    parser.add_argument("--output", type=Path, default=ROOT / "Figures" / "optimization.gif")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1, help="Number of optimization steps between frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    landscapes = [np.load(args.data_dir / "landscapes" / f"beta_{beta:.1f}.npz") for beta in BETAS]
    displayed = []
    for data in landscapes:
        x_mask = (data["x"] >= 0.4) & (data["x"] <= 1.35)
        y_mask = (data["y"] >= 0.85) & (data["y"] <= 1.80)
        values = data["loss_gap"][np.ix_(y_mask, x_mask)]
        displayed.append(values[values > 0])
    positive = np.concatenate(displayed)
    norm = LogNorm(float(positive.min()), float(positive.max()))
    levels = np.geomspace(norm.vmin, norm.vmax, 32)
    figure, axes = plt.subplots(1, 4, figsize=(9.2, 3.75))
    artists = []
    contour = None
    for column, (beta, data) in enumerate(zip(BETAS, landscapes)):
        axis = axes[column]
        xx, yy = np.meshgrid(data["x"], data["y"])
        contour = axis.contourf(xx, yy, np.maximum(data["loss_gap"], norm.vmin), levels=levels, norm=norm, cmap="viridis")
        for shots in (1_000, 100_000):
            trajectory = np.load(args.data_dir / "trajectory_slices" / f"beta_{beta:.1f}_m_{shots}.npz")
            x = trajectory["trajectory_j"].mean(axis=0)
            y = trajectory["trajectory_h"].mean(axis=0)
            line, = axis.plot([], [], color=COLORS[shots], linewidth=2.0, zorder=4)
            point, = axis.plot([], [], "o", color=COLORS[shots], markersize=4.8, markeredgewidth=0, zorder=5)
            artists.append((line, point, x, y))
        axis.set_xlim(0.4, 1.35)
        axis.set_ylim(0.85, 1.80)
        axis.set_title(fr"({chr(97 + column)}) $\beta={beta:.1f}$", loc="left", fontweight="bold")
        axis.set_xlabel(r"Coupling, $J_1$")
        style_axis(axis, grid=False)
    axes[0].set_ylabel(r"Field, $h_1^x$")
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], linewidth=2.1, label=fr"$N={m:,}$") for m in (1_000, 100_000)]
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.16, 0.055), ncol=2)
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.29, top=0.80, wspace=0.19)
    figure.text(0.58, 0.108, r"$J_Q(\theta)-J_Q(\theta^*)$", ha="right", va="center")
    colorbar_axis = figure.add_axes([0.595, 0.09, 0.25, 0.03])
    ticks = 10.0 ** np.arange(np.ceil(np.log10(norm.vmin)), np.floor(np.log10(norm.vmax)) + 1)
    colorbar = figure.colorbar(contour, cax=colorbar_axis, ticks=ticks, orientation="horizontal")
    colorbar.ax.invert_xaxis()
    title = figure.suptitle("Iteration 0", y=0.965, fontweight="bold")

    def update(step: int):
        for line, point, x, y in artists:
            line.set_data(x[: step + 1], y[: step + 1])
            point.set_data([x[step]], [y[step]])
        title.set_text(f"Iteration {step}")
        return [artist for line, point, _, _ in artists for artist in (line, point)] + [title]

    frames = list(range(0, args.steps + 1, args.stride))
    if frames[-1] != args.steps:
        frames.append(args.steps)
    frames += [args.steps] * 6
    movie = animation.FuncAnimation(figure, update, frames=frames, interval=1000 / 6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(args.output, writer=animation.PillowWriter(fps=6), dpi=140)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
