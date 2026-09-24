from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from plot_style import apply_style, style_axis  # noqa: E402


BETAS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
LANDSCAPE_BETAS = (0.2, 0.6, 1.0, 1.6)
DIAGNOSTIC_BETAS = (0.2, 0.8, 1.6)
SHOT_COLORS = {1_000: "#C84A17", 100_000: "#111111"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the finite-shot QSM figures.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Figures")
    return parser.parse_args()


def save(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def positive(values: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(values, floor)


def draw_mean_and_std(axis, values, color, smooth: int | None = None, positive_floor=None):
    if smooth is not None:
        cumulative = np.concatenate(
            [np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1
        )
        end = np.arange(1, values.shape[1] + 1)
        start = np.maximum(end - smooth, 0)
        values = (cumulative[:, end] - cumulative[:, start]) / (end - start)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    lower = mean - std
    if positive_floor is not None:
        lower = np.maximum(lower, positive_floor)
    axis.plot(np.arange(values.shape[1]), mean, color=color, linewidth=1.05)
    axis.fill_between(
        np.arange(values.shape[1]), lower, mean + std, color=color, alpha=0.12, linewidth=0
    )


def learning_figure(data_dir: Path, output_dir: Path) -> None:
    data = np.load(data_dir / "learning_summary.npz")
    betas = data["betas"]
    shots = data["shots"]
    endpoint = data["endpoint_error"]
    histories = (data["learning_error_far"], data["learning_error_local"])
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 0.92, len(betas)))
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.25), sharey="row")

    history_means = [history.mean(axis=1) for history in histories]
    history_uppers = [
        history.mean(axis=1) + history.std(axis=1, ddof=1) for history in histories
    ]
    history_floor = max(1e-4, 0.75 * min(float(values.min()) for values in history_means))
    history_ceiling = 1.12 * max(float(values.max()) for values in history_uppers)
    endpoint_mean = endpoint.mean(axis=-1)
    endpoint_upper = endpoint_mean + endpoint.std(axis=-1, ddof=1)
    endpoint_floor = 0.5 * float(endpoint_mean.min())
    endpoint_ceiling = 1.15 * float(endpoint_upper.max())

    for column, initialization in enumerate(("Far initialization", "Local initialization")):
        for beta_index, (beta, color) in enumerate(zip(betas, colors)):
            values = histories[column][beta_index]
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            axes[0, column].plot(np.arange(values.shape[1]), mean, color=color)
            axes[0, column].fill_between(
                np.arange(values.shape[1]),
                np.maximum(mean - std, history_floor),
                mean + std,
                color=color,
                alpha=0.11,
                linewidth=0,
            )
            errors = endpoint[column, beta_index]
            mean = errors.mean(axis=-1)
            std = errors.std(axis=-1, ddof=1)
            axes[1, column].plot(shots, mean, color=color, marker="o", label=fr"$\beta={beta:.1f}$")
            axes[1, column].fill_between(
                shots,
                np.maximum(mean - std, endpoint_floor),
                mean + std,
                color=color,
                alpha=0.10,
                linewidth=0,
            )

        axes[0, column].axhline(0.10, color="0.60", linestyle=":", linewidth=1.0)
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylim(history_floor, history_ceiling)
        axes[0, column].set_title(
            f"({chr(97 + column)}) {initialization}", loc="left", fontweight="bold"
        )
        axes[0, column].set_xlabel("Iteration")
        style_axis(axes[0, column])

        axes[1, column].axhline(0.10, color="0.5", linestyle="--", linewidth=1.0)
        axes[1, column].axhline(0.05, color="0.65", linestyle=":", linewidth=1.0)
        axes[1, column].set_xscale("log")
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylim(endpoint_floor, endpoint_ceiling)
        axes[1, column].set_title(
            f"({chr(99 + column)}) {initialization}", loc="left", fontweight="bold"
        )
        axes[1, column].set_xlabel("Measurement shots per iteration, $N$")
        style_axis(axes[1, column])

    guide_shots = 10.0 ** np.asarray([4.75, 5.25])
    log_floor, log_ceiling = np.log10(endpoint_floor), np.log10(endpoint_ceiling)
    guide_start = 10.0 ** (log_floor + 0.18 * (log_ceiling - log_floor))
    guide_error = guide_start * (guide_shots / guide_shots[0]) ** -0.5
    axes[1, 0].plot(guide_shots, guide_error, color="0.28", linestyle="--", linewidth=1.15)
    axes[1, 0].text(
        1e5,
        1.30 * guide_start * (1e5 / guide_shots[0]) ** -0.5,
        r"$N^{-1/2}$",
        color="0.24",
        fontsize=9.2,
    )
    axes[0, 0].set_ylabel(r"Rel. $\ell^2$ learning error, $e_{\mathrm{rel}}$")
    axes[1, 0].set_ylabel(r"Final rel. $\ell^2$ learning error, $e_{\mathrm{rel}}$")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=8)
    figure.tight_layout(rect=(0, 0.075, 1, 1), h_pad=1.45)
    save(figure, output_dir / "learning_dynamics_and_shot_scaling.pdf")


def add_endpoint_ellipse(axis, x: np.ndarray, y: np.ndarray, color: str) -> None:
    samples = np.column_stack((x[:, -1], y[:, -1]))
    mean = samples.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(samples, rowvar=False, ddof=1))
    eigenvalues = np.maximum(eigenvalues, 0)
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    axis.add_patch(
        Ellipse(
            mean,
            2 * np.sqrt(eigenvalues[-1]),
            2 * np.sqrt(eigenvalues[0]),
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=1.45,
            linestyle="--",
            alpha=0.85,
        )
    )
    axis.scatter(*mean, color=color, s=17, linewidth=0, zorder=4)


def landscape_figure(data_dir: Path, output_dir: Path, trajectories: bool = True) -> None:
    landscapes = [np.load(data_dir / "landscapes" / f"beta_{beta:.1f}.npz") for beta in LANDSCAPE_BETAS]
    x_limits, y_limits = (0.4, 1.35), (0.85, 1.80)
    displayed = []
    for data in landscapes:
        x_mask = (data["x"] >= x_limits[0]) & (data["x"] <= x_limits[1])
        y_mask = (data["y"] >= y_limits[0]) & (data["y"] <= y_limits[1])
        values = data["loss_gap"][np.ix_(y_mask, x_mask)]
        displayed.append(values[values > 0])
    positive_values = np.concatenate(displayed)
    vmin, vmax = float(positive_values.min()), float(positive_values.max())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, 32)
    figure, axes = plt.subplots(1, 4, figsize=(9.2, 3.0))
    contour = None
    for column, (beta, data) in enumerate(zip(LANDSCAPE_BETAS, landscapes)):
        axis = axes[column]
        xx, yy = np.meshgrid(data["x"], data["y"])
        contour = axis.contourf(
            xx,
            yy,
            positive(data["loss_gap"], vmin),
            levels=levels,
            norm=norm,
            cmap="viridis",
        )
        if trajectories:
            for shots in (1_000, 100_000):
                trajectory = np.load(
                    data_dir / "trajectory_slices" / f"beta_{beta:.1f}_m_{shots}.npz"
                )
                x, y = trajectory["trajectory_j"], trajectory["trajectory_h"]
                axis.plot(x.mean(axis=0), y.mean(axis=0), color=SHOT_COLORS[shots], linewidth=1.75)
                add_endpoint_ellipse(axis, x, y, SHOT_COLORS[shots])
            target = data["theta_star"][[int(data["coupling_index"]), int(data["field_index"])]]
            radius = 0.05 * np.linalg.norm(target)
            axis.add_patch(
                Ellipse(
                    target,
                    2 * radius,
                    2 * radius,
                    facecolor="none",
                    edgecolor="white",
                    linestyle="--",
                    linewidth=1.2,
                )
            )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_title(fr"({chr(97 + column)}) $\beta={beta:.1f}$", loc="left", fontweight="bold")
        axis.set_xlabel(r"Coupling, $J_1$")
        style_axis(axis, grid=False)
    axes[0].set_ylabel(r"Field, $h_1^x$")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.27, top=0.95, wspace=0.19)
    if trajectories:
        handles = [
            mpl.lines.Line2D([], [], color=SHOT_COLORS[shots], linewidth=2.1, label=fr"$N={shots:,}$")
            for shots in (1_000, 100_000)
        ]
        figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.16, -0.025), ncol=2)
        figure.text(0.58, 0.0325, r"$J_Q(\theta)-J_Q(\theta^*)$", ha="right", va="center")
        colorbar_axis = figure.add_axes([0.595, 0.015, 0.25, 0.035])
        filename = "mean_dynamics_landscape.pdf"
    else:
        figure.text(0.39, 0.0325, r"$J_Q(\theta)-J_Q(\theta^*)$", ha="right", va="center")
        colorbar_axis = figure.add_axes([0.405, 0.015, 0.34, 0.035])
        filename = "landscape_only.pdf"
    ticks = 10.0 ** np.arange(np.ceil(np.log10(vmin)), np.floor(np.log10(vmax)) + 1)
    colorbar = figure.colorbar(contour, cax=colorbar_axis, ticks=ticks, orientation="horizontal")
    colorbar.ax.invert_xaxis()
    save(figure, output_dir / filename)


def diagnostics_figure(data_dir: Path, output_dir: Path) -> None:
    data = np.load(data_dir / "gradient_diagnostics.npz")
    colors = mpl.colormaps["viridis"](np.linspace(0.12, 0.82, len(DIAGNOSTIC_BETAS)))
    exact_norm = data["exact_gradient_norm"]
    quantities = (
        data["absolute_gradient_error"],
        data["absolute_gradient_error"] / exact_norm,
        data["gradient_alignment"],
        data["learning_error"],
    )
    titles = ("Absolute error", "Relative error", "Gradient alignment", "Learning error")
    ylabels = (
        "Absolute gradient error",
        "Relative gradient error",
        "Cosine gradient similarity",
        r"Rel. $\ell^2$ learning error, $e_{\mathrm{rel}}$",
    )
    figure, grid = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    axes = grid.ravel()
    for panel, axis in enumerate(axes):
        for beta_index, (beta, color) in enumerate(zip(DIAGNOSTIC_BETAS, colors)):
            values = quantities[panel][beta_index]
            draw_mean_and_std(
                axis,
                values,
                color,
                smooth=None,
                positive_floor=1e-12 if panel != 2 else None,
            )
        axis.set_title(f"({chr(97 + panel)}) {titles[panel]}", loc="left", fontweight="bold")
        axis.set_ylabel(ylabels[panel])
        axis.set_xlabel("Iteration")
        style_axis(axis)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[1].axhline(1, color="0.55", linestyle=":", linewidth=1)
    axes[2].axhline(0, color="0.55", linestyle=":", linewidth=1)
    axes[2].set_ylim(-1, 1)
    axes[3].set_yscale("log")
    axes[3].axhline(0.1, color="0.55", linestyle=":", linewidth=1)
    handles = [
        mpl.lines.Line2D([], [], color=color, label=fr"$\beta={beta:.1f}$")
        for beta, color in zip(DIAGNOSTIC_BETAS, colors)
    ]
    figure.legend(handles, [item.get_label() for item in handles], loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3)
    figure.tight_layout(rect=(0, 0.075, 1, 1), h_pad=1.4, w_pad=1.2)
    save(figure, output_dir / "gradient_diagnostics_far_m1e5.pdf")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    learning_figure(args.data_dir, args.output_dir)
    landscape_figure(args.data_dir, args.output_dir, trajectories=True)
    landscape_figure(args.data_dir, args.output_dir, trajectories=False)
    diagnostics_figure(args.data_dir, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
