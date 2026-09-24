from __future__ import annotations

import matplotlib as mpl


BLUE = "#3B5BA5"
ORANGE = "#D47A2C"
GREEN = "#2A8F72"
PURPLE = "#7651A8"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 10.5,
            "axes.labelsize": 11.2,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.7,
            "axes.linewidth": 0.85,
            "lines.linewidth": 1.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "0.75",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis, grid: bool = True) -> None:
    for side in ("left", "right", "bottom", "top"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_linewidth(0.85)
    if grid:
        axis.grid(True, which="major", color="0.88", linestyle=":", linewidth=0.65)
        axis.grid(False, which="minor")

