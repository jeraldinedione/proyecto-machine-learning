from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, GROUP_COLORS, GROUP_LABELS, GROUP_ORDER, GROUP_SHORT

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 0.9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

COLUMN_WIDTH = 3.5
PAGE_WIDTH = 7.16


def use_project_style() -> None:
    mpl.rcParams.update(STYLE)


def use_paper_style() -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(PAPER_STYLE)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


def annotated_matrix(
    ax: plt.Axes,
    values: pd.DataFrame,
    fmt: str = "{:.2f}",
    cmap: str = "Greys",
    vmin: float = 0.0,
    vmax: float = 1.0,
    mask_diagonal: bool = False,
) -> plt.Axes:
    data = values.to_numpy(dtype=float)
    shown = np.ma.masked_where(np.eye(len(data), dtype=bool), data) if mask_diagonal else data
    ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask_diagonal and i == j:
                continue
            level = (data[i, j] - vmin) / (vmax - vmin)
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if level > 0.6 else "black")
    ax.set_xticks(range(data.shape[1]), values.columns)
    ax.set_yticks(range(data.shape[0]), values.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def present_groups(data: pd.DataFrame) -> list[str]:
    return [g for g in GROUP_ORDER if g in set(data.group)]


def labels_for(groups: list[str]) -> list[str]:
    return [GROUP_LABELS[g] for g in groups]


def short_labels_for(groups: list[str]) -> list[str]:
    return [GROUP_SHORT[g] for g in groups]


def colors_for(groups: list[str]) -> list[str]:
    return [GROUP_COLORS[g] for g in groups]


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")


def grouped_box(ax: plt.Axes, data: pd.DataFrame, column: str, title: str, ylabel: str) -> plt.Axes:
    groups = present_groups(data)
    samples = [data.loc[data.group == g, column].dropna() for g in groups]
    artists = ax.boxplot(
        samples, patch_artist=True, widths=0.6, showfliers=False, medianprops={"color": "black"}
    )
    for patch, color in zip(artists["boxes"], colors_for(groups)):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks(range(1, len(groups) + 1), labels_for(groups))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


def grouped_strip(ax: plt.Axes, data: pd.DataFrame, column: str, jitter: float = 0.08, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    groups = present_groups(data)
    for position, group in enumerate(groups, start=1):
        values = data.loc[data.group == group, column].dropna()
        offsets = position + rng.normal(0, jitter, len(values))
        ax.scatter(offsets, values, s=18, color=GROUP_COLORS[group], alpha=0.8, edgecolor="none")
    ax.set_xticks(range(1, len(groups) + 1), labels_for(groups))
    return ax


def group_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    groups: list[str] | None = None,
    jitter: float = 0.09,
    seed: int = 0,
) -> plt.Axes:
    rng = np.random.default_rng(seed)
    groups = groups or present_groups(data)
    for position, group in enumerate(groups):
        values = data.loc[data.group == group, column].dropna().to_numpy()
        offsets = position + rng.uniform(-jitter, jitter, len(values)) * 2
        ax.scatter(offsets, values, s=9, color=GROUP_COLORS[group], alpha=0.75, lw=0)
        ax.hlines(np.median(values), position - 0.28, position + 0.28, color="black", lw=1.0)
    ax.set_xticks(range(len(groups)), short_labels_for(groups))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    return ax


def stride_series(ax: plt.Axes, strides: pd.DataFrame, record: str, column: str = "stride_left_s"):
    subject = strides[strides.record == record]
    group = subject.group.iloc[0]
    ax.plot(subject.time_s, subject[column], lw=0.9, color=GROUP_COLORS[group])
    ax.set_title(f"{record} ({GROUP_LABELS[group]})")
    return ax


def group_legend(fig: plt.Figure, groups: list[str], **kwargs) -> None:
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=GROUP_COLORS[g], label=GROUP_LABELS[g])
        for g in groups
    ]
    fig.legend(handles=handles, **kwargs)
