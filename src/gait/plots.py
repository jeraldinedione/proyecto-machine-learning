from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, GROUP_COLORS, GROUP_LABELS, GROUP_ORDER

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


def use_project_style() -> None:
    mpl.rcParams.update(STYLE)


def present_groups(data: pd.DataFrame) -> list[str]:
    return [g for g in GROUP_ORDER if g in set(data.group)]


def labels_for(groups: list[str]) -> list[str]:
    return [GROUP_LABELS[g] for g in groups]


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
