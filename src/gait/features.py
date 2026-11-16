from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

MIN_BOX_SIZE = 4
BOXES_PER_DECADE = 12
SAMPLE_ENTROPY_ORDER = 2
SAMPLE_ENTROPY_TOLERANCE = 0.2

Extractor = Callable[..., dict[str, float]]


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = np.mean(values)
    return float(np.std(values, ddof=1) / mean) if mean else np.nan


def successive_difference_cv(values: np.ndarray) -> float:
    if len(values) < 3:
        return np.nan
    mean = np.mean(values)
    return float(np.sqrt(np.mean(np.diff(values) ** 2)) / mean) if mean else np.nan


def autocorrelation(values: np.ndarray, lag: int = 1) -> float:
    if len(values) <= lag:
        return np.nan
    centered = values - values.mean()
    denominator = np.dot(centered, centered)
    if denominator == 0:
        return np.nan
    return float(np.dot(centered[:-lag], centered[lag:]) / denominator)


def _box_sizes(n_points: int) -> np.ndarray:
    largest = n_points // 4
    if largest <= MIN_BOX_SIZE:
        return np.array([], dtype=int)
    decades = np.log10(largest / MIN_BOX_SIZE)
    count = max(int(np.ceil(decades * BOXES_PER_DECADE)), 2)
    sizes = np.unique(np.logspace(np.log10(MIN_BOX_SIZE), np.log10(largest), count).astype(int))
    return sizes[sizes >= MIN_BOX_SIZE]


def _fluctuation(profile: np.ndarray, box_size: int) -> float:
    n_boxes = len(profile) // box_size
    boxes = profile[: n_boxes * box_size].reshape(n_boxes, box_size)
    positions = np.arange(box_size)
    design = np.vstack([positions, np.ones(box_size)]).T
    coefficients, *_ = np.linalg.lstsq(design, boxes.T, rcond=None)
    residuals = boxes.T - design @ coefficients
    return float(np.sqrt(np.mean(residuals**2)))


def detrended_fluctuation_alpha(values: np.ndarray) -> float:
    sizes = _box_sizes(len(values))
    if len(sizes) < 2:
        return np.nan
    profile = np.cumsum(values - values.mean())
    fluctuations = np.array([_fluctuation(profile, size) for size in sizes])
    valid = fluctuations > 0
    if valid.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(sizes[valid]), np.log(fluctuations[valid]), 1)
    return float(slope)


def _template_matches(values: np.ndarray, length: int, count: int, radius: float) -> float:
    templates = sliding_window_view(values, length)[:count]
    distances = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
    return (np.count_nonzero(distances <= radius) - count) / 2


def sample_entropy(
    values: np.ndarray,
    order: int = SAMPLE_ENTROPY_ORDER,
    tolerance: float = SAMPLE_ENTROPY_TOLERANCE,
) -> float:
    values = np.asarray(values, dtype=float)
    count = len(values) - order
    if count < 2:
        return np.nan
    radius = tolerance * np.std(values)
    shorter = _template_matches(values, order, count, radius)
    longer = _template_matches(values, order + 1, count, radius)
    if shorter == 0 or longer == 0:
        return np.nan
    return float(-np.log(longer / shorter))


def asymmetry_index(left: np.ndarray, right: np.ndarray) -> float:
    total = left.mean() + right.mean()
    return float(200 * abs(left.mean() - right.mean()) / total) if total else np.nan


def stride_intervals(strides: pd.DataFrame, side: str = "left") -> np.ndarray:
    if side == "both":
        return strides[["stride_left_s", "stride_right_s"]].mean(axis=1).to_numpy()
    return strides[f"stride_{side}_s"].to_numpy()


def _dispersion_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_sd": float(np.std(values, ddof=1)),
        f"{prefix}_cv": coefficient_of_variation(values),
        f"{prefix}_iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
    }


def subject_features(strides: pd.DataFrame, side: str = "left") -> dict[str, float]:
    stride = stride_intervals(strides, side)
    swing = strides[["swing_left_pct", "swing_right_pct"]].mean(axis=1).to_numpy()
    support = strides.double_support_pct.to_numpy()
    span = strides.time_s.max() - strides.time_s.min()

    features = {"n_strides": len(strides), "observed_span_s": span}
    features["cadence_strides_per_min"] = 60 * len(strides) / span if span else np.nan
    features.update(_dispersion_stats(stride, "stride"))
    features.update(_dispersion_stats(swing, "swing_pct"))
    features.update(_dispersion_stats(support, "double_support_pct"))
    features["asymmetry_stride"] = asymmetry_index(
        strides.stride_left_s.to_numpy(), strides.stride_right_s.to_numpy()
    )
    features["asymmetry_swing"] = asymmetry_index(
        strides.swing_left_pct.to_numpy(), strides.swing_right_pct.to_numpy()
    )
    features["autocorr_lag1"] = autocorrelation(stride)
    features["dfa_alpha"] = detrended_fluctuation_alpha(stride)
    return features


def extended_features(strides: pd.DataFrame, side: str = "left") -> dict[str, float]:
    stride = stride_intervals(strides, side)
    features = subject_features(strides, side)
    features["stride_diff_cv"] = successive_difference_cv(stride)
    features["sample_entropy"] = sample_entropy(stride)
    return features


def build_feature_table(
    strides: pd.DataFrame, extractor: Extractor = subject_features, **options
) -> pd.DataFrame:
    rows = [
        {"group": group, "record": record, **extractor(subject, **options)}
        for (group, record), subject in strides.groupby(["group", "record"], sort=False)
    ]
    return pd.DataFrame(rows)


def window_features(
    strides: pd.DataFrame,
    window_strides: int = 30,
    min_strides: int = 20,
    extractor: Extractor = subject_features,
    **options,
) -> pd.DataFrame:
    rows = []
    for (group, record), subject in strides.groupby(["group", "record"], sort=False):
        subject = subject.reset_index(drop=True)
        for start in range(0, len(subject), window_strides):
            window = subject.iloc[start : start + window_strides]
            if len(window) < min_strides:
                continue
            rows.append(
                {
                    "group": group,
                    "record": record,
                    "window_index": start // window_strides,
                    **extractor(window, **options),
                }
            )
    return pd.DataFrame(rows)
