from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BOX_SIZE = 4
BOXES_PER_DECADE = 12


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = np.mean(values)
    return float(np.std(values, ddof=1) / mean) if mean else np.nan


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


def asymmetry_index(left: np.ndarray, right: np.ndarray) -> float:
    total = left.mean() + right.mean()
    return float(200 * abs(left.mean() - right.mean()) / total) if total else np.nan


def _dispersion_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_sd": float(np.std(values, ddof=1)),
        f"{prefix}_cv": coefficient_of_variation(values),
        f"{prefix}_iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
    }


def subject_features(strides: pd.DataFrame) -> dict[str, float]:
    left = strides.stride_left_s.to_numpy()
    right = strides.stride_right_s.to_numpy()
    swing = strides[["swing_left_pct", "swing_right_pct"]].mean(axis=1).to_numpy()
    support = strides.double_support_pct.to_numpy()
    span = strides.time_s.max() - strides.time_s.min()

    features = {"n_strides": len(strides), "observed_span_s": span}
    features["cadence_strides_per_min"] = 60 * len(strides) / span if span else np.nan
    features.update(_dispersion_stats(left, "stride"))
    features.update(_dispersion_stats(swing, "swing_pct"))
    features.update(_dispersion_stats(support, "double_support_pct"))
    features["asymmetry_stride"] = asymmetry_index(left, right)
    features["asymmetry_swing"] = asymmetry_index(
        strides.swing_left_pct.to_numpy(), strides.swing_right_pct.to_numpy()
    )
    features["autocorr_lag1"] = autocorrelation(left)
    features["dfa_alpha"] = detrended_fluctuation_alpha(left)
    return features


def build_feature_table(strides: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"group": group, "record": record, **subject_features(subject)}
        for (group, record), subject in strides.groupby(["group", "record"], sort=False)
    ]
    return pd.DataFrame(rows)


def window_features(
    strides: pd.DataFrame, window_strides: int = 30, min_strides: int = 20
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
                    **subject_features(window),
                }
            )
    return pd.DataFrame(rows)
