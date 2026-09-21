from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    PHYSIOLOGICAL_STRIDE_RANGE_S,
    ROBUST_OUTLIER_THRESHOLD,
    STRIDE_INTERVAL_COLUMNS,
)

MAD_TO_SIGMA = 1.4826
PERCENTAGE_COLUMNS = [
    "swing_left_pct",
    "swing_right_pct",
    "stance_left_pct",
    "stance_right_pct",
    "double_support_pct",
]
VIOLATION_COLUMNS = [
    "stride_out_of_range",
    "negative_double_support",
    "percentage_out_of_bounds",
]


def robust_z_score(values: pd.Series) -> pd.Series:
    median = values.median()
    scale = (values - median).abs().median() * MAD_TO_SIGMA
    if scale == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - median) / scale


def stride_violations(
    strides: pd.DataFrame,
    bounds: tuple[float, float] = PHYSIOLOGICAL_STRIDE_RANGE_S,
    threshold: float = ROBUST_OUTLIER_THRESHOLD,
) -> pd.DataFrame:
    low, high = bounds
    intervals = strides[STRIDE_INTERVAL_COLUMNS]
    percentages = strides[PERCENTAGE_COLUMNS]

    checked = strides.copy()
    checked["stride_out_of_range"] = ((intervals < low) | (intervals > high)).any(axis=1)
    checked["negative_double_support"] = strides.double_support_s < 0
    checked["percentage_out_of_bounds"] = ((percentages < 0) | (percentages > 100)).any(axis=1)
    checked["any_violation"] = checked[VIOLATION_COLUMNS].any(axis=1)

    checked["robust_z"] = checked.groupby("record").stride_left_s.transform(robust_z_score).abs()
    checked["dispersion_outlier"] = checked.robust_z > threshold
    return checked


def record_report(checked: pd.DataFrame) -> pd.DataFrame:
    report = checked.groupby(["group", "record"], sort=False).agg(
        n_strides=("stride_index", "size"),
        first_stride_s=("time_s", "min"),
        last_stride_s=("time_s", "max"),
        median_stride_left_s=("stride_left_s", "median"),
        median_stride_right_s=("stride_right_s", "median"),
        max_stride_s=("stride_left_s", "max"),
        n_out_of_range=("stride_out_of_range", "sum"),
        n_negative_support=("negative_double_support", "sum"),
        n_bad_percentage=("percentage_out_of_bounds", "sum"),
        n_violations=("any_violation", "sum"),
        n_dispersion_outliers=("dispersion_outlier", "sum"),
    )
    report["pct_violations"] = 100 * report.n_violations / report.n_strides
    report["observed_span_s"] = report.last_stride_s - report.first_stride_s
    report["left_right_ratio"] = report.median_stride_right_s / report.median_stride_left_s
    return report.reset_index()


def suspect_records(
    report: pd.DataFrame,
    max_pct_violations: float = 10.0,
    left_right_tolerance: float = 0.15,
) -> pd.DataFrame:
    asymmetric = (report.left_right_ratio - 1).abs() > left_right_tolerance
    corrupted = report.pct_violations > max_pct_violations
    flagged = report.loc[asymmetric | corrupted].copy()
    flagged["reason"] = np.where(
        asymmetric[flagged.index] & corrupted[flagged.index],
        "canal inconsistente y violaciones masivas",
        np.where(asymmetric[flagged.index], "canal izquierdo/derecho inconsistente", "violaciones masivas"),
    )
    return flagged


def drop_violations(checked: pd.DataFrame, use_dispersion: bool = False) -> pd.DataFrame:
    mask = checked.any_violation | checked.dispersion_outlier if use_dispersion else checked.any_violation
    audit_columns = VIOLATION_COLUMNS + ["any_violation", "robust_z", "dispersion_outlier"]
    return checked.loc[~mask].drop(columns=audit_columns).reset_index(drop=True)


def violation_counts(checked: pd.DataFrame) -> pd.DataFrame:
    counts = checked.groupby("group")[VIOLATION_COLUMNS + ["any_violation"]].sum()
    counts["n_strides"] = checked.groupby("group").size()
    counts["pct_any"] = 100 * counts.any_violation / counts.n_strides
    return counts
