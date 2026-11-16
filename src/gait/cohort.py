from __future__ import annotations

import pandas as pd

from . import features, quality
from .config import EXCLUDED_RECORDS

FILTER_MODES = ("none", "physiological", "strict")

SUBJECT_COLUMNS = [
    "record",
    "age",
    "sex",
    "speed_ms",
    "hoehn_yahr",
    "functional_capacity",
    "months_since_diagnosis",
]


def clean_strides(strides: pd.DataFrame, mode: str = "physiological") -> pd.DataFrame:
    if mode not in FILTER_MODES:
        raise ValueError(f"modo de filtrado desconocido: {mode}")
    strides = strides[~strides.record.isin(EXCLUDED_RECORDS)]
    if mode == "none":
        return strides.reset_index(drop=True)
    checked = quality.stride_violations(strides)
    return quality.drop_violations(checked, use_dispersion=mode == "strict")


def attach_subjects(table: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    merged = table.merge(subjects[SUBJECT_COLUMNS], on="record", how="left")
    merged["male"] = merged.sex.eq("m").astype(int)
    return merged


def subject_dataset(strides: pd.DataFrame, subjects: pd.DataFrame, side: str = "left") -> pd.DataFrame:
    table = features.build_feature_table(strides, features.extended_features, side=side)
    return attach_subjects(table, subjects)


def window_dataset(
    strides: pd.DataFrame,
    subjects: pd.DataFrame,
    window_strides: int = 30,
    side: str = "left",
) -> pd.DataFrame:
    table = features.window_features(
        strides,
        window_strides=window_strides,
        min_strides=int(0.7 * window_strides),
        extractor=features.extended_features,
        side=side,
    )
    return attach_subjects(table, subjects)


def first_strides(strides: pd.DataFrame, seconds: float) -> pd.DataFrame:
    start = strides.groupby("record").time_s.transform("min")
    return strides[strides.time_s - start <= seconds].reset_index(drop=True)
