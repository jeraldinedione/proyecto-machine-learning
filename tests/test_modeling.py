from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait import cohort, evaluation, features, gaitpdb, models


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_sample_entropy_is_higher_for_noise_than_for_periodic_signal(rng):
    noise = rng.normal(size=600)
    periodic = np.sin(np.linspace(0, 60 * np.pi, 600))
    assert features.sample_entropy(noise) > features.sample_entropy(periodic) + 1.0


def test_sample_entropy_of_white_noise_matches_reference(rng):
    values = [features.sample_entropy(rng.normal(size=1000)) for _ in range(5)]
    assert np.mean(values) == pytest.approx(2.2, abs=0.15)


def test_sample_entropy_undefined_for_tiny_series():
    assert np.isnan(features.sample_entropy(np.array([1.0, 2.0])))


def test_successive_difference_cv_ignores_slow_drift():
    drift = np.linspace(1.0, 1.2, 200)
    alternating = 1.1 + 0.05 * (-1) ** np.arange(200)
    assert features.successive_difference_cv(drift) < features.successive_difference_cv(alternating) / 10


def test_extended_features_keeps_original_descriptors(rng):
    stride = rng.normal(1.1, 0.04, 120)
    strides = pd.DataFrame(
        {
            "time_s": np.cumsum(stride),
            "stride_left_s": stride,
            "stride_right_s": stride,
            "swing_left_pct": 38.0,
            "swing_right_pct": 38.0,
            "double_support_pct": 25.0,
        }
    )
    base = features.subject_features(strides)
    extended = features.extended_features(strides)
    assert set(base) < set(extended)
    assert extended["stride_cv"] == base["stride_cv"]


def synthetic_forces(stride_s: float = 1.1, n_strides: int = 40, fs: int = 100) -> pd.DataFrame:
    time = np.arange(int(n_strides * stride_s * fs)) / fs
    phase_left = (time % stride_s) / stride_s
    phase_right = ((time + stride_s / 2) % stride_s) / stride_s
    left = np.where(phase_left < 0.62, 700.0, 0.0)
    right = np.where(phase_right < 0.62, 700.0, 0.0)
    return pd.DataFrame({"time_s": time, "left_total": left, "right_total": right})


def test_stride_table_recovers_synthetic_gait_cycle():
    strides = gaitpdb.stride_table(synthetic_forces())
    assert len(strides) > 30
    assert strides.stride_left_s.median() == pytest.approx(1.1, abs=0.011)
    assert strides.stride_right_s.median() == pytest.approx(1.1, abs=0.011)
    assert strides.stance_left_pct.median() == pytest.approx(62, abs=1.5)
    assert strides.double_support_pct.median() == pytest.approx(24, abs=2)


def test_contact_mask_removes_short_flickers():
    force = np.zeros(300)
    force[50:150] = 600.0
    force[100:103] = 0.0
    force[200:203] = 600.0
    mask = gaitpdb.contact_mask(force)
    assert mask[50:150].all()
    assert not mask[150:].any()


def test_clean_strides_excludes_broken_record(rng):
    stride = rng.normal(1.1, 0.04, 30)
    rows = pd.DataFrame(
        {
            "stride_index": np.arange(30),
            "time_s": np.cumsum(stride),
            "stride_left_s": stride,
            "stride_right_s": stride,
            "swing_left_pct": 38.0,
            "swing_right_pct": 38.0,
            "stance_left_pct": 62.0,
            "stance_right_pct": 62.0,
            "double_support_s": 0.3,
            "double_support_pct": 25.0,
        }
    )
    strides = pd.concat(
        [rows.assign(record="hunt20", group="hunt"), rows.assign(record="hunt1", group="hunt")]
    )
    assert set(cohort.clean_strides(strides).record) == {"hunt1"}


def subject_table(rng: np.random.Generator, per_class: int = 12) -> pd.DataFrame:
    groups = np.repeat(["control", "park", "hunt", "als"], per_class)
    shift = pd.Series(groups).map({"control": 0.0, "park": 1.0, "hunt": 2.0, "als": 3.0}).to_numpy()
    table = pd.DataFrame(
        {name: shift + rng.normal(0, 0.5, len(groups)) for name in models.GAIT_FEATURES}
    )
    table.insert(0, "record", [f"{g}{i}" for i, g in enumerate(groups)])
    table.insert(1, "group", groups)
    return table


def test_record_splits_never_share_subjects(rng):
    table = subject_table(rng)
    for train, test in evaluation.record_splits(table, n_repeats=2):
        assert not set(train) & set(test)
        assert len(set(train) | set(test)) == len(table)


def test_nested_cv_aggregates_windows_by_subject(rng):
    table = subject_table(rng)
    windows = pd.concat([table.assign(window_index=i) for i in range(3)], ignore_index=True)
    splits = evaluation.record_splits(table, n_repeats=1)
    result = evaluation.nested_cv(models.logistic().with_params(model__C=1.0), windows, splits, n_jobs=1)
    assert len(result.predictions) == len(table)
    assert result.predictions.record.is_unique


def test_nested_cv_separates_easy_classes(rng):
    table = subject_table(rng)
    splits = evaluation.record_splits(table, n_repeats=1)
    result = evaluation.nested_cv(models.logistic(), table, splits, n_jobs=1)
    metrics = evaluation.repeat_metrics(result)
    assert metrics.balanced_accuracy.mean() > 0.8


def test_majority_baseline_scores_chance(rng):
    table = subject_table(rng)
    extra = table[table.group == "hunt"].assign(record=lambda d: d.record + "b")
    table = pd.concat([table, extra], ignore_index=True)
    splits = evaluation.record_splits(table, n_repeats=1)
    result = evaluation.nested_cv(models.majority(), table, splits, n_jobs=1)
    assert evaluation.repeat_metrics(result).balanced_accuracy.item() == pytest.approx(0.25)


def test_corrected_ttest_is_more_conservative_than_naive():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.65, 0.05, 50))
    b = pd.Series(rng.normal(0.60, 0.05, 50))
    _, corrected = evaluation.corrected_ttest(a, b, n_train=50, n_test=13)
    _, naive = evaluation.corrected_ttest(a, b, n_train=10**9, n_test=1)
    assert corrected > naive


def test_most_frequent_params_handles_missing_depth():
    params = pd.DataFrame(
        {
            "repeat": [0, 0, 0],
            "fold": [0, 1, 2],
            "model__max_depth": [None, None, 4],
            "model__min_samples_leaf": [3, 3, 1],
        }
    )
    result = evaluation.NestedResult(models.random_forest(), pd.DataFrame(), params)
    assert evaluation.most_frequent_params(result) == {"model__max_depth": None, "model__min_samples_leaf": 3}


def test_absolute_threshold_overrides_relative():
    force = np.concatenate([np.zeros(50), np.full(100, 40.0), np.zeros(50), np.full(100, 800.0), np.zeros(50)])
    relative = gaitpdb.contact_mask(force)
    absolute = gaitpdb.contact_mask(force, threshold_n=20.0)
    assert not relative[50:150].any()
    assert absolute[50:150].all()
    assert not gaitpdb.contact_mask(force, threshold_n=50.0)[50:150].any()
