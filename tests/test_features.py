from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait import features, quality


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_dfa_recovers_white_noise_exponent(rng):
    alphas = [features.detrended_fluctuation_alpha(rng.normal(size=2000)) for _ in range(10)]
    assert np.mean(alphas) == pytest.approx(0.5, abs=0.06)


def test_dfa_recovers_brownian_exponent(rng):
    alphas = [features.detrended_fluctuation_alpha(np.cumsum(rng.normal(size=2000))) for _ in range(10)]
    assert np.mean(alphas) == pytest.approx(1.5, abs=0.1)


def test_dfa_returns_nan_for_short_series():
    assert np.isnan(features.detrended_fluctuation_alpha(np.ones(12)))


def test_autocorrelation_recovers_ar1_coefficient(rng):
    series = np.zeros(5000)
    for i in range(1, len(series)):
        series[i] = 0.7 * series[i - 1] + rng.normal()
    assert features.autocorrelation(series) == pytest.approx(0.7, abs=0.05)


def test_autocorrelation_is_zero_for_white_noise(rng):
    assert features.autocorrelation(rng.normal(size=5000)) == pytest.approx(0.0, abs=0.05)


def test_coefficient_of_variation_is_zero_for_constant_series():
    assert features.coefficient_of_variation(np.ones(10)) == 0.0


def test_asymmetry_index_is_zero_for_identical_sides():
    assert features.asymmetry_index(np.ones(10), np.ones(10)) == 0.0


def test_asymmetry_index_uses_symmetric_normalisation():
    assert features.asymmetry_index(np.ones(10), np.full(10, 1.1)) == pytest.approx(9.5238, abs=1e-3)


def synthetic_strides(record: str, group: str, n: int, mean: float, rng: np.random.Generator) -> pd.DataFrame:
    stride = rng.normal(mean, 0.05, n)
    return pd.DataFrame(
        {
            "record": record,
            "group": group,
            "stride_index": np.arange(n),
            "time_s": np.cumsum(stride) + 20,
            "stride_left_s": stride,
            "stride_right_s": stride + rng.normal(0, 0.01, n),
            "swing_left_pct": rng.normal(38, 1, n),
            "swing_right_pct": rng.normal(38, 1, n),
            "stance_left_pct": rng.normal(62, 1, n),
            "stance_right_pct": rng.normal(62, 1, n),
            "double_support_s": rng.normal(0.3, 0.02, n),
            "double_support_pct": rng.normal(28, 1, n),
        }
    )


def test_feature_table_has_one_row_per_record(rng):
    strides = pd.concat(
        [
            synthetic_strides("control1", "control", 200, 1.1, rng),
            synthetic_strides("park1", "park", 180, 1.3, rng),
        ]
    )
    table = features.build_feature_table(strides)
    assert len(table) == 2
    assert set(table.record) == {"control1", "park1"}
    assert table.loc[table.record == "park1", "stride_mean"].item() > 1.2


def test_window_features_keeps_subject_identity(rng):
    strides = synthetic_strides("control1", "control", 95, 1.1, rng)
    windows = features.window_features(strides, window_strides=30, min_strides=20)
    assert len(windows) == 3
    assert windows.record.nunique() == 1
    assert list(windows.window_index) == [0, 1, 2]


def test_window_features_drops_incomplete_tail(rng):
    strides = synthetic_strides("control1", "control", 65, 1.1, rng)
    windows = features.window_features(strides, window_strides=30, min_strides=20)
    assert len(windows) == 2


def test_violations_flag_impossible_strides(rng):
    strides = synthetic_strides("control1", "control", 50, 1.1, rng)
    strides.loc[10, "stride_left_s"] = 55.0
    strides.loc[20, "double_support_s"] = -0.4
    strides.loc[30, "swing_left_pct"] = 120.0

    checked = quality.stride_violations(strides)
    assert checked.stride_out_of_range.sum() == 1
    assert checked.negative_double_support.sum() == 1
    assert checked.percentage_out_of_bounds.sum() == 1
    assert checked.any_violation.sum() == 3
    assert len(quality.drop_violations(checked)) == 47


def test_record_report_detects_broken_channel(rng):
    strides = synthetic_strides("hunt20", "hunt", 50, 1.0, rng)
    strides["stride_right_s"] = 42.9

    report = quality.record_report(quality.stride_violations(strides))
    assert report.left_right_ratio.item() > 40
    assert not quality.suspect_records(report).empty
