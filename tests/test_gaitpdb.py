from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait import gaitpdb


def test_normalise_height_converts_centimetres():
    values = pd.Series([1.65, 160.0, 1.90, 185.0])
    assert list(gaitpdb.normalise_height(values)) == [1.65, 1.60, 1.90, 1.85]


def test_normalise_height_leaves_metres_untouched():
    values = pd.Series([1.45, 1.70, 2.04])
    assert list(gaitpdb.normalise_height(values)) == [1.45, 1.70, 2.04]


def test_implausible_heights_flags_out_of_range():
    subjects = pd.DataFrame(
        {
            "record_id": ["A", "B", "C"],
            "study": ["Ga", "Ju", "Si"],
            "group": ["park", "control", "park"],
            "sex": ["m", "f", "m"],
            "height_raw": [1.75, 0.9, 250.0],
            "height_m": [1.75, 0.9, 2.5],
        }
    )
    flagged = gaitpdb.implausible_heights(subjects)
    assert list(flagged.record_id) == ["B", "C"]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("GaPt03_01", {"subject": "GaPt03", "study": "Ga", "group": "park", "walk": 1, "dual_task": False}),
        ("JuCo11_01", {"subject": "JuCo11", "study": "Ju", "group": "control", "walk": 1, "dual_task": False}),
        ("GaPt07_10", {"subject": "GaPt07", "study": "Ga", "group": "park", "walk": 10, "dual_task": True}),
    ],
)
def test_parse_record(name, expected):
    parsed = gaitpdb.parse_record(name)
    for key, value in expected.items():
        assert parsed[key] == value


def test_record_pattern_rejects_malformed_identifier():
    assert gaitpdb.RECORD_PATTERN.match("Juc010") is None
    assert gaitpdb.RECORD_PATTERN.match("demographics") is None
