from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GAITPDB
from .download import require

DEMOGRAPHICS_FILE = "demographics.txt"
RECORD_PATTERN = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)$")

DEMOGRAPHIC_COLUMNS = [
    "record_id",
    "study",
    "group_code",
    "subject_number",
    "sex_code",
    "age",
    "height_raw",
    "weight_kg",
    "hoehn_yahr",
    "updrs",
    "updrs_motor",
    "tuag_s",
    "speed_walk01",
    "speed_walk02",
    "speed_walk03",
    "speed_walk04",
    "speed_walk05",
    "speed_walk06",
    "speed_walk07",
    "speed_walk10",
]

NUMERIC_COLUMNS = DEMOGRAPHIC_COLUMNS[5:]

GROUP_CODES = {1: "park", 2: "control"}
SEX_CODES = {1: "m", 2: "f"}
STUDY_LABELS = {
    "Ga": "doble tarea (Yogev et al., 2005)",
    "Ju": "estimulacion auditiva ritmica (Hausdorff et al., 2007)",
    "Si": "caminadora (Frenkel-Toledo et al., 2005)",
}

FORCE_COLUMNS = (
    ["time_s"]
    + [f"left_{i}" for i in range(1, 9)]
    + [f"right_{i}" for i in range(1, 9)]
    + ["left_total", "right_total"]
)

SAMPLING_HZ = 100
PLAUSIBLE_HEIGHT_RANGE_M = (1.3, 2.1)


def _root() -> Path:
    return require(GAITPDB)


def read_subject_table() -> pd.DataFrame:
    lines = (_root() / DEMOGRAPHICS_FILE).read_text().split("\n")
    rows = [line.split("\t")[: len(DEMOGRAPHIC_COLUMNS)] for line in lines[1:] if line.strip()]

    table = pd.DataFrame(rows, columns=DEMOGRAPHIC_COLUMNS)
    table[NUMERIC_COLUMNS] = table[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    table["group_code"] = pd.to_numeric(table.group_code)
    table["sex_code"] = pd.to_numeric(table.sex_code)

    table["group"] = table.group_code.map(GROUP_CODES)
    table["sex"] = table.sex_code.map(SEX_CODES)
    table["group_from_id"] = np.where(table.record_id.str[2:4] == "Pt", "park", "control")
    table["height_m"] = normalise_height(table.height_raw)

    ordered = ["record_id", "study", "group", "group_from_id", "sex", "age", "height_m", "weight_kg"]
    clinical = ["hoehn_yahr", "updrs", "updrs_motor", "tuag_s"]
    speeds = [c for c in DEMOGRAPHIC_COLUMNS if c.startswith("speed_")]
    return table[ordered + clinical + speeds + ["height_raw"]]


def normalise_height(values: pd.Series) -> pd.Series:
    return values.where(values < 3, values / 100)


def implausible_heights(subjects: pd.DataFrame) -> pd.DataFrame:
    low, high = PLAUSIBLE_HEIGHT_RANGE_M
    outside = (subjects.height_m < low) | (subjects.height_m > high)
    return subjects.loc[outside, ["record_id", "study", "group", "sex", "height_raw", "height_m"]]


def list_records() -> list[str]:
    return sorted(p.stem for p in _root().glob("*.txt") if RECORD_PATTERN.match(p.stem))


def parse_record(name: str) -> dict[str, object]:
    study, cohort, number, walk = RECORD_PATTERN.match(name).groups()
    return {
        "record": name,
        "subject": f"{study}{cohort}{number}",
        "study": study,
        "group": "park" if cohort == "Pt" else "control",
        "subject_number": int(number),
        "walk": int(walk),
        "dual_task": walk == "10",
    }


def record_index() -> pd.DataFrame:
    return pd.DataFrame([parse_record(name) for name in list_records()])


def read_forces(record: str) -> pd.DataFrame:
    values = np.loadtxt(_root() / f"{record}.txt")
    return pd.DataFrame(values, columns=FORCE_COLUMNS)


CONTACT_FRACTION = 0.05
MIN_PHASE_S = 0.1


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return edges[::2], edges[1::2]


def _debounce(mask: np.ndarray, min_samples: int) -> np.ndarray:
    mask = mask.copy()
    for start, end in zip(*_runs(~mask)):
        if end - start < min_samples and start > 0 and end < len(mask):
            mask[start:end] = True
    for start, end in zip(*_runs(mask)):
        if end - start < min_samples:
            mask[start:end] = False
    return mask


def contact_threshold(force: np.ndarray, fraction: float = CONTACT_FRACTION) -> float:
    return float(fraction * np.percentile(force, 99))


def contact_mask(
    force: np.ndarray,
    fraction: float = CONTACT_FRACTION,
    min_phase_s: float = MIN_PHASE_S,
    sampling_hz: int = SAMPLING_HZ,
    threshold_n: float | None = None,
) -> np.ndarray:
    threshold = contact_threshold(force, fraction) if threshold_n is None else threshold_n
    return _debounce(force > threshold, int(round(min_phase_s * sampling_hz)))


def gait_events(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    onsets, offsets = _runs(mask)
    return onsets[onsets > 0], offsets[offsets < len(mask)]


def _next_event(events: np.ndarray, after: int) -> int | None:
    position = np.searchsorted(events, after, side="right")
    return int(events[position]) if position < len(events) else None


def _stride_row(
    start: int,
    end: int,
    events: dict[str, np.ndarray],
    both: np.ndarray,
    sampling_hz: int,
) -> dict[str, float] | None:
    left_off = _next_event(events["left_off"], start)
    right_on = _next_event(events["right_on"], start)
    if left_off is None or right_on is None or left_off >= end or right_on >= end:
        return None
    right_next = _next_event(events["right_on"], right_on)
    right_off = _next_event(events["right_off"], right_on)
    if right_next is None or right_off is None or right_off >= right_next:
        return None

    stride_left = (end - start) / sampling_hz
    stride_right = (right_next - right_on) / sampling_hz
    stance_left = (left_off - start) / sampling_hz
    stance_right = (right_off - right_on) / sampling_hz
    double_support = both[start:end].sum() / sampling_hz
    return {
        "time_s": end / sampling_hz,
        "stride_left_s": stride_left,
        "stride_right_s": stride_right,
        "swing_left_s": stride_left - stance_left,
        "swing_right_s": stride_right - stance_right,
        "swing_left_pct": 100 * (stride_left - stance_left) / stride_left,
        "swing_right_pct": 100 * (stride_right - stance_right) / stride_right,
        "stance_left_s": stance_left,
        "stance_right_s": stance_right,
        "stance_left_pct": 100 * stance_left / stride_left,
        "stance_right_pct": 100 * stance_right / stride_right,
        "double_support_s": double_support,
        "double_support_pct": 100 * double_support / stride_left,
    }


def stride_table(forces: pd.DataFrame, sampling_hz: int = SAMPLING_HZ, **detection) -> pd.DataFrame:
    left = contact_mask(forces.left_total.to_numpy(), sampling_hz=sampling_hz, **detection)
    right = contact_mask(forces.right_total.to_numpy(), sampling_hz=sampling_hz, **detection)
    left_on, left_off = gait_events(left)
    right_on, right_off = gait_events(right)
    events = {"left_off": left_off, "right_on": right_on, "right_off": right_off}

    rows = [
        _stride_row(start, end, events, left & right, sampling_hz)
        for start, end in zip(left_on[:-1], left_on[1:])
    ]
    return pd.DataFrame([row for row in rows if row is not None])


def read_strides(record: str, **detection) -> pd.DataFrame:
    strides = stride_table(read_forces(record), **detection)
    info = parse_record(record)
    strides.insert(0, "record", info["subject"])
    strides.insert(1, "group", info["group"])
    strides.insert(2, "stride_index", np.arange(len(strides)))
    strides.insert(3, "study", info["study"])
    return strides


def read_baseline_strides(**detection) -> pd.DataFrame:
    records = record_index()
    baseline = records.loc[records.walk == 1, "record"]
    return pd.concat([read_strides(record, **detection) for record in baseline], ignore_index=True)
