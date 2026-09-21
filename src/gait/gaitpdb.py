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
