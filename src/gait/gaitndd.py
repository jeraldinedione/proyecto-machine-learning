from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GAITNDD, MISSING_TOKEN, STRIDE_COLUMNS
from .download import require

SUBJECT_FILE = "subject-description.txt"
RECORDS_FILE = "RECORDS"

RAW_SUBJECT_COLUMNS = [
    "record",
    "group_declared",
    "age",
    "height_m",
    "weight_kg",
    "sex",
    "speed_ms",
    "severity_raw",
]

NUMERIC_SUBJECT_COLUMNS = ["age", "height_m", "weight_kg", "speed_ms", "severity_raw"]


def _root() -> Path:
    return require(GAITNDD)


def list_records() -> list[str]:
    return (_root() / RECORDS_FILE).read_text().split()


def group_of(record: str) -> str:
    return re.match(r"^[a-z]+", record).group()


def read_subject_table() -> pd.DataFrame:
    text = (_root() / SUBJECT_FILE).read_text()
    repaired = re.sub(r"\n(?=\t)", "", text)
    rows = [line.split("\t") for line in repaired.strip().split("\n")[1:]]

    table = pd.DataFrame(rows, columns=RAW_SUBJECT_COLUMNS)
    table[NUMERIC_SUBJECT_COLUMNS] = (
        table[NUMERIC_SUBJECT_COLUMNS].replace(MISSING_TOKEN, np.nan).apply(pd.to_numeric)
    )
    table["group"] = table.record.map(group_of)
    table["sex"] = table.sex.str.lower()

    table["hoehn_yahr"] = table.severity_raw.where(table.group == "park")
    table["functional_capacity"] = table.severity_raw.where(table.group == "hunt")
    table["months_since_diagnosis"] = table.severity_raw.where(table.group == "als")

    ordered = ["record", "group", "group_declared", "age", "sex", "height_m", "weight_kg", "speed_ms"]
    return table[ordered + ["severity_raw", "hoehn_yahr", "functional_capacity", "months_since_diagnosis"]]


def read_strides(record: str) -> pd.DataFrame:
    values = np.loadtxt(_root() / f"{record}.ts")
    strides = pd.DataFrame(values, columns=STRIDE_COLUMNS)
    strides.insert(0, "record", record)
    strides.insert(1, "group", group_of(record))
    strides.insert(2, "stride_index", np.arange(len(strides)))
    return strides


def read_all_strides() -> pd.DataFrame:
    return pd.concat([read_strides(r) for r in list_records()], ignore_index=True)


def read_force_signal(record: str) -> pd.DataFrame:
    import wfdb

    signals, fields = wfdb.rdsamp(str(_root() / record))
    time = np.arange(len(signals)) / fields["fs"]
    return pd.DataFrame({"time_s": time, "left": signals[:, 0], "right": signals[:, 1]})


def read_recording_headers() -> pd.DataFrame:
    rows = []
    for record in list_records():
        name, n_signals, fs, n_samples = (_root() / f"{record}.hea").read_text().split("\n")[0].split()
        rows.append(
            {
                "record": name,
                "group": group_of(name),
                "n_signals": int(n_signals),
                "sampling_hz": int(fs),
                "n_samples": int(n_samples),
                "duration_s": int(n_samples) / int(fs),
            }
        )
    return pd.DataFrame(rows)
