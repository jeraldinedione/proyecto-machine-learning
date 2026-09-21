from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("GAIT_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    version: str
    folder: str

    @property
    def url(self) -> str:
        return f"https://physionet.org/static/published-projects/{self.slug}/{self.folder}.zip"

    @property
    def path(self) -> Path:
        return RAW_DIR / self.folder


GAITNDD = DatasetSpec("gaitndd", "1.0.0", "gait-in-neurodegenerative-disease-database-1.0.0")
GAITPDB = DatasetSpec("gaitpdb", "1.0.0", "gait-in-parkinsons-disease-1.0.0")

GROUP_ORDER = ["control", "park", "hunt", "als"]

GROUP_LABELS = {
    "control": "Control",
    "park": "Parkinson",
    "hunt": "Huntington",
    "als": "ELA",
}

GROUP_COLORS = {
    "control": "#4C72B0",
    "park": "#DD8452",
    "hunt": "#55A868",
    "als": "#C44E52",
}

SEVERITY_SCALES = {
    "control": "no aplica",
    "park": "Hoehn & Yahr (1-5, mayor = mas avanzado)",
    "hunt": "capacidad funcional total (0-13, menor = mas deteriorado)",
    "als": "meses desde el diagnostico",
}

STRIDE_COLUMNS = [
    "time_s",
    "stride_left_s",
    "stride_right_s",
    "swing_left_s",
    "swing_right_s",
    "swing_left_pct",
    "swing_right_pct",
    "stance_left_s",
    "stance_right_s",
    "stance_left_pct",
    "stance_right_pct",
    "double_support_s",
    "double_support_pct",
]

STRIDE_INTERVAL_COLUMNS = ["stride_left_s", "stride_right_s"]

RECORDING_SECONDS = 300.0
SETTLING_SECONDS = 20.0
PHYSIOLOGICAL_STRIDE_RANGE_S = (0.5, 2.5)
ROBUST_OUTLIER_THRESHOLD = 4.0

MISSING_TOKEN = "MISSING"
RANDOM_STATE = 42
