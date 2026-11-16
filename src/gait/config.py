from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("GAIT_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"


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

GROUP_SHORT = {
    "control": "Ctrl",
    "park": "EP",
    "hunt": "EH",
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

EXCLUDED_RECORDS = ["hunt20"]

TIMING_FEATURES = [
    "stride_mean",
    "stride_cv",
    "stride_iqr",
    "stride_diff_cv",
    "asymmetry_stride",
    "autocorr_lag1",
    "dfa_alpha",
    "sample_entropy",
]

PHASE_FEATURES = [
    "swing_pct_cv",
    "double_support_pct_mean",
    "double_support_pct_cv",
    "asymmetry_swing",
]

GAIT_FEATURES = TIMING_FEATURES + PHASE_FEATURES

CLASSIC_FEATURES = ["stride_cv", "dfa_alpha"]

P1_FEATURES = [
    "n_strides",
    "observed_span_s",
    "cadence_strides_per_min",
    "stride_mean",
    "stride_sd",
    "stride_cv",
    "stride_iqr",
    "swing_pct_mean",
    "swing_pct_sd",
    "swing_pct_cv",
    "swing_pct_iqr",
    "double_support_pct_mean",
    "double_support_pct_sd",
    "double_support_pct_cv",
    "double_support_pct_iqr",
    "asymmetry_stride",
    "asymmetry_swing",
    "autocorr_lag1",
    "dfa_alpha",
]

DURATION_FEATURES = ["n_strides", "observed_span_s", "cadence_strides_per_min"]

DEMOGRAPHIC_FEATURES = ["age", "male"]
SPEED_FEATURES = ["speed_ms"]

FEATURE_LABELS = {
    "stride_mean": "duración media",
    "stride_cv": "CV de zancada",
    "stride_iqr": "IQR de zancada",
    "stride_diff_cv": "variab. sucesiva",
    "asymmetry_stride": "asimetría de zancada",
    "autocorr_lag1": "autocorrelación",
    "dfa_alpha": "DFA α",
    "sample_entropy": "entropía muestral",
    "swing_pct_mean": "balanceo medio",
    "swing_pct_cv": "CV de balanceo",
    "double_support_pct_mean": "apoyo doble medio",
    "double_support_pct_cv": "CV de apoyo doble",
    "asymmetry_swing": "asimetría de balanceo",
    "age": "edad",
    "male": "sexo masculino",
    "speed_ms": "velocidad",
}
