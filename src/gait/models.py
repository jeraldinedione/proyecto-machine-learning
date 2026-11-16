from __future__ import annotations

from dataclasses import dataclass, field, replace

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CLASSIC_FEATURES, GAIT_FEATURES, RANDOM_STATE

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    estimator: Pipeline
    features: list[str]
    grid: dict[str, list] = field(default_factory=dict)

    def with_features(self, features: list[str], name: str | None = None) -> "ModelSpec":
        return replace(self, features=list(features), name=name or self.name)

    def labelled(self, label: str, name: str | None = None) -> "ModelSpec":
        return replace(self, label=label, name=name or self.name)

    def with_params(self, **params) -> "ModelSpec":
        estimator = clone(self.estimator).set_params(**params)
        return replace(self, estimator=estimator, grid={})


def _pipeline(model, scale: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def majority() -> ModelSpec:
    return ModelSpec(
        "mayoritaria",
        "Clase mayoritaria",
        _pipeline(DummyClassifier(strategy="most_frequent"), scale=False),
        GAIT_FEATURES,
    )


def classic_markers() -> ModelSpec:
    return ModelSpec(
        "clasico",
        "LR sobre CV y DFA",
        _pipeline(LogisticRegression(class_weight="balanced", max_iter=5000)),
        CLASSIC_FEATURES,
    )


def logistic(features: list[str] = GAIT_FEATURES) -> ModelSpec:
    model = LogisticRegression(
        solver="saga", class_weight="balanced", tol=1e-3, max_iter=20000, random_state=RANDOM_STATE
    )
    return ModelSpec(
        "logistica",
        "Regresión logística",
        _pipeline(model),
        features,
        {"model__C": C_GRID, "model__l1_ratio": [0.0, 0.5, 1.0]},
    )


def svm(features: list[str] = GAIT_FEATURES) -> ModelSpec:
    model = CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced"), ensemble=False, cv=3)
    return ModelSpec(
        "svm",
        "SVM (RBF)",
        _pipeline(model),
        features,
        {
            "model__estimator__C": [0.3, 1.0, 3.0, 10.0, 30.0],
            "model__estimator__gamma": ["scale", 0.01, 0.03, 0.1],
        },
    )


def random_forest(features: list[str] = GAIT_FEATURES) -> ModelSpec:
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
    )
    return ModelSpec(
        "bosque",
        "Bosque aleatorio",
        _pipeline(model, scale=False),
        features,
        {
            "model__max_depth": [None, 4],
            "model__min_samples_leaf": [1, 3, 5],
            "model__max_features": ["sqrt", 0.5],
        },
    )


def gradient_boosting(features: list[str] = GAIT_FEATURES) -> ModelSpec:
    model = GradientBoostingClassifier(subsample=0.8, random_state=RANDOM_STATE)
    return ModelSpec(
        "boosting",
        "Gradient boosting",
        _pipeline(model, scale=False),
        features,
        {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [1, 2, 3],
        },
    )


def comparison_suite() -> list[ModelSpec]:
    return [majority(), classic_markers(), logistic(), svm(), random_forest(), gradient_boosting()]
