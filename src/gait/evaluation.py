from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedGroupKFold

from .config import RANDOM_STATE
from .models import ModelSpec

Split = tuple[np.ndarray, np.ndarray]


@dataclass
class NestedResult:
    spec: ModelSpec
    predictions: pd.DataFrame
    params: pd.DataFrame
    importances: pd.DataFrame | None = None

    @property
    def classes(self) -> list[str]:
        return [c.removeprefix("p_") for c in self.predictions.columns if c.startswith("p_")]


def record_splits(
    subjects: pd.DataFrame,
    target: str = "group",
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> list[Split]:
    records = subjects.record.to_numpy()
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return [(records[train], records[test]) for train, test in splitter.split(records, subjects[target])]


def fit_model(
    spec: ModelSpec,
    train: pd.DataFrame,
    target: str = "group",
    inner_splits: int = 5,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
):
    X, y = train[spec.features], train[target].to_numpy()
    if not spec.grid:
        return clone(spec.estimator).fit(X, y), {}
    search = GridSearchCV(
        clone(spec.estimator),
        spec.grid,
        scoring="balanced_accuracy",
        cv=StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state),
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y, groups=train.record.to_numpy())
    return search.best_estimator_, search.best_params_


def _record_probabilities(model, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    proba = pd.DataFrame(
        model.predict_proba(test[features]),
        columns=[f"p_{c}" for c in model.classes_],
        index=test.index,
    )
    proba["record"] = test.record.to_numpy()
    return proba.groupby("record", sort=False).mean()


def _fold_importances(model, test: pd.DataFrame, spec: ModelSpec, target: str, seed: int) -> pd.Series:
    result = permutation_importance(
        model,
        test[spec.features],
        test[target].to_numpy(),
        scoring="balanced_accuracy",
        n_repeats=20,
        random_state=seed,
    )
    return pd.Series(result.importances_mean, index=spec.features)


def nested_cv(
    spec: ModelSpec,
    data: pd.DataFrame,
    splits: list[Split],
    target: str = "group",
    n_splits: int = 5,
    inner_splits: int = 5,
    n_jobs: int = -1,
    importances: bool = False,
    random_state: int = RANDOM_STATE,
) -> NestedResult:
    labels = data.groupby("record", sort=False)[target].first()
    predictions, chosen, weights = [], [], []

    for index, (train_records, test_records) in enumerate(splits):
        train = data[data.record.isin(train_records)]
        test = data[data.record.isin(test_records)]
        model, params = fit_model(spec, train, target, inner_splits, n_jobs, random_state + index)

        proba = _record_probabilities(model, test, spec.features)
        columns = list(proba.columns)
        proba["y_true"] = labels.loc[proba.index].to_numpy()
        proba["y_pred"] = [c.removeprefix("p_") for c in proba[columns].idxmax(axis=1)]
        proba["repeat"] = index // n_splits
        proba["fold"] = index % n_splits
        predictions.append(proba.reset_index())
        chosen.append({"repeat": index // n_splits, "fold": index % n_splits, **params})

        if importances:
            weights.append(_fold_importances(model, test, spec, target, random_state + index))

    ordered = ["repeat", "fold", "record", "y_true", "y_pred"]
    table = pd.concat(predictions, ignore_index=True)
    table = table[ordered + [c for c in table.columns if c.startswith("p_")]]
    importance_table = pd.DataFrame(weights) if importances else None
    return NestedResult(spec, table, pd.DataFrame(chosen), importance_table)


def _auc(part: pd.DataFrame, classes: list[str]) -> float:
    scores = part[[f"p_{c}" for c in classes]].to_numpy()
    if len(classes) == 2:
        return roc_auc_score(part.y_true == classes[1], scores[:, 1])
    return roc_auc_score(part.y_true, scores, multi_class="ovr", average="macro", labels=classes)


def repeat_metrics(result: NestedResult, order: list[str] | None = None) -> pd.DataFrame:
    classes = result.classes
    order = order or classes
    rows = []
    for repeat, part in result.predictions.groupby("repeat"):
        recalls = recall_score(part.y_true, part.y_pred, labels=order, average=None, zero_division=0)
        rows.append(
            {
                "repeat": repeat,
                "balanced_accuracy": balanced_accuracy_score(part.y_true, part.y_pred),
                "macro_f1": f1_score(part.y_true, part.y_pred, labels=order, average="macro", zero_division=0),
                "auc": _auc(part, classes),
                **{f"recall_{c}": r for c, r in zip(order, recalls)},
            }
        )
    return pd.DataFrame(rows)


def fold_scores(result: NestedResult) -> pd.Series:
    return result.predictions.groupby(["repeat", "fold"]).apply(
        lambda part: balanced_accuracy_score(part.y_true, part.y_pred), include_groups=False
    )


def summarize(results: list[NestedResult], order: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for result in results:
        metrics = repeat_metrics(result, order).drop(columns="repeat")
        row = {"modelo": result.spec.label}
        for column in metrics:
            row[f"{column}_mean"] = metrics[column].mean()
            row[f"{column}_sd"] = metrics[column].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("modelo")


def corrected_ttest(scores_a: pd.Series, scores_b: pd.Series, n_train: int, n_test: int) -> tuple[float, float]:
    differences = (scores_a - scores_b).to_numpy()
    k = len(differences)
    variance = np.var(differences, ddof=1)
    if variance == 0:
        return np.nan, np.nan
    statistic = differences.mean() / np.sqrt((1 / k + n_test / n_train) * variance)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=k - 1))


def bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    y_true, scores = np.asarray(y_true, dtype=bool), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(random_state)
    positives, negatives = np.flatnonzero(y_true), np.flatnonzero(~y_true)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = np.concatenate(
            [rng.choice(positives, len(positives)), rng.choice(negatives, len(negatives))]
        )
        values[i] = roc_auc_score(y_true[sample], scores[sample])
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(roc_auc_score(y_true, scores)), float(low), float(high)


def pooled_confusion(result: NestedResult, order: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(result.predictions.y_true, result.predictions.y_pred, labels=order)
    return pd.DataFrame(matrix, index=order, columns=order)


def pairwise_auc(result: NestedResult, order: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(np.nan, index=order, columns=order)
    for i, first in enumerate(order):
        for second in order[i + 1 :]:
            values = []
            for _, part in result.predictions.groupby("repeat"):
                pair = part[part.y_true.isin([first, second])]
                score = pair[f"p_{first}"] / (pair[f"p_{first}"] + pair[f"p_{second}"])
                values.append(roc_auc_score(pair.y_true == first, score))
            table.loc[first, second] = table.loc[second, first] = np.mean(values)
    return table


def subject_hit_rate(result: NestedResult) -> pd.DataFrame:
    predictions = result.predictions.assign(hit=lambda d: d.y_true == d.y_pred)
    summary = predictions.groupby(["record", "y_true"]).agg(
        hit_rate=("hit", "mean"),
        most_common=("y_pred", lambda s: s.mode().iloc[0]),
    )
    return summary.reset_index()


def most_frequent_params(result: NestedResult) -> dict:
    params = result.params.drop(columns=["repeat", "fold"])
    if params.empty:
        return {}
    keys = params.apply(lambda row: repr(tuple(row)), axis=1)
    row = params[keys == keys.mode().iloc[0]].iloc[0]
    return {key: _native(value) for key, value in row.items()}


def _native(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer() and value > 1:
        return int(value)
    return value
