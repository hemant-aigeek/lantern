"""Fragility curves: P(asset fails | local gust, asset attributes).

Rather than a single "will it break" classifier, Lantern fits a *fragility
curve* per asset family: the probability of failure as a function of the
gust the asset actually sees, shifted by age, canopy, construction class
and ground saturation. A curve is a thing an engineer can look at and argue
with; a feature-importance plot is not.

The curves are logistic in log-gust, fitted on (asset, storm) pairs from the
damage history. Validation is grouped by storm, because the question that
matters is "how well does this generalise to a storm we have not seen", not
"how well does it fit the storms we have".
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .grid import ASSET_TYPES, Asset, Grid
from .storms import History, Storm

FEATURES = ["log_gust", "age", "age_missing", "vegetation", "wet", "pole_class", "length_km", "underground",
            *[f"type_{t}" for t in ASSET_TYPES]]


def features(a: Asset, gust_mph: float, wet: float, median_age: float) -> list[float]:
    row = [
        math.log(max(gust_mph, 1.0) / 40.0),
        a.age_years if a.age_known else median_age,
        0.0 if a.age_known else 1.0,
        a.vegetation,
        wet,
        float(a.pole_class) if a.type == "pole" else 0.0,
        a.length_km,
        1.0 if a.underground else 0.0,
    ]
    row.extend(1.0 if a.type == t else 0.0 for t in ASSET_TYPES)
    return row


@dataclass
class Metrics:
    storms: int
    pairs: int
    positives: int
    base_rate: float
    cv_pr_auc: float
    cv_pr_auc_lift: float
    cv_roc_auc: float
    cv_brier: float
    reliability: list[dict] = field(default_factory=list)  # (predicted mean, observed rate, n) per bin

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class FragilityModel:
    coef: dict[str, float]
    intercept: float
    median_age: float
    metrics: Metrics | None = None

    # -- inference ---------------------------------------------------------
    def failure_probability(self, a: Asset, gust_mph: float, wet: float) -> float:
        x = features(a, gust_mph, wet, self.median_age)
        z = self.intercept + sum(self.coef[f] * v for f, v in zip(FEATURES, x, strict=True))
        return 1.0 / (1.0 + math.exp(-z))

    def curve(self, a: Asset, wet: float = 0.3, gusts: tuple[int, ...] = (30, 40, 50, 60, 70, 80, 90)) -> list[dict]:
        return [{"gust_mph": g, "p_fail": round(self.failure_probability(a, g, wet), 4)} for g in gusts]

    def drivers(self, a: Asset, gust_mph: float, wet: float) -> list[dict]:
        """Which attributes push this asset's failure odds, in log-odds, largest first."""
        x = features(a, gust_mph, wet, self.median_age)
        parts = [(f, self.coef[f] * v) for f, v in zip(FEATURES, x, strict=True) if abs(self.coef[f] * v) > 1e-6]
        parts.sort(key=lambda t: -abs(t[1]))
        return [{"factor": f, "log_odds": round(v, 3)} for f, v in parts[:5]]

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> FragilityModel:
        return pickle.loads(path.read_bytes())


def _pairs(grid: Grid, history: History, median_age: float):
    X, y, groups = [], [], []
    for s in history.storms:
        damaged = history.damaged_in(s.id)
        for a in grid.assets.values():
            X.append(features(a, s.gust_at(a.x, a.y), s.wet, median_age))
            y.append(1 if a.id in damaged else 0)
            groups.append(s.id)
    return np.array(X), np.array(y), np.array(groups)


def _fit(X, y) -> LogisticRegression:
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(X, y)
    return m


def fit(grid: Grid, history: History, folds: int = 5) -> FragilityModel:
    ages = [a.age_years for a in grid.assets.values() if a.age_known]
    median_age = float(np.median(ages)) if ages else 20.0
    X, y, groups = _pairs(grid, history, median_age)
    if y.sum() == 0:
        raise ValueError("history has no damage records to learn from")

    # grouped cross-validation: every fold holds out whole storms
    oof = np.zeros(len(y))
    n_groups = len(set(groups))
    k = max(2, min(folds, n_groups))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if y[tr].sum() == 0:
            continue
        oof[te] = _fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    base = float(y.mean())
    pr = float(average_precision_score(y, oof))
    metrics = Metrics(
        storms=n_groups, pairs=int(len(y)), positives=int(y.sum()), base_rate=base,
        cv_pr_auc=pr, cv_pr_auc_lift=pr / base if base else 0.0,
        cv_roc_auc=float(roc_auc_score(y, oof)), cv_brier=float(brier_score_loss(y, oof)),
        reliability=_reliability(oof, y),
    )
    m = _fit(X, y)
    coef = {f: float(c) for f, c in zip(FEATURES, m.coef_[0], strict=True)}
    return FragilityModel(coef=coef, intercept=float(m.intercept_[0]), median_age=median_age, metrics=metrics)


def _reliability(p: np.ndarray, y: np.ndarray, bins: int = 8) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out.append({"predicted": round(float(p[m].mean()), 5), "observed": round(float(y[m].mean()), 5),
                    "n": int(m.sum())})
    return out


def scenario_probabilities(model: FragilityModel, grid: Grid, storm: Storm) -> dict[str, float]:
    """Failure probability for every asset under one storm."""
    return {a.id: model.failure_probability(a, storm.gust_at(a.x, a.y), storm.wet) for a in grid.assets.values()}
