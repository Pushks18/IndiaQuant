"""Optuna sweep wrapper for the LightGBM forecaster.

Used by `scripts/train_global_forecaster.py --tune`. Keeps the runtime
artifact path identical: a tuned booster is still pickled to
`models/global_tab/{INDEX}_{target}.pkl` and loaded by `LightGBMArtifact`.

Hyperparameter ranges intentionally favour regularization — the global-tab
training set is small (~1000 rows), so `num_leaves` is capped at 63 and
`min_data_in_leaf` floored at 10 to keep individual trees from memorising.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

Target = Literal["direction", "magnitude"]


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


@dataclass
class SweepResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    target: str
    quantile: float | None


class OptunaSweep:
    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        target: Target,
        quantile: float | None = None,
        n_splits: int = 5,
        seed: int = 42,
    ) -> None:
        if target == "magnitude" and quantile is None:
            raise ValueError("quantile must be set for magnitude target")
        if target == "direction" and quantile is not None:
            raise ValueError("quantile must be None for direction target")
        self.X = features.to_numpy()
        self.y = labels.to_numpy()
        self.target = target
        self.quantile = quantile
        self.n_splits = n_splits
        self.seed = seed

    def _suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "num_leaves":          trial.suggest_int("num_leaves", 7, 63, log=True),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 10, 60),
            "n_estimators":        trial.suggest_int("n_estimators", 100, 500),
            "feature_fraction":    trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction":    trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq":        5,
            "min_gain_to_split":   trial.suggest_float("min_gain_to_split", 0.0, 0.1),
            "lambda_l2":           trial.suggest_float("lambda_l2", 1e-3, 1.0, log=True),
            "random_state":        self.seed,
            "verbose":             -1,
        }

    def _score_fold(self, params: dict[str, Any], tr: np.ndarray, va: np.ndarray) -> float:
        if self.target == "direction":
            params = {**params, "class_weight": "balanced"}
            clf = lgb.LGBMClassifier(**params)
            clf.fit(self.X[tr], self.y[tr].astype(int))
            proba = clf.predict_proba(self.X[va])[:, 1]
            yv = self.y[va].astype(int)
            if len(set(yv)) < 2:
                return float("nan")
            return float(log_loss(yv, proba, labels=[0, 1]))
        else:
            params = {**params, "objective": "quantile", "alpha": self.quantile}
            m = lgb.LGBMRegressor(**params)
            m.fit(self.X[tr], self.y[tr].astype(float))
            return _pinball_loss(self.y[va].astype(float), m.predict(self.X[va]), self.quantile)

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest(trial)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_scores: list[float] = []
        for fold_idx, (tr, va) in enumerate(tscv.split(self.X)):
            score = self._score_fold(params, tr, va)
            if np.isnan(score):
                continue
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not fold_scores:
            return float("inf")
        return float(np.mean(fold_scores))

    def run(self, n_trials: int = 30, storage: str | None = None,
            study_name: str | None = None) -> SweepResult:
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_warmup_steps=2)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=False)
        return SweepResult(
            best_params=dict(study.best_params),
            best_value=float(study.best_value),
            n_trials=len(study.trials),
            target=self.target,
            quantile=self.quantile,
        )
