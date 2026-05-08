"""Train the LightGBM forecaster artifacts for the global tab.

Usage
-----
    venv/bin/python scripts/train_global_forecaster.py \\
        --index NIFTY \\
        --target both \\
        --start 2023-01-01 \\
        --end 2025-12-31 \\
        --seed 42 \\
        --out models/global_tab/

Outputs (under --out):
    {INDEX}_direction.pkl
    {INDEX}_magnitude_q10.pkl
    {INDEX}_magnitude_q50.pkl
    {INDEX}_magnitude_q90.pkl
    {INDEX}_training_summary.json   (per-fold OOS metrics, seed, lightgbm version, feature list)

Reproducibility: same seed + same window + same data → identical
feature_importances arrays. LightGBM-pickle bytes can vary across runs
even when the model is identical, so the byte-equality check is replaced
by feature_importances equality (covered by `tests/global_tab/test_train_script.py`).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from india_quant.global_tab.training_features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    assemble_training_frame,
)


@dataclass
class FoldMetric:
    fold: int
    n_train: int
    n_valid: int
    metric_name: str
    metric_value: float


@dataclass
class TrainingSummary:
    index: str
    target: str
    start: str
    end: str
    seed: int
    n_samples: int
    n_features: int
    feature_columns: list[str]
    lightgbm_version: str
    fold_metrics: list[FoldMetric] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tuning: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["fold_metrics"] = [asdict(m) for m in self.fold_metrics]
        return d


# Sensible defaults; Phase 3c will Optuna-sweep these.
_DIRECTION_PARAMS = dict(
    n_estimators=200, learning_rate=0.05,
    num_leaves=31, min_data_in_leaf=20,
    class_weight="balanced", verbose=-1,
)
_MAGNITUDE_PARAMS = dict(
    objective="quantile",
    n_estimators=200, learning_rate=0.05,
    num_leaves=31, min_data_in_leaf=20,
    verbose=-1,
)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss; lower is better."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def _train_direction(
    df: pd.DataFrame, seed: int, n_splits: int, summary: TrainingSummary,
    *, override_params: dict[str, Any] | None = None,
) -> lgb.LGBMClassifier:
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label_direction"].to_numpy().astype(int)

    base_params = {**_DIRECTION_PARAMS, **(override_params or {}), "random_state": seed}

    if n_splits >= 2 and len(df) >= n_splits + 5:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (tr, va) in enumerate(tscv.split(X)):
            clf = lgb.LGBMClassifier(**base_params)
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[va])[:, 1]
            ll = log_loss(y[va], proba, labels=[0, 1]) if len(set(y[va])) > 1 else float("nan")
            summary.fold_metrics.append(FoldMetric(
                fold=i, n_train=len(tr), n_valid=len(va),
                metric_name="logloss", metric_value=ll,
            ))
            logger.info("direction fold {}: logloss={:.4f} (n_train={}, n_valid={})", i, ll, len(tr), len(va))
    else:
        summary.notes.append(
            f"Skipped walk-forward (n_splits={n_splits}, n_rows={len(df)})"
        )

    final = lgb.LGBMClassifier(**base_params)
    final.fit(X, y)
    return final


def _train_magnitude_quantile(
    df: pd.DataFrame, alpha: float, seed: int, n_splits: int,
    summary: TrainingSummary, label_key: str,
    *, override_params: dict[str, Any] | None = None,
) -> lgb.LGBMRegressor:
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label_return_bps"].to_numpy().astype(float)

    base_params = {
        **_MAGNITUDE_PARAMS, **(override_params or {}),
        "alpha": alpha, "random_state": seed,
    }

    if n_splits >= 2 and len(df) >= n_splits + 5:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (tr, va) in enumerate(tscv.split(X)):
            m = lgb.LGBMRegressor(**base_params)
            m.fit(X[tr], y[tr])
            pinball = _pinball_loss(y[va], m.predict(X[va]), alpha)
            summary.fold_metrics.append(FoldMetric(
                fold=i, n_train=len(tr), n_valid=len(va),
                metric_name=f"pinball_{label_key}",
                metric_value=pinball,
            ))
            logger.info(
                "magnitude {} fold {}: pinball={:.2f} bps (n_train={}, n_valid={})",
                label_key, i, pinball, len(tr), len(va),
            )

    final = lgb.LGBMRegressor(**base_params)
    final.fit(X, y)
    return final


def _run_tuning(
    df: pd.DataFrame, *, target_kind: str, n_splits: int, n_trials: int,
    storage: str | None, seed: int, index: str,
) -> dict[str, Any]:
    """Run Optuna for a given target_kind and return best params."""
    from india_quant.global_tab.tuning import OptunaSweep

    features = df[FEATURE_COLUMNS]
    if target_kind == "direction":
        labels = df["label_direction"].astype(int)
        sweep = OptunaSweep(features, labels, target="direction",
                            n_splits=n_splits, seed=seed)
        result = sweep.run(n_trials=n_trials, storage=storage,
                           study_name=f"global_tab_{index}_direction_{seed}")
        logger.info("tune direction: best logloss={:.4f}, params={}",
                    result.best_value, result.best_params)
        return {"direction": {"best_value": result.best_value,
                              "best_params": result.best_params,
                              "n_trials": result.n_trials}}
    else:
        out: dict[str, Any] = {}
        for alpha, key in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
            labels = df["label_return_bps"].astype(float)
            sweep = OptunaSweep(features, labels, target="magnitude",
                                quantile=alpha, n_splits=n_splits, seed=seed)
            result = sweep.run(n_trials=n_trials, storage=storage,
                               study_name=f"global_tab_{index}_magnitude_{key}_{seed}")
            logger.info("tune magnitude {}: best pinball={:.2f}, params={}",
                        key, result.best_value, result.best_params)
            out[key] = {"best_value": result.best_value,
                        "best_params": result.best_params,
                        "n_trials": result.n_trials}
        return out


def train(
    *,
    index: str,
    target: str,
    start: date,
    end: date,
    seed: int,
    out: Path,
    n_splits: int,
    session_factory: Callable,
    tune: bool = False,
    n_trials: int = 30,
    tune_storage: str | None = None,
) -> TrainingSummary:
    df = assemble_training_frame(
        index=index, start=start, end=end, session_factory=session_factory,
    )
    out.mkdir(parents=True, exist_ok=True)

    summary = TrainingSummary(
        index=index, target=target,
        start=start.isoformat(), end=end.isoformat(),
        seed=seed, n_samples=len(df),
        n_features=len(FEATURE_COLUMNS),
        feature_columns=list(FEATURE_COLUMNS),
        lightgbm_version=lgb.__version__,
    )

    direction_override: dict[str, Any] | None = None
    magnitude_overrides: dict[str, dict[str, Any]] = {}
    if tune:
        if target in {"direction", "both"}:
            tune_result = _run_tuning(
                df, target_kind="direction", n_splits=n_splits,
                n_trials=n_trials, storage=tune_storage, seed=seed, index=index,
            )
            summary.tuning.update(tune_result)
            direction_override = tune_result["direction"]["best_params"]
        if target in {"magnitude", "both"}:
            tune_result = _run_tuning(
                df, target_kind="magnitude", n_splits=n_splits,
                n_trials=n_trials, storage=tune_storage, seed=seed, index=index,
            )
            summary.tuning.update(tune_result)
            for key, payload in tune_result.items():
                magnitude_overrides[key] = payload["best_params"]

    if target in {"direction", "both"}:
        clf = _train_direction(df, seed, n_splits, summary,
                               override_params=direction_override)
        joblib.dump(clf, out / f"{index}_direction.pkl", compress=3)
        logger.info("wrote {}", out / f"{index}_direction.pkl")

    if target in {"magnitude", "both"}:
        for alpha, key in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
            m = _train_magnitude_quantile(
                df, alpha, seed, n_splits, summary, key,
                override_params=magnitude_overrides.get(key),
            )
            joblib.dump(m, out / f"{index}_magnitude_{key}.pkl", compress=3)
            logger.info("wrote {}", out / f"{index}_magnitude_{key}.pkl")

    summary_path = out / f"{index}_training_summary.json"
    summary_path.write_text(json.dumps(summary.to_json(), indent=2, default=str))
    logger.info("training summary → {}", summary_path)

    return summary


def _build_real_session_factory():
    """Lazy import — keeps tests free of .env / DB requirements."""
    from india_quant.data.db import get_engine
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train LightGBM forecaster for global tab.")
    p.add_argument("--index", choices=["NIFTY", "BANKNIFTY"], required=True)
    p.add_argument("--target", choices=["direction", "magnitude", "both"], default="both")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end",   required=True, help="YYYY-MM-DD")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--out",   default="models/global_tab/")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--tune", action="store_true",
                   help="Run Optuna sweep before final fit")
    p.add_argument("--n-trials", type=int, default=30,
                   help="Optuna trials per target (only with --tune)")
    p.add_argument("--tune-storage", default=None,
                   help="Optuna storage URI, e.g. sqlite:///optuna_global_tab.db")
    args = p.parse_args(argv)

    train(
        index=args.index,
        target=args.target,
        start=datetime.fromisoformat(args.start).date(),
        end=datetime.fromisoformat(args.end).date(),
        seed=args.seed,
        out=Path(args.out),
        n_splits=args.n_splits,
        session_factory=_build_real_session_factory(),
        tune=args.tune,
        n_trials=args.n_trials,
        tune_storage=args.tune_storage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
