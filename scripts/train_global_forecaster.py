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
    df: pd.DataFrame, seed: int, n_splits: int, summary: TrainingSummary
) -> lgb.LGBMClassifier:
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label_direction"].to_numpy().astype(int)

    if n_splits >= 2 and len(df) >= n_splits + 5:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (tr, va) in enumerate(tscv.split(X)):
            params = {**_DIRECTION_PARAMS, "random_state": seed}
            clf = lgb.LGBMClassifier(**params)
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

    final = lgb.LGBMClassifier(**{**_DIRECTION_PARAMS, "random_state": seed})
    final.fit(X, y)
    return final


def _train_magnitude_quantile(
    df: pd.DataFrame, alpha: float, seed: int, n_splits: int,
    summary: TrainingSummary, label_key: str,
) -> lgb.LGBMRegressor:
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label_return_bps"].to_numpy().astype(float)

    if n_splits >= 2 and len(df) >= n_splits + 5:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (tr, va) in enumerate(tscv.split(X)):
            params = {**_MAGNITUDE_PARAMS, "alpha": alpha, "random_state": seed}
            m = lgb.LGBMRegressor(**params)
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

    final = lgb.LGBMRegressor(**{**_MAGNITUDE_PARAMS, "alpha": alpha, "random_state": seed})
    final.fit(X, y)
    return final


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

    if target in {"direction", "both"}:
        clf = _train_direction(df, seed, n_splits, summary)
        joblib.dump(clf, out / f"{index}_direction.pkl", compress=3)
        logger.info("wrote {}", out / f"{index}_direction.pkl")

    if target in {"magnitude", "both"}:
        for alpha, key in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
            m = _train_magnitude_quantile(df, alpha, seed, n_splits, summary, key)
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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
