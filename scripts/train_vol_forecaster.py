"""Train the Phase 6b LightGBM realized-vol forecaster.

Quantile regressor at q=0.50 (point forecast for the straddle strategy).
Same 11-feature frame and walk-forward CV pattern as the direction model;
label is `label_realized_vol_5d_pct` (forward 5-day realized vol,
annualized %).

Usage
-----
    PYTHONPATH=. venv/bin/python scripts/train_vol_forecaster.py \\
        --index NIFTY --start 2021-01-01 --end $(date +%F) --seed 42 \\
        --tune --n-trials 30 \\
        --tune-storage sqlite:///optuna_global_tab.db \\
        --out models/global_tab/

Outputs:
    {INDEX}_vol_q50.pkl
    {INDEX}_vol_training_summary.json   (per-fold OOS pinball loss + best_params)
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
from sklearn.model_selection import TimeSeriesSplit

from india_quant.global_tab.training_features import (
    FEATURE_COLUMNS,
    assemble_training_frame,
)


_QUANTILE = 0.50


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
    quantile: float
    lightgbm_version: str
    fold_metrics: list[FoldMetric] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tuning: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["fold_metrics"] = [asdict(m) for m in self.fold_metrics]
        return d


_DEFAULT_PARAMS = dict(
    objective="quantile", alpha=_QUANTILE,
    n_estimators=300, learning_rate=0.05,
    num_leaves=31, min_data_in_leaf=20,
    verbose=-1,
)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def _train_walkforward(
    df: pd.DataFrame, seed: int, n_splits: int, summary: TrainingSummary,
    *, override_params: dict[str, Any] | None = None,
) -> lgb.LGBMRegressor:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["label_realized_vol_5d_pct"].to_numpy(dtype=np.float64)

    base_params = {**_DEFAULT_PARAMS, **(override_params or {}), "random_state": seed}

    if n_splits >= 2 and len(df) >= n_splits + 5:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (tr, va) in enumerate(tscv.split(X)):
            m = lgb.LGBMRegressor(**base_params)
            m.fit(X[tr], y[tr])
            pinball = _pinball_loss(y[va], m.predict(X[va]), _QUANTILE)
            summary.fold_metrics.append(FoldMetric(
                fold=i, n_train=len(tr), n_valid=len(va),
                metric_name="pinball_q50_pct", metric_value=pinball,
            ))
            logger.info("vol fold {}: pinball_q50={:.3f}% (n_train={}, n_valid={})",
                        i, pinball, len(tr), len(va))

    final = lgb.LGBMRegressor(**base_params)
    final.fit(X, y)
    return final


def _run_tuning(df, *, n_splits, n_trials, storage, seed, index) -> dict:
    from india_quant.global_tab.tuning import OptunaSweep
    features = df[FEATURE_COLUMNS]
    labels = df["label_realized_vol_5d_pct"].astype(float)
    sweep = OptunaSweep(
        features, labels, target="magnitude",
        quantile=_QUANTILE, n_splits=n_splits, seed=seed,
    )
    result = sweep.run(
        n_trials=n_trials, storage=storage,
        study_name=f"global_tab_{index}_vol_q50_{seed}",
    )
    logger.info("tune vol q50: best pinball={:.3f}, params={}",
                result.best_value, result.best_params)
    return {"q50": {"best_value": result.best_value,
                    "best_params": result.best_params,
                    "n_trials": result.n_trials}}


def train(
    *,
    index: str,
    start: date, end: date,
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
    df = df.dropna(subset=["label_realized_vol_5d_pct"])
    out.mkdir(parents=True, exist_ok=True)

    summary = TrainingSummary(
        index=index, target="vol_q50",
        start=start.isoformat(), end=end.isoformat(),
        seed=seed, n_samples=len(df),
        n_features=len(FEATURE_COLUMNS),
        feature_columns=list(FEATURE_COLUMNS),
        quantile=_QUANTILE,
        lightgbm_version=lgb.__version__,
    )

    override = None
    if tune:
        tune_result = _run_tuning(
            df, n_splits=n_splits, n_trials=n_trials,
            storage=tune_storage, seed=seed, index=index,
        )
        summary.tuning.update(tune_result)
        override = tune_result["q50"]["best_params"]

    final = _train_walkforward(df, seed, n_splits, summary, override_params=override)
    pkl_path = out / f"{index}_vol_q50.pkl"
    joblib.dump(final, pkl_path, compress=3)
    logger.info("wrote {}", pkl_path)

    summary_path = out / f"{index}_vol_training_summary.json"
    summary_path.write_text(json.dumps(summary.to_json(), indent=2, default=str))
    logger.info("training summary → {}", summary_path)
    return summary


def _build_real_session_factory():
    from india_quant.data.db import get_engine
    from sqlalchemy.orm import sessionmaker
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train LightGBM realized-vol forecaster.")
    p.add_argument("--index", choices=["NIFTY", "BANKNIFTY"], required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/global_tab/")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--tune-storage", default=None)
    args = p.parse_args(argv)

    train(
        index=args.index,
        start=datetime.fromisoformat(args.start).date(),
        end=datetime.fromisoformat(args.end).date(),
        seed=args.seed, out=Path(args.out),
        n_splits=args.n_splits,
        session_factory=_build_real_session_factory(),
        tune=args.tune, n_trials=args.n_trials,
        tune_storage=args.tune_storage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
