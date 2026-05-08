"""Tests for LightGBMVolArtifact — synthetic, offline."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from lightgbm import LGBMRegressor

from india_quant.global_tab.forecaster import FeatureRow
from india_quant.global_tab.lightgbm_vol_artifact import LightGBMVolArtifact
from india_quant.global_tab.training_features import FEATURE_COLUMNS


def _train_dummy_artifact(tmp_path: Path, *, index: str = "NIFTY") -> LightGBMVolArtifact:
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(80, len(FEATURE_COLUMNS)))
    y = np.abs(X[:, 0]) * 5 + rng.normal(10, 2, size=80)  # ~10% mean vol
    booster = LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=50, verbose=-1, random_state=42,
    )
    booster.fit(X, y)
    joblib.dump(booster, tmp_path / f"{index}_vol_q50.pkl")
    return LightGBMVolArtifact(models_dir=tmp_path)


def _row():
    return FeatureRow(
        gift_nifty_premium_bps=10.0, spx_overnight_pct=0.5,
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
    )


def test_predict_returns_finite_non_negative_float(tmp_path):
    art = _train_dummy_artifact(tmp_path)
    pred = art.predict_vol(_row(), index="NIFTY")
    assert pred is not None
    assert np.isfinite(pred)
    assert pred >= 0.0


def test_predict_none_when_pickle_missing(tmp_path):
    art = LightGBMVolArtifact(models_dir=tmp_path)
    assert art.predict_vol(_row(), index="BANKNIFTY") is None


def test_n_features_in_guard_falls_back_to_none(tmp_path):
    """A pickle trained on a different feature count must NOT predict."""
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(50, 5))   # only 5 features (mismatch)
    y = rng.normal(10, 2, size=50)
    booster = LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=20, verbose=-1)
    booster.fit(X, y)
    joblib.dump(booster, tmp_path / "NIFTY_vol_q50.pkl")
    art = LightGBMVolArtifact(models_dir=tmp_path)
    assert art.predict_vol(_row(), index="NIFTY") is None


def test_predict_floors_at_zero(tmp_path):
    """If the booster outputs a negative number (rare but possible at quantile
    edges with extreme inputs), the artifact floors it to 0.0."""
    art = _train_dummy_artifact(tmp_path)
    # Even a wildly out-of-distribution input shouldn't return < 0
    crazy_row = FeatureRow(
        gift_nifty_premium_bps=1e6, spx_overnight_pct=-1e6,
        dxy_delta_pct=1e6, india_vix_delta_pct=-1e6, brent_overnight_pct=1e6,
    )
    pred = art.predict_vol(crazy_row, index="NIFTY")
    assert pred is not None and pred >= 0.0
