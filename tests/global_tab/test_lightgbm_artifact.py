"""Tests for LightGBMArtifact: protocol conformance + missing-pickle fallback."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pytest

from india_quant.global_tab.forecaster import FeatureRow
from india_quant.global_tab.lightgbm_artifact import (
    ArtifactMissingError,
    LightGBMArtifact,
)
from india_quant.global_tab.training_features import FEATURE_COLUMNS
from india_quant.global_tab.types import Direction, Mode


def _train_tiny_bundle(tmp_path: Path, index: str = "NIFTY", seed: int = 42) -> Path:
    """Fit small LGBM models on synthetic data and pickle them.

    The classifier is fit so that proba_up correlates with the first feature
    column (so direction tests become deterministic without depending on
    training noise).
    """
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    # Strong signal: feature[0] (gift_nifty_premium_bps) determines direction
    y_dir = (X[:, 0] > 0).astype(int)
    y_mag = X[:, 0] * 30.0 + rng.normal(0, 5.0, size=n)  # bps

    clf = lgb.LGBMClassifier(
        n_estimators=50, num_leaves=15, min_data_in_leaf=5,
        learning_rate=0.1, random_state=seed, verbose=-1,
    )
    clf.fit(X, y_dir)

    quantile_models = {}
    for alpha, key in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha,
            n_estimators=50, num_leaves=15, min_data_in_leaf=5,
            learning_rate=0.1, random_state=seed, verbose=-1,
        )
        m.fit(X, y_mag)
        quantile_models[key] = m

    out = tmp_path / "models"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / f"{index}_direction.pkl", compress=3)
    for key, m in quantile_models.items():
        joblib.dump(m, out / f"{index}_magnitude_{key}.pkl", compress=3)
    return out


def _row(premium_bps: float = 50.0) -> FeatureRow:
    return FeatureRow(
        gift_nifty_premium_bps=premium_bps,
        spx_overnight_pct=0.005,
        dxy_delta_pct=-0.001,
        india_vix_delta_pct=0.02,
        brent_overnight_pct=0.001,
        nasdaq_overnight_pct=0.004,
        nifty_5d_momentum=0.01,
        nifty_realized_vol_20d=0.012,
        dow_int=2,
        is_expiry_week=1,
        days_to_rbi_policy=10,
    )


def test_protocol_conformance_long(tmp_path):
    out = _train_tiny_bundle(tmp_path)
    artifact = LightGBMArtifact(models_dir=out)

    # Strong positive premium → LONG
    direction, confidence = artifact.predict_direction(_row(100.0), Mode.BALANCED)
    assert direction == Direction.LONG
    assert 0.5 <= confidence <= 0.85

    median, p10, p90 = artifact.predict_magnitude(_row(100.0), Mode.BALANCED)
    assert isinstance(median, float)
    assert p10 <= median <= p90 or median == 0.0  # post-clamping invariant
    assert all(v >= 0 for v in (median, p10, p90))


def test_protocol_conformance_short(tmp_path):
    out = _train_tiny_bundle(tmp_path)
    artifact = LightGBMArtifact(models_dir=out)
    direction, confidence = artifact.predict_direction(_row(-100.0), Mode.BALANCED)
    assert direction == Direction.SHORT
    assert 0.5 <= confidence <= 0.85


def test_no_trade_in_dead_zone(tmp_path):
    """Threshold logic: with proba_up ≈ 0.55, conservative mode (threshold 0.10)
    should map to NO_TRADE while aggressive (0.02) maps to LONG.
    """
    out = _train_tiny_bundle(tmp_path)
    artifact = LightGBMArtifact(models_dir=out)
    artifact._load_index("NIFTY")  # warm cache
    # Replace the classifier with a stub that always returns proba_up = 0.55
    class _FakeClf:
        def predict_proba(self, X):
            return np.array([[0.45, 0.55]])
    artifact._cache["NIFTY"]["direction"] = _FakeClf()
    row = _row(premium_bps=10.0)
    assert artifact.predict_direction(row, Mode.CONSERVATIVE) == (Direction.NO_TRADE, 0.0)
    direction, _ = artifact.predict_direction(row, Mode.AGGRESSIVE)
    assert direction == Direction.LONG


def test_missing_pickle_raises(tmp_path):
    artifact = LightGBMArtifact(models_dir=tmp_path / "does_not_exist")
    with pytest.raises(ArtifactMissingError):
        artifact.predict_direction(_row(50.0), Mode.BALANCED)


def test_missing_pickle_magnitude_returns_zeros(tmp_path):
    artifact = LightGBMArtifact(models_dir=tmp_path / "does_not_exist")
    # predict_magnitude swallows the error and returns 0,0,0 so the sizer
    # can fall back without a 500.
    median, p10, p90 = artifact.predict_magnitude(_row(50.0), Mode.BALANCED)
    assert (median, p10, p90) == (0.0, 0.0, 0.0)


def test_lazy_load_caches_per_index(tmp_path):
    out = _train_tiny_bundle(tmp_path, index="NIFTY")
    _train_tiny_bundle(tmp_path, index="BANKNIFTY")
    # Reuse the same dir — _train_tiny_bundle writes into out=tmp_path/'models'
    artifact = LightGBMArtifact(models_dir=out)
    # First call loads pickles
    artifact.predict_direction(_row(50.0), Mode.BALANCED, index="NIFTY")
    artifact.predict_direction(_row(50.0), Mode.BALANCED, index="BANKNIFTY")
    assert set(artifact._cache.keys()) == {"NIFTY", "BANKNIFTY"}


def test_artifact_name():
    assert LightGBMArtifact.name == "lightgbm"


def test_none_features_imputed_to_zero(tmp_path):
    out = _train_tiny_bundle(tmp_path)
    artifact = LightGBMArtifact(models_dir=out)
    # Phase 3a-style FeatureRow with the new fields left at their None defaults
    sparse = FeatureRow(
        gift_nifty_premium_bps=80.0,
        spx_overnight_pct=None,
        dxy_delta_pct=None,
        india_vix_delta_pct=None,
        brent_overnight_pct=None,
    )
    direction, confidence = artifact.predict_direction(sparse, Mode.BALANCED)
    assert direction in {Direction.LONG, Direction.SHORT, Direction.NO_TRADE}
    assert 0.0 <= confidence <= 0.85
