"""Tests for the OptunaSweep wrapper and the --tune training path."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from india_quant.global_tab.lightgbm_artifact import LightGBMArtifact
from india_quant.global_tab.training_features import FEATURE_COLUMNS
from india_quant.global_tab.tuning import OptunaSweep, SweepResult

# Reuse the synthetic-DB fixture wired up in test_train_script
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_global_forecaster.py"
_spec = importlib.util.spec_from_file_location("train_global_forecaster_t", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_global_forecaster_t"] = _mod
_spec.loader.exec_module(_mod)
train = _mod.train

from tests.global_tab.test_train_script import _seed_synth_db


_EXPECTED_PARAM_KEYS = {
    "num_leaves", "learning_rate", "min_data_in_leaf", "n_estimators",
    "feature_fraction", "bagging_fraction", "min_gain_to_split", "lambda_l2",
}


def _synth_features_labels(n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, size=(n, len(FEATURE_COLUMNS))),
                     columns=FEATURE_COLUMNS)
    # Direction label depends weakly on spx_overnight_pct so the sweep has signal
    y_dir = (X["spx_overnight_pct"] + rng.normal(0, 0.5, size=n) > 0).astype(int)
    # Magnitude label in bps
    y_mag = (10 * X["spx_overnight_pct"] + rng.normal(0, 5, size=n))
    return X, y_dir, y_mag


def test_sweep_direction_returns_valid_params():
    X, y_dir, _ = _synth_features_labels(n=200)
    sweep = OptunaSweep(X, y_dir, target="direction", n_splits=3, seed=42)
    result = sweep.run(n_trials=5)
    assert isinstance(result, SweepResult)
    assert _EXPECTED_PARAM_KEYS.issubset(result.best_params.keys())
    assert np.isfinite(result.best_value)
    # ranges
    assert 7 <= result.best_params["num_leaves"] <= 63
    assert 0.01 <= result.best_params["learning_rate"] <= 0.2
    assert 10 <= result.best_params["min_data_in_leaf"] <= 60
    assert 100 <= result.best_params["n_estimators"] <= 500


def test_sweep_magnitude_returns_valid_params():
    X, _, y_mag = _synth_features_labels(n=200)
    sweep = OptunaSweep(X, y_mag, target="magnitude", quantile=0.5, n_splits=3, seed=42)
    result = sweep.run(n_trials=5)
    assert _EXPECTED_PARAM_KEYS.issubset(result.best_params.keys())
    assert np.isfinite(result.best_value)


def test_sweep_rejects_missing_quantile_for_magnitude():
    X, _, y_mag = _synth_features_labels(n=50)
    with pytest.raises(ValueError, match="quantile"):
        OptunaSweep(X, y_mag, target="magnitude")


def test_sweep_rejects_quantile_for_direction():
    X, y_dir, _ = _synth_features_labels(n=50)
    with pytest.raises(ValueError, match="quantile"):
        OptunaSweep(X, y_dir, target="direction", quantile=0.5)


def test_train_with_tune_writes_tuning_block(tmp_path):
    Session, sessions = _seed_synth_db(n_sessions=120)
    summary = train(
        index="NIFTY", target="direction",
        start=sessions[20], end=sessions[-1],
        seed=42, out=tmp_path, n_splits=3,
        session_factory=Session,
        tune=True, n_trials=5,
    )
    js = json.loads((tmp_path / "NIFTY_training_summary.json").read_text())
    assert "tuning" in js
    assert "direction" in js["tuning"]
    assert "best_params" in js["tuning"]["direction"]
    assert _EXPECTED_PARAM_KEYS.issubset(js["tuning"]["direction"]["best_params"].keys())
    # Pickle still loads as a usable LightGBM classifier
    clf = joblib.load(tmp_path / "NIFTY_direction.pkl")
    assert hasattr(clf, "predict_proba")


def test_train_without_tune_omits_tuning_block(tmp_path):
    Session, sessions = _seed_synth_db(n_sessions=120)
    train(
        index="NIFTY", target="direction",
        start=sessions[20], end=sessions[-1],
        seed=42, out=tmp_path, n_splits=3,
        session_factory=Session,
    )
    js = json.loads((tmp_path / "NIFTY_training_summary.json").read_text())
    assert js.get("tuning", {}) == {}


def test_tuned_artifact_satisfies_protocol(tmp_path):
    """End-to-end: tune -> pickle -> LightGBMArtifact loads and predicts."""
    Session, sessions = _seed_synth_db(n_sessions=120)
    train(
        index="NIFTY", target="both",
        start=sessions[20], end=sessions[-1],
        seed=42, out=tmp_path, n_splits=3,
        session_factory=Session,
        tune=True, n_trials=3,
    )
    artifact = LightGBMArtifact(models_dir=tmp_path)
    # Build a synthetic FeatureRow-like dict with the 11 columns in the
    # expected order. The artifact internally orders by FEATURE_COLUMNS.
    from india_quant.global_tab.forecaster import FeatureRow
    feats = FeatureRow(
        gift_nifty_premium_bps=0.0, spx_overnight_pct=0.0,
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
        nasdaq_overnight_pct=0.0, nifty_5d_momentum=0.0,
        nifty_realized_vol_20d=0.0, dow_int=2, is_expiry_week=1,
        days_to_rbi_policy=10,
    )
    direction, conf = artifact.predict_direction(feats, mode="balanced", index="NIFTY")
    assert 0.5 <= conf <= 0.85
    from india_quant.global_tab.types import Mode
    median, p10, p90 = artifact.predict_magnitude(feats, mode=Mode.BALANCED, index="NIFTY")
    # On synthetic noise the quantile regressors can cross; protocol only
    # guarantees 3 non-negative floats after the artifact's abs() clamp.
    assert all(np.isfinite(v) and v >= 0 for v in (median, p10, p90))
