"""Tests for scripts/train_global_forecaster.train() — synthetic, offline."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from india_quant.data.models import GlobalSignal, PriceData


# Load the script as a module by path (it lives outside the package tree).
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_global_forecaster.py"
_spec = importlib.util.spec_from_file_location("train_global_forecaster", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_global_forecaster"] = _mod
_spec.loader.exec_module(_mod)
train = _mod.train


GLOBAL_TICKERS = ["^GSPC", "^IXIC", "DX-Y.NYB", "^INDIAVIX", "BZ=F"]


def _seed_synth_db(n_sessions: int = 120, seed: int = 7):
    """Synthetic NIFTY + global_signals with a deliberate edge so the
    direction model does better than the always-up baseline.
    """
    rng = np.random.default_rng(seed)
    engine = create_engine("sqlite:///:memory:")
    PriceData.__table__.create(engine, checkfirst=True)
    GlobalSignal.__table__.create(engine, checkfirst=True)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    end = date(2025, 12, 31)
    sessions: list[date] = []
    d = end
    while len(sessions) < n_sessions:
        if d.weekday() < 5:
            sessions.append(d)
        d -= timedelta(days=1)
    sessions.reverse()

    # SPX overnight drives next-day NIFTY direction (synthetic edge).
    spx_pcts = rng.normal(0.0, 0.012, size=n_sessions)
    log_close = 10.0
    closes: list[float] = []
    for i in range(n_sessions):
        # Tomorrow's return correlates with today's spx_overnight + noise.
        nxt = i + 1 if i + 1 < n_sessions else i
        drift = 0.7 * spx_pcts[nxt]
        log_close += drift + rng.normal(0.0, 0.005)
        closes.append(math.exp(log_close))

    with Session() as s:
        for i, sd in enumerate(sessions):
            s.add(PriceData(
                ticker="^NSEI",
                datetime=datetime(sd.year, sd.month, sd.day, tzinfo=timezone.utc),
                interval="1d",
                open=closes[i] * 0.999, high=closes[i] * 1.005, low=closes[i] * 0.995,
                close=closes[i], volume=0,
            ))
            for tk in GLOBAL_TICKERS:
                pct = float(spx_pcts[i] if tk == "^GSPC" else rng.normal(0.0, 0.008))
                s.add(GlobalSignal(
                    date=sd, ticker=tk, pct_1d=pct, pct_5d=0.0,
                    corr_30d=0.0, corr_90d=0.0,
                    label=tk, group="test", regime="neutral",
                ))
        s.commit()

    return Session, sessions


def test_train_writes_pickles_and_summary(tmp_path):
    Session, sessions = _seed_synth_db(n_sessions=120)
    summary = train(
        index="NIFTY", target="both",
        start=sessions[20], end=sessions[-1],
        seed=42, out=tmp_path, n_splits=3,
        session_factory=Session,
    )
    assert (tmp_path / "NIFTY_direction.pkl").exists()
    assert (tmp_path / "NIFTY_magnitude_q10.pkl").exists()
    assert (tmp_path / "NIFTY_magnitude_q50.pkl").exists()
    assert (tmp_path / "NIFTY_magnitude_q90.pkl").exists()
    js = json.loads((tmp_path / "NIFTY_training_summary.json").read_text())
    assert js["index"] == "NIFTY"
    assert js["seed"] == 42
    assert js["n_features"] == 11
    assert len(js["fold_metrics"]) >= 3  # at least direction folds


def test_reproducibility_same_seed_same_feature_importances(tmp_path):
    Session, sessions = _seed_synth_db(n_sessions=120)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    train(
        index="NIFTY", target="direction",
        start=sessions[20], end=sessions[-1],
        seed=42, out=out_a, n_splits=2,
        session_factory=Session,
    )
    train(
        index="NIFTY", target="direction",
        start=sessions[20], end=sessions[-1],
        seed=42, out=out_b, n_splits=2,
        session_factory=Session,
    )
    import joblib
    a = joblib.load(out_a / "NIFTY_direction.pkl")
    b = joblib.load(out_b / "NIFTY_direction.pkl")
    np.testing.assert_array_equal(a.feature_importances_, b.feature_importances_)


def test_target_direction_only_skips_magnitude(tmp_path):
    Session, sessions = _seed_synth_db(n_sessions=80)
    train(
        index="NIFTY", target="direction",
        start=sessions[20], end=sessions[-1],
        seed=42, out=tmp_path, n_splits=2,
        session_factory=Session,
    )
    assert (tmp_path / "NIFTY_direction.pkl").exists()
    assert not (tmp_path / "NIFTY_magnitude_q50.pkl").exists()


def test_target_magnitude_only_skips_direction(tmp_path):
    Session, sessions = _seed_synth_db(n_sessions=80)
    train(
        index="NIFTY", target="magnitude",
        start=sessions[20], end=sessions[-1],
        seed=42, out=tmp_path, n_splits=2,
        session_factory=Session,
    )
    assert not (tmp_path / "NIFTY_direction.pkl").exists()
    assert (tmp_path / "NIFTY_magnitude_q10.pkl").exists()
    assert (tmp_path / "NIFTY_magnitude_q50.pkl").exists()
    assert (tmp_path / "NIFTY_magnitude_q90.pkl").exists()
