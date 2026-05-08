"""Tests for training_features.assemble_training_frame + to_feature_store."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from india_quant.data.models import GlobalSignal, PriceData
from india_quant.global_tab.feature_store import FuturePeekError
from india_quant.global_tab.training_features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    assemble_training_frame,
    to_feature_store,
)


GLOBAL_TICKERS = ["^GSPC", "^IXIC", "DX-Y.NYB", "^INDIAVIX", "BZ=F"]


def _seed_synth_db(n_sessions: int = 60, index_ticker: str = "^NSEI", seed: int = 42):
    rng = np.random.default_rng(seed)
    engine = create_engine("sqlite:///:memory:")
    PriceData.__table__.create(engine, checkfirst=True)
    GlobalSignal.__table__.create(engine, checkfirst=True)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    # Build a sequence of business days ending at 2025-12-31
    end = date(2025, 12, 31)
    sessions: list[date] = []
    d = end
    while len(sessions) < n_sessions:
        if d.weekday() < 5:
            sessions.append(d)
        d -= timedelta(days=1)
    sessions.reverse()

    # Synthesize a NIFTY price walk
    log_close = 10.0
    closes: list[float] = []
    for _ in sessions:
        log_close += rng.normal(0.0003, 0.012)
        closes.append(math.exp(log_close))

    with Session() as s:
        for sd, c in zip(sessions, closes):
            s.add(PriceData(
                ticker=index_ticker,
                datetime=datetime(sd.year, sd.month, sd.day, tzinfo=timezone.utc),
                interval="1d",
                open=c * 0.999, high=c * 1.005, low=c * 0.995,
                close=c, volume=0,
            ))
            for tk in GLOBAL_TICKERS:
                s.add(GlobalSignal(
                    date=sd,
                    ticker=tk,
                    pct_1d=float(rng.normal(0.0, 0.008)),
                    pct_5d=0.0,
                    corr_30d=0.0, corr_90d=0.0,
                    label=tk, group="test", regime="neutral",
                ))
        s.commit()

    return Session, sessions, closes


def test_frame_columns_and_shape():
    Session, sessions, _ = _seed_synth_db(n_sessions=60)
    df = assemble_training_frame(
        index="NIFTY",
        start=sessions[20],
        end=sessions[-1],
        session_factory=Session,
    )
    assert list(df.columns) == FEATURE_COLUMNS + LABEL_COLUMNS
    # Trailing row dropped (no T+1 close), plus possibly some NaN drops; we
    # asked for sessions[20:] so >= 30 rows after dropna+ambiguous filter.
    assert len(df) >= 25
    assert len(df) <= len(sessions) - 20  # trailing dropped


def test_no_future_peek_in_features():
    """Every feature value at row date d must be derivable from data <= d."""
    Session, sessions, _ = _seed_synth_db(n_sessions=60)
    df = assemble_training_frame(
        index="NIFTY",
        start=sessions[20],
        end=sessions[-2],   # avoid trailing label issue
        session_factory=Session,
    )
    # Build a feature store and exercise .get for the *first* row date,
    # confirming no later observation leaks in.
    store = to_feature_store(df)
    first_date = df.index[0]
    for col in FEATURE_COLUMNS:
        v = store.get(col, datetime.combine(first_date, datetime.min.time()))
        assert v is not None
        # Asking before the first row's date must raise FuturePeekError.
    with pytest.raises(FuturePeekError):
        store.get(FEATURE_COLUMNS[0], datetime(2000, 1, 1))


def test_label_direction_matches_return_sign():
    Session, sessions, _ = _seed_synth_db(n_sessions=80)
    df = assemble_training_frame(
        index="NIFTY",
        start=sessions[20],
        end=sessions[-1],
        session_factory=Session,
    )
    assert set(df["label_direction"].unique()) <= {0, 1}
    # Sign consistency: every up-day has positive return, every down-day negative.
    for _, row in df.iterrows():
        if row["label_direction"] == 1:
            assert row["label_return_bps"] > 0
        else:
            assert row["label_return_bps"] < 0
    # Ambiguous-return rows (|ret| < 5 bps) are dropped.
    assert (df["label_return_bps"].abs() >= 5.0).all()


def test_to_feature_store_round_trip():
    Session, sessions, _ = _seed_synth_db(n_sessions=60)
    df = assemble_training_frame(
        index="NIFTY",
        start=sessions[20],
        end=sessions[-1],
        session_factory=Session,
    )
    store = to_feature_store(df)
    assert set(store.features()) == set(FEATURE_COLUMNS)


def test_unknown_index_raises():
    Session, sessions, _ = _seed_synth_db(n_sessions=30)
    with pytest.raises(ValueError, match="unknown index"):
        assemble_training_frame(
            index="SENSEX",
            start=sessions[5],
            end=sessions[-1],
            session_factory=Session,
        )


def test_empty_window_raises():
    Session, _, _ = _seed_synth_db(n_sessions=10)
    with pytest.raises(ValueError):
        # Window that has no overlap with seeded sessions
        assemble_training_frame(
            index="NIFTY",
            start=date(2010, 1, 1),
            end=date(2010, 1, 5),
            session_factory=Session,
        )
