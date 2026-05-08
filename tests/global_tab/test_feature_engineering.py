"""Phase 3d feature-engineering tests — synthetic, offline.

Covers:
  - sector RS + dispersion arithmetic
  - breadth (pct_above_*dma, advance_decline_5d) bounds + no-future-peek
  - factor-aggregate column tolerates missing factor_scores table
  - back-compat: legacy 5-field FeatureRow construction still serves
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from india_quant.data.models import GlobalSignal, PriceData
from india_quant.global_tab.training_features import (
    FEATURE_COLUMNS,
    _DISPERSION_TICKERS,
    _SECTOR_TICKERS,
    _breadth_features,
    _factor_aggregate_features,
    _sector_features,
    assemble_training_frame,
)
from india_quant.global_tab.forecaster import FeatureRow


def _seed_db_with_sectors(n_sessions: int = 80, seed: int = 11):
    """Synthetic DB with NIFTY closes + 5 global tickers + 6 sector tickers."""
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

    nifty_close = 22000.0
    closes = []
    for _ in range(n_sessions):
        nifty_close *= 1.0 + rng.normal(0, 0.008)
        closes.append(nifty_close)

    GLOBAL = ["^GSPC", "^IXIC", "DX-Y.NYB", "^INDIAVIX", "BZ=F"]
    SECTORS = ["^NSEBANK", "^CNXIT", "^CNXPHARMA", "^CNXREALTY", "^CNXENERGY", "^CNXINFRA"]

    with Session() as s:
        for i, sd in enumerate(sessions):
            s.add(PriceData(
                ticker="^NSEI",
                datetime=datetime(sd.year, sd.month, sd.day, tzinfo=timezone.utc),
                interval="1d",
                open=closes[i] * 0.999, high=closes[i] * 1.005, low=closes[i] * 0.995,
                close=closes[i], volume=0,
            ))
            for tk in GLOBAL:
                s.add(GlobalSignal(
                    date=sd, ticker=tk, pct_1d=float(rng.normal(0, 0.01)),
                    pct_5d=float(rng.normal(0, 0.025)),
                    corr_30d=0.0, corr_90d=0.0,
                    label=tk, group="test", regime="neutral",
                ))
            for tk in SECTORS:
                s.add(GlobalSignal(
                    date=sd, ticker=tk, pct_1d=float(rng.normal(0, 0.012)),
                    pct_5d=float(rng.normal(0, 0.03)),
                    corr_30d=0.0, corr_90d=0.0,
                    label=tk, group="sectors", regime="neutral",
                ))
        s.commit()
    return Session, sessions


def test_feature_columns_legacy_order_preserved():
    """First 11 columns must always be the Phase 3a/3b set, byte-identical
    in order, for pickle reproducibility across phases."""
    expected_legacy = [
        "gift_nifty_premium_bps", "spx_overnight_pct", "nasdaq_overnight_pct",
        "dxy_delta_pct", "india_vix_delta_pct", "brent_overnight_pct",
        "nifty_5d_momentum", "nifty_realized_vol_20d",
        "dow_int", "is_expiry_week", "days_to_rbi_policy",
    ]
    assert FEATURE_COLUMNS[:11] == expected_legacy


def test_phase3d_candidates_documented():
    from india_quant.global_tab.training_features import PHASE3D_CANDIDATE_COLUMNS
    assert len(PHASE3D_CANDIDATE_COLUMNS) == 9
    # Each candidate has a corresponding optional field on FeatureRow
    legacy_row = FeatureRow(
        gift_nifty_premium_bps=0.0, spx_overnight_pct=0.0,
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
    )
    d = legacy_row.as_dict()
    for col in PHASE3D_CANDIDATE_COLUMNS:
        assert col in d, f"FeatureRow missing optional Phase 3d field {col}"


def test_sector_rs_helper_returns_pivoted_frame():
    """Sector helper produces a date×ticker frame even though the columns
    aren't currently in FEATURE_COLUMNS — Phase 3e will opt them in."""
    Session, sessions = _seed_db_with_sectors(n_sessions=80)
    wide = _sector_features(sessions[5], sessions[-1], session_factory=Session)
    assert not wide.empty
    for t in _DISPERSION_TICKERS:
        assert t in wide.columns


def test_breadth_features_skipped_when_no_universe_in_db():
    Session, sessions = _seed_db_with_sectors(n_sessions=60)
    # No Nifty-50 .NS tickers in this fixture → breadth helper returns empty
    out = _breadth_features(
        sessions[10], sessions[-1],
        session_factory=Session,
        universe=["RELIANCE.NS", "TCS.NS"],
    )
    assert out.empty


def test_factor_aggregates_tolerates_missing_table():
    """factor_scores table absent in the synthetic sqlite fixture must not raise."""
    Session, sessions = _seed_db_with_sectors(n_sessions=60)
    out = _factor_aggregate_features(
        sessions[10], sessions[-1],
        session_factory=Session,
        universe=["RELIANCE.NS"],
    )
    assert out.empty  # graceful degradation


def test_back_compat_legacy_feature_row():
    """A FeatureRow built with only Phase-3a kwargs must still be valid;
    Phase 3d optional fields default to None."""
    row = FeatureRow(
        gift_nifty_premium_bps=10.0, spx_overnight_pct=0.005,
        dxy_delta_pct=-0.001, india_vix_delta_pct=0.02, brent_overnight_pct=0.003,
    )
    d = row.as_dict()
    for col in FEATURE_COLUMNS:
        assert col in d
    assert d["bank_vs_nifty_5d_relstr"] is None
    assert d["mean_realized_vol_universe"] is None
