"""Shared fixtures for global_tab tests."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from india_quant.data.models import OptionChain, PriceData


def _create_minimal_schema(engine):
    """Create only the tables global_tab tests touch (skip Postgres-only ARRAY columns)."""
    OptionChain.__table__.create(engine, checkfirst=True)
    PriceData.__table__.create(engine, checkfirst=True)


@pytest.fixture
def tmp_session_factory_empty():
    """In-memory SQLite session factory with empty schema."""
    engine = create_engine("sqlite:///:memory:")
    _create_minimal_schema(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def tmp_session_factory_with_chain():
    """In-memory SQLite seeded with one weekly chain (5 strikes × CE/PE) and a NIFTY spot row.

    Returns (session_factory, spot).
    """
    engine = create_engine("sqlite:///:memory:")
    _create_minimal_schema(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    spot = 24500.0
    expiry = date(2026, 5, 7)
    other_expiry = date(2026, 5, 14)
    trade_date = date(2026, 5, 5)
    strikes = [24300, 24400, 24500, 24600, 24700]

    with Session() as s:
        # Underlying spot via PriceData (NIFTY index ticker convention used by repo: ^NSEI)
        s.add(PriceData(
            ticker="^NSEI",
            datetime=datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=timezone.utc),
            interval="1d",
            open=spot - 50, high=spot + 100, low=spot - 100,
            close=spot, volume=0,
        ))
        for k in strikes:
            for ot, base in (("CE", max(spot - k, 50)), ("PE", max(k - spot, 50))):
                s.add(OptionChain(
                    underlying="NIFTY",
                    trade_date=trade_date,
                    expiry=expiry,
                    strike=float(k),
                    option_type=ot,
                    last_price=base,
                    bid=base * 0.98,
                    ask=base * 1.02,
                    iv=15.0,
                    open_interest=10_000,
                    oi_change=0,
                    volume=1_000,
                ))
        # Other-expiry row that must NOT come back
        s.add(OptionChain(
            underlying="NIFTY",
            trade_date=trade_date,
            expiry=other_expiry,
            strike=99999.0,
            option_type="CE",
            last_price=1.0, bid=1.0, ask=1.0, iv=15.0,
            open_interest=0, oi_change=0, volume=0,
        ))
        s.commit()

    return Session, spot
