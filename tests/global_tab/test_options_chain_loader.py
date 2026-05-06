"""Tests for instruments.py and options_chain.py loader (Phase 3a Task 1)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from india_quant.global_tab.instruments import (
    LOT_SIZES,
    is_weekly_expiry,
    next_weekly_expiry,
)
from india_quant.global_tab.options_chain import (
    OptionsChainRow,
    OptionsChainSnapshot,
    load_chain_snapshot,
)


# ---------- instruments.py ----------

def test_lot_sizes_known_indices():
    assert LOT_SIZES["NIFTY"] == 25
    assert LOT_SIZES["BANKNIFTY"] == 15


def test_is_weekly_expiry_thursday():
    assert is_weekly_expiry(date(2026, 5, 7)) is True   # Thursday
    assert is_weekly_expiry(date(2026, 5, 8)) is False  # Friday
    assert is_weekly_expiry(date(2026, 5, 4)) is False  # Monday


def test_next_weekly_expiry_strictly_after():
    # 2026-05-05 is a Tuesday → next Thursday is 2026-05-07
    assert next_weekly_expiry(date(2026, 5, 5)) == date(2026, 5, 7)
    # If today IS Thursday, return next Thursday (strictly after)
    assert next_weekly_expiry(date(2026, 5, 7)) == date(2026, 5, 14)
    # Sunday → following Thursday
    assert next_weekly_expiry(date(2026, 5, 3)) == date(2026, 5, 7)


# ---------- options_chain.py ----------

def test_options_chain_snapshot_is_frozen():
    snap = OptionsChainSnapshot(
        index="NIFTY",
        as_of=datetime(2026, 5, 5, 9, 15, tzinfo=timezone.utc),
        expiry=date(2026, 5, 7),
        underlying_spot=24500.0,
        chain=[],
    )
    with pytest.raises(Exception):
        snap.underlying_spot = 25000.0  # type: ignore[misc]


def test_load_chain_snapshot_returns_none_on_empty(tmp_session_factory_empty):
    """Empty DB → None (sizer treats as DATA_GAP)."""
    result = load_chain_snapshot(
        index="NIFTY",
        as_of=datetime(2026, 5, 5, 9, 15, tzinfo=timezone.utc),
        expiry=date(2026, 5, 7),
        session_factory=tmp_session_factory_empty,
    )
    assert result is None


def test_load_chain_snapshot_filters_by_index_and_expiry(tmp_session_factory_with_chain):
    """Loader returns only rows matching (index, expiry); ignores other expiries."""
    factory, spot = tmp_session_factory_with_chain
    snap = load_chain_snapshot(
        index="NIFTY",
        as_of=datetime(2026, 5, 5, 9, 15, tzinfo=timezone.utc),
        expiry=date(2026, 5, 7),
        session_factory=factory,
    )
    assert snap is not None
    assert snap.index == "NIFTY"
    assert snap.expiry == date(2026, 5, 7)
    assert snap.underlying_spot == pytest.approx(spot)
    # 5 strikes × 2 option types = 10 rows
    assert len(snap.chain) == 10
    # Different expiry filtered out
    other_strikes = {r.strike for r in snap.chain}
    assert 99999.0 not in other_strikes


def test_load_chain_snapshot_rows_carry_expected_fields(tmp_session_factory_with_chain):
    factory, _ = tmp_session_factory_with_chain
    snap = load_chain_snapshot(
        index="NIFTY",
        as_of=datetime(2026, 5, 5, 9, 15, tzinfo=timezone.utc),
        expiry=date(2026, 5, 7),
        session_factory=factory,
    )
    assert snap is not None
    row = snap.chain[0]
    assert isinstance(row, OptionsChainRow)
    assert row.option_type in ("CE", "PE")
    assert row.strike > 0
