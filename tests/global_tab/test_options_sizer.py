"""Tests for options_sizer.size_trade() (Phase 3a Task 2)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from india_quant.global_tab.forecaster import IndexForecast
from india_quant.global_tab.instruments import LOT_SIZES
from india_quant.global_tab.options_chain import OptionsChainRow, OptionsChainSnapshot
from india_quant.global_tab.options_sizer import pick_strike, size_trade
from india_quant.global_tab.types import Direction, Mode


SPOT = 24500.0
STRIKES = [24300, 24400, 24500, 24600, 24700]


def _make_chain() -> OptionsChainSnapshot:
    rows = []
    for k in STRIKES:
        for ot in ("CE", "PE"):
            mid = max(SPOT - k, 50) if ot == "CE" else max(k - SPOT, 50)
            rows.append(OptionsChainRow(
                strike=float(k),
                option_type=ot,
                last_price=mid,
                bid=mid * 0.98,
                ask=mid * 1.02,
                iv=15.0,
                oi=10_000,
            ))
    return OptionsChainSnapshot(
        index="NIFTY",
        as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
        expiry=date(2026, 5, 7),
        underlying_spot=SPOT,
        chain=rows,
    )


def _forecast(direction: Direction, confidence: float = 0.7, move_bps: float = 60.0) -> IndexForecast:
    sign = 1.0 if direction == Direction.LONG else -1.0
    return IndexForecast(
        index="NIFTY",
        direction=direction,
        confidence=confidence,
        expected_move_bps=sign * move_bps,
        expected_move_low_bps=sign * 30.0,
        expected_move_high_bps=sign * 100.0,
        feature_attributions=[],
        no_trade_reason_code=None,
    )


# ---------- pick_strike ----------

@pytest.mark.parametrize("direction,rule,expected", [
    (Direction.LONG, "atm", 24500.0),
    (Direction.LONG, "otm_1", 24600.0),   # CE OTM = above spot
    (Direction.LONG, "itm_1", 24400.0),   # CE ITM = below spot
    (Direction.SHORT, "atm", 24500.0),
    (Direction.SHORT, "otm_1", 24400.0),  # PE OTM = below spot
    (Direction.SHORT, "itm_1", 24600.0),  # PE ITM = above spot
])
def test_pick_strike_rules(direction, rule, expected):
    chain = _make_chain()
    row = pick_strike(SPOT, chain.chain, rule, direction)
    assert row is not None
    assert row.strike == expected
    assert row.option_type == ("CE" if direction == Direction.LONG else "PE")


# ---------- size_trade ----------

def test_no_trade_returns_none():
    fc = IndexForecast(
        index="NIFTY",
        direction=Direction.NO_TRADE,
        confidence=0.0,
        expected_move_bps=0.0,
        expected_move_low_bps=0.0,
        expected_move_high_bps=0.0,
        feature_attributions=[],
        no_trade_reason_code="no_overnight_catalyst",
    )
    assert size_trade(fc, 100_000, Mode.BALANCED, _make_chain()) is None


def test_chain_none_returns_none():
    fc = _forecast(Direction.LONG)
    assert size_trade(fc, 100_000, Mode.BALANCED, None) is None


def test_long_picks_ce_at_atm_for_balanced():
    fc = _forecast(Direction.LONG)
    out = size_trade(fc, 1_000_000, Mode.BALANCED, _make_chain())
    assert out is not None
    leg, rr, timing = out
    assert leg.option_type == "CE"
    assert leg.strike == 24500.0
    assert leg.lots >= 1
    assert leg.lot_size == LOT_SIZES["NIFTY"]


def test_short_picks_pe_at_otm_for_aggressive():
    fc = _forecast(Direction.SHORT, confidence=0.65)
    out = size_trade(fc, 1_000_000, Mode.AGGRESSIVE, _make_chain())
    assert out is not None
    leg, _, _ = out
    assert leg.option_type == "PE"
    assert leg.strike == 24400.0  # one strike below spot for SHORT/OTM


def test_lots_arithmetic():
    """Hand-checked: capital 1_000_000, balanced 1.5% loss budget, ATM CE premium=50,
    lot_size 25, stop_loss_multiple=0.7 → max_loss_per_lot = 25*50*0.3 ≈ 375.
    lots_budget ≈ 40 (floor may give 39 due to floating-point in 1-0.7)."""
    fc = _forecast(Direction.LONG)
    out = size_trade(fc, 1_000_000, Mode.BALANCED, _make_chain())
    assert out is not None
    leg, _, _ = out
    assert leg.premium_estimate == pytest.approx(50.0)
    assert leg.lots in (39, 40)


def test_ev_gate_rejects_low_win_probability():
    """Balanced requires p >= 0.55. Confidence 0.5 → reject."""
    fc = _forecast(Direction.LONG, confidence=0.5)
    assert size_trade(fc, 1_000_000, Mode.BALANCED, _make_chain()) is None


def test_aggressive_accepts_below_balanced_threshold():
    """Aggressive has no win-prob gate (Balanced requires >=0.55).
    p=0.50 still clears Aggressive's EV>=0 gate (0.5*0.5 − 0.5*0.4 = +0.05 of premium)."""
    fc = _forecast(Direction.LONG, confidence=0.50)
    out = size_trade(fc, 1_000_000, Mode.AGGRESSIVE, _make_chain())
    assert out is not None
    # Same forecast against Balanced fails the win-prob gate.
    assert size_trade(fc, 1_000_000, Mode.BALANCED, _make_chain()) is None


def test_premium_zone_brackets_estimate():
    fc = _forecast(Direction.LONG)
    out = size_trade(fc, 1_000_000, Mode.BALANCED, _make_chain())
    assert out is not None
    leg, _, _ = out
    lo, hi = leg.premium_zone
    assert lo < leg.premium_estimate < hi
    assert lo == pytest.approx(leg.premium_estimate * 0.97)


def test_timing_window_per_mode():
    fc = _forecast(Direction.LONG)
    out_a = size_trade(fc, 1_000_000, Mode.AGGRESSIVE, _make_chain())
    out_b = size_trade(fc, 1_000_000, Mode.BALANCED, _make_chain())
    out_c = size_trade(fc, 1_000_000, Mode.CONSERVATIVE, _make_chain())
    assert out_a is not None and out_b is not None and out_c is not None
    assert out_a[2] != out_b[2]  # different windows
    assert out_b[2].invalidation_time.hour == 14  # 14:55


def test_zero_capital_returns_none():
    fc = _forecast(Direction.LONG)
    assert size_trade(fc, 0, Mode.BALANCED, _make_chain()) is None


def test_unknown_index_returns_none():
    fc = _forecast(Direction.LONG)
    object.__setattr__(fc, "index", "FINNIFTY")  # frozen but bypass for test
    chain = _make_chain()
    object.__setattr__(chain, "index", "FINNIFTY")
    assert size_trade(fc, 1_000_000, Mode.BALANCED, chain) is None
