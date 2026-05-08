"""Tests for vol_strategy.build_straddle_ticket — synthetic, no DB."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from india_quant.global_tab.types import Direction, Mode
from india_quant.global_tab.vol_strategy import build_straddle_ticket


def _kwargs(**over):
    base = dict(
        index="NIFTY",
        spot=24500.0,
        vol_forecast_pct=15.0,
        vol_implied_pct=12.0,    # ratio 1.25 — clears balanced (1.15)
        mode=Mode.BALANCED,
        capital=1_000_000.0,
        expiry=date(2026, 5, 14),
        as_of=datetime(2026, 5, 7, 9, 30, tzinfo=timezone.utc),
    )
    base.update(over)
    return base


def test_clears_balanced_buffer_emits_straddle():
    t = build_straddle_ticket(**_kwargs())
    assert t.kind == "straddle"
    assert t.straddle is not None
    assert t.direction == Direction.LONG
    assert t.straddle.lots >= 1
    assert t.straddle.breakeven_high > t.straddle.strike > t.straddle.breakeven_low
    assert t.straddle.max_loss > 0


def test_below_balanced_buffer_emits_no_trade():
    """ratio 1.10 < 1.15 buffer → NO_TRADE."""
    t = build_straddle_ticket(**_kwargs(vol_forecast_pct=13.2, vol_implied_pct=12.0))
    assert t.kind == "straddle"
    assert t.straddle is None
    assert t.direction == Direction.NO_TRADE
    assert t.reasoning.no_trade_reason_code == "vol_below_threshold"


def test_aggressive_buffer_is_lower():
    """ratio 1.10 clears aggressive (1.05) but not balanced (1.15)."""
    t_agg = build_straddle_ticket(**_kwargs(vol_forecast_pct=13.2, vol_implied_pct=12.0, mode=Mode.AGGRESSIVE))
    assert t_agg.straddle is not None
    t_bal = build_straddle_ticket(**_kwargs(vol_forecast_pct=13.2, vol_implied_pct=12.0, mode=Mode.BALANCED))
    assert t_bal.straddle is None


def test_strike_rounds_to_50_for_nifty():
    t = build_straddle_ticket(**_kwargs(spot=24523.0))
    assert t.straddle.strike == 24500.0


def test_strike_rounds_to_100_for_banknifty():
    t = build_straddle_ticket(**_kwargs(index="BANKNIFTY", spot=56078.0))
    assert t.straddle.strike == 56100.0


def test_breakeven_distance_equals_total_premium():
    t = build_straddle_ticket(**_kwargs())
    s = t.straddle
    assert (s.breakeven_high - s.strike) == pytest.approx(s.total_premium, abs=1e-6)
    assert (s.strike - s.breakeven_low) == pytest.approx(s.total_premium, abs=1e-6)


def test_zero_capital_yields_no_trade():
    t = build_straddle_ticket(**_kwargs(capital=100.0))  # too small for one lot
    assert t.straddle is None
    assert t.reasoning.no_trade_reason_code in {"below_mode_threshold", "vol_below_threshold"}


def test_conservative_low_analog_hitrate_blocks():
    class _StubAnalog:
        def lookup_breakeven(self, features, breakeven_bps, k=20):
            return 0.4  # below 0.6 conservative floor

    t = build_straddle_ticket(
        **_kwargs(mode=Mode.CONSERVATIVE, vol_forecast_pct=20.0, vol_implied_pct=12.0),
        features=object(), analog_index=_StubAnalog(),
    )
    assert t.straddle is None
    assert t.reasoning.no_trade_reason_code == "vol_analog_low_hitrate"


def test_conservative_with_high_hitrate_passes():
    class _StubAnalog:
        def lookup_breakeven(self, features, breakeven_bps, k=20):
            return 0.75
    t = build_straddle_ticket(
        **_kwargs(mode=Mode.CONSERVATIVE, vol_forecast_pct=20.0, vol_implied_pct=12.0),
        features=object(), analog_index=_StubAnalog(),
    )
    assert t.straddle is not None
