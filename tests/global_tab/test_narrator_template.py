"""Tests for narrator.py (Phase 3a Task 4)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from india_quant.global_tab.narrator import blurb_for_ticket
from india_quant.global_tab.types import Direction, ReasoningContext


def _ctx(reason: str | None = None, drivers=None, analogs=(0, 0.0, 0.0)):
    return ReasoningContext(
        top_drivers=drivers if drivers is not None else [("gift_nifty_premium_bps", 45.0)],
        analog_count=analogs[0],
        analog_winrate=analogs[1],
        analog_avg_pnl=analogs[2],
        no_trade_reason_code=reason,
    )


def test_trade_template_interpolation():
    ctx = _ctx(drivers=[("gift_nifty_premium_bps", 45.0)], analogs=(20, 0.65, 12500.0))
    blurb = blurb_for_ticket(ctx, Direction.LONG, "NIFTY")
    assert "NIFTY long" in blurb
    assert "gift_nifty_premium_bps (+45bps)" in blurb
    assert "20 analog sessions" in blurb
    assert "65%" in blurb
    assert "₹12,500" in blurb


def test_short_uses_short_word():
    ctx = _ctx(drivers=[("dxy_delta_pct", -0.6)])
    blurb = blurb_for_ticket(ctx, Direction.SHORT, "BANKNIFTY")
    assert "BANKNIFTY short" in blurb
    assert "(-1bps)" in blurb  # -0.6 formatted as +.0f → -1


@pytest.mark.parametrize("code,expected", [
    ("no_overnight_catalyst", "no overnight catalyst"),
    ("below_mode_threshold", "expected value below mode threshold"),
    ("data_gap", "options chain unavailable"),
])
def test_no_trade_reason_pretty(code, expected):
    ctx = _ctx(reason=code)
    blurb = blurb_for_ticket(ctx, Direction.NO_TRADE, "NIFTY")
    assert blurb == f"NIFTY: no trade. {expected}."


def test_no_trade_unknown_reason_falls_back_to_humanized():
    ctx = _ctx(reason="some_new_code")
    blurb = blurb_for_ticket(ctx, Direction.NO_TRADE, "NIFTY")
    assert "some new code" in blurb


def test_no_trade_with_none_reason():
    ctx = _ctx(reason=None)
    blurb = blurb_for_ticket(ctx, Direction.NO_TRADE, "NIFTY")
    assert "no signal" in blurb


def test_llm_arg_is_not_called_in_phase_3a():
    """Phase 3a contract: llm is accepted but never invoked."""
    mock_llm = MagicMock()
    ctx = _ctx()
    blurb_for_ticket(ctx, Direction.LONG, "NIFTY", llm=mock_llm)
    mock_llm.assert_not_called()


def test_no_drivers_renders_em_dash():
    ctx = _ctx(drivers=[])
    blurb = blurb_for_ticket(ctx, Direction.LONG, "NIFTY")
    assert "—" in blurb  # em dash placeholder for missing driver
