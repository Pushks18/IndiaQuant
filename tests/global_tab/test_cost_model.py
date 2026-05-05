"""Tests for cost_model: Indian options realistic costs."""
import pytest

from india_quant.global_tab.cost_model import (
    CostBreakdown,
    compute_costs,
    realized_pnl,
)
from india_quant.global_tab.types import Direction


def test_brokerage_capped_at_20_per_side():
    # qty=150, premium=200 → 0.03% × 30000 = 9, well below cap; brokerage = 9 each side.
    cb = compute_costs(entry_premium=200.0, exit_premium=200.0, qty=150, bid_ask_spread=0.0)
    assert cb.brokerage == pytest.approx(2 * 9.0)


def test_brokerage_uses_cap_for_large_notional():
    # qty=1000, premium=200 → 0.03% × 200000 = 60 > 20, so brokerage = 20 each side.
    cb = compute_costs(entry_premium=200.0, exit_premium=210.0, qty=1000, bid_ask_spread=0.0)
    assert cb.brokerage == pytest.approx(2 * 20.0)


def test_stt_only_on_sell_side():
    # STT = 0.0625% × exit_premium × qty
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    assert cb.stt == pytest.approx(0.000625 * 120.0 * 75)


def test_exchange_charges_both_sides():
    # NSE = 0.03503% on each side
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    expected = 0.0003503 * 100.0 * 75 + 0.0003503 * 120.0 * 75
    assert cb.exchange == pytest.approx(expected)


def test_sebi_charges_both_sides():
    # SEBI = ₹10 per crore = 1e-6 of turnover
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    expected = 1e-6 * 100.0 * 75 + 1e-6 * 120.0 * 75
    assert cb.sebi == pytest.approx(expected)


def test_stamp_duty_only_on_buy_side():
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    assert cb.stamp_duty == pytest.approx(0.00003 * 100.0 * 75)


def test_gst_is_18_percent_of_brokerage_plus_exchange_plus_sebi():
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    expected = 0.18 * (cb.brokerage + cb.exchange + cb.sebi)
    assert cb.gst == pytest.approx(expected)


def test_slippage_is_half_spread_per_side_total_two_sides():
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.4)
    # 0.5 * 0.4 * 75 per side, 2 sides
    assert cb.slippage == pytest.approx(2 * 0.5 * 0.4 * 75)


def test_total_is_sum_of_all_components():
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.4)
    assert cb.total == pytest.approx(
        cb.brokerage + cb.stt + cb.exchange + cb.sebi + cb.stamp_duty + cb.gst + cb.slippage
    )


def test_realized_pnl_long_winning_trade():
    pnl = realized_pnl(
        direction=Direction.LONG,
        entry_premium=100.0,
        exit_premium=120.0,
        qty=150,
        bid_ask_spread=0.4,
    )
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=150, bid_ask_spread=0.4)
    assert pnl == pytest.approx(20.0 * 150 - cb.total)


def test_realized_pnl_short_winning_trade():
    pnl = realized_pnl(
        direction=Direction.SHORT,
        entry_premium=120.0,
        exit_premium=100.0,
        qty=150,
        bid_ask_spread=0.4,
    )
    cb = compute_costs(entry_premium=120.0, exit_premium=100.0, qty=150, bid_ask_spread=0.4)
    assert pnl == pytest.approx(20.0 * 150 - cb.total)


def test_realized_pnl_no_trade_returns_zero():
    pnl = realized_pnl(
        direction=Direction.NO_TRADE,
        entry_premium=100.0,
        exit_premium=120.0,
        qty=150,
        bid_ask_spread=0.4,
    )
    assert pnl == 0.0


def test_cost_breakdown_is_frozen():
    import dataclasses

    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cb.brokerage = 0.0  # type: ignore[misc]
