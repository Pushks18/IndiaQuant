"""Tests for live_status.compute_status — pure time-based transitions."""
from __future__ import annotations

from datetime import date, datetime, time, timezone

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    from datetime import timedelta
    IST = timezone(timedelta(hours=5, minutes=30))

from india_quant.global_tab.live_status import compute_status
from india_quant.global_tab.types import (
    Direction, LiveTicket, OptionsLeg, ReasoningContext, Status,
    TimingWindow, TradeTicket, RiskReward,
)


def _ticket(direction=Direction.LONG, timing=None) -> TradeTicket:
    if timing is None:
        timing = TimingWindow(
            entry_window_start=time(9, 20),
            entry_window_end=time(9, 45),
            exit_window_start=time(14, 30),
            exit_window_end=time(15, 15),
            invalidation_time=time(15, 25),
        )
    leg = OptionsLeg(
        underlying="NIFTY", strike=24500.0, option_type="CE",
        expiry=date(2026, 5, 8), lot_size=50, lots=1, premium_estimate=120.0,
        premium_zone=(115.0, 125.0),
        target_t1=150.0, target_t2=180.0, stop_loss=100.0,
        underlying_entry_trigger=24500.0, underlying_target_t1=24600.0,
        underlying_target_t2=24700.0, underlying_stop_trigger=24400.0,
    ) if direction != Direction.NO_TRADE else None
    rr = RiskReward(
        capital_deployed=6000.0, max_loss=1000.0,
        target_pnl_t1=1500.0, target_pnl_t2=3000.0,
        win_probability=0.6, expected_value=15.0, risk_reward_ratio=1.5,
    ) if direction != Direction.NO_TRADE else None
    ctx = ReasoningContext(
        top_drivers=[], analog_count=0, analog_winrate=0.0,
        analog_avg_pnl=0.0, no_trade_reason_code=None,
    )
    return TradeTicket(
        index="NIFTY", direction=direction, confidence=0.7,
        leg=leg, timing=timing if direction != Direction.NO_TRADE else None,
        risk_reward=rr, reasoning=ctx,
        live=LiveTicket(status=Status.WAITING, live_pnl=None,
                        last_update=datetime.now(timezone.utc)),
        blurb="",
    )


def _at(h: int, m: int = 0) -> datetime:
    return datetime(2026, 5, 7, h, m, tzinfo=IST)


def test_before_entry_window_is_waiting():
    t = _ticket()
    assert compute_status(t, _at(9, 0)) == Status.WAITING


def test_in_entry_window_is_active():
    t = _ticket()
    assert compute_status(t, _at(9, 30)) == Status.ENTRY_ZONE_ACTIVE


def test_post_entry_pre_exit_is_in_position():
    t = _ticket()
    assert compute_status(t, _at(11, 0)) == Status.IN_POSITION
    assert compute_status(t, _at(14, 0)) == Status.IN_POSITION


def test_past_exit_window_expires():
    t = _ticket()
    assert compute_status(t, _at(15, 30)) == Status.EXPIRED_NO_ENTRY
    assert compute_status(t, _at(16, 0)) == Status.EXPIRED_NO_ENTRY


def test_no_trade_ticket_stays_waiting():
    t = _ticket(direction=Direction.NO_TRADE)
    assert compute_status(t, _at(11, 0)) == Status.WAITING
    assert compute_status(t, _at(16, 0)) == Status.WAITING


def test_naive_datetime_treated_as_ist():
    """A datetime without tzinfo must be interpreted as IST so cards built from
    test fixtures (often naive datetimes) get correct flips."""
    t = _ticket()
    naive = datetime(2026, 5, 7, 9, 30)  # no tzinfo
    assert compute_status(t, naive) == Status.ENTRY_ZONE_ACTIVE


def test_utc_datetime_converted_correctly():
    """A UTC datetime must be converted to IST before comparison.
    9:30 IST = 04:00 UTC."""
    t = _ticket()
    utc_at_930_ist = datetime(2026, 5, 7, 4, 0, tzinfo=timezone.utc)
    assert compute_status(t, utc_at_930_ist) == Status.ENTRY_ZONE_ACTIVE


def test_invalidation_time_caps_in_position():
    """If invalidation_time fires before exit_window_end, that's the cap."""
    early_invalid = TimingWindow(
        entry_window_start=time(9, 20),
        entry_window_end=time(9, 45),
        exit_window_start=time(14, 30),
        exit_window_end=time(15, 15),
        invalidation_time=time(13, 0),  # before exit_window_end
    )
    t = _ticket(timing=early_invalid)
    # 13:30 IST is past invalidation_time → EXPIRED_NO_ENTRY
    assert compute_status(t, _at(13, 30)) == Status.EXPIRED_NO_ENTRY
