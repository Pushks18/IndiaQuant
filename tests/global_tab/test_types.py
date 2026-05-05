"""Tests for india_quant.global_tab.types."""
from india_quant.global_tab.types import Direction, Mode, Status


def test_mode_values():
    assert Mode.AGGRESSIVE.value == "aggressive"
    assert Mode.BALANCED.value == "balanced"
    assert Mode.CONSERVATIVE.value == "conservative"


def test_mode_membership():
    assert set(Mode) == {Mode.AGGRESSIVE, Mode.BALANCED, Mode.CONSERVATIVE}


def test_direction_values():
    assert Direction.LONG.value == "long"
    assert Direction.SHORT.value == "short"
    assert Direction.NO_TRADE.value == "no_trade"


def test_status_values():
    assert Status.WAITING.value == "waiting"
    assert Status.ENTRY_ZONE_ACTIVE.value == "entry_zone_active"
    assert Status.IN_POSITION.value == "in_position"
    assert Status.TARGET_HIT.value == "target_hit"
    assert Status.STOPPED_OUT.value == "stopped_out"
    assert Status.EXPIRED_NO_ENTRY.value == "expired_no_entry"
    assert Status.DATA_GAP.value == "data_gap"


def test_enums_are_str_subclasses():
    """Round-trip through json without a custom encoder."""
    import json
    payload = {"mode": Mode.BALANCED, "direction": Direction.LONG, "status": Status.WAITING}
    assert json.loads(json.dumps(payload)) == {"mode": "balanced", "direction": "long", "status": "waiting"}


import dataclasses
from datetime import date, datetime, time

import pytest

from india_quant.global_tab.types import (
    BriefingStrip,
    BriefingTile,
    CorrelationCell,
    CorrelationHeatmap,
    GlobalTabView,
    LiveTicket,
    OptionsLeg,
    ReasoningContext,
    RiskReward,
    TimingWindow,
    TradeTicket,
)


def _sample_leg() -> OptionsLeg:
    return OptionsLeg(
        underlying="NIFTY",
        strike=24350.0,
        option_type="CE",
        expiry=date(2026, 5, 8),
        lot_size=75,
        lots=2,
        premium_estimate=142.0,
        premium_zone=(138.0, 148.0),
        target_t1=199.0,
        target_t2=284.0,
        stop_loss=99.0,
        underlying_entry_trigger=24310.0,
        underlying_target_t1=24420.0,
        underlying_target_t2=24500.0,
        underlying_stop_trigger=24220.0,
    )


def test_options_leg_is_frozen():
    leg = _sample_leg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        leg.lots = 99  # type: ignore[misc]


def test_options_leg_premium_zone_is_tuple():
    leg = _sample_leg()
    assert isinstance(leg.premium_zone, tuple)
    assert leg.premium_zone[0] < leg.premium_zone[1]


def test_risk_reward_round_trip():
    rr = RiskReward(
        capital_deployed=10000.0,
        max_loss=14850.0,
        target_pnl_t1=8550.0,
        target_pnl_t2=21300.0,
        win_probability=0.62,
        expected_value=4860.0,
        risk_reward_ratio=1.43,
    )
    assert dataclasses.asdict(rr)["win_probability"] == pytest.approx(0.62)


def test_timing_window_construction():
    tw = TimingWindow(
        entry_window_start=time(9, 18),
        entry_window_end=time(9, 25),
        exit_window_start=time(14, 30),
        exit_window_end=time(15, 15),
        invalidation_time=time(11, 0),
    )
    assert tw.entry_window_start < tw.entry_window_end


def test_reasoning_context_no_trade_optional():
    ctx = ReasoningContext(
        top_drivers=[("gift_nifty_premium_bps", 60.0)],
        analog_count=47,
        analog_winrate=0.64,
        analog_avg_pnl=3200.0,
        no_trade_reason_code=None,
    )
    assert ctx.no_trade_reason_code is None


def test_trade_ticket_no_trade_has_none_leg():
    ticket = TradeTicket(
        index="NIFTY",
        direction=Direction.NO_TRADE,
        confidence=12.0,
        leg=None,
        timing=None,
        risk_reward=None,
        reasoning=ReasoningContext(
            top_drivers=[],
            analog_count=0,
            analog_winrate=0.0,
            analog_avg_pnl=0.0,
            no_trade_reason_code="gift_premium_in_noise_band",
        ),
        live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=datetime(2026, 5, 5, 8, 45)),
        blurb="No trade today: GIFT Nifty premium is within the noise band.",
    )
    assert ticket.leg is None
    assert ticket.risk_reward is None


def test_global_tab_view_assembles():
    view = GlobalTabView(
        as_of=datetime(2026, 5, 5, 8, 45),
        mode=Mode.BALANCED,
        capital=10000.0,
        briefing=BriefingStrip(
            as_of=datetime(2026, 5, 5, 8, 45),
            tiles=[BriefingTile(label="SPX", value="5,612.40", change_pct=1.10, sentiment="bullish")],
            predicted_gap_bps={"NIFTY": 35.0, "BANKNIFTY": 48.0},
        ),
        heatmap=CorrelationHeatmap(
            as_of=date(2026, 5, 5),
            cells=[CorrelationCell(asset_a="NIFTY", asset_b="SPX", rho_20d=0.45, rho_60d=0.52)],
        ),
        cards=[],
        artifact_paths={},
        staleness={},
    )
    assert view.mode == Mode.BALANCED
    assert view.cards == []
