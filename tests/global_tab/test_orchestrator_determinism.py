"""Tests for orchestrator.build_global_view (Phase 3a Task 5)."""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote
from india_quant.global_tab.options_chain import OptionsChainRow, OptionsChainSnapshot
from india_quant.global_tab.orchestrator import build_global_view
from india_quant.global_tab.types import Direction, Mode
from india_quant.signals.global_context import GlobalContext, SignalRow


def _ctx_provider():
    rows = [
        SignalRow(ticker="^GSPC", label="SPX", group="US", pct_1d=0.5, pct_5d=1.2,
                  direction="up", corr_30d=0.5, corr_90d=0.4, price=5800.0, atr_5d=12.0),
        SignalRow(ticker="^IXIC", label="Nasdaq", group="US", pct_1d=0.7, pct_5d=1.5,
                  direction="up", corr_30d=0.5, corr_90d=0.4, price=20000.0, atr_5d=80.0),
        SignalRow(ticker="DX-Y.NYB", label="DXY", group="FX", pct_1d=-0.2, pct_5d=-0.5,
                  direction="down", corr_30d=-0.3, corr_90d=-0.2, price=104.5, atr_5d=0.4),
        SignalRow(ticker="BZ=F", label="Brent", group="Commodities", pct_1d=0.8, pct_5d=2.0,
                  direction="up", corr_30d=0.1, corr_90d=0.1, price=78.0, atr_5d=1.5),
        SignalRow(ticker="^INDIAVIX", label="India VIX", group="IN", pct_1d=-1.5, pct_5d=-3.0,
                  direction="down", corr_30d=-0.4, corr_90d=-0.3, price=14.0, atr_5d=0.5),
    ]
    return lambda: GlobalContext(
        fetched_at=datetime(2026, 5, 5, 9, 0, tzinfo=timezone.utc),
        regime="risk-on",
        regime_drivers=["spx_up", "vix_down"],
        signals=rows,
        nifty_bias_text="constructive",
        nifty_pct_1d=0.4,
        nifty_pct_5d=1.1,
        usdinr=83.5,
    )


def _gift_provider(premium_pct: float = 0.45):
    return lambda: GiftNiftyQuote(last_price=24600.0, change_pct=premium_pct)


def _gift_none_provider():
    return lambda: None


def _history_provider():
    """Synthetic correlated history covering the heatmap columns."""
    idx = pd.date_range("2026-01-01", periods=80, freq="B", tz="UTC")
    rng = np.random.default_rng(seed=42)
    base = rng.standard_normal(len(idx)).cumsum() * 0.5
    cols = {
        "NIFTY": 24500 + base,
        "BANKNIFTY": 52000 + base * 2,
        "SPX": 5800 + base * 3,
        "NASDAQ": 20000 + base * 8,
        "DXY": 104.0 - base * 0.05,
        "BRENT": 78 + base * 0.3,
        "INDIA_VIX": 14 - base * 0.1,
        "US10Y": 4.2 + base * 0.01,
        "NIKKEI": 39000 + base * 6,
        "HSI": 18000 + base * 4,
        "FTSE": 8200 + base * 1.5,
    }
    return lambda: pd.DataFrame(cols, index=idx)


def _chain_loader_factory(spot: float = 24500.0):
    def _loader(index: str, as_of: datetime, expiry: date):
        scale = 1 if index == "NIFTY" else (52000 / 24500)
        s = spot * scale
        strikes = [round(s + delta, 0) for delta in (-200, -100, 0, 100, 200)]
        rows = []
        for k in strikes:
            for ot in ("CE", "PE"):
                base = max(s - k, 50) if ot == "CE" else max(k - s, 50)
                rows.append(OptionsChainRow(
                    strike=float(k), option_type=ot,
                    last_price=base, bid=base * 0.98, ask=base * 1.02,
                    iv=15.0, oi=10_000,
                ))
        return OptionsChainSnapshot(
            index=index, as_of=as_of, expiry=expiry,
            underlying_spot=s, chain=rows,
        )
    return _loader


def _empty_chain_loader():
    return lambda index, as_of, expiry: None


# ---------- Tests ----------

def test_build_view_returns_two_cards():
    view = build_global_view(
        as_of=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        mode=Mode.BALANCED,
        capital=1_000_000,
        context_provider=_ctx_provider(),
        gift_provider=_gift_provider(0.45),
        history_provider=_history_provider(),
        chain_loader=_chain_loader_factory(),
    )
    assert len(view.cards) == 2
    assert {c.index for c in view.cards} == {"NIFTY", "BANKNIFTY"}
    # GIFT premium 45 bps > 20 → LONG
    assert all(c.direction == Direction.LONG for c in view.cards)


def test_no_trade_card_when_premium_within_band():
    view = build_global_view(
        as_of=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        mode=Mode.BALANCED, capital=1_000_000,
        context_provider=_ctx_provider(),
        gift_provider=_gift_provider(0.05),  # 5 bps → NO_TRADE
        history_provider=_history_provider(),
        chain_loader=_chain_loader_factory(),
    )
    for c in view.cards:
        assert c.direction == Direction.NO_TRADE
        assert c.leg is None
        assert c.reasoning.no_trade_reason_code == "no_overnight_catalyst"


def test_data_gap_when_chain_missing():
    view = build_global_view(
        as_of=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        mode=Mode.BALANCED, capital=1_000_000,
        context_provider=_ctx_provider(),
        gift_provider=_gift_provider(0.45),
        history_provider=_history_provider(),
        chain_loader=_empty_chain_loader(),
    )
    for c in view.cards:
        assert c.direction == Direction.NO_TRADE
        assert c.reasoning.no_trade_reason_code == "data_gap"


def test_determinism_double_run():
    """Frozen as_of + identical providers + llm=None → byte-identical view."""
    kwargs = dict(
        as_of=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        mode=Mode.BALANCED,
        capital=1_000_000,
        context_provider=_ctx_provider(),
        gift_provider=_gift_provider(0.45),
        history_provider=_history_provider(),
        chain_loader=_chain_loader_factory(),
    )
    v1 = build_global_view(**kwargs)
    v2 = build_global_view(**kwargs)
    assert dataclasses.asdict(v1) == dataclasses.asdict(v2)


@settings(max_examples=15, deadline=None)
@given(
    capital=st.floats(min_value=10_000, max_value=10_000_000, allow_nan=False, allow_infinity=False),
    mode=st.sampled_from(list(Mode)),
)
def test_determinism_property(capital, mode):
    kwargs = dict(
        as_of=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        mode=mode, capital=capital,
        context_provider=_ctx_provider(),
        gift_provider=_gift_provider(0.45),
        history_provider=_history_provider(),
        chain_loader=_chain_loader_factory(),
    )
    v1 = build_global_view(**kwargs)
    v2 = build_global_view(**kwargs)
    assert dataclasses.asdict(v1) == dataclasses.asdict(v2)


def test_conservative_mode_blocks_when_no_top_decile_analog():
    """In Conservative mode, a forecast that would otherwise produce a card
    must instead emit a no_trade_ticket with reason 'no_top_decile_analog'
    when the AnalogIndex says the closest analog isn't in the top decile."""
    from india_quant.global_tab.analog_index import AnalogIndex, AnalogStats

    # Stub index that always reports top_decile_match=False
    class _StubAnalog:
        def lookup(self, features, predicted_direction, k: int = 20):
            return AnalogStats(count=20, winrate=0.5, avg_return_bps=0.0,
                               top_decile_match=False)

    view = build_global_view(
        as_of=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        mode=Mode.CONSERVATIVE, capital=1_000_000,
        context_provider=_ctx_provider(),
        gift_provider=_gift_provider(0.45),  # premium >> 20bps → would-be LONG
        history_provider=_history_provider(),
        chain_loader=_chain_loader_factory(),
        analog_index=_StubAnalog(),
    )
    # Conservative gates on top-decile analog → all NO_TRADE
    assert all(c.direction == Direction.NO_TRADE for c in view.cards)
    assert all(
        c.reasoning.no_trade_reason_code == "no_top_decile_analog"
        for c in view.cards
    )


def test_balanced_mode_does_not_gate_on_top_decile_analog():
    """Balanced mode must still produce trades even when no top-decile match
    exists — the gate is Conservative-only."""
    from india_quant.global_tab.analog_index import AnalogStats

    class _StubAnalog:
        def lookup(self, features, predicted_direction, k: int = 20):
            return AnalogStats(20, 0.5, 0.0, top_decile_match=False)

    view = build_global_view(
        as_of=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        mode=Mode.BALANCED, capital=1_000_000,
        context_provider=_ctx_provider(),
        gift_provider=_gift_provider(0.45),
        history_provider=_history_provider(),
        chain_loader=_chain_loader_factory(),
        analog_index=_StubAnalog(),
    )
    # Balanced does not require top-decile → at least one LONG card
    assert any(c.direction == Direction.LONG for c in view.cards)


def test_view_contains_briefing_and_heatmap():
    view = build_global_view(
        as_of=datetime(2026, 5, 5, 9, 30, tzinfo=timezone.utc),
        mode=Mode.BALANCED, capital=1_000_000,
        context_provider=_ctx_provider(),
        gift_provider=_gift_provider(0.45),
        history_provider=_history_provider(),
        chain_loader=_chain_loader_factory(),
    )
    assert len(view.briefing.tiles) == 10
    assert len(view.heatmap.cells) > 0
    assert view.mode == Mode.BALANCED
    assert view.capital == 1_000_000
