"""build_global_view — assemble the full GlobalTabView for a given (as_of, mode, capital).

Pure given the providers passed in. The Flask route wires real DB/network
providers; tests pass deterministic stubs to verify byte-identical re-runs.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable, Optional

import pandas as pd

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote
from india_quant.global_tab.briefing import build_briefing
from india_quant.global_tab.correlation import build_heatmap
from india_quant.global_tab.forecaster import (
    FeatureRow,
    ModelArtifact,
    StubArtifact,
    forecast_index,
)
from india_quant.global_tab.instruments import next_weekly_expiry
from india_quant.global_tab.narrator import blurb_for_ticket
from india_quant.global_tab.options_chain import OptionsChainSnapshot
from india_quant.global_tab.options_sizer import size_trade
from india_quant.global_tab.types import (
    Direction,
    GlobalTabView,
    LiveTicket,
    Mode,
    ReasoningContext,
    Status,
    TradeTicket,
)

# Tickers we read from the GlobalContext to populate FeatureRow.
_FEAT_TICKERS = {
    "spx_overnight_pct": "^GSPC",
    "dxy_delta_pct": "DX-Y.NYB",
    "india_vix_delta_pct": "^INDIAVIX",
    "brent_overnight_pct": "BZ=F",
}


def _row_by_ticker(rows, ticker):
    for r in rows:
        if getattr(r, "ticker", None) == ticker:
            return r
    return None


def _build_features(context, gift_nifty: GiftNiftyQuote | None) -> FeatureRow:
    rows = list(getattr(context, "signals", []) or [])
    gift_bps = (gift_nifty.change_pct * 100.0) if (gift_nifty and gift_nifty.change_pct is not None) else None

    def _pct(ticker: str) -> float | None:
        row = _row_by_ticker(rows, ticker)
        return getattr(row, "pct_1d", None) if row is not None else None

    return FeatureRow(
        gift_nifty_premium_bps=gift_bps,
        spx_overnight_pct=_pct(_FEAT_TICKERS["spx_overnight_pct"]),
        dxy_delta_pct=_pct(_FEAT_TICKERS["dxy_delta_pct"]),
        india_vix_delta_pct=_pct(_FEAT_TICKERS["india_vix_delta_pct"]),
        brent_overnight_pct=_pct(_FEAT_TICKERS["brent_overnight_pct"]),
    )


def _no_trade_ticket(
    index: str,
    reason_code: str,
    features: FeatureRow,
    as_of: datetime,
    confidence: float = 0.0,
) -> TradeTicket:
    ctx = ReasoningContext(
        top_drivers=_top_attribs(features),
        analog_count=0,
        analog_winrate=0.0,
        analog_avg_pnl=0.0,
        no_trade_reason_code=reason_code,
    )
    return TradeTicket(
        index=index,
        direction=Direction.NO_TRADE,
        confidence=confidence,
        leg=None,
        timing=None,
        risk_reward=None,
        reasoning=ctx,
        live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=as_of),
        blurb=blurb_for_ticket(ctx, Direction.NO_TRADE, index),
    )


def _top_attribs(features: FeatureRow, k: int = 3) -> list[tuple[str, float]]:
    items = [(n, v) for n, v in features.as_dict().items() if v is not None]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items[:k]


def build_global_view(
    as_of: datetime,
    mode: Mode,
    capital: float,
    *,
    context_provider: Callable[[], Any],
    gift_provider: Callable[[], Optional[GiftNiftyQuote]],
    history_provider: Callable[[], pd.DataFrame],
    chain_loader: Callable[[str, datetime, date], Optional[OptionsChainSnapshot]],
    model_artifact: ModelArtifact | None = None,
    llm: Any | None = None,
    indices: tuple[str, ...] = ("NIFTY", "BANKNIFTY"),
) -> GlobalTabView:
    artifact = model_artifact if model_artifact is not None else StubArtifact()

    context = context_provider()
    gift = gift_provider()
    briefing = build_briefing(as_of=as_of, context=context, gift_nifty=gift)

    history = history_provider()
    heatmap = build_heatmap(history=history, as_of=as_of.date())

    features = _build_features(context, gift)
    expiry = next_weekly_expiry(as_of.date())

    cards: list[TradeTicket] = []
    for index in indices:
        chain = chain_loader(index, as_of, expiry)
        forecast = forecast_index(index, as_of, mode, features, artifact)

        if forecast.direction == Direction.NO_TRADE:
            cards.append(_no_trade_ticket(index, forecast.no_trade_reason_code or "no_overnight_catalyst", features, as_of))
            continue

        if chain is None:
            cards.append(_no_trade_ticket(index, "data_gap", features, as_of, confidence=forecast.confidence))
            continue

        sized = size_trade(forecast, capital, mode, chain)
        if sized is None:
            cards.append(_no_trade_ticket(index, "below_mode_threshold", features, as_of, confidence=forecast.confidence))
            continue

        leg, rr, timing = sized
        ctx = ReasoningContext(
            top_drivers=list(forecast.feature_attributions),
            analog_count=0,        # Phase 4: real analog index
            analog_winrate=0.0,
            analog_avg_pnl=0.0,
            no_trade_reason_code=None,
        )
        ticket = TradeTicket(
            index=index,
            direction=forecast.direction,
            confidence=forecast.confidence,
            leg=leg,
            timing=timing,
            risk_reward=rr,
            reasoning=ctx,
            live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=as_of),
            blurb=blurb_for_ticket(ctx, forecast.direction, index, llm=llm),
        )
        cards.append(ticket)

    staleness = {
        "briefing": briefing.as_of,
        "heatmap": as_of,
    }
    return GlobalTabView(
        as_of=as_of,
        mode=mode,
        capital=capital,
        briefing=briefing,
        heatmap=heatmap,
        cards=cards,
        artifact_paths={"name": getattr(artifact, "name", "stub")},
        staleness=staleness,
    )
