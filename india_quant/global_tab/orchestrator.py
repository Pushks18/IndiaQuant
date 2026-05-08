"""build_global_view — assemble the full GlobalTabView for a given (as_of, mode, capital).

Pure given the providers passed in. The Flask route wires real DB/network
providers; tests pass deterministic stubs to verify byte-identical re-runs.
"""
from __future__ import annotations

import dataclasses
from datetime import date, datetime
from typing import Any, Callable, Optional

import pandas as pd

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote
from india_quant.global_tab.analog_index import AnalogIndex, AnalogStats
from india_quant.global_tab.live_status import compute_status
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

    def _pct5(ticker: str) -> float | None:
        row = _row_by_ticker(rows, ticker)
        return getattr(row, "pct_5d", None) if row is not None else None

    # Phase 3d: sector relative-strength + dispersion at serve time.
    nifty_pct_5d = getattr(context, "nifty_pct_5d", None)
    def _rs(ticker: str) -> float | None:
        s = _pct5(ticker)
        if s is None or nifty_pct_5d is None:
            return None
        return float(s) - float(nifty_pct_5d)

    sector_disp: float | None = None
    sector_pcts = [_pct5(t) for t in ("^NSEBANK", "^CNXIT", "^CNXPHARMA",
                                       "^CNXREALTY", "^CNXENERGY", "^CNXINFRA")]
    sector_clean = [float(v) for v in sector_pcts if v is not None]
    if len(sector_clean) >= 4:
        import numpy as _np
        sector_disp = float(_np.std(sector_clean, ddof=0))

    return FeatureRow(
        gift_nifty_premium_bps=gift_bps,
        spx_overnight_pct=_pct(_FEAT_TICKERS["spx_overnight_pct"]),
        dxy_delta_pct=_pct(_FEAT_TICKERS["dxy_delta_pct"]),
        india_vix_delta_pct=_pct(_FEAT_TICKERS["india_vix_delta_pct"]),
        brent_overnight_pct=_pct(_FEAT_TICKERS["brent_overnight_pct"]),
        bank_vs_nifty_5d_relstr=_rs("^NSEBANK"),
        it_vs_nifty_5d_relstr=_rs("^CNXIT"),
        pharma_vs_nifty_5d_relstr=_rs("^CNXPHARMA"),
        realty_vs_nifty_5d_relstr=_rs("^CNXREALTY"),
        sector_dispersion_5d=sector_disp,
        # Breadth + factor-aggregate features stay None at serve time;
        # artifact zero-imputes (Phase 3e: thread session_factory through orchestrator
        # to populate them properly and close the train/serve skew).
    )


def _no_trade_ticket(
    index: str,
    reason_code: str,
    features: FeatureRow,
    as_of: datetime,
    confidence: float = 0.0,
    analog_stats: AnalogStats | None = None,
) -> TradeTicket:
    stats = analog_stats or AnalogStats(0, 0.0, 0.0, False)
    ctx = ReasoningContext(
        top_drivers=_top_attribs(features),
        analog_count=stats.count,
        analog_winrate=stats.winrate,
        analog_avg_pnl=stats.avg_return_bps,
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
    analog_index: AnalogIndex | None = None,
    spot_provider: Callable[[str], Optional[float]] | None = None,
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

        # Phase 4a: real analog stats. None index → zero-stat fallback (preserves
        # the Phase 3a behaviour for callers that haven't wired the index yet).
        if analog_index is not None:
            stats = analog_index.lookup(features, forecast.direction)
        else:
            stats = AnalogStats(0, 0.0, 0.0, False)

        if forecast.direction == Direction.NO_TRADE:
            cards.append(_no_trade_ticket(
                index, forecast.no_trade_reason_code or "no_overnight_catalyst",
                features, as_of, analog_stats=stats,
            ))
            continue

        # Conservative mode: require a top-decile analog match. The spec gates
        # the most expensive setup on "this looks like a known winning regime"
        # — if the closest historical session isn't in the top 10% of the
        # similarity distribution, refuse the trade.
        from india_quant.global_tab.modes import MODE_CONFIGS
        mcfg = MODE_CONFIGS.get(mode)
        if mcfg is not None and getattr(mcfg, "require_top_decile_analog", False):
            if not stats.top_decile_match:
                cards.append(_no_trade_ticket(
                    index, "no_top_decile_analog", features, as_of,
                    confidence=forecast.confidence, analog_stats=stats,
                ))
                continue

        if chain is None:
            cards.append(_no_trade_ticket(
                index, "data_gap", features, as_of,
                confidence=forecast.confidence, analog_stats=stats,
            ))
            continue

        sized = size_trade(forecast, capital, mode, chain)
        if sized is None:
            cards.append(_no_trade_ticket(
                index, "below_mode_threshold", features, as_of,
                confidence=forecast.confidence, analog_stats=stats,
            ))
            continue

        leg, rr, timing = sized
        ctx = ReasoningContext(
            top_drivers=list(forecast.feature_attributions),
            analog_count=stats.count,
            analog_winrate=stats.winrate,
            analog_avg_pnl=stats.avg_return_bps,
            no_trade_reason_code=None,
        )
        # Build the ticket first with WAITING so compute_status can read its
        # timing window; then re-stamp with the time-derived status.
        provisional = TradeTicket(
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
        spot = spot_provider(index) if spot_provider is not None else None
        live_status = compute_status(provisional, as_of, current_spot=spot)
        ticket = dataclasses.replace(
            provisional,
            live=LiveTicket(status=live_status, live_pnl=None, last_update=as_of),
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
