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
from india_quant.global_tab.vol_forecaster import forecast_realized_vol
from india_quant.global_tab.vol_strategy import build_straddle_ticket
from india_quant.global_tab.briefing import build_briefing
from india_quant.global_tab.correlation import build_heatmap
from india_quant.global_tab.forecaster import (
    FeatureRow,
    ModelArtifact,
    StubArtifact,
    forecast_index,
)
from india_quant.global_tab.instruments import next_weekly_expiry
from india_quant.global_tab.narrator import blurb_for_straddle, blurb_for_ticket
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
    "spx_overnight_pct":    "^GSPC",
    "nasdaq_overnight_pct": "^IXIC",
    "dxy_delta_pct":        "DX-Y.NYB",
    "india_vix_delta_pct":  "^INDIAVIX",
    "brent_overnight_pct":  "BZ=F",
}

# RBI policy dates — duplicated from training_features so orchestrator
# stays free of training-side imports. Update both lists together.
_RBI_POLICY_DATES_FOR_SERVE = (
    date(2025, 2, 7),  date(2025, 4, 9),  date(2025, 6, 6),
    date(2025, 8, 8),  date(2025, 10, 8), date(2025, 12, 5),
    date(2026, 2, 6),  date(2026, 4, 8),  date(2026, 6, 5),
    date(2026, 8, 7),  date(2026, 10, 7), date(2026, 12, 4),
)


def _days_to_next_rbi_policy(d: date) -> int:
    upcoming = [p for p in _RBI_POLICY_DATES_FOR_SERVE if p >= d]
    return (upcoming[0] - d).days if upcoming else 999


def _nifty_momentum_and_vol(closes: list[float]) -> tuple[float | None, float | None]:
    """Compute (nifty_5d_momentum, nifty_realized_vol_20d) from oldest-first closes.

    Matches the training-time computation in training_features.assemble_training_frame:
      momentum = log(close[-1]) - log(close[-6])     (5-session log return)
      vol      = std(diff(log(close)))[-20:]         (20-session log-return std)
    """
    import math
    if not closes or len(closes) < 6:
        return (None, None)
    c = [float(x) for x in closes if x and x > 0]
    if len(c) < 6:
        return (None, None)
    momentum = math.log(c[-1]) - math.log(c[-6])
    vol = None
    if len(c) >= 21:
        log_rets = [math.log(c[i]) - math.log(c[i - 1]) for i in range(1, len(c))]
        tail = log_rets[-20:]
        m = sum(tail) / len(tail)
        var = sum((r - m) ** 2 for r in tail) / len(tail)
        vol = var ** 0.5
    return (momentum, vol)


def _row_by_ticker(rows, ticker):
    for r in rows:
        if getattr(r, "ticker", None) == ticker:
            return r
    return None


def _build_features(
    context,
    gift_nifty: GiftNiftyQuote | None,
    *,
    as_of: datetime | None = None,
    nifty_closes: list[float] | None = None,
) -> FeatureRow:
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

    # Phase 5c: serve-time train/serve-skew fix. Populate the 6 fields the
    # original Phase 3a orchestrator left at None so the LightGBM input
    # vector matches the train-time distribution.
    momentum, vol_20d = _nifty_momentum_and_vol(nifty_closes or [])
    serve_date = (as_of.date() if isinstance(as_of, datetime) else date.today())
    dow_int_v = serve_date.weekday()  # Mon=0..Sun=6; matches training trainer
    is_expiry_v = 1                   # placeholder matches training (always 1 Mon..Fri)
    days_to_rbi = _days_to_next_rbi_policy(serve_date)

    return FeatureRow(
        gift_nifty_premium_bps=gift_bps,
        spx_overnight_pct=_pct(_FEAT_TICKERS["spx_overnight_pct"]),
        nasdaq_overnight_pct=_pct(_FEAT_TICKERS["nasdaq_overnight_pct"]),
        dxy_delta_pct=_pct(_FEAT_TICKERS["dxy_delta_pct"]),
        india_vix_delta_pct=_pct(_FEAT_TICKERS["india_vix_delta_pct"]),
        brent_overnight_pct=_pct(_FEAT_TICKERS["brent_overnight_pct"]),
        nifty_5d_momentum=momentum,
        nifty_realized_vol_20d=vol_20d,
        dow_int=dow_int_v,
        is_expiry_week=is_expiry_v,
        days_to_rbi_policy=days_to_rbi,
        bank_vs_nifty_5d_relstr=_rs("^NSEBANK"),
        it_vs_nifty_5d_relstr=_rs("^CNXIT"),
        pharma_vs_nifty_5d_relstr=_rs("^CNXPHARMA"),
        realty_vs_nifty_5d_relstr=_rs("^CNXREALTY"),
        sector_dispersion_5d=sector_disp,
        # Breadth + factor-aggregate features stay None at serve time;
        # artifact zero-imputes (Phase 3e candidate columns weren't shown to
        # be information-additive — see PHASE3D_CANDIDATE_COLUMNS).
    )


def _vol_no_trade(index: str, reason: str, as_of: datetime) -> TradeTicket:
    """A no-trade variant for the straddle path — flagged kind=straddle so
    the renderer routes it to the vol card layout rather than the directional
    layout."""
    ctx = ReasoningContext(
        top_drivers=[], analog_count=0, analog_winrate=0.0,
        analog_avg_pnl=0.0, no_trade_reason_code=reason,
    )
    return TradeTicket(
        index=index, direction=Direction.NO_TRADE, confidence=0.0,
        leg=None, timing=None, risk_reward=None, reasoning=ctx,
        live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=as_of),
        blurb="", kind="straddle", straddle=None,
    )


def _default_vol_implied(context, index: str) -> float | None:
    """Read India VIX from context.signals (publishes annualized %).
    BANKNIFTY proxy = India VIX × 1.20 (rough historical multiplier).
    """
    rows = list(getattr(context, "signals", []) or [])
    vix_row = _row_by_ticker(rows, "^INDIAVIX")
    if vix_row is None:
        return None
    vix = getattr(vix_row, "price", None)
    if vix is None or vix <= 0:
        return None
    return float(vix) * (1.20 if index == "BANKNIFTY" else 1.0)


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
    nifty_closes_provider: Callable[[], list[float]] | None = None,
    vol_implied_provider: Callable[[str], Optional[float]] | None = None,
    llm: Any | None = None,
    indices: tuple[str, ...] = ("NIFTY", "BANKNIFTY"),
) -> GlobalTabView:
    artifact = model_artifact if model_artifact is not None else StubArtifact()

    context = context_provider()
    gift = gift_provider()
    briefing = build_briefing(as_of=as_of, context=context, gift_nifty=gift)

    history = history_provider()
    heatmap = build_heatmap(history=history, as_of=as_of.date())

    nifty_closes = nifty_closes_provider() if nifty_closes_provider is not None else []
    features = _build_features(context, gift, as_of=as_of, nifty_closes=nifty_closes)
    expiry = next_weekly_expiry(as_of.date())

    def _build_directional_ticket(index: str) -> TradeTicket:
        """Returns exactly one TradeTicket for the directional path."""
        chain = chain_loader(index, as_of, expiry)
        forecast = forecast_index(index, as_of, mode, features, artifact)

        if analog_index is not None:
            stats = analog_index.lookup(features, forecast.direction)
        else:
            stats = AnalogStats(0, 0.0, 0.0, False)

        if forecast.direction == Direction.NO_TRADE:
            return _no_trade_ticket(
                index, forecast.no_trade_reason_code or "no_overnight_catalyst",
                features, as_of, analog_stats=stats,
            )

        from india_quant.global_tab.modes import MODE_CONFIGS as _MC
        mcfg = _MC.get(mode)
        if mcfg is not None and getattr(mcfg, "require_top_decile_analog", False):
            if not stats.top_decile_match:
                return _no_trade_ticket(
                    index, "no_top_decile_analog", features, as_of,
                    confidence=forecast.confidence, analog_stats=stats,
                )

        if chain is None:
            return _no_trade_ticket(
                index, "data_gap", features, as_of,
                confidence=forecast.confidence, analog_stats=stats,
            )

        sized = size_trade(forecast, capital, mode, chain)
        if sized is None:
            return _no_trade_ticket(
                index, "below_mode_threshold", features, as_of,
                confidence=forecast.confidence, analog_stats=stats,
            )

        leg, rr, timing = sized
        ctx = ReasoningContext(
            top_drivers=list(forecast.feature_attributions),
            analog_count=stats.count,
            analog_winrate=stats.winrate,
            analog_avg_pnl=stats.avg_return_bps,
            no_trade_reason_code=None,
        )
        provisional = TradeTicket(
            index=index, direction=forecast.direction, confidence=forecast.confidence,
            leg=leg, timing=timing, risk_reward=rr, reasoning=ctx,
            live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=as_of),
            blurb=blurb_for_ticket(ctx, forecast.direction, index, llm=llm),
        )
        spot = spot_provider(index) if spot_provider is not None else None
        live_status = compute_status(provisional, as_of, current_spot=spot)
        return dataclasses.replace(
            provisional,
            live=LiveTicket(status=live_status, live_pnl=None, last_update=as_of),
        )

    def _build_straddle_card(index: str) -> TradeTicket:
        """Phase 6a: long-vol straddle card per index. Always emits one ticket
        (either the straddle or a vol-no-trade variant)."""
        spot = spot_provider(index) if spot_provider is not None else None
        if spot is None or spot <= 0:
            return _vol_no_trade(index, "data_gap", as_of)

        # Vol forecast: HAR-RV blend on NIFTY closes (Phase 6a — same proxy used
        # for both indices because the BANKNIFTY closes provider isn't wired yet).
        vf = forecast_realized_vol(nifty_closes or [])
        if vf is None:
            return _vol_no_trade(index, "data_gap", as_of)
        vol_forecast_pct = vf.annualized_pct

        # Implied vol per index. NIFTY: India VIX. BANKNIFTY: India VIX × 1.20.
        if vol_implied_provider is not None:
            vol_implied_pct = vol_implied_provider(index)
        else:
            vol_implied_pct = _default_vol_implied(context, index)
        if vol_implied_pct is None or vol_implied_pct <= 0:
            return _vol_no_trade(index, "data_gap", as_of)

        chain = chain_loader(index, as_of, expiry)
        return build_straddle_ticket(
            index=index, spot=spot,
            vol_forecast_pct=vol_forecast_pct,
            vol_implied_pct=vol_implied_pct,
            mode=mode, capital=capital, expiry=expiry, as_of=as_of,
            chain=chain, features=features, analog_index=analog_index,
        )

    cards: list[TradeTicket] = []
    for index in indices:
        cards.append(_build_directional_ticket(index))
        straddle = _build_straddle_card(index)
        cards.append(dataclasses.replace(straddle, blurb=blurb_for_straddle(straddle)))

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
