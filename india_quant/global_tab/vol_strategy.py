"""Phase 6a — daily ATM straddle strategy (long volatility).

Pure assembler: given the spot, vol forecast, vol implied, mode, and
capital, produce a TradeTicket of kind="straddle". When the forecast/implied
gap doesn't beat the mode buffer, returns a no_trade ticket with a
specific reason code so the UI can explain the gate.

No DB access, no network. The orchestrator wires up live data via
providers.
"""
from __future__ import annotations

import math
from dataclasses import replace as _dc_replace
from datetime import date, datetime
from typing import Optional

from india_quant.global_tab.instruments import LOT_SIZES
from india_quant.global_tab.modes import MODE_CONFIGS
from india_quant.global_tab.types import (
    Direction, LiveTicket, Mode, ReasoningContext, StraddleLeg, Status, TradeTicket,
)


# Mode → minimum (forecast / implied) ratio to trigger the straddle.
_VOL_BUFFER: dict[Mode, float] = {
    Mode.AGGRESSIVE:   1.05,
    Mode.BALANCED:     1.15,
    Mode.CONSERVATIVE: 1.30,
}

# Conservative-only: also require analog hit-rate ≥ this fraction.
_CONSERVATIVE_HITRATE_FLOOR = 0.60

# Strike rounding per index.
_STRIKE_STEP: dict[str, int] = {"NIFTY": 50, "BANKNIFTY": 100}


def _round_strike(spot: float, step: int) -> float:
    return round(spot / step) * step


def _atm_premium_per_leg(spot: float, sigma_annual_pct: float, days_to_expiry: int) -> float:
    """Black-Scholes ATM approximation per leg.

    For ATM, T-day options at vol σ:  prem ≈ S × σ × √(T/365) × 0.4
    The 0.4 factor approximates 2 × Φ'(0) × √(2π) / something — close
    enough for sizing when option_chain isn't populated. Refine in 6b
    using chain mids when present.
    """
    if spot <= 0 or sigma_annual_pct <= 0 or days_to_expiry <= 0:
        return 0.0
    sigma = sigma_annual_pct / 100.0
    return spot * sigma * math.sqrt(days_to_expiry / 365.0) * 0.4


def _no_trade_straddle(
    index: str, reason: str, *, as_of: datetime,
    analog_hitrate: float | None = None, analog_count: int = 0,
) -> TradeTicket:
    ctx = ReasoningContext(
        top_drivers=[],
        analog_count=analog_count,
        analog_winrate=float(analog_hitrate or 0.0),
        analog_avg_pnl=0.0,
        no_trade_reason_code=reason,
    )
    return TradeTicket(
        index=index, direction=Direction.NO_TRADE, confidence=0.0,
        leg=None, timing=None, risk_reward=None,
        reasoning=ctx,
        live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=as_of),
        blurb="",  # narrator fills based on kind
        kind="straddle", straddle=None,
    )


def build_straddle_ticket(
    *,
    index: str,
    spot: float,
    vol_forecast_pct: float,
    vol_implied_pct: float,
    mode: Mode,
    capital: float,
    expiry: date,
    as_of: datetime,
    chain=None,                   # optional OptionsChainSnapshot
    features=None,                # FeatureRow for analog lookup
    analog_index=None,            # AnalogIndex
) -> TradeTicket:
    """Produce a straddle TradeTicket. Returns a no_trade variant when the
    vol gap doesn't clear the mode buffer or sizing comes back empty."""
    if spot <= 0 or vol_implied_pct <= 0:
        return _no_trade_straddle(index, "data_gap", as_of=as_of)

    # Vol gate
    buffer = _VOL_BUFFER.get(mode, 1.15)
    if vol_forecast_pct < vol_implied_pct * buffer:
        return _no_trade_straddle(index, "vol_below_threshold", as_of=as_of)

    # Strike + premium
    step = _STRIKE_STEP.get(index, 50)
    strike = float(_round_strike(spot, step))
    days_to_expiry = max((expiry - as_of.date()).days, 1)

    if chain is not None:
        # Prefer live chain mids when the ATM strike is present.
        rows = list(getattr(chain, "chain", []) or [])
        atm_call = next((r for r in rows if abs(r.strike - strike) < 1e-6 and r.option_type == "CE"), None)
        atm_put  = next((r for r in rows if abs(r.strike - strike) < 1e-6 and r.option_type == "PE"), None)
    else:
        atm_call = atm_put = None

    if atm_call and atm_put:
        call_prem = float((atm_call.bid + atm_call.ask) / 2.0 if atm_call.ask else atm_call.last_price)
        put_prem  = float((atm_put.bid  + atm_put.ask)  / 2.0 if atm_put.ask  else atm_put.last_price)
    else:
        approx = _atm_premium_per_leg(spot, vol_implied_pct, days_to_expiry)
        call_prem = put_prem = approx

    if call_prem <= 0 or put_prem <= 0:
        return _no_trade_straddle(index, "data_gap", as_of=as_of)

    total_prem_per_unit = call_prem + put_prem
    lot_size = LOT_SIZES.get(index, 50)
    max_loss_per_lot = total_prem_per_unit * lot_size

    mcfg = MODE_CONFIGS.get(mode)
    risk_budget = capital * (mcfg.max_loss_fraction if mcfg else 0.015)
    lots = int(risk_budget // max_loss_per_lot) if max_loss_per_lot > 0 else 0
    if lots < 1:
        return _no_trade_straddle(index, "below_mode_threshold", as_of=as_of)

    breakeven_high = spot + total_prem_per_unit
    breakeven_low  = spot - total_prem_per_unit
    max_loss_total = total_prem_per_unit * lot_size * lots
    breakeven_bps  = (total_prem_per_unit / spot) * 10_000.0

    # Analog confirmation (optional)
    analog_count = 0
    analog_hitrate: float | None = None
    if analog_index is not None and features is not None:
        try:
            analog_hitrate = float(analog_index.lookup_breakeven(features, breakeven_bps, k=20))
            analog_count = 20
        except Exception:  # noqa: BLE001
            analog_hitrate = None

    if mode == Mode.CONSERVATIVE and analog_hitrate is not None and analog_hitrate < _CONSERVATIVE_HITRATE_FLOOR:
        return _no_trade_straddle(
            index, "vol_analog_low_hitrate", as_of=as_of,
            analog_hitrate=analog_hitrate, analog_count=analog_count,
        )

    leg = StraddleLeg(
        underlying=index, strike=strike, expiry=expiry,
        lot_size=lot_size, lots=lots,
        call_premium=call_prem, put_premium=put_prem,
        total_premium=total_prem_per_unit,
        breakeven_high=breakeven_high, breakeven_low=breakeven_low,
        max_loss=max_loss_total,
        vol_forecast_pct=vol_forecast_pct,
        vol_implied_pct=vol_implied_pct,
    )

    ctx = ReasoningContext(
        top_drivers=[("vol_gap_pct", vol_forecast_pct - vol_implied_pct)],
        analog_count=analog_count,
        analog_winrate=float(analog_hitrate or 0.0),
        analog_avg_pnl=0.0,
        no_trade_reason_code=None,
    )

    confidence = min(0.85, 0.5 + (vol_forecast_pct / vol_implied_pct - 1.0))
    return TradeTicket(
        index=index, direction=Direction.LONG,    # "long volatility"
        confidence=float(confidence),
        leg=None, timing=None, risk_reward=None,
        reasoning=ctx,
        live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=as_of),
        blurb="",
        kind="straddle", straddle=leg,
    )
