"""Translate an IndexForecast into a concrete options leg + risk/reward + timing window.

Returns `None` when:
  - forecast.direction is NO_TRADE
  - chain snapshot is None (DATA_GAP)
  - the EV / win-probability gate for the mode is not cleared
"""
from __future__ import annotations

from datetime import time
from math import floor
from typing import Optional

from india_quant.global_tab.forecaster import IndexForecast
from india_quant.global_tab.instruments import LOT_SIZES
from india_quant.global_tab.modes import MODE_CONFIGS
from india_quant.global_tab.options_chain import OptionsChainRow, OptionsChainSnapshot
from india_quant.global_tab.types import (
    Direction,
    Mode,
    OptionsLeg,
    RiskReward,
    TimingWindow,
)

# Per-mode timing windows. Hardcoded in Phase 3a; promoted to data when the
# scheduler/UI needs them. NSE session: 09:15–15:30 IST.
_TIMING: dict[Mode, TimingWindow] = {
    Mode.AGGRESSIVE: TimingWindow(
        entry_window_start=time(9, 20),
        entry_window_end=time(10, 30),
        exit_window_start=time(14, 30),
        exit_window_end=time(15, 20),
        invalidation_time=time(14, 55),
    ),
    Mode.BALANCED: TimingWindow(
        entry_window_start=time(9, 25),
        entry_window_end=time(11, 0),
        exit_window_start=time(14, 0),
        exit_window_end=time(15, 25),
        invalidation_time=time(14, 55),
    ),
    Mode.CONSERVATIVE: TimingWindow(
        entry_window_start=time(9, 30),
        entry_window_end=time(10, 30),
        exit_window_start=time(13, 30),
        exit_window_end=time(15, 25),
        invalidation_time=time(14, 55),
    ),
}


def pick_strike(
    spot: float,
    chain: list[OptionsChainRow],
    rule: str,
    direction: Direction,
) -> OptionsChainRow | None:
    """Pick a single chain row per mode rule + direction.

    LONG → CE; SHORT → PE.  ATM = closest to spot.
    OTM_1 = first strike past ATM in the unfavourable direction (further OTM).
    ITM_1 = first strike past ATM in the favourable direction (further ITM).
    """
    target_type = "CE" if direction == Direction.LONG else "PE"
    typed = [r for r in chain if r.option_type == target_type]
    if not typed:
        return None

    strikes = sorted({r.strike for r in typed})
    atm = min(strikes, key=lambda k: abs(k - spot))
    atm_idx = strikes.index(atm)

    if rule == "atm":
        chosen_strike = atm
    elif rule == "otm_1":
        # CE OTM = strike > spot; PE OTM = strike < spot
        if direction == Direction.LONG:
            chosen_strike = strikes[atm_idx + 1] if atm_idx + 1 < len(strikes) else atm
        else:
            chosen_strike = strikes[atm_idx - 1] if atm_idx - 1 >= 0 else atm
    elif rule == "itm_1":
        if direction == Direction.LONG:
            chosen_strike = strikes[atm_idx - 1] if atm_idx - 1 >= 0 else atm
        else:
            chosen_strike = strikes[atm_idx + 1] if atm_idx + 1 < len(strikes) else atm
    else:
        raise ValueError(f"unknown strike_rule {rule!r}")

    for r in typed:
        if r.strike == chosen_strike:
            return r
    return None


def _premium(row: OptionsChainRow) -> float:
    if row.bid is not None and row.ask is not None and row.bid > 0 and row.ask > 0:
        return (row.bid + row.ask) / 2.0
    return row.last_price


def size_trade(
    forecast: IndexForecast,
    capital: float,
    mode: Mode,
    chain: Optional[OptionsChainSnapshot],
) -> tuple[OptionsLeg, RiskReward, TimingWindow] | None:
    if forecast.direction == Direction.NO_TRADE:
        return None
    if chain is None:
        return None

    cfg = MODE_CONFIGS[mode]
    lot_size = LOT_SIZES.get(forecast.index)
    if lot_size is None:
        return None

    row = pick_strike(chain.underlying_spot, chain.chain, cfg.strike_rule, forecast.direction)
    if row is None:
        return None

    premium = _premium(row)
    if premium <= 0:
        return None

    max_loss_per_lot = lot_size * premium * (1 - cfg.stop_loss_multiple)
    if max_loss_per_lot <= 0:
        return None

    lots_budget = (capital * cfg.max_loss_fraction) / max_loss_per_lot
    lots = int(floor(lots_budget))
    if lots < 1:
        return None

    contracts = lots * lot_size
    capital_deployed = contracts * premium
    max_loss = contracts * premium * (1 - cfg.stop_loss_multiple)
    target_pnl_t1 = contracts * premium * (cfg.target_t1_multiple - 1)
    target_pnl_t2 = contracts * premium * (cfg.target_t2_multiple - 1)

    p = float(forecast.confidence)
    expected_value = p * target_pnl_t1 - (1 - p) * max_loss
    rr_ratio = target_pnl_t1 / max_loss if max_loss > 0 else 0.0

    if expected_value < cfg.min_expected_value:
        return None
    if p < cfg.min_win_probability:
        return None

    spot = chain.underlying_spot
    move = spot * forecast.expected_move_bps / 10_000.0
    move_t1 = spot * forecast.expected_move_bps / 10_000.0
    move_t2 = spot * forecast.expected_move_high_bps / 10_000.0
    move_stop = spot * forecast.expected_move_low_bps / 10_000.0

    leg = OptionsLeg(
        underlying=forecast.index,
        strike=row.strike,
        option_type=row.option_type,  # type: ignore[arg-type]
        expiry=chain.expiry,
        lot_size=lot_size,
        lots=lots,
        premium_estimate=premium,
        premium_zone=(premium * 0.97, premium * 1.03),
        target_t1=premium * cfg.target_t1_multiple,
        target_t2=premium * cfg.target_t2_multiple,
        stop_loss=premium * cfg.stop_loss_multiple,
        underlying_entry_trigger=spot,
        underlying_target_t1=spot + move_t1,
        underlying_target_t2=spot + move_t2,
        underlying_stop_trigger=spot + move_stop,
    )
    rr = RiskReward(
        capital_deployed=capital_deployed,
        max_loss=max_loss,
        target_pnl_t1=target_pnl_t1,
        target_pnl_t2=target_pnl_t2,
        win_probability=p,
        expected_value=expected_value,
        risk_reward_ratio=rr_ratio,
    )
    return leg, rr, _TIMING[mode]
