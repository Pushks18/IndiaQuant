"""Realistic Indian options cost model.

All numbers calibrated to NSE F&O / Zerodha as of 2025. The backtester uses
this for every trade so equity curves reflect after-cost reality. No "before
cost" P&L is ever surfaced on the Global tab.

References:
- Brokerage: min(20, 0.03% of turnover) per executed order, each side.
- STT: 0.0625% of premium turnover, sell side only.
- Exchange transaction (NSE F&O): 0.03503% of premium, both sides.
- SEBI charges: ₹10 per crore of turnover (1e-6), both sides.
- Stamp duty: 0.003% of premium, buy side only (state-imposed; uniform 2020+).
- GST: 18% on (brokerage + exchange + SEBI).
- Slippage: estimated as half the bid-ask spread, applied per side.

Phase-1 takes bid_ask_spread as an argument; the chain snapshot will supply
it in Phase 2.
"""
from dataclasses import dataclass

from india_quant.global_tab.types import Direction

_BROKERAGE_RATE = 0.0003           # 0.03%
_BROKERAGE_CAP = 20.0
_STT_RATE = 0.000625               # 0.0625% sell-side
_EXCHANGE_RATE = 0.0003503         # 0.03503% both sides
_SEBI_RATE = 1e-6                  # ₹10 / crore
_STAMP_DUTY_RATE = 0.00003         # 0.003% buy-side
_GST_RATE = 0.18                   # 18%


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp_duty: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange
            + self.sebi
            + self.stamp_duty
            + self.gst
            + self.slippage
        )


def compute_costs(
    *,
    entry_premium: float,
    exit_premium: float,
    qty: int,
    bid_ask_spread: float,
) -> CostBreakdown:
    buy_turnover = entry_premium * qty
    sell_turnover = exit_premium * qty

    brokerage_buy = min(_BROKERAGE_CAP, _BROKERAGE_RATE * buy_turnover)
    brokerage_sell = min(_BROKERAGE_CAP, _BROKERAGE_RATE * sell_turnover)
    brokerage = brokerage_buy + brokerage_sell

    stt = _STT_RATE * sell_turnover  # sell-side only

    exch_buy = _EXCHANGE_RATE * buy_turnover
    exch_sell = _EXCHANGE_RATE * sell_turnover
    exchange = exch_buy + exch_sell

    sebi_buy = _SEBI_RATE * buy_turnover
    sebi_sell = _SEBI_RATE * sell_turnover
    sebi = sebi_buy + sebi_sell

    stamp_duty = _STAMP_DUTY_RATE * buy_turnover  # buy-side only

    gst = _GST_RATE * (brokerage + exchange + sebi)

    slippage_per_side = 0.5 * bid_ask_spread * qty
    slippage = 2 * slippage_per_side

    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange=exchange,
        sebi=sebi,
        stamp_duty=stamp_duty,
        gst=gst,
        slippage=slippage,
    )


def realized_pnl(
    *,
    direction: Direction,
    entry_premium: float,
    exit_premium: float,
    qty: int,
    bid_ask_spread: float,
) -> float:
    if direction == Direction.NO_TRADE:
        return 0.0

    if direction == Direction.LONG:
        gross = (exit_premium - entry_premium) * qty
    else:  # SHORT
        gross = (entry_premium - exit_premium) * qty

    costs = compute_costs(
        entry_premium=entry_premium,
        exit_premium=exit_premium,
        qty=qty,
        bid_ask_spread=bid_ask_spread,
    )
    return gross - costs.total
