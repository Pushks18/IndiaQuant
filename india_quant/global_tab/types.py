"""Type definitions for the global_tab package.

This module is pure: only dataclasses and enums, no I/O, no logic.
Every dataclass is frozen so views are immutable once produced.
"""
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Literal


class Mode(str, Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


class Status(str, Enum):
    WAITING = "waiting"
    ENTRY_ZONE_ACTIVE = "entry_zone_active"
    IN_POSITION = "in_position"
    TARGET_HIT = "target_hit"
    STOPPED_OUT = "stopped_out"
    EXPIRED_NO_ENTRY = "expired_no_entry"
    DATA_GAP = "data_gap"


@dataclass(frozen=True)
class BriefingTile:
    label: str
    value: str
    change_pct: float
    sentiment: Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True)
class BriefingStrip:
    as_of: datetime
    tiles: list[BriefingTile]
    predicted_gap_bps: dict[str, float]


@dataclass(frozen=True)
class CorrelationCell:
    asset_a: str
    asset_b: str
    rho_20d: float
    rho_60d: float


@dataclass(frozen=True)
class CorrelationHeatmap:
    as_of: date
    cells: list[CorrelationCell]


@dataclass(frozen=True)
class OptionsLeg:
    underlying: str
    strike: float
    option_type: Literal["CE", "PE"]
    expiry: date
    lot_size: int
    lots: int
    premium_estimate: float
    premium_zone: tuple[float, float]
    target_t1: float
    target_t2: float
    stop_loss: float
    underlying_entry_trigger: float
    underlying_target_t1: float
    underlying_target_t2: float
    underlying_stop_trigger: float


@dataclass(frozen=True)
class RiskReward:
    capital_deployed: float
    max_loss: float
    target_pnl_t1: float
    target_pnl_t2: float
    win_probability: float
    expected_value: float
    risk_reward_ratio: float


@dataclass(frozen=True)
class TimingWindow:
    entry_window_start: time
    entry_window_end: time
    exit_window_start: time
    exit_window_end: time
    invalidation_time: time


@dataclass(frozen=True)
class ReasoningContext:
    top_drivers: list[tuple[str, float]]
    analog_count: int
    analog_winrate: float
    analog_avg_pnl: float
    no_trade_reason_code: str | None


@dataclass(frozen=True)
class LiveTicket:
    status: Status
    live_pnl: float | None
    last_update: datetime


@dataclass(frozen=True)
class StraddleLeg:
    """Phase 6a: ATM long straddle (call + put) for the volatility strategy.

    Both legs are bought at the same strike (ATM). Max loss is the total
    premium paid. Profit when |spot_at_expiry - strike| > total_premium.
    """
    underlying: str
    strike: float
    expiry: date
    lot_size: int
    lots: int
    call_premium: float
    put_premium: float
    total_premium: float       # call_premium + put_premium per lot
    breakeven_high: float
    breakeven_low: float
    max_loss: float            # total_premium × lot_size × lots (rupees)
    vol_forecast_pct: float    # annualized %
    vol_implied_pct: float     # annualized %


@dataclass(frozen=True)
class TradeTicket:
    index: str
    direction: Direction
    confidence: float
    leg: OptionsLeg | None
    timing: TimingWindow | None
    risk_reward: RiskReward | None
    reasoning: ReasoningContext
    live: LiveTicket
    blurb: str
    # Phase 6a: variant marker so the renderer can branch. "directional" is
    # the legacy ticket type produced by forecast_index + size_trade.
    # "straddle" is the volatility variant — populated via build_straddle_ticket.
    kind: str = "directional"
    straddle: StraddleLeg | None = None


@dataclass(frozen=True)
class GlobalTabView:
    as_of: datetime
    mode: Mode
    capital: float
    briefing: BriefingStrip
    heatmap: CorrelationHeatmap
    cards: list[TradeTicket]
    artifact_paths: dict[str, str]
    staleness: dict[str, datetime]
