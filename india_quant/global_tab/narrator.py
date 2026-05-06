"""Deterministic-template blurb for trade tickets.

Phase 3a: pure templating. The `llm` parameter is accepted but ignored — Phase 6
will plug in an LLM rewrite + validator (length ≤ 400, no foreign digits, no
foreign instrument names) behind the same signature.
"""
from __future__ import annotations

from typing import Any, Literal

from india_quant.global_tab.types import Direction, ReasoningContext

BlurbKind = Literal["trade", "no_trade"]

_REASON_PRETTY = {
    "no_overnight_catalyst": "no overnight catalyst",
    "below_mode_threshold": "expected value below mode threshold",
    "data_gap": "options chain unavailable",
    "stale_data": "stale data",
    "outside_window": "outside entry window",
}

_DIR_WORD = {
    Direction.LONG: "long",
    Direction.SHORT: "short",
    Direction.NO_TRADE: "no trade",
}


def _reason_pretty(code: str | None) -> str:
    if code is None:
        return "no signal"
    return _REASON_PRETTY.get(code, code.replace("_", " "))


def _render_template(
    ctx: ReasoningContext,
    direction: Direction,
    index: str,
    blurb_kind: BlurbKind,
) -> str:
    if blurb_kind == "no_trade":
        return f"{index}: no trade. {_reason_pretty(ctx.no_trade_reason_code)}."

    if ctx.top_drivers:
        drv0_name, drv0_val = ctx.top_drivers[0]
    else:
        drv0_name, drv0_val = "—", 0.0

    return (
        f"{index} {_DIR_WORD[direction]}: top driver {drv0_name} "
        f"({drv0_val:+.0f}bps); {ctx.analog_count} analog sessions averaged "
        f"{ctx.analog_winrate:.0%} win rate, ₹{ctx.analog_avg_pnl:,.0f} avg P&L."
    )


def blurb_for_ticket(
    ctx: ReasoningContext,
    direction: Direction,
    index: str,
    llm: Any | None = None,
) -> str:
    """Phase 3a: always returns the deterministic template.

    The `llm` arg is accepted for forward-compat with Phase 6; it is never
    invoked here (Phase 3 determinism contract).
    """
    kind: BlurbKind = "no_trade" if direction == Direction.NO_TRADE else "trade"
    return _render_template(ctx, direction, index, kind)
