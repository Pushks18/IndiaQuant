"""Intraday strategy module — Opening Range Breakout (ORB) for NSE equities.

Strategy:
  09:15-09:30 IST: observe (no trade); record HIGH and LOW of first 15-min candle.
  09:30-09:35:    place buy-stop above 9:30 high, sell-stop below 9:30 low.
  Once triggered:  hold with stop = opposite end of opening range.
                   target = 1.5x the range size from entry.
  15:15:           hard exit (avoid auto-square-off congestion at 15:20).

Position sizing uses MIS (intraday margin) leverage of ~5x.

A pre-market plan (built before 09:15) substitutes ATR-based estimates for
the unknown opening range. Once the market is open, replace those estimates
with the real 9:30 high/low.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, time

from sqlalchemy import text

from india_quant.data.db import get_session

# Indian intraday cost / leverage assumptions
MIS_LEVERAGE = 5            # 5x intraday margin for liquid equity
INTRADAY_COST_PCT = 0.0006  # ~0.06% round-trip (STT-on-sell + brokerage + GST + slippage)


@dataclass
class IntradayPlan:
    ticker: str
    prev_close: float
    atr: float
    expected_range_pct: float       # ATR / prev_close
    bias: str                        # "long_only" | "short_only" | "either"
    conviction: int | None
    # Long-side
    long_trigger: float              # buy-stop above this
    long_stop: float
    long_target: float
    long_risk_per_share: float
    long_reward_per_share: float
    long_rr: float
    # Short-side
    short_trigger: float             # sell-stop below this
    short_stop: float
    short_target: float
    short_risk_per_share: float
    short_reward_per_share: float
    short_rr: float
    # Position sizing on a given capital
    capital_inr: float
    risk_per_trade_pct: float        # e.g. 0.01 = 1% of capital per trade
    long_qty: int
    short_qty: int
    long_position_value: float
    short_position_value: float
    margin_required_long: float
    margin_required_short: float
    long_max_loss_inr: float
    long_target_profit_inr: float
    short_max_loss_inr: float
    short_target_profit_inr: float

    def to_dict(self) -> dict:
        return asdict(self)


def _latest_atr_and_close(ticker: str) -> tuple[float | None, float | None]:
    """Cheap 14-day ATR proxy from price_data — close-to-close range."""
    with get_session() as s:
        rows = s.execute(text("""
            SELECT high, low, close FROM price_data
            WHERE ticker = :t AND interval = '1d'
            ORDER BY datetime DESC LIMIT 14
        """), {"t": ticker}).fetchall()
    if not rows or len(rows) < 5:
        return None, None
    trs = [(float(h) - float(l)) for h, l, _ in rows]
    atr = sum(trs) / len(trs)
    last_close = float(rows[0][2])
    return atr, last_close


def _bias_from_verdict(verdict: str | None, conviction: int | None) -> str:
    if verdict == "Bullish" and (conviction or 0) >= 6:
        return "long_only"
    if verdict == "Bearish" and (conviction or 0) >= 6:
        return "short_only"
    return "either"


def build_plan(
    ticker: str,
    verdict: str | None = None,
    conviction: int | None = None,
    capital_inr: float = 100_000,
    risk_per_trade_pct: float = 0.01,        # risk 1% of capital per trade
    range_factor: float = 0.4,               # opening range ≈ 40% of daily ATR
    target_multiple: float = 1.5,            # target = 1.5 × range
) -> IntradayPlan | None:
    """Build a pre-market intraday plan from yesterday's data.

    Replace the implied 9:30 high/low with the real values once market opens.
    """
    atr, prev_close = _latest_atr_and_close(ticker)
    if atr is None or prev_close is None or prev_close <= 0:
        return None

    bias = _bias_from_verdict(verdict, conviction)

    # Expected first 15-min range (rule of thumb: ~40% of daily ATR)
    or_size = atr * range_factor
    or_high = prev_close + 0.5 * or_size
    or_low = prev_close - 0.5 * or_size

    # Buffer above/below to confirm breakout (0.05% above range)
    buf = prev_close * 0.0005
    long_trigger = round(or_high + buf, 2)
    long_stop = round(or_low, 2)
    long_target = round(long_trigger + target_multiple * or_size, 2)

    short_trigger = round(or_low - buf, 2)
    short_stop = round(or_high, 2)
    short_target = round(short_trigger - target_multiple * or_size, 2)

    long_risk = max(long_trigger - long_stop, 0.01)
    long_rew = max(long_target - long_trigger, 0.01)
    short_risk = max(short_stop - short_trigger, 0.01)
    short_rew = max(short_trigger - short_target, 0.01)

    # Position sizing — risk a fixed % of capital per trade
    risk_inr = capital_inr * risk_per_trade_pct
    long_qty = max(1, int(risk_inr // long_risk))
    short_qty = max(1, int(risk_inr // short_risk))

    long_pos_value = long_qty * long_trigger
    short_pos_value = short_qty * short_trigger
    margin_long = long_pos_value / MIS_LEVERAGE
    margin_short = short_pos_value / MIS_LEVERAGE

    # Cap qty so that margin <= 90% of capital (don't burn all margin on one trade)
    if margin_long > capital_inr * 0.9:
        long_qty = int((capital_inr * 0.9 * MIS_LEVERAGE) // long_trigger)
        long_pos_value = long_qty * long_trigger
        margin_long = long_pos_value / MIS_LEVERAGE
    if margin_short > capital_inr * 0.9:
        short_qty = int((capital_inr * 0.9 * MIS_LEVERAGE) // short_trigger)
        short_pos_value = short_qty * short_trigger
        margin_short = short_pos_value / MIS_LEVERAGE

    long_max_loss = long_qty * long_risk + long_pos_value * INTRADAY_COST_PCT
    long_tgt_profit = long_qty * long_rew - long_pos_value * INTRADAY_COST_PCT
    short_max_loss = short_qty * short_risk + short_pos_value * INTRADAY_COST_PCT
    short_tgt_profit = short_qty * short_rew - short_pos_value * INTRADAY_COST_PCT

    return IntradayPlan(
        ticker=ticker,
        prev_close=round(prev_close, 2),
        atr=round(atr, 2),
        expected_range_pct=round(atr / prev_close * 100, 2),
        bias=bias,
        conviction=conviction,
        long_trigger=long_trigger,
        long_stop=long_stop,
        long_target=long_target,
        long_risk_per_share=round(long_risk, 2),
        long_reward_per_share=round(long_rew, 2),
        long_rr=round(long_rew / long_risk, 2),
        short_trigger=short_trigger,
        short_stop=short_stop,
        short_target=short_target,
        short_risk_per_share=round(short_risk, 2),
        short_reward_per_share=round(short_rew, 2),
        short_rr=round(short_rew / short_risk, 2),
        capital_inr=capital_inr,
        risk_per_trade_pct=risk_per_trade_pct,
        long_qty=long_qty,
        short_qty=short_qty,
        long_position_value=round(long_pos_value, 2),
        short_position_value=round(short_pos_value, 2),
        margin_required_long=round(margin_long, 2),
        margin_required_short=round(margin_short, 2),
        long_max_loss_inr=round(long_max_loss, 2),
        long_target_profit_inr=round(long_tgt_profit, 2),
        short_max_loss_inr=round(short_max_loss, 2),
        short_target_profit_inr=round(short_tgt_profit, 2),
    )


def shortlist_tickers(top_n: int = 10) -> list[dict]:
    """Pick top-N tickers from latest debate verdicts: high conviction first.

    Returns list of {ticker, verdict, conviction}.
    """
    with get_session() as s:
        rows = s.execute(text("""
            WITH latest AS (
              SELECT ticker, judge_verdict, created_at,
                     ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY created_at DESC) AS rn
              FROM debate_result
            )
            SELECT ticker, judge_verdict FROM latest WHERE rn = 1
        """)).fetchall()
    import json as _json
    out = []
    for tkr, jv in rows:
        try:
            d = _json.loads(jv) if isinstance(jv, str) else jv
        except Exception:
            continue
        v = d.get("verdict")
        c = d.get("conviction") or 0
        if v in ("Bullish", "Bearish") and c >= 5:
            out.append({"ticker": tkr, "verdict": v, "conviction": c})
    out.sort(key=lambda r: -r["conviction"])
    return out[:top_n]


def todays_intraday_plan(capital_inr: float = 100_000,
                         risk_per_trade_pct: float = 0.01,
                         top_n: int = 10,
                         target_multiple: float = 1.5) -> list[dict]:
    """End-to-end: shortlist + ORB plan for each ticker, ready for the dashboard.

    target_multiple = how far above the breakout level the target sits, in
    units of the opening range. 1.5 is conservative; 2.5 is aggressive.
    """
    short = shortlist_tickers(top_n=top_n)
    plans: list[dict] = []
    for s in short:
        plan = build_plan(
            s["ticker"],
            verdict=s["verdict"],
            conviction=s["conviction"],
            capital_inr=capital_inr,
            risk_per_trade_pct=risk_per_trade_pct,
            target_multiple=target_multiple,
        )
        if plan is not None:
            plans.append(plan.to_dict())
    return plans
