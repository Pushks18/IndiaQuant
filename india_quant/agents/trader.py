"""Trader agent — converts judge verdict into a concrete trade proposal.

Deterministic: uses CMP + 14-day ATR + judge verdict + nearest support/resistance
to derive entry, stop, target_1, target_2, R:R, position size.
"""
from datetime import date, timedelta
from typing import Literal

from loguru import logger
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert

from india_quant.data.db import get_session
from india_quant.data.models import TradeProposal as TradeProposalModel


class TradeProposalSchema(BaseModel):
    ticker: str
    instrument: Literal["equity", "futures", "options_buy", "options_spread", "options_sell"]
    direction: Literal["long", "short"]
    entry_price: float
    entry_zone: tuple[float, float]
    stop_loss: float
    target_1: float
    target_2: float | None = None
    risk_reward: float
    time_horizon: str
    position_size_pct: float
    rationale: str
    invalidation: str
    nse_risks: list[str]
    expiry_date: str | None = None
    strike: float | None = None


def _safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


class TraderAgent:
    """Build a concrete trade proposal from the debate result deterministically."""

    def propose_trade(self, debate_result: dict, options_signals: dict | None = None) -> dict:
        ticker = debate_result.get("ticker", "")
        run_date = debate_result.get("date") or date.today().isoformat()
        judge = debate_result.get("judge", {}) or {}
        analyst_summary = debate_result.get("analyst_summary", {}) or {}
        ml_signal = debate_result.get("ml_signal", {}) or {}
        options_signals = options_signals or {}

        tech = (
            analyst_summary.get("technical_analyst")
            or analyst_summary.get("technical")
            or {}
        )
        macro = (
            analyst_summary.get("macro_analyst")
            or analyst_summary.get("macro")
            or {}
        )

        verdict = judge.get("verdict", "Neutral")
        conviction = int(judge.get("conviction") or 5)
        horizon = judge.get("suggested_horizon", "swing")

        if verdict == "Neutral" or verdict == "Mixed":
            return {
                "ticker": ticker,
                "date": run_date,
                "instrument": "equity",
                "direction": "long",
                "skipped": True,
                "rationale": f"Judge verdict {verdict} — no actionable setup.",
            }

        cmp_ = _safe(tech, "indicators", "cmp")
        atr = _safe(tech, "indicators", "atr")
        support = _safe(tech, "key_levels", "support")
        resistance = _safe(tech, "key_levels", "resistance")
        lower_circuit = _safe(tech, "key_levels", "lower_circuit")
        upper_circuit = _safe(tech, "key_levels", "upper_circuit")

        if cmp_ is None:
            return {
                "ticker": ticker,
                "date": run_date,
                "instrument": "equity",
                "direction": "long",
                "skipped": True,
                "rationale": "No CMP available.",
            }

        atr = atr if atr and atr > 0 else cmp_ * 0.015
        direction = "long" if verdict == "Bullish" else "short"

        # Stop = the CLOSER of (1.5×ATR) and (just past S/R) — never wider than 1.5×ATR.
        # Target = the FURTHER of (nearby S/R) and (2.5×ATR) — never tighter than 2.5×ATR.
        atr_stop_dist = 1.5 * atr
        atr_target_dist = 2.5 * atr
        if direction == "long":
            entry_price = round(cmp_, 2)
            entry_zone = (round(cmp_ - 0.4 * atr, 2), round(cmp_ + 0.2 * atr, 2))
            atr_stop = cmp_ - atr_stop_dist
            if support and support < cmp_:
                stop_loss = round(max(atr_stop, support * 0.997), 2)
            else:
                stop_loss = round(atr_stop, 2)
            atr_target = cmp_ + atr_target_dist
            target_1 = round(max(resistance or atr_target, atr_target), 2)
            target_2 = round(target_1 + 1.5 * atr, 2)
            invalidation = f"Close below {stop_loss}"
        else:
            entry_price = round(cmp_, 2)
            entry_zone = (round(cmp_ - 0.2 * atr, 2), round(cmp_ + 0.4 * atr, 2))
            atr_stop = cmp_ + atr_stop_dist
            if resistance and resistance > cmp_:
                stop_loss = round(min(atr_stop, resistance * 1.003), 2)
            else:
                stop_loss = round(atr_stop, 2)
            atr_target = cmp_ - atr_target_dist
            target_1 = round(min(support or atr_target, atr_target), 2)
            target_2 = round(target_1 - 1.5 * atr, 2)
            invalidation = f"Close above {stop_loss}"

        if entry_price == stop_loss:
            stop_loss = round(stop_loss * (0.99 if direction == "long" else 1.01), 2)
        risk = abs(entry_price - stop_loss)
        reward = abs(target_1 - entry_price)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        # Instrument selection
        instrument = "equity"
        vrp = options_signals.get("vrp")
        iv_skew = options_signals.get("iv_skew")
        if conviction >= 8 and horizon != "intraday":
            instrument = "futures"
        if vrp is not None and vrp > 1.5:
            instrument = "options_sell"
        elif iv_skew is not None and abs(iv_skew) > 0.05 and horizon != "intraday":
            instrument = "options_spread"

        # Size: scale with conviction, capped
        base_size = 0.01 + 0.003 * conviction
        position_size_pct = round(min(base_size, 0.05), 4)

        nse_risks = []
        today = date.today()
        if today.weekday() in (2, 3) and ticker in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
            nse_risks.append("Weekly F&O expiry — pin/vol risk")
        if upper_circuit and entry_price >= upper_circuit * 0.97:
            nse_risks.append(f"Within 3% of upper circuit ({upper_circuit})")
        if lower_circuit and entry_price <= lower_circuit * 1.03:
            nse_risks.append(f"Within 3% of lower circuit ({lower_circuit})")
        if (macro.get("regime_label") or "") in ("High-Vol", "Bear"):
            nse_risks.append(f"Regime: {macro.get('regime_label')}")

        bullish_pts = judge.get("bull_points_accepted") or []
        bearish_pts = judge.get("bear_points_accepted") or []
        rationale_pts = bullish_pts if direction == "long" else bearish_pts
        rationale = (
            f"Judge {verdict} with conviction {conviction}/10. "
            + ("Drivers: " + "; ".join(rationale_pts[:3]) + "." if rationale_pts else "")
            + (f" ML 1d return: {ml_signal.get('predicted_return_1d'):+.3f}." if ml_signal.get('predicted_return_1d') is not None else "")
        )

        proposal = {
            "ticker": ticker,
            "date": run_date,
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "entry_zone": list(entry_zone),
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "risk_reward": rr,
            "time_horizon": horizon,
            "position_size_pct": position_size_pct,
            "rationale": rationale.strip(),
            "invalidation": invalidation,
            "nse_risks": nse_risks,
            "expiry_date": None,
            "strike": None,
        }

        # Persist (only if not skipped)
        try:
            d = date.fromisoformat(run_date) if isinstance(run_date, str) else run_date
            with get_session() as session:
                stmt = insert(TradeProposalModel).values(
                    ticker=ticker,
                    date=d,
                    instrument=instrument,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    risk_reward=rr,
                    position_size_pct=position_size_pct,
                    rationale=rationale,
                    risk_status="PENDING",
                )
                session.execute(stmt)
        except Exception as e:
            logger.error(f"TraderAgent persist failed: {e}")

        return proposal
