"""Judge agent — synthesises Bull/Bear into a structured verdict and orchestrates the debate."""
import json as _json
from datetime import date, timedelta

from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from india_quant.data.db import get_session
from india_quant.data.models import DebateResult
from india_quant.agents.researcher import BullAgent, BearAgent


class JudgeAgent:
    """Net-score the bull/bear arguments, classify verdict, suggest horizon."""

    def run(
        self,
        bull_report: dict,
        bear_report: dict,
        analyst_summary: dict,
        ml_signal: dict,
    ) -> dict:
        bull_conf = int(bull_report.get("confidence") or 0)
        bear_conf = int(bear_report.get("confidence") or 0)
        bull_pts = list(bull_report.get("key_catalysts") or [])
        bear_pts = list(bear_report.get("key_risks") or [])

        # ML weight: t-stat > 3 effectively means high-conviction
        pred = ml_signal.get("predicted_return_1d") or 0
        ml_bonus = 0
        if pred > 0.005:
            ml_bonus = 2
        elif pred < -0.005:
            ml_bonus = -2

        net = (bull_conf - bear_conf) + ml_bonus
        if net >= 4:
            verdict = "Bullish"
        elif net <= -4:
            verdict = "Bearish"
        elif abs(net) <= 1 and bull_conf >= 4 and bear_conf >= 4:
            verdict = "Mixed"
        elif net > 0:
            verdict = "Bullish"
        elif net < 0:
            verdict = "Bearish"
        else:
            verdict = "Neutral"

        conviction = max(1, min(10, abs(net) + 3))

        # Horizon: ML pred magnitude → intraday/swing/positional
        horizon = "swing"
        if pred and abs(pred) > 0.02:
            horizon = "positional"
        elif pred and abs(pred) < 0.005:
            horizon = "intraday"

        # Watchout: surface most pressing risk
        macro = analyst_summary.get("macro_analyst") or analyst_summary.get("macro") or {}
        sent = analyst_summary.get("sentiment_analyst") or analyst_summary.get("sentiment") or {}
        watchout = "Standard market risk."
        today = date.today()
        if today.weekday() == 3:
            watchout = "Thursday weekly F&O expiry — elevated vol."
        elif macro.get("regime_label") in ("High-Vol", "Bear"):
            watchout = f"Regime: {macro.get('regime_label')} (VIX {macro.get('india_vix')})."
        elif sent.get("rbi_relevance"):
            watchout = f"RBI policy in {sent.get('rbi_days_away')} days."

        uncertainty = "Mixed signals" if verdict == "Mixed" else (
            f"Bull confidence {bull_conf} vs Bear {bear_conf}; ML pred {pred:+.3f}." if pred else
            f"Bull {bull_conf} vs Bear {bear_conf}."
        )

        return {
            "verdict": verdict,
            "conviction": conviction,
            "bull_points_accepted": bull_pts[:5],
            "bear_points_accepted": bear_pts[:5],
            "key_uncertainty": uncertainty,
            "suggested_horizon": horizon,
            "watchout": watchout,
            "net_score": net,
        }


# ── Debate orchestrator ───────────────────────────────────────────────────────

def run_debate(ticker: str, run_date: str = None) -> dict:
    """Run analysts (if missing), Bull, Bear, Judge — store and return."""
    from sqlalchemy import text

    run_date = run_date or date.today().isoformat()
    logger.info(f"[Debate] {ticker} on {run_date}")

    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT agent_name, report_json FROM analyst_report
                WHERE ticker = :t AND date = :d
            """),
            {"t": ticker, "d": run_date},
        ).fetchall()

    analyst_summary = {row[0]: _json.loads(row[1]) for row in rows} if rows else {}

    if not analyst_summary:
        logger.info(f"[Debate] No reports for {ticker} on {run_date} — running analysts")
        from india_quant.agents.technical_analyst import TechnicalAnalystAgent
        from india_quant.agents.fundamental_analyst import FundamentalAnalystAgent
        from india_quant.agents.sentiment_analyst import SentimentAnalystAgent
        from india_quant.agents.macro_analyst import MacroAnalystAgent
        analyst_summary = {
            "technical_analyst": TechnicalAnalystAgent().run({"ticker": ticker}, run_date),
            "fundamental_analyst": FundamentalAnalystAgent().run({"ticker": ticker}, run_date),
            "sentiment_analyst": SentimentAnalystAgent().run({"ticker": ticker}, run_date),
            "macro_analyst": MacroAnalystAgent().run({"ticker": ticker}, run_date),
        }

    with get_session() as session:
        from sqlalchemy import text as _t
        ml_row = session.execute(
            _t("""
                SELECT predicted_return, signal_rank FROM signal_labels
                WHERE ticker = :t AND horizon = '1d' AND predicted_return IS NOT NULL
                  AND date <= :d
                ORDER BY date DESC LIMIT 1
            """),
            {"t": ticker, "d": run_date},
        ).fetchone()

    ml_signal = {}
    if ml_row and ml_row[0] is not None:
        ml_signal = {
            "predicted_return_1d": float(ml_row[0]),
            "signal_rank": int(ml_row[1]) if ml_row[1] is not None else None,
        }

    bull_report = BullAgent().run(analyst_summary, ml_signal)
    bear_report = BearAgent().run(analyst_summary, ml_signal)
    judge_verdict = JudgeAgent().run(bull_report, bear_report, analyst_summary, ml_signal)

    debate_date = date.fromisoformat(run_date)
    with get_session() as session:
        stmt = insert(DebateResult).values(
            ticker=ticker,
            date=debate_date,
            bull_report=_json.dumps(bull_report, default=str),
            bear_report=_json.dumps(bear_report, default=str),
            judge_verdict=_json.dumps(judge_verdict, default=str),
        )
        session.execute(stmt)

    logger.info(f"[Debate] {ticker}: {judge_verdict.get('verdict')} (conviction {judge_verdict.get('conviction')})")
    return {
        "ticker": ticker,
        "date": run_date,
        "bull": bull_report,
        "bear": bear_report,
        "judge": judge_verdict,
        "analyst_summary": analyst_summary,
        "ml_signal": ml_signal,
    }
