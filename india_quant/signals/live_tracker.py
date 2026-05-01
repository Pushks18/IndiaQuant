"""Live intraday tracker for Alpha-50 screener picks.

Workflow:
  1. Pre-market: persist_today_predictions(plans)  ← saves screener output to DB
  2. During session: live_state(date) → returns each prediction with current
     status (PENDING/TRIGGERED/TARGET1/TARGET2/STOPPED) plus 5-min bars for
     charting. Called by /api/live/data on a 60s refresh.
  3. EOD: same call after 15:30 IST gives final outcomes.
  4. Accuracy: accuracy_summary() rolls up historical hit rates.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, date as date_cls, timedelta, time as time_cls
from typing import Iterable

import yfinance as yf
from loguru import logger
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    from datetime import timezone
    IST = timezone.utc

from india_quant.data.db import get_session
from india_quant.data.models import IntradayPrediction


# ── Persistence ────────────────────────────────────────────────────────────────

def persist_today_predictions(
    plans: list[dict],
    trade_date: date_cls | None = None,
) -> int:
    """Upsert today's screener picks. Returns number of rows written.

    Skips entries with no actionable bias (skip_day or NO_TRADE)."""
    if trade_date is None:
        trade_date = datetime.now(IST).date()

    actionable = [p for p in plans if p.get("bias") in ("LONG", "SHORT")]
    if not actionable:
        logger.info("[Tracker] No actionable picks to persist.")
        return 0

    with get_session() as s:
        for p in actionable:
            stmt = pg_insert(IntradayPrediction).values(
                id=uuid.uuid4(),
                date=trade_date,
                ticker=p["ticker"],
                bias=p["bias"],
                score=p.get("score"),
                score_long=p.get("score_long"),
                score_short=p.get("score_short"),
                prev_close=p.get("prev_close"),
                atr=p.get("atr"),
                atr_pct=p.get("atr_pct"),
                trigger=p["trigger"],
                stop=p["stop"],
                target1=p["target1"],
                target2=p["target2"],
                qty=p.get("qty"),
                max_loss_inr=p.get("max_loss_inr"),
                profit1_inr=p.get("profit1_inr"),
                profit2_inr=p.get("profit2_inr"),
                verdict=p.get("verdict"),
                conviction=p.get("conviction"),
                status="PENDING",
                meta=json.dumps(p, default=str),
            )
            # Idempotent: re-running the screener on the same day refreshes levels
            stmt = stmt.on_conflict_do_update(
                index_elements=["date", "ticker", "bias"],
                set_={
                    "score":       stmt.excluded.score,
                    "trigger":     stmt.excluded.trigger,
                    "stop":        stmt.excluded.stop,
                    "target1":     stmt.excluded.target1,
                    "target2":     stmt.excluded.target2,
                    "qty":         stmt.excluded.qty,
                    "verdict":     stmt.excluded.verdict,
                    "conviction":  stmt.excluded.conviction,
                    "meta":        stmt.excluded.meta,
                    "updated_at":  func_now(),
                },
            )
            s.execute(stmt)
    logger.info(f"[Tracker] Persisted {len(actionable)} predictions for {trade_date}")
    return len(actionable)


def func_now():
    from sqlalchemy import func
    return func.now()


# ── Loading ────────────────────────────────────────────────────────────────────

def get_predictions(trade_date: date_cls | None = None) -> list[dict]:
    if trade_date is None:
        trade_date = datetime.now(IST).date()
    with get_session() as s:
        rows = s.execute(text("""
            SELECT id, date, ticker, bias, score, score_long, score_short,
                   prev_close, atr, atr_pct, trigger, stop, target1, target2,
                   qty, max_loss_inr, profit1_inr, profit2_inr, verdict, conviction,
                   status, triggered_at, exit_at, exit_price, exit_reason,
                   max_favorable_pct, max_adverse_pct, realized_pnl_inr
            FROM intraday_prediction WHERE date = :d ORDER BY score DESC NULLS LAST
        """), {"d": trade_date}).mappings().all()
    return [dict(r) for r in rows]


# ── Intraday bar fetch ─────────────────────────────────────────────────────────

def fetch_intraday_bars(ticker: str) -> list[dict]:
    """Today's 5-minute bars in IST chronological order. Returns [] if empty."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="5m")
        if df.empty:
            return []
        bars: list[dict] = []
        for ts, row in df.iterrows():
            ts_ist = ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(IST)
            bars.append({
                "time":   int(ts_ist.timestamp()),
                "iso":    ts_ist.isoformat(),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row.get("Volume", 0) or 0),
            })
        return bars
    except Exception as e:
        logger.warning(f"[Tracker] {ticker} intraday fetch failed: {e}")
        return []


# ── Outcome evaluation ─────────────────────────────────────────────────────────

EOD_CUTOFF = time_cls(15, 15)   # square-off time per house rules


def evaluate_outcome(pred: dict, bars: list[dict]) -> dict:
    """Walk bars chronologically, compute current status + extrema.

    Conservative single-bar tie-break: STOPPED beats TARGET when a bar's range
    spans both. Returns a dict of fields to update on the prediction row.
    """
    bias    = pred["bias"]
    trigger = float(pred["trigger"])
    stop    = float(pred["stop"])
    t1      = float(pred["target1"])
    t2      = float(pred["target2"])

    status        = "PENDING"
    triggered_at  = None
    exit_at       = None
    exit_price    = None
    exit_reason   = None
    mfe_pct       = 0.0   # max favorable excursion (post-entry)
    mae_pct       = 0.0   # max adverse excursion  (post-entry)

    for b in bars:
        bar_dt = datetime.fromisoformat(b["iso"])
        h, l   = b["high"], b["low"]

        # Stage 1 — wait for trigger touch
        if status == "PENDING":
            hit_trig = (bias == "LONG" and h >= trigger) or (bias == "SHORT" and l <= trigger)
            if hit_trig:
                status       = "TRIGGERED"
                triggered_at = bar_dt
                # Within the trigger bar, check stop/target order conservatively
                _status, _exit_p, _exit_r = _check_exits_in_bar(bias, b, trigger, stop, t1, t2)
                if _status:
                    status      = _status
                    exit_at     = bar_dt
                    exit_price  = _exit_p
                    exit_reason = _exit_r
                else:
                    # Track excursion from trigger
                    mfe_pct, mae_pct = _excursion(bias, trigger, h, l, mfe_pct, mae_pct)
            continue

        # Stage 2 — already triggered, look for exit
        if status in ("TRIGGERED", "TARGET1"):
            mfe_pct, mae_pct = _excursion(bias, trigger, h, l, mfe_pct, mae_pct)
            _status, _exit_p, _exit_r = _check_exits_in_bar(bias, b, trigger, stop, t1, t2)
            if _status == "STOPPED":
                status, exit_at, exit_price, exit_reason = "STOPPED", bar_dt, _exit_p, "STOPPED"
                break
            if _status == "TARGET2":
                status, exit_at, exit_price, exit_reason = "TARGET2", bar_dt, _exit_p, "TARGET2"
                break
            if _status == "TARGET1" and status == "TRIGGERED":
                status = "TARGET1"   # touched T1 but keep watching for T2/stop

    # Force EOD square-off if still open after market close
    if status in ("TRIGGERED", "TARGET1") and bars:
        last_bar_t = datetime.fromisoformat(bars[-1]["iso"]).time()
        if last_bar_t >= EOD_CUTOFF:
            exit_at     = datetime.fromisoformat(bars[-1]["iso"])
            exit_price  = bars[-1]["close"]
            exit_reason = "EOD"

    realized_pnl = None
    if exit_price is not None and pred.get("qty"):
        qty = int(pred["qty"])
        if bias == "LONG":
            realized_pnl = (exit_price - trigger) * qty
        else:
            realized_pnl = (trigger - exit_price) * qty

    return {
        "status": status,
        "triggered_at": triggered_at,
        "exit_at": exit_at,
        "exit_price": round(exit_price, 2) if exit_price else None,
        "exit_reason": exit_reason,
        "max_favorable_pct": round(mfe_pct, 3),
        "max_adverse_pct":   round(mae_pct, 3),
        "realized_pnl_inr":  round(realized_pnl, 0) if realized_pnl is not None else None,
    }


def _check_exits_in_bar(bias: str, bar: dict, trigger: float,
                         stop: float, t1: float, t2: float):
    """Returns (status, exit_price, exit_reason) or (None, None, None)."""
    h, l = bar["high"], bar["low"]
    if bias == "LONG":
        hit_stop = l <= stop
        hit_t2   = h >= t2
        hit_t1   = h >= t1
        if hit_stop:                 # conservative: stop wins ties
            return "STOPPED", stop, "STOPPED"
        if hit_t2:
            return "TARGET2", t2, "TARGET2"
        if hit_t1:
            return "TARGET1", t1, "TARGET1"
    else:  # SHORT
        hit_stop = h >= stop
        hit_t2   = l <= t2
        hit_t1   = l <= t1
        if hit_stop:
            return "STOPPED", stop, "STOPPED"
        if hit_t2:
            return "TARGET2", t2, "TARGET2"
        if hit_t1:
            return "TARGET1", t1, "TARGET1"
    return None, None, None


def _excursion(bias: str, trigger: float, high: float, low: float,
               mfe: float, mae: float) -> tuple[float, float]:
    if bias == "LONG":
        fav = (high - trigger) / trigger * 100
        adv = (low  - trigger) / trigger * 100
        return max(mfe, fav), min(mae, adv)
    else:
        fav = (trigger - low)  / trigger * 100
        adv = (trigger - high) / trigger * 100
        return max(mfe, fav), min(mae, adv)


# ── Live state assembly (called by /api/live/data) ─────────────────────────────

def live_state(trade_date: date_cls | None = None,
               include_bars: bool = True) -> dict:
    """One-shot snapshot for the Live tab: predictions + bars + outcomes."""
    if trade_date is None:
        trade_date = datetime.now(IST).date()

    preds = get_predictions(trade_date)
    items = []
    for pred in preds:
        bars = fetch_intraday_bars(pred["ticker"]) if include_bars else []
        outcome = evaluate_outcome(pred, bars) if bars else {
            "status": pred["status"], "triggered_at": pred.get("triggered_at"),
            "exit_at": pred.get("exit_at"), "exit_price": pred.get("exit_price"),
            "exit_reason": pred.get("exit_reason"),
            "max_favorable_pct": pred.get("max_favorable_pct") or 0,
            "max_adverse_pct":   pred.get("max_adverse_pct") or 0,
            "realized_pnl_inr":  pred.get("realized_pnl_inr"),
        }
        # Persist outcome updates if they changed
        _maybe_update_pred(pred["id"], pred, outcome)

        items.append({
            "id":        str(pred["id"]),
            "ticker":    pred["ticker"],
            "bias":      pred["bias"],
            "score":     pred.get("score"),
            "verdict":   pred.get("verdict"),
            "conviction": pred.get("conviction"),
            "prev_close": pred.get("prev_close"),
            "atr":       pred.get("atr"),
            "trigger":   pred["trigger"],
            "stop":      pred["stop"],
            "target1":   pred["target1"],
            "target2":   pred["target2"],
            "qty":       pred.get("qty"),
            "outcome":   outcome,
            "bars":      bars,
        })

    summary = _day_summary(items)
    return {
        "date": trade_date.isoformat(),
        "items": items,
        "summary": summary,
    }


def _maybe_update_pred(pred_id, pred: dict, outcome: dict):
    """Persist outcome changes to DB so EOD stats are reliable."""
    fields_changed = (
        outcome["status"] != pred.get("status")
        or outcome.get("exit_reason") != pred.get("exit_reason")
        or outcome.get("realized_pnl_inr") != pred.get("realized_pnl_inr")
    )
    if not fields_changed:
        return
    with get_session() as s:
        s.execute(text("""
            UPDATE intraday_prediction
            SET status=:st, triggered_at=:ta, exit_at=:xa, exit_price=:xp,
                exit_reason=:xr, max_favorable_pct=:mfe, max_adverse_pct=:mae,
                realized_pnl_inr=:pnl, updated_at=now()
            WHERE id = :id
        """), {
            "id":  pred_id,
            "st":  outcome["status"],
            "ta":  outcome.get("triggered_at"),
            "xa":  outcome.get("exit_at"),
            "xp":  outcome.get("exit_price"),
            "xr":  outcome.get("exit_reason"),
            "mfe": outcome.get("max_favorable_pct"),
            "mae": outcome.get("max_adverse_pct"),
            "pnl": outcome.get("realized_pnl_inr"),
        })


def _day_summary(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    triggered = sum(1 for i in items if i["outcome"]["status"] != "PENDING")
    t1_hits   = sum(1 for i in items if i["outcome"]["status"] in ("TARGET1", "TARGET2"))
    t2_hits   = sum(1 for i in items if i["outcome"]["status"] == "TARGET2")
    stopped   = sum(1 for i in items if i["outcome"]["status"] == "STOPPED")
    pnl       = sum((i["outcome"].get("realized_pnl_inr") or 0) for i in items)
    return {
        "n": n,
        "triggered": triggered,
        "t1_hits": t1_hits,
        "t2_hits": t2_hits,
        "stopped": stopped,
        "pending": n - triggered,
        "trigger_rate":  round(triggered / n * 100, 1) if n else 0,
        "t1_hit_rate":   round(t1_hits / triggered * 100, 1) if triggered else 0,
        "stop_rate":     round(stopped / triggered * 100, 1) if triggered else 0,
        "net_pnl_inr":   round(pnl, 0),
    }


# ── Historical accuracy ────────────────────────────────────────────────────────

def accuracy_summary(start: date_cls | None = None,
                     end: date_cls | None = None) -> dict:
    """Cumulative accuracy across persisted predictions."""
    end = end or datetime.now(IST).date()
    start = start or (end - timedelta(days=30))
    with get_session() as s:
        rows = s.execute(text("""
            SELECT date, ticker, bias, status, exit_reason, realized_pnl_inr,
                   max_favorable_pct, max_adverse_pct, score
            FROM intraday_prediction
            WHERE date BETWEEN :s AND :e
            ORDER BY date DESC, score DESC NULLS LAST
        """), {"s": start, "e": end}).mappings().all()

    rows = [dict(r) for r in rows]
    n = len(rows)
    triggered = [r for r in rows if r["status"] != "PENDING"]
    t1 = [r for r in rows if r["status"] in ("TARGET1", "TARGET2")]
    t2 = [r for r in rows if r["status"] == "TARGET2"]
    stopped = [r for r in rows if r["status"] == "STOPPED"]
    wins = [r for r in triggered if (r.get("realized_pnl_inr") or 0) > 0]

    by_day: dict[str, dict] = {}
    for r in rows:
        d = r["date"].isoformat()
        b = by_day.setdefault(d, {"n": 0, "t1": 0, "stopped": 0, "pnl": 0.0})
        b["n"] += 1
        if r["status"] in ("TARGET1", "TARGET2"): b["t1"] += 1
        if r["status"] == "STOPPED":              b["stopped"] += 1
        b["pnl"] += float(r.get("realized_pnl_inr") or 0)

    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "totals": {
            "n":              n,
            "triggered":      len(triggered),
            "t1_hits":        len(t1),
            "t2_hits":        len(t2),
            "stopped":        len(stopped),
            "win_rate":       round(len(wins) / len(triggered) * 100, 1) if triggered else 0,
            "trigger_rate":   round(len(triggered) / n * 100, 1) if n else 0,
            "t1_hit_rate":    round(len(t1) / len(triggered) * 100, 1) if triggered else 0,
            "stop_rate":      round(len(stopped) / len(triggered) * 100, 1) if triggered else 0,
            "net_pnl_inr":    round(sum((r.get("realized_pnl_inr") or 0) for r in rows), 0),
        },
        "by_day": [{"date": d, **v} for d, v in sorted(by_day.items(), reverse=True)],
        "rows": [
            {**r, "date": r["date"].isoformat()} for r in rows[:200]
        ],
    }
