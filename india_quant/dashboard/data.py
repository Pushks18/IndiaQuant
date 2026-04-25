"""Read-only DB helpers for the dashboard."""
import json
from datetime import date, timedelta

from sqlalchemy import text

from india_quant.data.db import get_session


def latest_trading_date() -> date:
    with get_session() as s:
        row = s.execute(text(
            "SELECT MAX(datetime)::date FROM price_data WHERE interval='1d'"
        )).fetchone()
    return row[0] if row and row[0] else date.today()


def macro_snapshot() -> dict:
    """Quick macro card from yfinance — cached lightly inside the call."""
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI").history(period="2d")
        vix = yf.Ticker("^INDIAVIX").history(period="2d")
        usdinr = yf.Ticker("USDINR=X").history(period="2d")
        out = {}
        if not nifty.empty:
            cur = float(nifty["Close"].iloc[-1])
            prv = float(nifty["Close"].iloc[0]) if len(nifty) > 1 else cur
            out["nifty"] = {"value": round(cur, 2), "change_pct": round((cur/prv - 1) * 100, 2)}
        if not vix.empty:
            out["vix"] = {"value": round(float(vix["Close"].iloc[-1]), 2)}
        if not usdinr.empty:
            cur = float(usdinr["Close"].iloc[-1])
            prv = float(usdinr["Close"].iloc[0]) if len(usdinr) > 1 else cur
            out["usdinr"] = {"value": round(cur, 4), "change_pct": round((cur/prv - 1) * 100, 2)}
        return out
    except Exception:
        return {}


def all_tickers() -> list[str]:
    with get_session() as s:
        rows = s.execute(text(
            "SELECT DISTINCT ticker FROM price_data WHERE interval='1d' ORDER BY ticker"
        )).fetchall()
    return [r[0] for r in rows]


def top_movers(limit: int = 10) -> dict:
    """Return advancers + decliners over the last available 2 trading days."""
    with get_session() as s:
        rows = s.execute(text("""
            WITH last2 AS (
              SELECT ticker, datetime, close,
                     ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY datetime DESC) AS rn
              FROM price_data WHERE interval='1d'
            )
            SELECT ticker,
                   MAX(CASE WHEN rn=1 THEN close END) AS today_close,
                   MAX(CASE WHEN rn=2 THEN close END) AS prev_close
            FROM last2 WHERE rn <= 2
            GROUP BY ticker
            HAVING MAX(CASE WHEN rn=1 THEN close END) IS NOT NULL
               AND MAX(CASE WHEN rn=2 THEN close END) IS NOT NULL
        """)).fetchall()
    movers = []
    for ticker, today, prev in rows:
        if prev and prev > 0:
            change = (float(today) / float(prev) - 1) * 100
            movers.append({"ticker": ticker, "close": round(float(today), 2),
                           "change_pct": round(change, 2)})
    movers.sort(key=lambda r: r["change_pct"], reverse=True)
    return {"advancers": movers[:limit], "decliners": list(reversed(movers[-limit:]))}


def latest_factor_scores(limit: int = 60) -> list[dict]:
    """Return latest factor scores per ticker."""
    with get_session() as s:
        rows = s.execute(text("""
            WITH ranked AS (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
              FROM factor_scores
            )
            SELECT * FROM ranked WHERE rn = 1 ORDER BY ticker LIMIT :n
        """), {"n": limit}).mappings().all()
    return [dict(r) for r in rows]


def latest_analyst_reports(ticker: str | None = None, limit: int = 30) -> list[dict]:
    if ticker:
        sql = """
            SELECT ticker, date, agent_name, report_json, created_at
            FROM analyst_report WHERE ticker = :t
            ORDER BY date DESC, created_at DESC LIMIT :n
        """
        params = {"t": ticker, "n": limit}
    else:
        sql = """
            SELECT ticker, date, agent_name, report_json, created_at
            FROM analyst_report ORDER BY created_at DESC LIMIT :n
        """
        params = {"n": limit}
    with get_session() as s:
        rows = s.execute(text(sql), params).fetchall()
    out = []
    for r in rows:
        try:
            report = json.loads(r[3])
        except Exception:
            report = {"raw": r[3]}
        out.append({
            "ticker": r[0], "date": str(r[1]), "agent_name": r[2],
            "report": report, "created_at": str(r[4]),
        })
    return out


def latest_debate(ticker: str | None = None, limit: int = 20) -> list[dict]:
    if ticker:
        sql = """
            SELECT ticker, date, bull_report, bear_report, judge_verdict, created_at
            FROM debate_result WHERE ticker = :t
            ORDER BY date DESC, created_at DESC LIMIT :n
        """
        params = {"t": ticker, "n": limit}
    else:
        sql = """
            SELECT ticker, date, bull_report, bear_report, judge_verdict, created_at
            FROM debate_result ORDER BY created_at DESC LIMIT :n
        """
        params = {"n": limit}
    with get_session() as s:
        rows = s.execute(text(sql), params).fetchall()
    out = []
    for r in rows:
        try:
            bull = json.loads(r[2]) if r[2] else {}
            bear = json.loads(r[3]) if r[3] else {}
            judge = json.loads(r[4]) if r[4] else {}
        except Exception:
            bull = bear = judge = {}
        out.append({
            "ticker": r[0], "date": str(r[1]),
            "bull": bull, "bear": bear, "judge": judge,
            "created_at": str(r[5]),
        })
    return out


def latest_proposals(ticker: str | None = None, limit: int = 30) -> list[dict]:
    if ticker:
        sql = """
            SELECT id, ticker, date, instrument, direction, entry_price, stop_loss,
                   target_1, target_2, risk_reward, position_size_pct, rationale,
                   risk_status, created_at
            FROM trade_proposal WHERE ticker = :t
            ORDER BY date DESC, created_at DESC LIMIT :n
        """
        params = {"t": ticker, "n": limit}
    else:
        sql = """
            SELECT id, ticker, date, instrument, direction, entry_price, stop_loss,
                   target_1, target_2, risk_reward, position_size_pct, rationale,
                   risk_status, created_at
            FROM trade_proposal ORDER BY created_at DESC LIMIT :n
        """
        params = {"n": limit}
    with get_session() as s:
        rows = s.execute(text(sql), params).fetchall()
    cols = ["id","ticker","date","instrument","direction","entry_price","stop_loss",
            "target_1","target_2","risk_reward","position_size_pct","rationale",
            "risk_status","created_at"]
    return [dict(zip(cols, r)) for r in rows]


def price_history(ticker: str, days: int = 180) -> list[dict]:
    with get_session() as s:
        rows = s.execute(text("""
            SELECT datetime::date, open, high, low, close, volume
            FROM price_data
            WHERE ticker = :t AND interval = '1d'
            AND datetime >= NOW() - INTERVAL '%d days'
            ORDER BY datetime
        """ % int(days)), {"t": ticker}).fetchall()
    return [
        {"date": str(r[0]), "open": float(r[1] or 0), "high": float(r[2] or 0),
         "low": float(r[3] or 0), "close": float(r[4] or 0), "volume": float(r[5] or 0)}
        for r in rows
    ]


def data_health() -> dict:
    """Return pipeline / data freshness for the dashboard health card."""
    with get_session() as s:
        price_row = s.execute(text(
            "SELECT COUNT(*), MAX(datetime), COUNT(DISTINCT ticker) FROM price_data"
        )).fetchone()
        opt_row = s.execute(text(
            "SELECT COUNT(*), MAX(timestamp) FROM option_chain"
        )).fetchone()
        news_row = s.execute(text(
            "SELECT COUNT(*), MAX(timestamp) FROM news_article"
        )).fetchone()
        sent_row = s.execute(text(
            "SELECT COUNT(*), MAX(date) FROM sentiment_aggregate"
        )).fetchone()
        fact_row = s.execute(text(
            "SELECT COUNT(*), MAX(date) FROM factor_scores"
        )).fetchone()
        sig_row = s.execute(text(
            "SELECT COUNT(*), MAX(date) FROM signal_labels"
        )).fetchone()
        rep_row = s.execute(text(
            "SELECT COUNT(*), MAX(created_at) FROM analyst_report"
        )).fetchone()
        deb_row = s.execute(text(
            "SELECT COUNT(*), MAX(created_at) FROM debate_result"
        )).fetchone()
        prop_row = s.execute(text(
            "SELECT COUNT(*), MAX(created_at) FROM trade_proposal"
        )).fetchone()
    return {
        "price_data": {"rows": price_row[0], "last": str(price_row[1]) if price_row[1] else None,
                       "tickers": price_row[2]},
        "option_chain": {"rows": opt_row[0], "last": str(opt_row[1]) if opt_row[1] else None},
        "news_article": {"rows": news_row[0], "last": str(news_row[1]) if news_row[1] else None},
        "sentiment_aggregate": {"rows": sent_row[0], "last": str(sent_row[1]) if sent_row[1] else None},
        "factor_scores": {"rows": fact_row[0], "last": str(fact_row[1]) if fact_row[1] else None},
        "signal_labels": {"rows": sig_row[0], "last": str(sig_row[1]) if sig_row[1] else None},
        "analyst_report": {"rows": rep_row[0], "last": str(rep_row[1]) if rep_row[1] else None},
        "debate_result": {"rows": deb_row[0], "last": str(deb_row[1]) if deb_row[1] else None},
        "trade_proposal": {"rows": prop_row[0], "last": str(prop_row[1]) if prop_row[1] else None},
    }


def signal_summary() -> list[dict]:
    """Per-ticker signal summary: latest close, change, factor scores, ML prediction, latest verdict."""
    with get_session() as s:
        rows = s.execute(text("""
            WITH latest_prices AS (
              SELECT ticker, datetime, close,
                     ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY datetime DESC) AS rn
              FROM price_data WHERE interval='1d'
            ),
            today_p AS (SELECT ticker, close FROM latest_prices WHERE rn=1),
            prev_p  AS (SELECT ticker, close FROM latest_prices WHERE rn=2),
            latest_factors AS (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
              FROM factor_scores
            ),
            latest_verdict AS (
              SELECT ticker, judge_verdict, created_at,
                     ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY created_at DESC) AS rn
              FROM debate_result
            ),
            latest_pred AS (
              SELECT ticker, predicted_return, signal_rank,
                     ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
              FROM signal_labels WHERE horizon='1d' AND predicted_return IS NOT NULL
            )
            SELECT t.ticker, t.close, p.close AS prev_close,
                   f.momentum_12_1, f.momentum_3, f.realized_vol,
                   v.judge_verdict,
                   pr.predicted_return, pr.signal_rank
            FROM today_p t
            LEFT JOIN prev_p p ON p.ticker = t.ticker
            LEFT JOIN latest_factors f ON f.ticker = t.ticker AND f.rn = 1
            LEFT JOIN latest_verdict v ON v.ticker = t.ticker AND v.rn = 1
            LEFT JOIN latest_pred pr ON pr.ticker = t.ticker AND pr.rn = 1
            ORDER BY pr.signal_rank NULLS LAST, t.ticker
        """)).fetchall()
    out = []
    for r in rows:
        ticker, close, prev_close, m12_1, m3, rv, jv, pred, rank = r
        change_pct = None
        if close and prev_close:
            change_pct = round((float(close) / float(prev_close) - 1) * 100, 2)
        verdict = None
        conviction = None
        if jv:
            try:
                jv_d = json.loads(jv) if isinstance(jv, str) else jv
                verdict = jv_d.get("verdict")
                conviction = jv_d.get("conviction")
            except Exception:
                pass
        out.append({
            "ticker": ticker,
            "close": round(float(close), 2) if close else None,
            "change_pct": change_pct,
            "momentum_12_1": round(float(m12_1), 3) if m12_1 is not None else None,
            "momentum_3": round(float(m3), 3) if m3 is not None else None,
            "realized_vol": round(float(rv), 3) if rv is not None else None,
            "predicted_return_1d": round(float(pred), 5) if pred is not None else None,
            "signal_rank": int(rank) if rank is not None else None,
            "verdict": verdict,
            "conviction": conviction,
        })
    return out
