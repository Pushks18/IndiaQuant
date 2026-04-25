"""Macro analyst — NIFTY regime, India VIX, USD/INR, RBI stance, market breadth."""
from datetime import date

from india_quant.agents.base import BaseAnalystAgent
from india_quant.config import cfg


def _get_india_vix() -> dict:
    try:
        import yfinance as yf
        vix = yf.Ticker("^INDIAVIX").history(period="5d")
        if not vix.empty:
            cur = float(vix["Close"].iloc[-1])
            prv = float(vix["Close"].iloc[0])
            return {
                "india_vix": round(cur, 2),
                "vix_5d_ago": round(prv, 2),
                "trend": "rising" if cur > prv else "falling",
            }
    except Exception:
        pass
    return {"india_vix": None}


def _get_usd_inr() -> dict:
    try:
        import yfinance as yf
        h = yf.Ticker("USDINR=X").history(period="5d")
        if not h.empty:
            cur = float(h["Close"].iloc[-1])
            chg = (cur / float(h["Close"].iloc[0]) - 1) * 100
            return {"usd_inr": round(cur, 4), "5d_change_pct": round(chg, 2)}
    except Exception:
        pass
    return {"usd_inr": None}


def _get_market_breadth() -> dict:
    """Use prices already in DB for last two trading days."""
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(text("""
                SELECT ticker, datetime, close FROM price_data
                WHERE interval = '1d'
                AND datetime >= NOW() - INTERVAL '7 days'
                ORDER BY ticker, datetime DESC
            """)).fetchall()
        per_ticker = {}
        for tkr, dt, c in rows:
            per_ticker.setdefault(tkr, []).append(float(c))
        adv = dec = unc = 0
        for tkr, closes in per_ticker.items():
            if len(closes) < 2:
                continue
            r = closes[0] / closes[1] - 1
            if r > 0.001:
                adv += 1
            elif r < -0.001:
                dec += 1
            else:
                unc += 1
        return {
            "advances": adv,
            "declines": dec,
            "unchanged": unc,
            "adv_dec_ratio": round(adv / max(dec, 1), 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_macro_regime() -> dict:
    """NIFTY 200-EMA + India VIX → regime label."""
    try:
        import pandas as pd
        import yfinance as yf
        nifty = yf.Ticker("^NSEI").history(period="1y")
        vix = yf.Ticker("^INDIAVIX").history(period="1d")
        if nifty.empty:
            return {"regime_label": "Unknown"}
        close = nifty["Close"]
        ema_200 = float(close.ewm(span=200).mean().iloc[-1])
        cur = float(close.iloc[-1])
        vix_val = float(vix["Close"].iloc[-1]) if not vix.empty else 15.0

        if cur > ema_200 and vix_val < 15:
            label = "Bull"
        elif cur < ema_200 and vix_val > 20:
            label = "Bear"
        elif vix_val > 20:
            label = "High-Vol"
        else:
            label = "Sideways"

        return {
            "regime_label": label,
            "nifty_current": round(cur, 2),
            "nifty_200ema": round(ema_200, 2),
            "india_vix": round(vix_val, 2),
            "above_200ema": bool(cur > ema_200),
        }
    except Exception as e:
        return {"error": str(e), "regime_label": "Unknown"}


def _rbi_stance() -> str:
    """Heuristic RBI stance from policy proximity (no live MPC scrape)."""
    rbi_dates = [date.fromisoformat(d) for d in cfg.rbi_policy_dates]
    today = date.today()
    upcoming = [d for d in rbi_dates if d >= today]
    if not upcoming:
        return "neutral"
    days = (min(upcoming) - today).days
    return "watch" if days <= 7 else "neutral"


class MacroAnalystAgent(BaseAnalystAgent):
    def __init__(self):
        super().__init__("macro_analyst")

    def _tool_registry(self) -> dict:
        return {
            "get_india_vix": _get_india_vix,
            "get_usd_inr": _get_usd_inr,
            "get_market_breadth": _get_market_breadth,
            "get_macro_regime": _get_macro_regime,
        }

    def analyze(self, context: dict, date_str: str) -> dict:
        regime = _get_macro_regime()
        vix = _get_india_vix()
        fx = _get_usd_inr()
        breadth = _get_market_breadth()
        stance = _rbi_stance()

        return {
            "date": date_str,
            "nifty_regime": regime.get("regime_label", "Unknown"),
            "rbi_stance": stance,
            "fii_flow_7d": None,
            "usd_inr": fx.get("usd_inr"),
            "india_vix": vix.get("india_vix"),
            "market_breadth": breadth if "error" not in breadth else {},
            "regime_label": regime.get("regime_label", "Unknown"),
            "nifty_current": regime.get("nifty_current"),
            "nifty_200ema": regime.get("nifty_200ema"),
            "above_200ema": regime.get("above_200ema"),
        }
