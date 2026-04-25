"""Technical analyst — deterministic, rule-based scoring of NSE/BSE equities."""
import pandas as pd
from loguru import logger

from india_quant.agents.base import BaseAnalystAgent


def _get_price_data(ticker: str, days: int = 90) -> dict:
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(
                text(f"""
                    SELECT datetime, open, high, low, close, volume
                    FROM price_data
                    WHERE ticker = :t AND interval = '1d'
                    AND datetime >= NOW() - INTERVAL '{int(days)} days'
                    ORDER BY datetime DESC LIMIT :n
                """),
                {"t": ticker, "n": days},
            ).fetchall()
        return {"data": [dict(zip(["datetime","open","high","low","close","volume"], r)) for r in rows]}
    except Exception as e:
        return {"error": str(e)}


def _compute_indicators(ticker: str) -> dict:
    """RSI, MACD, ATR, Bollinger Bands, EMAs, volume ratio."""
    try:
        import pandas_ta as ta
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT datetime, open, high, low, close, volume
                    FROM price_data
                    WHERE ticker = :t AND interval = '1d'
                    ORDER BY datetime DESC LIMIT 250
                """),
                {"t": ticker},
            ).fetchall()
        if not rows:
            return {"error": "No data"}

        df = pd.DataFrame(rows, columns=["datetime","open","high","low","close","volume"])
        df = df.sort_values("datetime")
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        rsi_series = ta.rsi(close, length=14)
        rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_val = float(macd_df["MACD_12_26_9"].iloc[-1]) if macd_df is not None and not macd_df.empty else None
        macd_sig_val = float(macd_df["MACDs_12_26_9"].iloc[-1]) if macd_df is not None and not macd_df.empty else None

        atr_series = ta.atr(high, low, close, length=14)
        atr = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.empty else None

        bb = ta.bbands(close, length=20)
        bb_upper = float(bb["BBU_20_2.0"].iloc[-1]) if bb is not None and "BBU_20_2.0" in bb else None
        bb_lower = float(bb["BBL_20_2.0"].iloc[-1]) if bb is not None and "BBL_20_2.0" in bb else None

        ema_20 = float(ta.ema(close, length=20).iloc[-1]) if len(close) >= 20 else None
        ema_50 = float(ta.ema(close, length=50).iloc[-1]) if len(close) >= 50 else None
        ema_200 = float(ta.ema(close, length=200).iloc[-1]) if len(close) >= 200 else None
        vol_ratio = float(volume.iloc[-1] / volume.iloc[-20:].mean()) if len(volume) >= 20 else None

        return {
            "rsi": round(rsi, 2) if rsi is not None else None,
            "macd": round(macd_val, 4) if macd_val is not None else None,
            "macd_signal": round(macd_sig_val, 4) if macd_sig_val is not None else None,
            "atr": round(atr, 2) if atr is not None else None,
            "bb_upper": round(bb_upper, 2) if bb_upper is not None else None,
            "bb_lower": round(bb_lower, 2) if bb_lower is not None else None,
            "ema_20": round(ema_20, 2) if ema_20 is not None else None,
            "ema_50": round(ema_50, 2) if ema_50 is not None else None,
            "ema_200": round(ema_200, 2) if ema_200 is not None else None,
            "volume_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
            "cmp": round(float(close.iloc[-1]), 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_support_resistance(ticker: str) -> dict:
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT high, low, close FROM price_data
                    WHERE ticker = :t AND interval = '1d'
                    ORDER BY datetime DESC LIMIT 60
                """),
                {"t": ticker},
            ).fetchall()
        if not rows:
            return {}
        highs = [float(r[0]) for r in rows]
        lows = [float(r[1]) for r in rows]
        closes = [float(r[2]) for r in rows]
        pivot = (max(highs[:20]) + min(lows[:20]) + closes[0]) / 3
        return {
            "pivot": round(pivot, 2),
            "resistance_1": round(2 * pivot - min(lows[:5]), 2),
            "support_1": round(2 * pivot - max(highs[:5]), 2),
            "52w_high": round(max(highs), 2),
            "52w_low": round(min(lows), 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _check_fo_ban(ticker: str) -> dict:
    """Check NSE F&O ban list. Network failure is non-fatal."""
    import requests
    try:
        resp = requests.get(
            "https://www.nseindia.com/api/fo-ban-list",
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = resp.json()
        ban_list = data.get("data", [])
        clean_ticker = ticker.replace(".NS", "").replace("-EQ", "").upper()
        is_banned = any(clean_ticker in str(item) for item in ban_list)
        return {"in_ban_list": is_banned, "ticker": clean_ticker}
    except Exception:
        return {"in_ban_list": False, "note": "ban list unavailable"}


def _get_circuit_limits(ticker: str) -> dict:
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            row = session.execute(
                text("SELECT close FROM price_data WHERE ticker=:t AND interval='1d' ORDER BY datetime DESC LIMIT 1"),
                {"t": ticker},
            ).fetchone()
        if row:
            prev_close = float(row[0])
            return {
                "lower_circuit": round(prev_close * 0.80, 2),
                "upper_circuit": round(prev_close * 1.20, 2),
                "prev_close": round(prev_close, 2),
            }
    except Exception:
        pass
    return {}


def _classify_trend(ind: dict) -> str:
    cmp_ = ind.get("cmp")
    e20, e50, e200 = ind.get("ema_20"), ind.get("ema_50"), ind.get("ema_200")
    if cmp_ is None:
        return "sideways"
    if e50 and e200 and cmp_ > e50 > e200:
        return "uptrend"
    if e50 and e200 and cmp_ < e50 < e200:
        return "downtrend"
    if e20 and cmp_ > e20:
        return "uptrend"
    if e20 and cmp_ < e20:
        return "downtrend"
    return "sideways"


def _macd_state(ind: dict) -> str:
    macd, sig = ind.get("macd"), ind.get("macd_signal")
    if macd is None or sig is None:
        return "neutral"
    if macd > sig and macd > 0:
        return "bullish_crossover"
    if macd < sig and macd < 0:
        return "bearish_crossover"
    return "neutral"


def _rsi_pattern(rsi: float | None) -> str | None:
    if rsi is None:
        return None
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    if 55 <= rsi < 70:
        return "strong_momentum"
    if 30 < rsi <= 45:
        return "weak_momentum"
    return None


TOOLS = [
    {"name": "compute_indicators"},
    {"name": "get_support_resistance"},
    {"name": "check_fo_ban"},
    {"name": "get_circuit_limits"},
]


class TechnicalAnalystAgent(BaseAnalystAgent):
    def __init__(self):
        super().__init__("technical_analyst", TOOLS)

    def _tool_registry(self) -> dict:
        return {
            "get_price_data": _get_price_data,
            "compute_indicators": _compute_indicators,
            "get_support_resistance": _get_support_resistance,
            "check_fo_ban": _check_fo_ban,
            "get_circuit_limits": _get_circuit_limits,
        }

    def analyze(self, context: dict, date_str: str) -> dict:
        ticker = context.get("ticker", "")
        ind = _compute_indicators(ticker)
        sr = _get_support_resistance(ticker)
        circuit = _get_circuit_limits(ticker)

        trend = _classify_trend(ind)
        macd_signal = _macd_state(ind)
        rsi = ind.get("rsi")
        pattern = _rsi_pattern(rsi)

        score = 0
        if trend == "uptrend":
            score += 3
        elif trend == "downtrend":
            score -= 3
        if macd_signal == "bullish_crossover":
            score += 2
        elif macd_signal == "bearish_crossover":
            score -= 2
        if rsi is not None:
            if 50 <= rsi <= 65:
                score += 1
            elif 35 <= rsi < 50:
                score -= 1
            elif rsi > 70:
                score -= 1
            elif rsi < 30:
                score += 1
        if ind.get("volume_ratio") and ind["volume_ratio"] > 1.5:
            score += 1 if trend == "uptrend" else -1

        confidence = max(1, min(10, 5 + score))

        if trend == "uptrend" and macd_signal == "bullish_crossover":
            outlook = f"Bullish setup: price above EMAs, MACD positive crossover, RSI {rsi}."
        elif trend == "downtrend" and macd_signal == "bearish_crossover":
            outlook = f"Bearish setup: price below EMAs, MACD negative crossover, RSI {rsi}."
        elif trend == "sideways":
            outlook = f"Range-bound; wait for breakout above {sr.get('resistance_1')} or below {sr.get('support_1')}."
        else:
            outlook = f"{trend.capitalize()} bias with mixed momentum (MACD {macd_signal}, RSI {rsi})."

        return {
            "ticker": ticker,
            "date": date_str,
            "trend": trend,
            "rsi": rsi,
            "macd_signal": macd_signal,
            "key_levels": {
                "support": sr.get("support_1"),
                "resistance": sr.get("resistance_1"),
                "pivot": sr.get("pivot"),
                "52w_high": sr.get("52w_high"),
                "52w_low": sr.get("52w_low"),
                "lower_circuit": circuit.get("lower_circuit"),
                "upper_circuit": circuit.get("upper_circuit"),
            },
            "pattern": pattern,
            "outlook": outlook,
            "confidence": confidence,
            "indicators": ind,
        }
