"""Global cross-market context module.

Fetches 25 signals (US, Europe, Asia, FX, rates, commodities) via yfinance,
computes rolling correlations vs Nifty, classifies daily regime, and provides
instrument-level trade levels for the dashboard.

Public API:
    get_global_context() -> GlobalContext   (15-min TTL cached)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    from datetime import timezone
    IST = timezone.utc

# ─── Signal universe ──────────────────────────────────────────────────────────

GROUPS: dict[str, dict[str, str]] = {
    "US": {
        "^GSPC":  "S&P 500",
        "^IXIC":  "Nasdaq",
        "YM=F":   "Dow Futures",
    },
    "Europe": {
        "^GDAXI": "DAX",
        "^FTSE":  "FTSE 100",
    },
    "Asia": {
        "^N225":      "Nikkei 225",
        "^HSI":       "Hang Seng",
        "^KS11":      "KOSPI",
        "^TWII":      "TAIEX",
        "^AXJO":      "ASX 200",
        "000001.SS":  "Shanghai",
        "^CNXIT":     "Nifty IT",
        "^NSEBANK":   "Bank Nifty",
        "^CNXINFRA":  "Nifty Infra",
        "^CNXPHARMA": "Nifty Pharma",
        "^CNXREALTY": "Nifty Realty",
        "^CNXENERGY": "Nifty Energy",
    },
    "FX": {
        "USDINR=X":  "USD/INR",
        "DX-Y.NYB":  "DXY",
        "USDJPY=X":  "USD/JPY",
        "^TNX":      "US 10Y Yield",
        "^VIX":      "CBOE VIX",
        "^INDIAVIX": "India VIX",
    },
    "Commodities": {
        "CL=F": "Crude WTI",
        "BZ=F":  "Brent",
        "GC=F":  "Gold",
        "NG=F":  "Natural Gas",
    },
}

REFERENCE_ONLY = {"^TNX", "^VIX"}

SECTOR_ETF_MAP = {
    "^CNXIT":     "IT",
    "^NSEBANK":   "BANK",
    "^CNXINFRA":  "INFRA",
    "^CNXPHARMA": "PHARMA",
    "^CNXREALTY": "REALTY",
    "^CNXENERGY": "ENERGY",
}

# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SignalRow:
    ticker:    str
    label:     str
    group:     str
    pct_1d:    Optional[float]
    pct_5d:    Optional[float]
    direction: str
    corr_30d:  Optional[float]
    corr_90d:  Optional[float]
    price:     Optional[float]
    atr_5d:    Optional[float]


@dataclass
class GlobalContext:
    fetched_at:      datetime
    regime:          str
    regime_drivers:  list[str]
    signals:         list[SignalRow]
    nifty_bias_text: str
    nifty_pct_1d:    Optional[float] = None
    nifty_pct_5d:    Optional[float] = None
    usdinr:          float = 83.0


# ─── TTL cache (module-level) ──────────────────────────────────────────────────

_CACHE: dict = {"data": None, "fetched_at": None}
TTL_SECONDS = 900  # 15 minutes


def get_global_context() -> GlobalContext:
    """Return cached GlobalContext, refreshing if older than TTL_SECONDS."""
    now = datetime.now(IST)
    if (
        _CACHE["data"] is not None
        and _CACHE["fetched_at"] is not None
        and (now - _CACHE["fetched_at"]).total_seconds() < TTL_SECONDS
    ):
        return _CACHE["data"]
    result = _fetch_all()
    _CACHE["data"] = result
    _CACHE["fetched_at"] = now
    return result


import pandas as pd
import yfinance as yf


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_direction(pct_1d: Optional[float], corr_30d: Optional[float]) -> str:
    if pct_1d is None or corr_30d is None:
        return "neutral"
    product = pct_1d * corr_30d
    if product > 0:
        return "bullish"
    if product < 0:
        return "bearish"
    return "neutral"


def _normalize_index(s: pd.Series) -> pd.Series:
    """Strip tz and time-of-day so series from different yfinance calls align."""
    if s.empty:
        return s
    idx = s.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    try:
        idx = idx.normalize()
    except Exception:
        pass
    out = s.copy()
    out.index = idx
    return out


def _compute_corr(
    signal_returns: pd.Series,
    nifty_returns: pd.Series,
    window: int,
) -> Optional[float]:
    """Pearson correlation of last `window` aligned daily returns."""
    a = _normalize_index(signal_returns)
    b = _normalize_index(nifty_returns)
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < window:
        return None
    tail = aligned.tail(window)
    val = float(tail.iloc[:, 0].corr(tail.iloc[:, 1]))
    return round(val, 3) if not pd.isna(val) else None


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 5) -> Optional[float]:
    """Simple ATR over `period` bars."""
    try:
        trs = []
        for i in range(1, len(close)):
            tr = max(
                float(high.iloc[i]) - float(low.iloc[i]),
                abs(float(high.iloc[i]) - float(close.iloc[i - 1])),
                abs(float(low.iloc[i]) - float(close.iloc[i - 1])),
            )
            trs.append(tr)
        if len(trs) < period:
            return None
        return round(sum(trs[-period:]) / period, 4)
    except Exception:
        return None


def _fetch_nifty_returns() -> tuple[pd.Series, Optional[float], Optional[float]]:
    """Returns (daily_returns, pct_1d, pct_5d) for ^NSEI."""
    df = yf.Ticker("^NSEI").history(period="100d")
    close = df["Close"].dropna()
    returns = close.pct_change().dropna()
    pct_1d = round(float(returns.iloc[-1]) * 100, 3) if len(returns) >= 1 else None
    pct_5d = round((float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100, 3) if len(close) >= 6 else None
    return returns, pct_1d, pct_5d


def _fetch_group(
    ticker_map: dict[str, str],
    group: str,
    nifty_returns: pd.Series,
) -> list[SignalRow]:
    """Download 90d OHLCV for all tickers in a group, compute per-ticker metrics."""
    tickers = list(ticker_map.keys())
    df = yf.download(tickers, period="90d", auto_adjust=True, progress=False, threads=True)
    if df.empty:
        return []

    rows: list[SignalRow] = []
    multi = isinstance(df.columns, pd.MultiIndex)

    for ticker, label in ticker_map.items():
        try:
            if multi:
                close = df["Close"][ticker].dropna()
                high  = df["High"][ticker].dropna()
                low   = df["Low"][ticker].dropna()
            else:
                close = df["Close"].dropna()
                high  = df["High"].dropna()
                low   = df["Low"].dropna()

            if len(close) < 6:
                continue

            returns = close.pct_change().dropna()
            pct_1d  = round(float(returns.iloc[-1]) * 100, 3)
            pct_5d  = round((float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100, 3)
            price   = round(float(close.iloc[-1]), 4)
            atr_5d  = _compute_atr(high, low, close, period=5)

            corr_30d = _compute_corr(returns, nifty_returns, 30)
            corr_90d = _compute_corr(returns, nifty_returns, 90)
            direction = _compute_direction(pct_1d, corr_30d)

            rows.append(SignalRow(
                ticker=ticker, label=label, group=group,
                pct_1d=pct_1d, pct_5d=pct_5d,
                direction=direction,
                corr_30d=corr_30d, corr_90d=corr_90d,
                price=price, atr_5d=atr_5d,
            ))
        except Exception:
            continue

    return rows


def _fetch_usdinr(signals: list[SignalRow]) -> float:
    for s in signals:
        if s.ticker == "USDINR=X" and s.price:
            return s.price
    return 83.0


def _fetch_all() -> GlobalContext:
    """Fetch all groups, classify regime, return GlobalContext."""
    try:
        nifty_returns, nifty_pct_1d, nifty_pct_5d = _fetch_nifty_returns()
    except Exception:
        nifty_returns = pd.Series(dtype=float)
        nifty_pct_1d = nifty_pct_5d = None

    all_signals: list[SignalRow] = []
    for group_name, ticker_map in GROUPS.items():
        try:
            rows = _fetch_group(ticker_map, group_name, nifty_returns)
            all_signals.extend(rows)
        except Exception:
            pass

    regime, drivers = _classify_regime(all_signals)
    bias_text = _bias_text(regime)
    usdinr = _fetch_usdinr(all_signals)

    return GlobalContext(
        fetched_at=datetime.now(IST),
        regime=regime,
        regime_drivers=drivers,
        signals=all_signals,
        nifty_bias_text=bias_text,
        nifty_pct_1d=nifty_pct_1d,
        nifty_pct_5d=nifty_pct_5d,
        usdinr=usdinr,
    )


# ─── Regime classification ────────────────────────────────────────────────────

_BIAS_TEXT = {
    "RISK_OFF": (
        "Global markets bearish overnight. Nifty likely to open weak. "
        "Favour short setups; reduce long position sizes."
    ),
    "RISK_ON": (
        "Global markets bullish overnight. Nifty likely to open strong. "
        "Long setups favoured; standard position sizes apply."
    ),
    "NEUTRAL": (
        "Mixed global signals. No strong directional bias today. "
        "Trade individual setups on their own merit."
    ),
}


def _bias_text(regime: str) -> str:
    return _BIAS_TEXT.get(regime, _BIAS_TEXT["NEUTRAL"])


def _classify_regime(signals: list[SignalRow]) -> tuple[str, list[str]]:
    """
    RISK_OFF: VIX > 22  OR  (S&P < -1.0% AND DXY > +0.3% AND INR > +0.3%)
    RISK_ON:  VIX < 15  AND  S&P > +0.5%  AND  DXY <= 0
    NEUTRAL:  everything else
    """
    by_ticker = {s.ticker: s for s in signals}

    vix_sig = by_ticker.get("^VIX")
    sp_sig  = by_ticker.get("^GSPC")
    dxy_sig = by_ticker.get("DX-Y.NYB")
    inr_sig = by_ticker.get("USDINR=X")

    vix_price = vix_sig.price  if vix_sig else None
    sp_1d     = sp_sig.pct_1d  if sp_sig  else None
    dxy_1d    = dxy_sig.pct_1d if dxy_sig else None
    inr_1d    = inr_sig.pct_1d if inr_sig else None

    if vix_price is not None and vix_price > 22:
        return "RISK_OFF", [f"VIX {vix_price:.1f}"]

    if (
        sp_1d  is not None and sp_1d  < -1.0 and
        dxy_1d is not None and dxy_1d >  0.3 and
        inr_1d is not None and inr_1d >  0.3
    ):
        return "RISK_OFF", [
            f"S&P {sp_1d:+.1f}%",
            f"DXY {dxy_1d:+.1f}%",
            f"INR {inr_1d:+.1f}%",
        ]

    if (
        vix_price is not None and vix_price < 15 and
        sp_1d     is not None and sp_1d     >  0.5 and
        dxy_1d    is not None and dxy_1d    <= 0.0
    ):
        return "RISK_ON", [
            f"VIX {vix_price:.1f}",
            f"S&P {sp_1d:+.1f}%",
            f"DXY {dxy_1d:+.1f}%",
        ]

    drivers = []
    if sp_1d  is not None: drivers.append(f"S&P {sp_1d:+.1f}%")
    if vix_price is not None: drivers.append(f"VIX {vix_price:.1f}")
    if dxy_1d is not None: drivers.append(f"DXY {dxy_1d:+.1f}%")
    return "NEUTRAL", drivers[:3]


# ─── Instrument trade levels ──────────────────────────────────────────────────

_LEVERAGE: dict[str, float] = {
    "US":          10.0,
    "Europe":      10.0,
    "Asia":        10.0,
    "FX":          50.0,
    "Commodities": 16.0,
}
_TICKER_LEVERAGE: dict[str, float] = {
    "GC=F": 25.0,
    "NG=F": 20.0,
}


# ─── Time-ordered signal vector for playbook engine ───────────────────────────

TIME_ORDERED_WEIGHTS: dict[str, float] = {
    "Asia":        0.50,
    "US":          0.30,
    "FX":          0.15,
    "Europe":      0.05,
    "Commodities": 0.10,
}


def time_ordered_signal_vector(ctx: "GlobalContext") -> dict:
    """6-dim signal snapshot consumed by the playbook KNN engine.

    Output keys are stable so they align with the global_signals history
    table for z-score / KNN distance calculations.
    """
    by_ticker = {s.ticker: s for s in ctx.signals}

    def _f(ticker: str, attr: str = "pct_1d") -> Optional[float]:
        s = by_ticker.get(ticker)
        if s is None:
            return None
        return getattr(s, attr, None)

    raw = {
        "sp_pct_1d":     _f("^GSPC"),
        "nikkei_pct_1d": _f("^N225"),
        "dxy_pct_1d":    _f("DX-Y.NYB"),
        "usdinr_pct_1d": _f("USDINR=X"),
        "vix_price":     _f("^VIX", "price"),
        "crude_pct_1d":  _f("CL=F"),
    }

    # Group-weighted directional contribution (signed; positive = bullish for Nifty)
    group_directional = {
        "US":          (raw["sp_pct_1d"]     or 0.0),
        "Asia":        (raw["nikkei_pct_1d"] or 0.0),
        "FX":          -(raw["dxy_pct_1d"]   or 0.0)
                       - (raw["usdinr_pct_1d"] or 0.0),
        "Commodities": -(raw["crude_pct_1d"] or 0.0),
        "Europe":      0.0,
    }
    weighted = {g: round(group_directional[g] * w, 4)
                for g, w in TIME_ORDERED_WEIGHTS.items()
                if g in group_directional}

    raw["group_weighted"] = weighted
    raw["weighted_sum"]   = round(sum(weighted.values()), 4)
    return raw


def instrument_levels(
    sig: SignalRow,
    usdinr: float,
    capital: float,
    risk_pct: float = 0.01,
) -> dict:
    """ATR-based ORB-style trade levels for a global instrument."""
    if sig.ticker in REFERENCE_ONLY:
        return {}
    if sig.price is None or sig.atr_5d is None or sig.atr_5d <= 0:
        return {}

    atr   = sig.atr_5d
    price = sig.price
    buf   = price * 0.001

    is_long = sig.direction == "bullish"

    if is_long:
        entry = round(price + buf, 4)
        stop  = round(price - atr * 0.5, 4)
        t1    = round(entry + atr * 1.5, 4)
        t2    = round(entry + atr * 2.8, 4)
    else:
        entry = round(price - buf, 4)
        stop  = round(price + atr * 0.5, 4)
        t1    = round(entry - atr * 1.5, 4)
        t2    = round(entry - atr * 2.8, 4)

    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return {}

    leverage = _TICKER_LEVERAGE.get(sig.ticker, _LEVERAGE.get(sig.group, 10.0))

    risk_inr = capital * risk_pct
    is_inr = sig.group == "Asia" and sig.ticker.startswith("^CNX")
    entry_inr = entry if is_inr else entry * usdinr
    risk_per_unit_inr = risk_per_unit if is_inr else risk_per_unit * usdinr

    qty = max(1, int(risk_inr / risk_per_unit_inr)) if risk_per_unit_inr > 0 else 1
    margin_inr   = round(qty * entry_inr / leverage, 0)
    max_loss_inr = round(qty * risk_per_unit_inr, 0)
    profit1_inr  = round(qty * abs(t1 - entry) * (1 if is_inr else usdinr), 0)
    rr1 = round(abs(t1 - entry) / risk_per_unit, 2)
    rr2 = round(abs(t2 - entry) / risk_per_unit, 2)

    return {
        "side":         "LONG" if is_long else "SHORT",
        "entry":        entry,
        "stop":         stop,
        "t1":           t1,
        "t2":           t2,
        "rr1":          rr1,
        "rr2":          rr2,
        "qty":          qty,
        "margin_inr":   margin_inr,
        "max_loss_inr": max_loss_inr,
        "profit1_inr":  profit1_inr,
        "currency":     "INR" if is_inr else "USD",
    }
