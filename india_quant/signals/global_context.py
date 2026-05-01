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
    },
    "Commodities": {
        "CL=F": "Crude WTI",
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


def _fetch_all() -> GlobalContext:
    """Stub - implemented in Task 2."""
    raise NotImplementedError("Implemented in Task 2")
