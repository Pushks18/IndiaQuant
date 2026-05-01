# Global Context Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable `GlobalContext` module that fetches 25 global market signals (US, Europe, Asia, FX, rates, commodities), computes rolling correlations vs Nifty, classifies a daily market regime (RISK_ON / RISK_OFF / NEUTRAL), integrates into the screener's scoring, stores signals to DB for ML feature engineering, and powers a new `/global` dashboard page.

**Architecture:** Group-based yfinance fetching (5 groups, each try/except isolated) with a 15-minute in-memory TTL cache. `global_context.py` is pure data — regime classification only. Score multipliers and hard-block logic live in `screener.py`. DB persistence runs as a pre-market pipeline step.

**Tech Stack:** yfinance, pandas, SQLAlchemy (PostgreSQL), Flask/Jinja2, XGBoost/LightGBM (existing), pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `india_quant/signals/global_context.py` | **Create** | Dataclasses, fetch groups, ATR, correlations, regime, trade levels, TTL cache |
| `india_quant/data/models.py` | **Modify** | Add `GlobalSignal` ORM model |
| `india_quant/data/pipeline.py` | **Modify** | Add `fetch_global_signals()` step to `run_pre_market()` |
| `india_quant/data/backfill_global.py` | **Create** | One-time historical backfill for ML training |
| `india_quant/signals/screener.py` | **Modify** | Replace `_get_market_context()`, add `_market_ctx_from_global()`, `_compute_global_delta()`, update `_score()`, add hard block |
| `india_quant/signals/ml_models.py` | **Modify** | Add global signal columns to `prepare_dataset()` and `FEATURE_COLS` |
| `india_quant/dashboard/app.py` | **Modify** | Add `/global` route |
| `india_quant/dashboard/templates/global_context.html` | **Create** | 4-section dashboard page |
| `india_quant/dashboard/templates/base.html` | **Modify** | Add Global nav link |
| `tests/test_integration.py` | **Modify** | Add offline unit tests for new logic |

---

## Task 1: GlobalContext dataclasses + GROUPS constant

**Files:**
- Create: `india_quant/signals/global_context.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_integration.py`:

```python
def test_global_context_imports():
    from india_quant.signals.global_context import (
        GlobalContext, SignalRow, GROUPS, get_global_context
    )
    assert "US" in GROUPS
    assert "Asia" in GROUPS
    assert "FX" in GROUPS
    assert "Commodities" in GROUPS
    assert "Europe" in GROUPS
    # Sector ETFs must be in Asia group for screener compatibility
    assert "^CNXIT" in GROUPS["Asia"]
    assert "^NSEBANK" in GROUPS["Asia"]
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /Users/pushkaraj/Documents/Trading
pytest tests/test_integration.py::test_global_context_imports -v
```

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create `india_quant/signals/global_context.py` with dataclasses and GROUPS**

```python
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

# Tickers shown as reference only on dashboard (no trade levels computed)
REFERENCE_ONLY = {"^TNX", "^VIX"}

# Indian sector ETFs — mapped to screener sector names
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
    direction: str              # bullish | bearish | neutral (re: Nifty)
    corr_30d:  Optional[float]
    corr_90d:  Optional[float]
    price:     Optional[float]
    atr_5d:    Optional[float]  # used for instrument trade level computation


@dataclass
class GlobalContext:
    fetched_at:      datetime
    regime:          str            # RISK_ON | RISK_OFF | NEUTRAL
    regime_drivers:  list[str]
    signals:         list[SignalRow]
    nifty_bias_text: str
    nifty_pct_1d:    Optional[float] = None
    nifty_pct_5d:    Optional[float] = None
    usdinr:          float = 83.0   # live USDINR rate for INR conversion


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
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_integration.py::test_global_context_imports -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/signals/global_context.py tests/test_integration.py
git commit -m "feat: add GlobalContext dataclasses, GROUPS constant, TTL cache skeleton"
```

---

## Task 2: Nifty reference fetch + per-group signal fetch

**Files:**
- Modify: `india_quant/signals/global_context.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_integration.py`:

```python
def test_signal_row_direction_bullish():
    """direction = bullish when pct_1d and corr_30d have same sign."""
    from india_quant.signals.global_context import _compute_direction
    assert _compute_direction(pct_1d=1.2, corr_30d=0.65) == "bullish"

def test_signal_row_direction_bearish():
    from india_quant.signals.global_context import _compute_direction
    assert _compute_direction(pct_1d=-0.8, corr_30d=0.65) == "bearish"

def test_signal_row_direction_neutral_on_none():
    from india_quant.signals.global_context import _compute_direction
    assert _compute_direction(pct_1d=None, corr_30d=0.65) == "neutral"

def test_compute_corr_returns_none_when_insufficient_data():
    import pandas as pd
    from india_quant.signals.global_context import _compute_corr
    short = pd.Series([0.01, -0.02, 0.03])
    nifty = pd.Series([0.01, -0.01, 0.02])
    # window=30 requires 30 points; only 3 provided
    assert _compute_corr(short, nifty, window=30) is None

def test_compute_corr_value():
    import pandas as pd
    import numpy as np
    from india_quant.signals.global_context import _compute_corr
    rng = np.random.default_rng(42)
    n = 50
    a = pd.Series(rng.normal(0, 1, n))
    b = a + pd.Series(rng.normal(0, 0.1, n))  # highly correlated
    result = _compute_corr(a, b, window=30)
    assert result is not None
    assert 0.9 < result <= 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_integration.py::test_signal_row_direction_bullish tests/test_integration.py::test_compute_corr_value -v
```

Expected: `ImportError` — `_compute_direction` and `_compute_corr` not defined yet

- [ ] **Step 3: Add fetch helpers and `_fetch_all()` stub to `global_context.py`**

Add after the `_CACHE` block:

```python
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


def _compute_corr(
    signal_returns: pd.Series,
    nifty_returns: pd.Series,
    window: int,
) -> Optional[float]:
    """Pearson correlation of last `window` aligned daily returns."""
    aligned = pd.concat([signal_returns, nifty_returns], axis=1).dropna()
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
    """
    Returns (daily_returns, pct_1d, pct_5d) for ^NSEI.
    Raises on total failure so caller can fall back.
    """
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
    """Download 90d OHLCV for all tickers in a group, compute all per-ticker metrics."""
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
                # single-ticker download — columns are flat
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
    """Extract live USDINR from already-fetched signals, fallback 83.0."""
    for s in signals:
        if s.ticker == "USDINR=X" and s.price:
            return s.price
    return 83.0


def _fetch_all() -> GlobalContext:
    """Fetch all groups, classify regime, return GlobalContext."""
    # 1. Nifty reference
    try:
        nifty_returns, nifty_pct_1d, nifty_pct_5d = _fetch_nifty_returns()
    except Exception:
        nifty_returns = pd.Series(dtype=float)
        nifty_pct_1d = nifty_pct_5d = None

    # 2. Fetch each group independently
    all_signals: list[SignalRow] = []
    for group_name, ticker_map in GROUPS.items():
        try:
            rows = _fetch_group(ticker_map, group_name, nifty_returns)
            all_signals.extend(rows)
        except Exception:
            pass

    # 3. Classify regime
    regime, drivers = _classify_regime(all_signals)

    # 4. Bias text
    bias_text = _bias_text(regime)

    # 5. USDINR for INR conversion
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
```

Note: `_classify_regime` and `_bias_text` are stubs — added in Task 3.

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/test_integration.py::test_signal_row_direction_bullish tests/test_integration.py::test_signal_row_direction_bearish tests/test_integration.py::test_signal_row_direction_neutral_on_none tests/test_integration.py::test_compute_corr_returns_none_when_insufficient_data tests/test_integration.py::test_compute_corr_value -v
```

Expected: All `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/signals/global_context.py tests/test_integration.py
git commit -m "feat: add group fetch helpers, correlation and direction computation"
```

---

## Task 3: Regime classification + bias text

**Files:**
- Modify: `india_quant/signals/global_context.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing tests**

```python
def _make_signal(ticker: str, pct_1d: float, price: float = 100.0) -> "SignalRow":
    from india_quant.signals.global_context import SignalRow
    return SignalRow(
        ticker=ticker, label=ticker, group="TEST",
        pct_1d=pct_1d, pct_5d=None, direction="neutral",
        corr_30d=None, corr_90d=None, price=price, atr_5d=None,
    )

def test_regime_risk_off_high_vix():
    from india_quant.signals.global_context import _classify_regime
    signals = [
        _make_signal("^VIX",   pct_1d=18.0, price=25.0),   # VIX > 22
        _make_signal("^GSPC",  pct_1d=-1.5),
        _make_signal("DX-Y.NYB", pct_1d=0.5),
        _make_signal("USDINR=X", pct_1d=0.4),
    ]
    regime, drivers = _classify_regime(signals)
    assert regime == "RISK_OFF"
    assert any("VIX" in d for d in drivers)

def test_regime_risk_on():
    from india_quant.signals.global_context import _classify_regime
    signals = [
        _make_signal("^VIX",    pct_1d=-5.0, price=12.0),  # VIX < 15
        _make_signal("^GSPC",   pct_1d=0.8),                # S&P > 0.5%
        _make_signal("DX-Y.NYB", pct_1d=-0.2),              # DXY falling
    ]
    regime, drivers = _classify_regime(signals)
    assert regime == "RISK_ON"

def test_regime_neutral_when_mixed():
    from india_quant.signals.global_context import _classify_regime
    signals = [
        _make_signal("^VIX",    pct_1d=0.0, price=17.0),
        _make_signal("^GSPC",   pct_1d=0.2),
        _make_signal("DX-Y.NYB", pct_1d=0.1),
    ]
    regime, _ = _classify_regime(signals)
    assert regime == "NEUTRAL"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_integration.py::test_regime_risk_off_high_vix tests/test_integration.py::test_regime_risk_on tests/test_integration.py::test_regime_neutral_when_mixed -v
```

Expected: `NameError` — `_classify_regime` not defined

- [ ] **Step 3: Add `_classify_regime` and `_bias_text` to `global_context.py`**

Add after `_fetch_all`:

```python
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
    Returns (regime, drivers).
    RISK_OFF: VIX > 22  OR  (S&P < -1.0% AND DXY > +0.3% AND INR > +0.3%)
    RISK_ON:  VIX < 15  AND  S&P > +0.5%  AND  DXY <= 0
    NEUTRAL:  everything else
    """
    by_ticker = {s.ticker: s for s in signals}

    vix_sig   = by_ticker.get("^VIX")
    sp_sig    = by_ticker.get("^GSPC")
    dxy_sig   = by_ticker.get("DX-Y.NYB")
    inr_sig   = by_ticker.get("USDINR=X")

    vix_price = vix_sig.price  if vix_sig  else None
    sp_1d     = sp_sig.pct_1d  if sp_sig   else None
    dxy_1d    = dxy_sig.pct_1d if dxy_sig  else None
    inr_1d    = inr_sig.pct_1d if inr_sig  else None

    # ── RISK_OFF ──────────────────────────────────────────────────────────────
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

    # ── RISK_ON ───────────────────────────────────────────────────────────────
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

    # ── NEUTRAL ───────────────────────────────────────────────────────────────
    drivers = []
    if sp_1d  is not None: drivers.append(f"S&P {sp_1d:+.1f}%")
    if vix_price is not None: drivers.append(f"VIX {vix_price:.1f}")
    if dxy_1d is not None: drivers.append(f"DXY {dxy_1d:+.1f}%")
    return "NEUTRAL", drivers[:3]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_integration.py::test_regime_risk_off_high_vix tests/test_integration.py::test_regime_risk_on tests/test_integration.py::test_regime_neutral_when_mixed -v
```

Expected: All `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/signals/global_context.py tests/test_integration.py
git commit -m "feat: add regime classification (RISK_ON/RISK_OFF/NEUTRAL) and bias text"
```

---

## Task 4: Instrument trade level computation

**Files:**
- Modify: `india_quant/signals/global_context.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_instrument_levels_long():
    from india_quant.signals.global_context import SignalRow, instrument_levels
    sig = SignalRow(
        ticker="^GSPC", label="S&P 500", group="US",
        pct_1d=1.0, pct_5d=2.0, direction="bullish",
        corr_30d=0.7, corr_90d=0.65,
        price=5800.0, atr_5d=60.0,
    )
    levels = instrument_levels(sig, usdinr=83.0, capital=200_000)
    assert levels["side"] == "LONG"
    assert levels["entry"] > sig.price          # entry above last price
    assert levels["stop"]  < levels["entry"]    # stop below entry
    assert levels["t1"]    > levels["entry"]    # T1 above entry
    assert levels["t2"]    > levels["t1"]       # T2 above T1
    assert levels["rr1"]   > 1.0               # R:R > 1
    assert levels["margin_inr"] > 0
    assert levels["max_loss_inr"] > 0

def test_instrument_levels_short():
    from india_quant.signals.global_context import SignalRow, instrument_levels
    sig = SignalRow(
        ticker="CL=F", label="Crude WTI", group="Commodities",
        pct_1d=-1.5, pct_5d=-2.0, direction="bearish",
        corr_30d=-0.3, corr_90d=-0.28,
        price=78.0, atr_5d=1.5,
    )
    levels = instrument_levels(sig, usdinr=83.0, capital=200_000)
    assert levels["side"] == "SHORT"
    assert levels["entry"] < sig.price
    assert levels["stop"]  > levels["entry"]
    assert levels["t1"]    < levels["entry"]

def test_instrument_levels_none_on_missing_atr():
    from india_quant.signals.global_context import SignalRow, instrument_levels
    sig = SignalRow(
        ticker="^GSPC", label="S&P 500", group="US",
        pct_1d=0.5, pct_5d=1.0, direction="bullish",
        corr_30d=0.7, corr_90d=0.65,
        price=5800.0, atr_5d=None,   # missing ATR
    )
    assert instrument_levels(sig, usdinr=83.0, capital=200_000) == {}
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_integration.py::test_instrument_levels_long tests/test_integration.py::test_instrument_levels_short tests/test_integration.py::test_instrument_levels_none_on_missing_atr -v
```

Expected: `ImportError`

- [ ] **Step 3: Add `instrument_levels()` to `global_context.py`**

```python
# Leverage by asset class (for margin estimation)
_LEVERAGE: dict[str, float] = {
    "US":          10.0,
    "Europe":      10.0,
    "Asia":        10.0,
    "FX":          50.0,
    "Commodities": 16.0,  # crude default; overridden below per ticker
}
_TICKER_LEVERAGE: dict[str, float] = {
    "GC=F": 25.0,   # gold MCX
    "NG=F": 20.0,   # natural gas MCX
}


def instrument_levels(
    sig: SignalRow,
    usdinr: float,
    capital: float,
    risk_pct: float = 0.01,
) -> dict:
    """
    Compute ORB-style trade levels for a global instrument.
    Returns empty dict if price or ATR is missing, or for reference-only tickers.
    Margin is always returned in INR (non-INR prices multiplied by usdinr).
    """
    if sig.ticker in REFERENCE_ONLY:
        return {}
    if sig.price is None or sig.atr_5d is None or sig.atr_5d <= 0:
        return {}

    atr   = sig.atr_5d
    price = sig.price
    buf   = price * 0.001   # 0.1% buffer

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

    # Position size: risk capital / risk per unit
    risk_inr = capital * risk_pct
    # Convert entry to INR for non-INR instruments (rough: all non-India prices in USD)
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
        "side":        "LONG" if is_long else "SHORT",
        "entry":       entry,
        "stop":        stop,
        "t1":          t1,
        "t2":          t2,
        "rr1":         rr1,
        "rr2":         rr2,
        "qty":         qty,
        "margin_inr":  margin_inr,
        "max_loss_inr":  max_loss_inr,
        "profit1_inr": profit1_inr,
        "currency":    "INR" if is_inr else "USD",
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_integration.py::test_instrument_levels_long tests/test_integration.py::test_instrument_levels_short tests/test_integration.py::test_instrument_levels_none_on_missing_atr -v
```

Expected: All `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/signals/global_context.py tests/test_integration.py
git commit -m "feat: add instrument_levels() for global trade level computation"
```

---

## Task 5: GlobalSignal ORM model

**Files:**
- Modify: `india_quant/data/models.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_global_signal_model_imports():
    from india_quant.data.models import GlobalSignal
    import inspect, sqlalchemy
    # Must have the expected columns
    cols = {c.name for c in GlobalSignal.__table__.columns}
    for expected in ("id", "date", "ticker", "label", "group",
                     "pct_1d", "pct_5d", "corr_30d", "corr_90d", "regime"):
        assert expected in cols, f"Missing column: {expected}"
    # Must have unique constraint on (date, ticker)
    constraints = {c.name for c in GlobalSignal.__table__.constraints}
    assert any("global_signal" in (n or "") for n in constraints)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_integration.py::test_global_signal_model_imports -v
```

Expected: `ImportError`

- [ ] **Step 3: Add `GlobalSignal` to `india_quant/data/models.py`**

Add at the end of `models.py`:

```python
class GlobalSignal(Base):
    __tablename__ = "global_signals"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date       = Column(Date, nullable=False)
    ticker     = Column(String(20), nullable=False)
    label      = Column(String(50))
    group      = Column(String(20))
    pct_1d     = Column(Float)
    pct_5d     = Column(Float)
    corr_30d   = Column(Float)
    corr_90d   = Column(Float)
    regime     = Column(String(10))   # denormalised: same value for all rows on a date
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_global_signal_date_ticker"),
        Index("ix_global_signal_date", "date"),
    )
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_integration.py::test_global_signal_model_imports -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/data/models.py tests/test_integration.py
git commit -m "feat: add GlobalSignal ORM model"
```

---

## Task 6: Pipeline integration

**Files:**
- Modify: `india_quant/data/pipeline.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pipeline_has_fetch_global_signals():
    """fetch_global_signals must exist as a static method on DataPipeline."""
    from india_quant.data.pipeline import DataPipeline
    assert hasattr(DataPipeline, "fetch_global_signals")
    import inspect
    sig = inspect.signature(DataPipeline.fetch_global_signals)
    # Must accept optional trade_date param
    assert "trade_date" in sig.parameters
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_integration.py::test_pipeline_has_fetch_global_signals -v
```

Expected: `AssertionError`

- [ ] **Step 3: Add `fetch_global_signals()` to `pipeline.py` and call it from `run_pre_market()`**

Add the static method to `DataPipeline`:

```python
@staticmethod
def fetch_global_signals(trade_date: str = None):
    """Fetch global context and upsert all 25 signal rows to global_signals table."""
    from datetime import date as date_cls
    from india_quant.signals.global_context import get_global_context
    from india_quant.data.models import GlobalSignal
    from india_quant.data.db import get_session
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    if trade_date is None:
        trade_date = date_cls.today().isoformat()

    ctx = get_global_context()
    trade_date_obj = date_cls.fromisoformat(trade_date)

    with get_session() as session:
        for sig in ctx.signals:
            stmt = pg_insert(GlobalSignal).values(
                date=trade_date_obj,
                ticker=sig.ticker,
                label=sig.label,
                group=sig.group,
                pct_1d=sig.pct_1d,
                pct_5d=sig.pct_5d,
                corr_30d=sig.corr_30d,
                corr_90d=sig.corr_90d,
                regime=ctx.regime,
            ).on_conflict_do_update(
                index_elements=["date", "ticker"],
                set_={
                    "pct_1d":   sig.pct_1d,
                    "pct_5d":   sig.pct_5d,
                    "corr_30d": sig.corr_30d,
                    "corr_90d": sig.corr_90d,
                    "regime":   ctx.regime,
                },
            )
            session.execute(stmt)
    logger.info(f"[Pipeline] GlobalSignals: {len(ctx.signals)} rows upserted for {trade_date}")
    return len(ctx.signals)
```

Then add this block at the end of `run_pre_market()`, before the final logger line:

```python
        # 4. Fetch and store global cross-market signals
        try:
            n = DataPipeline.fetch_global_signals(run_date)
            logger.info(f"[Pipeline] GlobalSignals: {n} rows updated")
        except Exception as e:
            logger.error(f"[Pipeline] GlobalSignals fetch failed: {e}")
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_integration.py::test_pipeline_has_fetch_global_signals -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/data/pipeline.py tests/test_integration.py
git commit -m "feat: add fetch_global_signals() pipeline step"
```

---

## Task 7: Backfill script

**Files:**
- Create: `india_quant/data/backfill_global.py`

No offline unit test — requires live yfinance + DB. Manual verification step included.

- [ ] **Step 1: Create the backfill script**

```python
"""
One-time backfill of global_signals table.
Run once before the first ML retrain so ReturnPredictor has historical features.

Usage:
    python -m india_quant.data.backfill_global            # 365 days
    python -m india_quant.data.backfill_global --days 730 # 2 years
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta, datetime
import pandas as pd
import yfinance as yf
from loguru import logger

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    from datetime import timezone
    IST = timezone.utc

ALL_TICKERS = [
    t for group in {
        "US":          ["^GSPC", "^IXIC", "YM=F"],
        "Europe":      ["^GDAXI", "^FTSE"],
        "Asia":        ["^N225", "^HSI", "^KS11", "^TWII", "^AXJO", "000001.SS",
                        "^CNXIT", "^NSEBANK", "^CNXINFRA", "^CNXPHARMA", "^CNXREALTY", "^CNXENERGY"],
        "FX":          ["USDINR=X", "DX-Y.NYB", "USDJPY=X", "^TNX", "^VIX"],
        "Commodities": ["CL=F", "GC=F", "NG=F"],
    }.values()
    for t in group
]

TICKER_GROUP = {}
for g, tickers in {
    "US":          ["^GSPC", "^IXIC", "YM=F"],
    "Europe":      ["^GDAXI", "^FTSE"],
    "Asia":        ["^N225", "^HSI", "^KS11", "^TWII", "^AXJO", "000001.SS",
                    "^CNXIT", "^NSEBANK", "^CNXINFRA", "^CNXPHARMA", "^CNXREALTY", "^CNXENERGY"],
    "FX":          ["USDINR=X", "DX-Y.NYB", "USDJPY=X", "^TNX", "^VIX"],
    "Commodities": ["CL=F", "GC=F", "NG=F"],
}.items():
    for t in tickers:
        TICKER_GROUP[t] = g

TICKER_LABEL = {}
from india_quant.signals.global_context import GROUPS as _GROUPS
for g_tickers in _GROUPS.values():
    TICKER_LABEL.update(g_tickers)


def _compute_corr_series(returns: pd.Series, nifty_ret: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation of returns vs nifty_ret."""
    combined = pd.concat([returns, nifty_ret], axis=1).dropna()
    if len(combined) < window:
        return pd.Series(dtype=float)
    r1 = combined.iloc[:, 0].rolling(window)
    r2 = combined.iloc[:, 1].rolling(window)
    return r1.corr(r2)


def backfill(days: int = 365):
    from india_quant.data.models import GlobalSignal
    from india_quant.data.db import get_session
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    period = f"{days + 30}d"   # extra buffer for rolling windows
    logger.info(f"Downloading {len(ALL_TICKERS)} tickers, period={period} ...")

    # Nifty reference
    nifty_close = yf.download("^NSEI", period=period, auto_adjust=True, progress=False)["Close"]
    nifty_ret   = nifty_close.pct_change().dropna()

    # All signals in one batch
    df = yf.download(ALL_TICKERS, period=period, auto_adjust=True, progress=True, threads=True)
    if df.empty:
        logger.error("Download returned empty dataframe")
        return

    close_df = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]

    inserted = 0
    with get_session() as session:
        for ticker in ALL_TICKERS:
            try:
                close = close_df[ticker].dropna() if isinstance(close_df.columns, pd.Index) and ticker in close_df.columns else None
                if close is None or len(close) < 32:
                    continue

                returns = close.pct_change().dropna()
                corr30  = _compute_corr_series(returns, nifty_ret, 30)
                corr90  = _compute_corr_series(returns, nifty_ret, 90)

                for dt in returns.index[-days:]:
                    trade_date = dt.date() if hasattr(dt, "date") else dt
                    pct_1d = round(float(returns.loc[dt]) * 100, 3)
                    # pct_5d: 5-day return ending at dt
                    pos = list(close.index).index(dt)
                    pct_5d = None
                    if pos >= 5:
                        prev5 = float(close.iloc[pos - 5])
                        pct_5d = round((float(close.loc[dt]) / prev5 - 1) * 100, 3) if prev5 else None

                    c30 = float(corr30.loc[dt]) if dt in corr30.index and not pd.isna(corr30.loc[dt]) else None
                    c90 = float(corr90.loc[dt]) if dt in corr90.index and not pd.isna(corr90.loc[dt]) else None

                    stmt = pg_insert(GlobalSignal).values(
                        date=trade_date,
                        ticker=ticker,
                        label=TICKER_LABEL.get(ticker, ticker),
                        group=TICKER_GROUP.get(ticker, "OTHER"),
                        pct_1d=pct_1d,
                        pct_5d=pct_5d,
                        corr_30d=round(c30, 3) if c30 is not None else None,
                        corr_90d=round(c90, 3) if c90 is not None else None,
                        regime="NEUTRAL",  # regime backfill not required for ML features
                    ).on_conflict_do_nothing()
                    session.execute(stmt)
                    inserted += 1

            except Exception as e:
                logger.warning(f"Skipping {ticker}: {e}")
                continue

    logger.info(f"Backfill complete: {inserted} rows inserted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    backfill(days=args.days)
```

- [ ] **Step 2: Verify script is importable**

```bash
python -c "from india_quant.data.backfill_global import backfill; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add india_quant/data/backfill_global.py
git commit -m "feat: add global signal historical backfill script"
```

---

## Task 8: Screener — replace `_get_market_context()` and add global delta

**Files:**
- Modify: `india_quant/signals/screener.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_global_delta_positive():
    """Bullish global signals produce positive delta."""
    from india_quant.signals.global_context import SignalRow
    from india_quant.signals.screener import _compute_global_delta

    def _s(ticker, pct_1d, price=100.0):
        return SignalRow(
            ticker=ticker, label=ticker, group="TEST",
            pct_1d=pct_1d, pct_5d=None, direction="neutral",
            corr_30d=None, corr_90d=None, price=price, atr_5d=None,
        )

    signals = [
        _s("^GSPC",    0.8),    # +6 (S&P > 0.5%)
        _s("^N225",    0.6),    # +4 (Nikkei > 0.5%)
        _s("USDINR=X", -0.3),   # +3 (INR strengthening)
        _s("DX-Y.NYB", -0.1),   # DXY neutral (< 0.3%)
        _s("DX-Y.NYB", -0.1),   # no negative trigger
    ]
    delta = _compute_global_delta(signals)
    assert delta > 0
    assert delta <= 10  # capped

def test_compute_global_delta_negative():
    from india_quant.signals.global_context import SignalRow
    from india_quant.signals.screener import _compute_global_delta

    def _s(ticker, pct_1d, price=100.0):
        return SignalRow(
            ticker=ticker, label=ticker, group="TEST",
            pct_1d=pct_1d, pct_5d=None, direction="neutral",
            corr_30d=None, corr_90d=None, price=price, atr_5d=None,
        )

    signals = [
        _s("^GSPC",    -0.8),   # -6 (S&P < -0.5%)
        _s("DX-Y.NYB",  0.5),   # -4 (DXY > 0.3%)
        _s("CL=F",      2.5),   # -3 (crude > 2%)
    ]
    delta = _compute_global_delta(signals)
    assert delta < 0
    assert delta >= -10  # capped

def test_compute_global_delta_capped_at_ten():
    from india_quant.signals.global_context import SignalRow
    from india_quant.signals.screener import _compute_global_delta

    def _s(ticker, pct_1d, price=100.0):
        return SignalRow(
            ticker=ticker, label=ticker, group="TEST",
            pct_1d=pct_1d, pct_5d=None, direction="neutral",
            corr_30d=None, corr_90d=None, price=price, atr_5d=None,
        )

    # Everything screaming bullish — still capped at 10
    signals = [
        _s("^GSPC",    2.0),    # +6
        _s("^N225",    1.5),    # +4
        _s("USDINR=X", -0.5),   # +3
    ]
    assert _compute_global_delta(signals) == 10
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_integration.py::test_compute_global_delta_positive tests/test_integration.py::test_compute_global_delta_negative tests/test_integration.py::test_compute_global_delta_capped_at_ten -v
```

Expected: `ImportError`

- [ ] **Step 3: Add helpers to `screener.py`**

Add after the `_get_market_context` function (which you will replace):

```python
def _market_ctx_from_global(gctx: "GlobalContext") -> dict:
    """
    Convert GlobalContext to the ctx dict shape expected by _build_plan().
    Extracts VIX, Nifty changes, and sector ETF returns from the signal list.
    """
    from india_quant.signals.global_context import SECTOR_ETF_MAP
    by_ticker = {s.ticker: s for s in gctx.signals}

    vix_sig = by_ticker.get("^VIX")
    vix     = vix_sig.price if vix_sig else None

    sector_returns: dict[str, float] = {}
    for etf_ticker, sector_name in SECTOR_ETF_MAP.items():
        sig = by_ticker.get(etf_ticker)
        if sig and sig.pct_1d is not None:
            sector_returns[sector_name] = sig.pct_1d

    return {
        "vix":                vix,
        "nifty_prev_chg_pct": gctx.nifty_pct_1d or 0.0,
        "nifty_5d_return":    gctx.nifty_pct_5d or 0.0,
        "sector_returns":     sector_returns,
    }


def _compute_global_delta(signals: list) -> int:
    """
    Additive score delta from global signals. Range: [-10, +10].
    Positive = bullish for Nifty, negative = bearish.
    """
    from india_quant.signals.global_context import SignalRow
    by_ticker = {s.ticker: s for s in signals}

    sp_1d   = (by_ticker["^GSPC"].pct_1d    if "^GSPC"    in by_ticker else None)
    nk_1d   = (by_ticker["^N225"].pct_1d    if "^N225"    in by_ticker else None)
    inr_1d  = (by_ticker["USDINR=X"].pct_1d if "USDINR=X" in by_ticker else None)
    dxy_1d  = (by_ticker["DX-Y.NYB"].pct_1d if "DX-Y.NYB" in by_ticker else None)
    cl_1d   = (by_ticker["CL=F"].pct_1d     if "CL=F"     in by_ticker else None)
    vix_p   = (by_ticker["^VIX"].price      if "^VIX"     in by_ticker else None)

    delta = 0

    # Bullish signals
    if sp_1d is not None and sp_1d > 0.5 and (dxy_1d is None or dxy_1d < 0):
        delta += 6
    if nk_1d is not None and nk_1d > 0.5:
        delta += 4
    if inr_1d is not None and inr_1d < -0.2:
        delta += 3

    # Bearish signals
    if (sp_1d is not None and sp_1d < -0.5) or (vix_p is not None and vix_p > 20):
        delta -= 6
    if dxy_1d is not None and dxy_1d > 0.3:
        delta -= 4
    if cl_1d is not None and cl_1d > 2.0:
        delta -= 3

    return max(-10, min(10, delta))
```

Then replace the old `_get_market_context()` call inside `run_screener()`:

Find this block (around line where `_get_market_context()` is called):
```python
ctx = _get_market_context()
```

Replace with:
```python
    from india_quant.signals.global_context import get_global_context
    try:
        _gctx       = get_global_context()
        ctx         = _market_ctx_from_global(_gctx)
        global_delta = _compute_global_delta(_gctx.signals)
        regime       = _gctx.regime
    except Exception:
        ctx          = _get_market_context()   # fallback to original
        global_delta = 0
        regime       = "NEUTRAL"
```

Also add the hard-block check right after regime is set (before the stock loop):

```python
    # Hard block on extreme RISK_OFF
    if (
        regime == "RISK_OFF"
        and (ctx.get("vix") or 0) > 25
        and (ctx.get("nifty_prev_chg_pct") or 0) < -1.5
    ):
        msg = f"Global RISK_OFF extreme — VIX {ctx.get('vix'):.1f}, skip all trades"
        print(_col(f"  ⛔  {msg}", "RED"))
        return [{"skip_day": True, "reason": msg, "vix": ctx.get("vix")}]
```

Then pass `global_delta` and `regime` into `_build_plan()`:

In `_build_plan()` signature, add:
```python
    global_delta: int = 0,
    regime:       str = "NEUTRAL",
```

And pass them through to `_score()`:
```python
    sl, ss = _score(
        nifty_chg, stk, adx, pdi, ndi, rsi_val, macd_h,
        rs, pd_sig, sect_mom, mom5, w52hp, w52lp,
        vol_surge, vwap, prev_close, regime,
        global_delta=global_delta,   # ← new
    )
```

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/test_integration.py::test_compute_global_delta_positive tests/test_integration.py::test_compute_global_delta_negative tests/test_integration.py::test_compute_global_delta_capped_at_ten -v
```

Expected: All `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/signals/screener.py tests/test_integration.py
git commit -m "feat: replace _get_market_context with global context, add _compute_global_delta"
```

---

## Task 9: Screener — `_score()` regime gate + global delta

**Files:**
- Modify: `india_quant/signals/screener.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_score_risk_off_halves_long_score():
    """RISK_OFF regime should roughly halve the long score."""
    from india_quant.signals.screener import _score

    base_kwargs = dict(
        nifty_chg=0.5, ema_stack="bullish", adx=30.0,
        plus_di=25.0, minus_di=15.0, rsi=62.0, macd_hist=0.5,
        rs_vs_nifty=2.0, prev_day_sig="above_high", sect_mom=1.5,
        mom_5d=3.5, w52h_pct=-2.0, w52l_pct=20.0,
        vol_surge=1.8, vwap=None, prev_close=1000.0,
        regime="NEUTRAL", global_delta=0,
    )
    sl_neutral, _ = _score(**base_kwargs)

    risk_off_kwargs = {**base_kwargs, "regime": "RISK_OFF"}
    sl_risk_off, _ = _score(**risk_off_kwargs)

    assert sl_risk_off < sl_neutral * 0.7   # at least 30% reduction

def test_score_global_delta_applied_directionally():
    """Positive global_delta raises LONG score and lowers SHORT score."""
    from india_quant.signals.screener import _score

    base = dict(
        nifty_chg=0.0, ema_stack="mixed", adx=None,
        plus_di=None, minus_di=None, rsi=50.0, macd_hist=None,
        rs_vs_nifty=0.0, prev_day_sig="inside", sect_mom=None,
        mom_5d=None, w52h_pct=None, w52l_pct=None,
        vol_surge=1.0, vwap=None, prev_close=1000.0,
        regime="NEUTRAL",
    )
    sl_0, ss_0 = _score(**{**base, "global_delta": 0})
    sl_p, ss_p = _score(**{**base, "global_delta": 8})

    assert sl_p > sl_0
    assert ss_p < ss_0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_integration.py::test_score_risk_off_halves_long_score tests/test_integration.py::test_score_global_delta_applied_directionally -v
```

Expected: `TypeError` — `_score()` doesn't accept `regime` or `global_delta` yet

- [ ] **Step 3: Update `_score()` signature and add two new layers**

In `screener.py`, modify `_score()`:

1. Add `global_delta: int = 0` to the parameter list (it already has `regime: str`).

2. At the end of `_score()`, before the `return` statement, add:

```python
    # ── Layer 1: Regime multiplier (extreme events) ─────────────────────
    if regime == "RISK_OFF":
        sl = round(sl * 0.5, 1)
        ss = round(ss * 1.2, 1)
    elif regime == "RISK_ON":
        sl = round(sl * 1.1, 1)
        ss = round(ss * 0.8, 1)

    # ── Layer 2: Global delta (additive, directional) ────────────────────
    sl = round(min(100, max(0, sl + global_delta)), 1)
    ss = round(min(100, max(0, ss - global_delta)), 1)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_integration.py::test_score_risk_off_halves_long_score tests/test_integration.py::test_score_global_delta_applied_directionally -v
```

Expected: All `PASSED`

- [ ] **Step 5: Run all existing offline tests to check for regressions**

```bash
pytest tests/test_integration.py -k "not angel and not finbert and not timescale and not ml_train and not walkforward and not hypertable" -v
```

Expected: All `PASSED`

- [ ] **Step 6: Commit**

```bash
git add india_quant/signals/screener.py tests/test_integration.py
git commit -m "feat: add regime gate and global delta layers to _score()"
```

---

## Task 10: ML feature integration

**Files:**
- Modify: `india_quant/signals/ml_models.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ml_feature_cols_include_global():
    """FEATURE_COLS must include at least one global signal column."""
    from india_quant.signals.ml_models import ReturnPredictor
    rp = ReturnPredictor()
    global_cols = [c for c in rp.FEATURE_COLS if c.startswith("^") or c.startswith("CL") or c == "global_regime"]
    assert len(global_cols) > 0, "No global signal columns found in FEATURE_COLS"
    assert "global_regime" in rp.FEATURE_COLS
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_integration.py::test_ml_feature_cols_include_global -v
```

Expected: `AssertionError`

- [ ] **Step 3: Update `ml_models.py`**

1. Extend `FEATURE_COLS` in `ReturnPredictor`:

```python
    FEATURE_COLS = [
        # existing factor features
        "momentum_12_1", "momentum_1", "momentum_3",
        "realized_vol", "vol_of_vol", "idiosyncratic_vol",
        "liquidity_amihud", "turnover",
        "iv_skew", "iv_spread", "vrp", "oi_flow",
        "profitability_roe", "gross_profitability",
        "value_bm", "earnings_yield",
        # global context features (joined from global_signals table)
        "global_regime",
        "^GSPC_pct_1d",  "^GSPC_pct_5d",  "^GSPC_corr_30d",
        "^IXIC_pct_1d",  "^IXIC_pct_5d",
        "^N225_pct_1d",  "^N225_pct_5d",  "^N225_corr_30d",
        "^HSI_pct_1d",   "^HSI_corr_30d",
        "USDINR=X_pct_1d", "DX-Y.NYB_pct_1d",
        "^VIX_pct_1d",   "^VIX_price",
        "CL=F_pct_1d",   "CL=F_pct_5d",
        "GC=F_pct_1d",
    ]
```

2. In `prepare_dataset()`, add the global signal join after the `labels` query and before the `merged = factors.merge(...)` line:

```python
            # Global signals pivot (one row per date, columns = ticker_metric)
            global_raw = pd.DataFrame(
                session.execute(
                    text("""
                        SELECT date,
                               ticker,
                               pct_1d,
                               pct_5d,
                               corr_30d,
                               corr_90d,
                               regime
                        FROM global_signals
                        WHERE date BETWEEN :start AND :end
                    """),
                    {"start": start_date, "end": end_date},
                ).fetchall()
            )

            if not global_raw.empty:
                global_raw.columns = ["date", "ticker", "pct_1d", "pct_5d", "corr_30d", "corr_90d", "regime"]
                # Pivot: one row per date, columns like "^GSPC_pct_1d"
                pivot_vals = global_raw.pivot(index="date", columns="ticker",
                                              values=["pct_1d", "pct_5d", "corr_30d", "corr_90d"])
                pivot_vals.columns = [f"{ticker}_{metric}" for metric, ticker in pivot_vals.columns]
                pivot_vals = pivot_vals.reset_index()

                # Special: VIX price (store as pct_1d proxy for level)
                vix_rows = global_raw[global_raw["ticker"] == "^VIX"][["date", "pct_1d"]].copy()
                vix_rows = vix_rows.rename(columns={"pct_1d": "^VIX_price"})

                # Regime ordinal (one per date)
                regime_map = {"RISK_OFF": 0, "NEUTRAL": 1, "RISK_ON": 2}
                regime_df = (
                    global_raw.groupby("date")["regime"]
                    .first()
                    .map(regime_map)
                    .reset_index()
                    .rename(columns={"regime": "global_regime"})
                )

                factors = factors.merge(pivot_vals,   on="date", how="left")
                factors = factors.merge(regime_df,    on="date", how="left")
                factors = factors.merge(vix_rows,     on="date", how="left")
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_integration.py::test_ml_feature_cols_include_global -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/signals/ml_models.py tests/test_integration.py
git commit -m "feat: add global signal columns to ReturnPredictor feature set"
```

---

## Task 11: Dashboard `/global` route

**Files:**
- Modify: `india_quant/dashboard/app.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_global_route_exists():
    """The /global route must be registered on the Flask app."""
    from india_quant.dashboard.app import create_app
    app = create_app()
    rules = [r.rule for r in app.url_map.iter_rules()]
    assert "/global" in rules
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_integration.py::test_global_route_exists -v
```

Expected: `AssertionError`

- [ ] **Step 3: Add `/global` route to `app.py`**

Add inside `create_app()`, after the `/live_accuracy` route:

```python
    @app.route("/global")
    def global_context_page():
        from india_quant.signals.global_context import get_global_context, instrument_levels
        from india_quant.signals.screener import run_screener

        try:    capital  = float(request.args.get("capital", 200_000))
        except ValueError: capital = 200_000
        try:    risk_pct = float(request.args.get("risk", 1.0)) / 100.0
        except ValueError: risk_pct = 0.01
        try:    top_n    = int(request.args.get("top", 10))
        except ValueError: top_n = 10

        ctx = get_global_context()

        # Compute instrument trade levels for each signal
        signal_levels = {}
        for sig in ctx.signals:
            lvl = instrument_levels(sig, usdinr=ctx.usdinr, capital=capital, risk_pct=risk_pct)
            if lvl:
                signal_levels[sig.ticker] = lvl

        # Screener output — filtered and annotated with global alignment
        plans = run_screener(capital_inr=capital, risk_per_trade_pct=risk_pct, top_n=top_n)
        actionable = []
        for p in plans:
            if p.get("bias") not in ("LONG", "SHORT"):
                continue
            # Global alignment: does the stock bias match the day's regime?
            if ctx.regime == "RISK_ON"  and p["bias"] == "LONG":  p["global_aligned"] = True
            elif ctx.regime == "RISK_OFF" and p["bias"] == "SHORT": p["global_aligned"] = True
            elif ctx.regime == "NEUTRAL":                           p["global_aligned"] = True
            else:                                                   p["global_aligned"] = False
            actionable.append(p)

        return render_template(
            "global_context.html",
            ctx=ctx,
            signal_levels=signal_levels,
            plans=actionable,
            capital=capital,
            risk_pct=risk_pct * 100,
            top_n=top_n,
            today=ddata.latest_trading_date().isoformat(),
        )
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_integration.py::test_global_route_exists -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add india_quant/dashboard/app.py tests/test_integration.py
git commit -m "feat: add /global dashboard route"
```

---

## Task 12: Dashboard template + nav link

**Files:**
- Create: `india_quant/dashboard/templates/global_context.html`
- Modify: `india_quant/dashboard/templates/base.html`

- [ ] **Step 1: Read the existing `base.html` to find the nav pattern**

```bash
grep -n "href.*intraday\|href.*signals\|href.*proposals" /Users/pushkaraj/Documents/Trading/india_quant/dashboard/templates/base.html | head -10
```

Note the nav link pattern, then match it exactly in Step 3.

- [ ] **Step 2: Create `global_context.html`**

```bash
cat > /Users/pushkaraj/Documents/Trading/india_quant/dashboard/templates/global_context.html << 'TEMPLATE'
{% extends "base.html" %}
{% block title %}Global Context · India Quant{% endblock %}
{% block content %}

<div class="page-head">
  <div>
    <h1>Global Market Context</h1>
    <div class="desc">25 cross-market signals · rolling correlations vs Nifty · {{ today }}</div>
  </div>
  <form method="get" class="flex gap-2 items-center">
    <label class="text-sm text-2">Capital ₹</label>
    <input class="input" style="width:120px" type="number" name="capital" value="{{ capital|int }}" min="10000" step="10000">
    <label class="text-sm text-2">Risk %</label>
    <input class="input" style="width:70px"  type="number" name="risk"    value="{{ '%.1f' % risk_pct }}" min="0.1" max="5" step="0.1">
    <label class="text-sm text-2">Top</label>
    <input class="input" style="width:60px"  type="number" name="top"     value="{{ top_n }}" min="1" max="20">
    <button class="btn primary" type="submit">Refresh</button>
  </form>
</div>

{# ── Section 1: Day Bias Card ────────────────────────────────────────────── #}
<section class="card section">
  <div class="card-hd"><span class="title">Today's Global Bias</span>
    <span class="meta">fetched {{ ctx.fetched_at.strftime('%H:%M IST') }}</span>
  </div>
  <div class="card-bd" style="display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap">
    <div>
      {% if ctx.regime == 'RISK_ON' %}
        <span class="badge bull" style="font-size:1.4rem; padding:8px 20px">RISK-ON</span>
      {% elif ctx.regime == 'RISK_OFF' %}
        <span class="badge bear" style="font-size:1.4rem; padding:8px 20px">RISK-OFF</span>
      {% else %}
        <span class="badge" style="font-size:1.4rem; padding:8px 20px; background:#334155; color:#cbd5e1">NEUTRAL</span>
      {% endif %}
    </div>
    <div>
      <div class="muted" style="font-size:12px; margin-bottom:4px">KEY DRIVERS</div>
      <ul style="margin:0; padding-left:16px; font-size:13px; line-height:1.8">
        {% for d in ctx.regime_drivers %}<li>{{ d }}</li>{% endfor %}
      </ul>
    </div>
    <div style="max-width:420px; font-size:13px; line-height:1.6; color:#94a3b8">
      {{ ctx.nifty_bias_text }}
    </div>
  </div>
</section>

{# ── Section 2: Global Instrument Trade Levels ───────────────────────────── #}
{% set groups_order = ["US", "Europe", "Asia", "FX", "Commodities"] %}
{% for grp in groups_order %}
  {% set grp_signals = ctx.signals | selectattr("group", "equalto", grp) | list %}
  {% if grp_signals %}
  <section class="card section">
    <div class="card-hd"><span class="title">{{ grp }} Markets</span></div>
    <div class="card-bd flush"><div class="table-wrap"><table class="x" style="font-size:12px">
      <thead><tr>
        <th>Signal</th>
        <th class="text-right">Price</th>
        <th class="text-right">1D %</th>
        <th class="text-right">5D %</th>
        <th class="text-right">Nifty Impact</th>
        <th class="text-right">Corr 30d</th>
        <th class="text-right">Corr 90d</th>
        <th class="text-right">Side</th>
        <th class="text-right">Entry</th>
        <th class="text-right">Stop</th>
        <th class="text-right">T1</th>
        <th class="text-right">R:R</th>
        <th class="text-right">Margin ₹</th>
        <th class="text-right">Max Loss ₹</th>
        <th class="text-right">T1 Profit ₹</th>
      </tr></thead>
      <tbody>
      {% for sig in grp_signals %}
        {% set lvl = signal_levels.get(sig.ticker, {}) %}
        <tr>
          <td><b>{{ sig.label }}</b><br><span class="muted" style="font-size:10px">{{ sig.ticker }}</span></td>
          <td class="text-right num">{{ '%.2f'|format(sig.price) if sig.price else '—' }}</td>
          <td class="text-right num {% if sig.pct_1d and sig.pct_1d > 0 %}pos{% elif sig.pct_1d %}neg{% endif %}">
            {{ '%+.2f%%'|format(sig.pct_1d) if sig.pct_1d is not none else '—' }}
          </td>
          <td class="text-right num {% if sig.pct_5d and sig.pct_5d > 0 %}pos{% elif sig.pct_5d %}neg{% endif %}">
            {{ '%+.2f%%'|format(sig.pct_5d) if sig.pct_5d is not none else '—' }}
          </td>
          <td class="text-right">
            {% if sig.direction == 'bullish' %}<span class="badge bull">Bullish</span>
            {% elif sig.direction == 'bearish' %}<span class="badge bear">Bearish</span>
            {% else %}<span class="muted">—</span>{% endif %}
          </td>
          <td class="text-right num" style="{% if sig.corr_30d and sig.corr_30d|abs > 0.5 %}font-weight:bold{% endif %}">
            {{ '%.2f'|format(sig.corr_30d) if sig.corr_30d is not none else '—' }}
          </td>
          <td class="text-right num">{{ '%.2f'|format(sig.corr_90d) if sig.corr_90d is not none else '—' }}</td>
          {% if lvl %}
            <td class="text-right">
              {% if lvl.side == 'LONG' %}<span class="badge bull">Long</span>{% else %}<span class="badge bear">Short</span>{% endif %}
            </td>
            <td class="text-right num {% if lvl.side == 'LONG' %}pos{% else %}neg{% endif %}">{{ '%.2f'|format(lvl.entry) }}</td>
            <td class="text-right num {% if lvl.side == 'LONG' %}neg{% else %}pos{% endif %}">{{ '%.2f'|format(lvl.stop) }}</td>
            <td class="text-right num pos">{{ '%.2f'|format(lvl.t1) }}</td>
            <td class="text-right num">{{ lvl.rr1 }}R</td>
            <td class="text-right num">₹{{ '{:,.0f}'.format(lvl.margin_inr) }}</td>
            <td class="text-right num neg">−₹{{ '{:,.0f}'.format(lvl.max_loss_inr) }}</td>
            <td class="text-right num pos">+₹{{ '{:,.0f}'.format(lvl.profit1_inr) }}</td>
          {% else %}
            <td colspan="8" class="text-center muted" style="font-size:11px">Reference only</td>
          {% endif %}
        </tr>
      {% endfor %}
      </tbody>
    </table></div></div>
  </section>
  {% endif %}
{% endfor %}

{# ── Section 3: Nifty Stock Setups (Global Filtered) ─────────────────────── #}
<section class="card section">
  <div class="card-hd">
    <span class="title">{{ plans|length }} Nifty setups · global-context filtered</span>
    <span class="meta">✓ = bias aligns with regime · ⚠ = conflicts</span>
  </div>
  <div class="card-bd flush"><div class="table-wrap"><table class="x" style="font-size:12px">
    <thead><tr>
      <th>Ticker</th><th>Bias</th><th class="text-right">Score</th><th class="text-right">L/S</th>
      <th class="text-right">Close</th><th class="text-right">ATR%</th>
      <th class="text-right">RSI</th><th class="text-right">Regime</th>
      <th class="text-right">Trigger</th><th class="text-right">Stop</th>
      <th class="text-right">T1</th><th class="text-right">R:R</th>
      <th class="text-right">Qty</th><th class="text-right">Margin</th>
      <th class="text-right">Max Loss</th><th class="text-right">T1 Profit</th>
      <th class="text-right">Global ✓</th>
    </tr></thead>
    <tbody>
    {% for p in plans %}
    <tr>
      <td><a class="ticker-pill" href="/ticker/{{ p.ticker.replace('.NS','') }}">{{ p.ticker.replace('.NS','') }}</a></td>
      <td>{% if p.bias == 'LONG' %}<span class="badge bull">long</span>{% else %}<span class="badge bear">short</span>{% endif %}</td>
      <td class="text-right num">{{ '%.0f' % p.score }}</td>
      <td class="text-right num" style="font-size:10px; color:#8aa1c4">{{ '%.0f' % p.score_long }}/{{ '%.0f' % p.score_short }}</td>
      <td class="text-right num">{{ p.prev_close }}</td>
      <td class="text-right num">{{ p.atr_pct }}%</td>
      <td class="text-right num">{{ '%.0f'|format(p.rsi) if p.rsi else '—' }}</td>
      <td class="text-right num {% if p.regime == 'TRENDING' %}pos{% else %}muted{% endif %}">{{ p.regime[:5] if p.regime else '—' }}</td>
      <td class="text-right num {% if p.bias == 'LONG' %}pos{% else %}neg{% endif %}">{{ p.trigger }}</td>
      <td class="text-right num {% if p.bias == 'LONG' %}neg{% else %}pos{% endif %}">{{ p.stop }}</td>
      <td class="text-right num pos">{{ p.target1 }}</td>
      <td class="text-right num">{{ p.rr1 }}</td>
      <td class="text-right num">{{ p.qty }}</td>
      <td class="text-right num">₹{{ '{:,.0f}'.format(p.margin_inr) }}</td>
      <td class="text-right num neg">−₹{{ '{:,.0f}'.format(p.max_loss_inr) }}</td>
      <td class="text-right num pos">+₹{{ '{:,.0f}'.format(p.profit1_inr) }}</td>
      <td class="text-right">{% if p.global_aligned %}✓{% else %}<span class="warn">⚠</span>{% endif %}</td>
    </tr>
    {% else %}
    <tr><td colspan="17" class="text-center muted">No setups passed today's screener.</td></tr>
    {% endfor %}
    </tbody>
  </table></div></div>
</section>

{# ── Section 4: Correlation Heatmap ──────────────────────────────────────── #}
<section class="card section">
  <div class="card-hd"><span class="title">Correlation heatmap — vs Nifty</span><span class="meta">sorted by |30d corr| · green = positive · red = negative</span></div>
  <div class="card-bd flush"><div class="table-wrap">
    <table class="x" style="font-size:12px">
      <thead><tr><th>Signal</th><th>Ticker</th><th class="text-right">Group</th><th class="text-right" style="width:140px">Corr 30d</th><th class="text-right" style="width:140px">Corr 90d</th></tr></thead>
      <tbody>
      {% set sorted_signals = ctx.signals | sort(attribute='corr_30d', reverse=true) %}
      {% for sig in sorted_signals %}
        {% if sig.corr_30d is not none %}
        <tr>
          <td>{{ sig.label }}</td>
          <td class="muted" style="font-size:10px">{{ sig.ticker }}</td>
          <td class="text-right muted">{{ sig.group }}</td>
          <td class="text-right">
            {% set c = sig.corr_30d %}
            {% set pct = ((c + 1) / 2 * 100)|int %}
            <div style="background:linear-gradient(to right,
              {% if c >= 0 %}#1e293b {{ (50 - pct//2)|int }}%, #22c55e {{ pct }}%{% else %}#ef4444 {{ (100 - pct)|int }}%, #1e293b {{ (50 + pct//2)|int }}%{% endif %}
            ); padding:2px 8px; border-radius:3px; color:white; font-weight:bold">
              {{ '%.3f'|format(c) }}
            </div>
          </td>
          <td class="text-right">
            {% set c = sig.corr_90d %}
            {% if c is not none %}
            <div style="background:linear-gradient(to right,
              {% if c >= 0 %}#1e293b {{ (50 - ((c+1)/2*100)//2)|int }}%, #22c55e {{ ((c+1)/2*100)|int }}%{% else %}#ef4444 {{ (100 - ((c+1)/2*100))|int }}%, #1e293b {{ (50 + ((c+1)/2*100)//2)|int }}%{% endif %}
            ); padding:2px 8px; border-radius:3px; color:white">
              {{ '%.3f'|format(c) }}
            </div>
            {% else %}—{% endif %}
          </td>
        </tr>
        {% endif %}
      {% endfor %}
      </tbody>
    </table>
  </div></div>
</section>

{% endblock %}
TEMPLATE
```

- [ ] **Step 3: Add the nav link to `base.html`**

Find the existing nav links pattern (e.g. the Intraday or Signals link) and add Global after it:

```bash
grep -n "intraday\|/signals\|/proposals" /Users/pushkaraj/Documents/Trading/india_quant/dashboard/templates/base.html | head -5
```

Then add this nav item in the same pattern used by the existing links (match exact HTML structure):

```html
<a href="/global" class="nav-link {% if request.path == '/global' %}active{% endif %}">Global</a>
```

- [ ] **Step 4: Start the dashboard and verify manually**

```bash
cd /Users/pushkaraj/Documents/Trading
python main.py --dashboard
```

Open `http://localhost:5050/global`. Verify:
- Day bias card renders with regime badge and drivers
- At least some market sections appear (US, Europe, Asia)
- Correlation heatmap rows visible
- No 500 errors in terminal

- [ ] **Step 5: Commit**

```bash
git add india_quant/dashboard/templates/global_context.html india_quant/dashboard/templates/base.html
git commit -m "feat: add global context dashboard page (4 sections)"
```

---

## Task 13: Final integration verification

**Files:** none (verification only)

- [ ] **Step 1: Run all offline tests**

```bash
cd /Users/pushkaraj/Documents/Trading
pytest tests/test_integration.py -k "not angel and not finbert and not timescale and not ml_train and not walkforward and not hypertable" -v
```

Expected: All pass. Any failure is a regression — fix before proceeding.

- [ ] **Step 2: Run the screener standalone to verify global context is wired**

```bash
python -m india_quant.signals.screener --capital 200000 --top 5
```

Expected output includes:
- `NSE INTRADAY SCREENER v4` header
- No Python traceback
- Regime information visible in the pre-market block (or falls back silently)

- [ ] **Step 3: Verify `get_global_context()` round-trip**

```bash
python -c "
from india_quant.signals.global_context import get_global_context
ctx = get_global_context()
print('Regime:', ctx.regime)
print('Drivers:', ctx.regime_drivers)
print('Signals fetched:', len(ctx.signals))
print('USDINR:', ctx.usdinr)
# Second call — should hit cache
import time; t = time.time()
get_global_context()
print(f'Cache hit: {time.time()-t:.3f}s')
"
```

Expected: Regime printed, signals > 15, cache hit < 0.01s

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "feat: global context module complete — screener, ML, dashboard integrated"
```

---

## Spec Coverage Checklist

| Spec requirement | Task |
|---|---|
| 25-ticker signal universe in 5 groups | Task 1 |
| Group-isolated fetch with try/except | Task 2 |
| 30d and 90d rolling correlations vs Nifty | Task 2 |
| Regime: RISK_ON / RISK_OFF / NEUTRAL | Task 3 |
| Instrument trade levels (entry/stop/T1/T2/margin/P&L) | Task 4 |
| GlobalSignal ORM model | Task 5 |
| Pipeline step at `run_pre_market()` | Task 6 |
| Historical backfill script | Task 7 |
| Replace `_get_market_context()` in screener | Task 8 |
| `_compute_global_delta()` capped at ±10 | Task 8 |
| Regime gate in `_score()` (Layer 1) | Task 9 |
| Additive delta in `_score()` (Layer 2) | Task 9 |
| Hard block on extreme RISK_OFF | Task 8 |
| ML feature join in `prepare_dataset()` | Task 10 |
| `FEATURE_COLS` extended with global columns | Task 10 |
| `/global` route | Task 11 |
| Dashboard section 1: Day bias card | Task 12 |
| Dashboard section 2: Instrument trade levels | Task 12 |
| Dashboard section 3: Nifty setups + global alignment column | Task 12 |
| Dashboard section 4: Correlation heatmap | Task 12 |
| Nav link in base.html | Task 12 |
| TTL cache (15 min, module-level) | Task 1 |
| Error handling: group failure doesn't abort others | Task 2 |
| Screener fallback if global context fails | Task 8 |
