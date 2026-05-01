# Global Context Module — Design Spec
**Date:** 2026-05-01
**Status:** Approved

---

## Goal

Add cross-market awareness to the India Quant trading system. Global markets (US, Europe, Asia, FX, commodities) demonstrably lead or correlate with Nifty. This module makes those signals first-class citizens — feeding the screener, the ML models, the backtest, and a new dashboard page.

---

## Scope

Two phases delivered as one build:

- **Phase 1:** `global_context.py` data module + screener integration (regime gate + score delta) + dashboard page
- **Phase 2:** DB persistence + pipeline integration + ML feature engineering + backfill

---

## Signal Universe (25 tickers, all via yfinance)

| Group | Tickers | Rationale |
|---|---|---|
| US | `^GSPC`, `^IXIC`, `YM=F` | Strongest Nifty lead; ~300% corr jump post-2003 |
| Europe | `^GDAXI`, `^FTSE` | FTSE is second-strongest error-correction leader per VECM studies |
| Asia | `^N225`, `^HSI`, `^KS11`, `^TWII`, `^AXJO`, `000001.SS`, `^CNXIT`, `^NSEBANK`, `^CNXINFRA`, `^CNXPHARMA`, `^CNXREALTY`, `^CNXENERGY` | Regional session context; Nikkei closes 3:30 AM IST; Indian sector ETFs for screener sector momentum |
| FX & Rates | `USDINR=X`, `DX-Y.NYB`, `USDJPY=X`, `^TNX`, `^VIX` | Rupee cointegrated with Nifty (IIMB); DXY inverse EM proxy |
| Commodities | `CL=F`, `GC=F`, `NG=F` | India imports ~85% oil; gold is domestic sentiment signal |

`^TNX` (US 10Y yield) and `^VIX` shown as reference only on dashboard — no trade levels computed.

---

## Architecture

```
yfinance (19 tickers, 5 groups)
        │
        ▼
global_context.py — fetch()        ← group-isolated, 15-min TTL in-memory cache
        │
        ├── signals[]              SignalRow per ticker
        ├── regime                 RISK_ON | RISK_OFF | NEUTRAL
        ├── regime_drivers         ["^VIX +18%", "S&P −1.2%", ...]
        └── nifty_bias_text        plain-English summary sentence
        │
        ├──▶ screener.py           regime gate + score delta per stock
        ├──▶ app.py /global        new dashboard page
        └──▶ pipeline.py           daily DB upsert → ML features
```

**Key boundary:** `global_context.py` is pure data — regime classification only. Score multipliers and deltas live in `screener.py`. Dashboard and ML pipeline consume raw signals + regime label.

---

## File Changes

| File | Change |
|---|---|
| `india_quant/signals/global_context.py` | **New** — core module |
| `india_quant/data/models.py` | Add `GlobalSignal` ORM model |
| `india_quant/data/pipeline.py` | Add `fetch_global_signals()` step to `run_pre_market()` |
| `india_quant/data/backfill_global.py` | **New** — one-time historical backfill |
| `india_quant/signals/screener.py` | Replace `_get_market_context()`, update `_score()` |
| `india_quant/signals/ml_models.py` | Add global signal columns to feature set |
| `india_quant/dashboard/app.py` | Add `/global` route |
| `india_quant/dashboard/templates/global_context.html` | **New** — dashboard page |
| `india_quant/dashboard/templates/base.html` | Add nav link |

---

## `global_context.py` — Module Design

### Dataclasses

```python
@dataclass
class SignalRow:
    ticker:    str
    label:     str          # "S&P 500"
    group:     str          # US | Europe | Asia | FX | Commodities
    pct_1d:    float | None
    pct_5d:    float | None
    direction: str          # bullish | bearish | neutral (re: Nifty impact)
    corr_30d:  float | None
    corr_90d:  float | None
    price:     float | None # latest close/price

@dataclass
class GlobalContext:
    fetched_at:      datetime
    regime:          str           # RISK_ON | RISK_OFF | NEUTRAL
    regime_drivers:  list[str]     # top 2-3 driver strings
    signals:         list[SignalRow]
    nifty_bias_text: str           # plain-English sentence
```

### Fetch Strategy

Five groups, each wrapped in try/except. A failed group returns empty rows for that group — does not abort other groups.

```python
GROUPS = {
    "US":          ["^GSPC", "^IXIC", "YM=F"],
    "Europe":      ["^GDAXI", "^FTSE"],
    "Asia":        ["^N225", "^HSI", "^KS11", "^TWII", "^AXJO", "000001.SS"],
    "FX":          ["USDINR=X", "DX-Y.NYB", "USDJPY=X", "^TNX", "^VIX"],
    "Commodities": ["CL=F", "GC=F", "NG=F"],
}
```

Each group fetches 90 days of daily history via `yf.download(tickers, period="90d")`. This single call per group covers both overnight returns and rolling correlation windows.

### Rolling Correlations

Nifty (`^NSEI`) fetched separately (90d). For each signal:
- `corr_30d` = Pearson correlation of last 30 daily returns vs Nifty
- `corr_90d` = Pearson correlation of last 90 daily returns vs Nifty

`direction` for each signal = `"bullish"` if `pct_1d * corr_30d > 0`, `"bearish"` if `< 0`, else `"neutral"`.

### Regime Classification

```
RISK_OFF  → VIX > 22
            OR (S&P pct_1d < -1.0 AND DXY pct_1d > +0.3 AND USDINR pct_1d > +0.3)
RISK_ON   → VIX < 15 AND S&P pct_1d > +0.5 AND DXY pct_1d <= 0
NEUTRAL   → everything else
```

`regime_drivers` = top 2-3 signals that triggered the classification, formatted as `"^VIX +18%"`.

`nifty_bias_text` = templated sentence:
- RISK_OFF → `"Global markets bearish overnight. Nifty likely to open weak. Favour short setups; reduce long sizes."`
- RISK_ON  → `"Global markets bullish overnight. Nifty likely to open strong. Long setups favoured."`
- NEUTRAL  → `"Mixed global signals. No strong directional bias. Trade individual setups on merit."`

### TTL Cache

```python
_CACHE: dict = {"data": None, "fetched_at": None}
TTL_SECONDS = 900  # 15 minutes

def get_global_context() -> GlobalContext:
    now = datetime.now(IST)
    if _CACHE["data"] and (now - _CACHE["fetched_at"]).seconds < TTL_SECONDS:
        return _CACHE["data"]
    result = _fetch_all()
    _CACHE.update({"data": result, "fetched_at": now})
    return result
```

Cache is module-level — survives across requests within a Flask worker process, reset on process restart.

---

## Screener Integration

### Replace `_get_market_context()`

`run_screener()` calls `get_global_context()` instead of `_get_market_context()`. The `GlobalContext` object provides a superset of what `_get_market_context()` returned (VIX, Nifty change, sector returns all derivable from signals list). Sector ETF returns extracted from the Asia/US groups by ticker lookup.

### Regime Gate + Score Delta in `_score()`

Two new parameters added to `_score()`:

**Layer 1 — Regime multiplier (extreme events):**
```
RISK_OFF → long_score  *= 0.5, short_score *= 1.2
RISK_ON  → long_score  *= 1.1, short_score *= 0.8
NEUTRAL  → no change
```

**Layer 2 — Additive global delta (±10 pts max):**
```
+6  S&P pct_1d > +0.5% AND DXY falling
+4  Nikkei pct_1d > +0.5%
+3  USDINR pct_1d < -0.2% (INR strengthening)
-6  S&P pct_1d < -0.5% OR VIX > 20
-4  DXY pct_1d > +0.3%
-3  Crude pct_1d > +2.0%
```

Delta applied directionally: LONG score += delta, SHORT score -= delta. Capped at ±10.

**Hard block:** if `regime == RISK_OFF` AND `VIX > 25` AND `S&P pct_1d < -1.5%`, `run_screener()` returns early with `[{"skip_day": True, "reason": "Global RISK_OFF extreme", "vix": ...}]`. No per-stock processing.

---

## DB Schema

### `GlobalSignal` model (`india_quant/data/models.py`)

```python
class GlobalSignal(Base):
    __tablename__ = "global_signals"
    id         = Column(UUID, primary_key=True, default=uuid4)
    date       = Column(Date, nullable=False)
    ticker     = Column(String(20), nullable=False)
    label      = Column(String(50))
    group      = Column(String(20))
    pct_1d     = Column(Float)
    pct_5d     = Column(Float)
    corr_30d   = Column(Float)
    corr_90d   = Column(Float)
    regime     = Column(String(10))   # denormalised, same for all rows on a date
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("date", "ticker"),)
```

### Pipeline (`india_quant/data/pipeline.py`)

New step at end of `run_pre_market()`:

```python
def fetch_global_signals(session, trade_date):
    ctx = get_global_context()
    for sig in ctx.signals:
        stmt = pg_insert(GlobalSignal).values(
            date=trade_date, ticker=sig.ticker, label=sig.label,
            group=sig.group, pct_1d=sig.pct_1d, pct_5d=sig.pct_5d,
            corr_30d=sig.corr_30d, corr_90d=sig.corr_90d,
            regime=ctx.regime,
        ).on_conflict_do_update(
            index_elements=["date", "ticker"],
            set_={"pct_1d": ..., "corr_30d": ..., "regime": ..., ...}
        )
        session.execute(stmt)
```

### Backfill (`india_quant/data/backfill_global.py`)

One-time script. Fetches up to 3 years of daily history for all 19 tickers. Computes rolling correlations per date using an expanding window (min 30 days). Upserts all rows. Run once on setup before first ML retrain.

---

## ML Integration (`india_quant/signals/ml_models.py`)

`ReturnPredictor.build_features()` adds a JOIN to `global_signals` on `date`:

New feature columns (19 tickers × 4 values = 76 columns):
- `{ticker}_pct_1d`, `{ticker}_pct_5d`, `{ticker}_corr_30d`, `{ticker}_corr_90d`

Plus one ordinal column:
- `global_regime` → RISK_OFF=0, NEUTRAL=1, RISK_ON=2

Walk-forward validation requires no changes — it splits by date, and historical global signals are available for any past training window after backfill.

---

## Dashboard Page (`/global`)

### Route (`app.py`)

```python
@app.route("/global")
def global_context_page():
    from india_quant.signals.global_context import get_global_context
    from india_quant.signals.screener import run_screener
    ctx = get_global_context()
    plans = run_screener(capital_inr=capital, ...)   # cached result if pre-market ran
    actionable = [p for p in plans if p.get("bias") in ("LONG", "SHORT")]
    return render_template("global_context.html", ctx=ctx, plans=actionable)
```

### Template — 4 Sections

**Section 1 — Day Bias Card**
- Large regime badge (green=RISK_ON, red=RISK_OFF, grey=NEUTRAL)
- Bullet list of `regime_drivers`
- `nifty_bias_text` plain-English sentence
- Cache freshness: "Fetched 08:03 IST · Next refresh in 12 min"

**Section 2 — Global Instrument Trade Levels**

One row per signal (excluding `^TNX` and `^VIX` which show price + direction only).
Trade levels computed from 5-day ATR using same `_build_orb_levels()` from `screener.py`.

Asset-class leverage for margin calculation:

| Class | Leverage | Note |
|---|---|---|
| Equity indices | 10× | Reference — not directly tradeable on NSE |
| FX (USDINR) | 50× | NSE currency futures, lot = 1000 units |
| Commodities (Crude) | ~16× (6% margin) | MCX standard |
| Commodities (Gold) | ~25× (4% margin) | MCX standard |
| Commodities (NG) | ~20× (5% margin) | MCX standard |

Margin converted to INR using live `USDINR` rate for non-INR instruments.

Columns: Signal | Price | 1D% | 5D% | Side | Entry | Stop | T1 | T2 | R:R | Margin ₹ | Max Loss ₹ | T1 Profit ₹

**Section 3 — Nifty Stock Setups (Global-Context Filtered)**

Full screener table (same columns as `/intraday`) with one extra column:
- **Global Alignment** — ✓ (stock bias matches regime) or ⚠ (conflicts with regime)

The `/global` route calls `run_screener()` fresh each page load. The expensive part (yfinance fetches inside `get_global_context()`) is TTL-cached, so repeated page loads are fast.

**Section 4 — Correlation Heatmap**

HTML table, no JS. Rows = 19 signals sorted by `|corr_30d|` descending. Columns = `corr_30d`, `corr_90d`. Cell background interpolated: dark green (≥0.7) → white (0) → dark red (≤−0.7).

---

## Error Handling

- Each fetch group silently returns empty rows on failure — page renders with available data
- If `get_global_context()` fails entirely, screener falls back to original `_get_market_context()` logic (no regime gate, no delta) — trading continues
- Dashboard `/global` shows a warning banner if any group failed to fetch
- Backfill script logs per-date errors and skips, does not abort

---

## Testing

No new test infrastructure needed. Existing `tests/test_integration.py` categories apply:

**No infra needed (offline):**
- `test_global_context_imports` — module imports cleanly
- `test_regime_classification` — unit test RISK_OFF/RISK_ON/NEUTRAL logic with mock signal data
- `test_score_delta_capped` — verify delta never exceeds ±10
- `test_hard_block_returns_sentinel` — extreme RISK_OFF returns skip_day dict
- `test_signal_row_direction` — direction = bullish/bearish/neutral from pct × corr sign

**Requires live data / DB:**
- `test_fetch_all_groups` — real yfinance fetch, assert 17+ signals returned (allowing 2 to fail)
- `test_pipeline_upsert` — DB round-trip for GlobalSignal
- `test_ml_features_include_global` — feature matrix has expected global columns after JOIN

---

## Open Questions (resolved)

| Question | Decision |
|---|---|
| Screener logic in global_context.py? | No — pure data. Score math stays in screener.py |
| Phase 1 only or both phases? | Both phases in one build |
| Runtime overlay vs ML features? | Both (Option 3) — regime gate for extremes, ML for nuance |
| Approach A/B/C for fetching? | Approach B — group-based with 15-min TTL cache |
| Dashboard content | 4 sections: bias card, global trade levels, Nifty setups, corr heatmap |
