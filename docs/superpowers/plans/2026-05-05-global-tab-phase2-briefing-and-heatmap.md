# Global Tab Revamp — Phase 2 (Briefing + Heatmap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the briefing strip (overnight tiles + predicted gap) and the 2×10 correlation heatmap, render them on the existing Flask `/global` route. No trade-ticket cards yet — those are Phase 3.

**Architecture:** Two new pure modules (`briefing.py`, `correlation.py`) plus a Plotly figure factory (`heatmap_view.py`) and a small extension of the existing global-data infrastructure to add Brent and GIFT Nifty. The Flask route rewrite reads from the existing TimescaleDB tables and `signals.global_context.get_global_context()` — no parquet artifacts in this phase. GIFT Nifty is fetched best-effort via NSE-IFSC scrape (`curl-cffi`); failure path renders a `—` tile.

**Tech Stack:** Python 3.12, Flask 3.1, Jinja2, Plotly, pandas 2.3, yfinance 0.2.51, curl-cffi (already wired for NSE bypass), pytest 8.3.

**Spec reference:** `docs/superpowers/specs/2026-05-05-global-tab-revamp-design.md` §5.1, §5.2, §6.1 (BriefingStrip / CorrelationHeatmap), §11.2.

**Architectural deviation from spec (intentional, see §3 of the previous discussion):** Phase 2 reads from `global_signals` DB and `get_global_context()` at request time with a 60-second Flask cache. Parquet artifacts and the nightly/preopen scripts (spec Flow A/B) are deferred to Phase 4 when the backtester needs them. This collapses ~40% of Phase 2 work without losing functionality.

**Phase end-state demo:** Open `http://localhost:5050/global` and see:
- A top strip with 10 tiles for SPX, Nasdaq, GIFT Nifty (or `—` if scrape failed), DXY, Brent, India VIX, US10Y, Nikkei, HSI, FTSE — each showing live value and overnight % change.
- A predicted-gap line: `NIFTY +35 bps · BANKNIFTY +48 bps`.
- A 2×10 correlation heatmap (rows = NIFTY, BANKNIFTY; cols = same 10 assets) with rolling 20d/60d windows.
- The old playbook section either preserved below or removed (Task 6 decides).

**No trade tickets, no backtest sub-tab, no LLM blurb.**

---

## File Structure

**New files:**
- `india_quant/data/fetchers/gift_nifty_fetcher.py` — best-effort NSE-IFSC scraper.
- `india_quant/global_tab/briefing.py` — `build_briefing()` adapter.
- `india_quant/global_tab/correlation.py` — `load_history()` + `build_heatmap()`.
- `india_quant/global_tab/heatmap_view.py` — Plotly figure factory returning HTML embed string.
- `india_quant/dashboard/templates/global_v2.html` — new Jinja template.
- `tests/global_tab/test_briefing.py`
- `tests/global_tab/test_correlation.py`
- `tests/global_tab/test_heatmap_view.py`
- `tests/global_tab/test_gift_nifty_fetcher.py`
- `tests/global_tab/test_route_smoke.py`
- `tests/global_tab/fixtures/gift_nifty_sample.html` — fixture HTML for the GIFT Nifty parser test.

**Modified files:**
- `india_quant/data/backfill_global.py` — add `BZ=F` (Brent) to FX/Commodities group.
- `india_quant/signals/global_context.py` — add `BZ=F` label to GROUPS dict.
- `india_quant/dashboard/app.py` — `/global` route rewritten to call the new pipeline.

**Files NOT touched in Phase 2:**
- Phase 1 modules (`types.py`, `modes.py`, `feature_store.py`, `cost_model.py`).
- The 7 existing fetchers other than the new `gift_nifty_fetcher.py`.
- Other Flask routes.

---

## Conventions

- Run pytest as `venv/bin/python -m pytest …` — `pytest` on PATH may resolve to anaconda.
- Imports follow the existing repo pattern: absolute (`from india_quant.global_tab.briefing import …`).
- Logging via `loguru` (existing convention).
- Currency stays as `float` rupees; bps is integer (e.g., `35` for 0.35%).
- Commits: one per task, conventional-commits style. User runs `git push` themselves.
- Subagents may run `git add` and `git commit` per the standing authorization for this session. No push/force/amend.

---

## Task 1: Add Brent (`BZ=F`) to global ticker universe

**Why:** Brent is on the briefing strip but `backfill_global.py` only has WTI (`CL=F`). One-line additions.

**Files:**
- Modify: `india_quant/data/backfill_global.py:24-25`
- Modify: `india_quant/signals/global_context.py` (the `GROUPS` dict, around the `Commodities` block)

- [ ] **Step 1: Add `BZ=F` to backfill ticker list**

In `india_quant/data/backfill_global.py`, change the `Commodities` line in `_RAW_GROUPS`:

```python
    "Commodities": ["CL=F", "GC=F", "NG=F"],
```
to:
```python
    "Commodities": ["CL=F", "BZ=F", "GC=F", "NG=F"],
```

- [ ] **Step 2: Add label to `GROUPS` in `global_context.py`**

Find the `Commodities` sub-dict (around line ~58 in `global_context.py`) and add `BZ=F`:

```python
    "Commodities": {
        "CL=F": "Crude WTI",
        "BZ=F":  "Brent",
        "GC=F":  "Gold",
        "NG=F":  "Natural Gas",
    },
```

- [ ] **Step 3: Verify yfinance recognises the symbol**

Run:
```bash
cd /Users/pushkaraj/Documents/Trading && venv/bin/python -c "import yfinance as yf; t = yf.Ticker('BZ=F'); h = t.history(period='5d'); print(h.tail(2))"
```
Expected: a small DataFrame with `Close` values around $80. If it fails (zero rows, network error), report and we'll fall back to USO ETF or skip Brent entirely.

- [ ] **Step 4: Commit**

```bash
cd /Users/pushkaraj/Documents/Trading && git add india_quant/data/backfill_global.py india_quant/signals/global_context.py && git commit -m "feat(data): add Brent (BZ=F) to global ticker universe"
```

---

## Task 2: GIFT Nifty fetcher (best-effort NSE-IFSC scrape)

**Why:** No yfinance ticker for GIFT Nifty post-2023. Scrape `nseifsc.com` with `curl-cffi` (already used for NSE Cloudflare bypass). Failure mode: return `None`; briefing renders `—`.

**Files:**
- Create: `india_quant/data/fetchers/gift_nifty_fetcher.py`
- Create: `tests/global_tab/test_gift_nifty_fetcher.py`
- Create: `tests/global_tab/fixtures/gift_nifty_sample.html` (small HTML snippet for parsing test)

- [ ] **Step 1: Capture a sample HTML page**

The implementer should manually run:

```bash
cd /Users/pushkaraj/Documents/Trading && venv/bin/python -c "
from curl_cffi import requests
r = requests.get('https://www.nseifsc.com/products-services/derivatives', impersonate='chrome120', timeout=15)
print('status', r.status_code)
print('len', len(r.text))
"
```

If status 200 and length > 5000, save the response to `tests/global_tab/fixtures/gift_nifty_sample.html`. Then locate the GIFT Nifty front-month price in the HTML (search for "GIFT" or "NIFTY" near a numeric value); copy a ~1000-byte slice that contains it into a smaller test fixture if the full page is too large.

If the page does not contain GIFT Nifty pricing (NSE-IFSC may serve it via a separate API/JSON endpoint), pivot: try `https://www.nseifsc.com/api/derivatives-quote/NIFTY` (educated guess). If neither works, set status to `BLOCKED` and report — we'll either pick another data source or defer GIFT Nifty to a later phase. Do not invent data.

- [ ] **Step 2: Write failing tests against the fixture**

`tests/global_tab/test_gift_nifty_fetcher.py`:

```python
"""Tests for the GIFT Nifty NSE-IFSC scraper."""
from pathlib import Path
from unittest.mock import patch

import pytest

from india_quant.data.fetchers.gift_nifty_fetcher import (
    GiftNiftyQuote,
    fetch_gift_nifty_quote,
    parse_gift_nifty_page,
)


FIXTURE = Path(__file__).parent / "fixtures" / "gift_nifty_sample.html"


def test_parse_extracts_price_from_fixture():
    html = FIXTURE.read_text()
    quote = parse_gift_nifty_page(html)
    assert isinstance(quote, GiftNiftyQuote)
    assert quote.last_price > 20000  # NIFTY range
    assert quote.last_price < 30000


def test_parse_returns_none_on_unparseable_html():
    assert parse_gift_nifty_page("<html><body>nothing here</body></html>") is None


def test_fetch_returns_none_on_http_error():
    class FakeResp:
        status_code = 503
        text = ""

    with patch("india_quant.data.fetchers.gift_nifty_fetcher.requests.get", return_value=FakeResp()):
        assert fetch_gift_nifty_quote() is None


def test_fetch_returns_none_on_network_exception():
    with patch(
        "india_quant.data.fetchers.gift_nifty_fetcher.requests.get",
        side_effect=Exception("network down"),
    ):
        assert fetch_gift_nifty_quote() is None
```

- [ ] **Step 3: Implement the fetcher**

`india_quant/data/fetchers/gift_nifty_fetcher.py`:

```python
"""GIFT Nifty fetcher via NSE-IFSC public website.

GIFT Nifty migrated from SGX to NSE-IFSC in mid-2023. There is no clean
yfinance ticker for it. This module scrapes the public NSE-IFSC pages with
curl-cffi (Chrome TLS impersonation) and returns a GiftNiftyQuote, or None
on any failure. Callers must handle None gracefully.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from curl_cffi import requests
from loguru import logger

_IFSC_URL = "https://www.nseifsc.com/products-services/derivatives"
_TIMEOUT = 15


@dataclass(frozen=True)
class GiftNiftyQuote:
    last_price: float
    change_pct: float | None  # None if not available on the page


def parse_gift_nifty_page(html: str) -> GiftNiftyQuote | None:
    # The NSE-IFSC page format may shift; this is a best-effort extraction.
    # Strategy: find a sequence "NIFTY ... 24123.45 ... +0.45%" or similar.
    # The implementer captures the actual fixture and adjusts the regex.
    # IMPORTANT: do not invent values; return None if the regex misses.
    price_match = re.search(r"NIFTY[^0-9<>]{0,200}?(2[0-9]{4}\.[0-9]{1,2})", html)
    if price_match is None:
        return None
    try:
        last_price = float(price_match.group(1))
    except ValueError:
        return None

    pct_match = re.search(r"NIFTY[^%<>]{0,300}?([+-]?[0-9]{1,2}\.[0-9]{1,2})\s*%", html)
    change_pct: float | None = None
    if pct_match is not None:
        try:
            change_pct = float(pct_match.group(1))
        except ValueError:
            change_pct = None

    return GiftNiftyQuote(last_price=last_price, change_pct=change_pct)


def fetch_gift_nifty_quote() -> GiftNiftyQuote | None:
    try:
        resp = requests.get(_IFSC_URL, impersonate="chrome120", timeout=_TIMEOUT)
    except Exception as exc:
        logger.warning("gift_nifty fetch failed: {}", exc)
        return None

    if resp.status_code != 200:
        logger.warning("gift_nifty bad status: {}", resp.status_code)
        return None

    return parse_gift_nifty_page(resp.text)
```

The regex above is a starting point. After capturing the fixture in Step 1, the implementer adjusts the regex to match the actual HTML and re-runs the parse test. The test asserts `last_price > 20000 and < 30000` — this gives the implementer freedom to refine the regex without changing the test contract.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab/test_gift_nifty_fetcher.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/pushkaraj/Documents/Trading && git add india_quant/data/fetchers/gift_nifty_fetcher.py tests/global_tab/test_gift_nifty_fetcher.py tests/global_tab/fixtures/gift_nifty_sample.html && git commit -m "feat(data): add best-effort GIFT Nifty fetcher (NSE-IFSC scrape)"
```

If Step 1 escalated to BLOCKED (no usable HTML source found), skip this task entirely and proceed to Task 3 — `briefing.py` will fill the GIFT Nifty tile with `None` for now, rendered as `—`. Open a follow-up to revisit the GIFT Nifty source in Phase 7.

---

## Task 3: `briefing.py` — adapter from `GlobalContext` to `BriefingStrip`

**Why:** `signals/global_context.get_global_context()` already produces a rich object with directional signals and per-ticker price/% data. Phase 2 just adapts it into the `BriefingStrip` typed dataclass from Phase 1.

**Files:**
- Create: `india_quant/global_tab/briefing.py`
- Create: `tests/global_tab/test_briefing.py`

**Tile definition (the 10 columns of the heatmap, plus rows for context):**

| Tile label | Source ticker | yfinance / fetcher |
|---|---|---|
| SPX | `^GSPC` | yfinance |
| Nasdaq | `^IXIC` | yfinance |
| GIFT Nifty | (scrape) | `gift_nifty_fetcher` |
| DXY | `DX-Y.NYB` | yfinance |
| Brent | `BZ=F` | yfinance (Task 1) |
| India VIX | `^INDIAVIX` (yfinance) — or fallback to `^VIX` | yfinance |
| US 10Y | `^TNX` | yfinance |
| Nikkei | `^N225` | yfinance |
| HSI | `^HSI` | yfinance |
| FTSE | `^FTSE` | yfinance |

The order above is the rendering order on the strip.

**Predicted gap (predicted_gap_bps)**: a simple regression-based estimate fits later. For Phase 2, use a placeholder formula:
```
predicted_gap_bps[NIFTY]    = 0.5 * gift_nifty_change_pct_bps  if gift available else None
predicted_gap_bps[BANKNIFTY] = 0.6 * gift_nifty_change_pct_bps if gift available else None
```
This gets replaced with a real model in Phase 3. Document inline that this is a placeholder.

- [ ] **Step 1: Write failing tests**

`tests/global_tab/test_briefing.py`:

```python
"""Tests for the briefing-strip adapter."""
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote
from india_quant.global_tab.briefing import build_briefing
from india_quant.global_tab.types import BriefingStrip, BriefingTile


def _fake_signal_row(ticker: str, label: str, group: str, pct_1d: float, price: float):
    """Mimic SignalRow without importing it."""
    obj = MagicMock()
    obj.ticker = ticker
    obj.label = label
    obj.group = group
    obj.pct_1d = pct_1d
    obj.price = price
    return obj


def _fake_global_context():
    rows = [
        _fake_signal_row("^GSPC", "S&P 500", "US", 1.10, 5612.40),
        _fake_signal_row("^IXIC", "Nasdaq", "US", 1.40, 17800.0),
        _fake_signal_row("DX-Y.NYB", "DXY", "FX", -0.30, 104.20),
        _fake_signal_row("BZ=F", "Brent", "Commodities", 0.85, 82.10),
        _fake_signal_row("^INDIAVIX", "India VIX", "FX", -2.10, 13.40),
        _fake_signal_row("^TNX", "US 10Y Yield", "FX", 0.50, 4.45),
        _fake_signal_row("^N225", "Nikkei 225", "Asia", 0.70, 39200.0),
        _fake_signal_row("^HSI", "Hang Seng", "Asia", -0.20, 18800.0),
        _fake_signal_row("^FTSE", "FTSE 100", "Europe", 0.30, 8200.0),
    ]
    ctx = MagicMock()
    ctx.signals = rows
    return ctx


def test_build_briefing_returns_typed_strip():
    ctx = _fake_global_context()
    gift = GiftNiftyQuote(last_price=24412.0, change_pct=0.60)
    as_of = datetime(2026, 5, 5, 8, 45)

    strip = build_briefing(as_of=as_of, context=ctx, gift_nifty=gift)
    assert isinstance(strip, BriefingStrip)
    assert strip.as_of == as_of


def test_build_briefing_has_all_ten_tiles_in_order():
    ctx = _fake_global_context()
    gift = GiftNiftyQuote(last_price=24412.0, change_pct=0.60)
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=gift)
    labels = [t.label for t in strip.tiles]
    assert labels == [
        "SPX", "Nasdaq", "GIFT Nifty", "DXY", "Brent",
        "India VIX", "US 10Y", "Nikkei", "Hang Seng", "FTSE",
    ]


def test_build_briefing_gift_tile_uses_scraper_value():
    ctx = _fake_global_context()
    gift = GiftNiftyQuote(last_price=24412.0, change_pct=0.60)
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=gift)
    gift_tile = next(t for t in strip.tiles if t.label == "GIFT Nifty")
    assert gift_tile.change_pct == pytest.approx(0.60)
    assert "24412" in gift_tile.value or "24,412" in gift_tile.value


def test_build_briefing_gift_tile_renders_em_dash_when_unavailable():
    ctx = _fake_global_context()
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=None)
    gift_tile = next(t for t in strip.tiles if t.label == "GIFT Nifty")
    assert gift_tile.value == "—"
    assert gift_tile.change_pct == 0.0  # neutral
    assert gift_tile.sentiment == "neutral"


def test_build_briefing_sentiment_follows_pct_sign():
    ctx = _fake_global_context()
    strip = build_briefing(
        as_of=datetime(2026, 5, 5, 8, 45),
        context=ctx,
        gift_nifty=GiftNiftyQuote(last_price=24412.0, change_pct=0.60),
    )
    spx = next(t for t in strip.tiles if t.label == "SPX")
    dxy = next(t for t in strip.tiles if t.label == "DXY")
    vix = next(t for t in strip.tiles if t.label == "India VIX")
    assert spx.sentiment == "bullish"
    assert dxy.sentiment == "bearish"   # DXY -0.30 in fixture
    assert vix.sentiment == "bearish"   # VIX -2.10 → falling vol is good but tile is "neutral on price direction"; we'll treat <0 as bearish-tone for the tile color


def test_predicted_gap_uses_gift_change():
    ctx = _fake_global_context()
    gift = GiftNiftyQuote(last_price=24412.0, change_pct=0.60)
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=gift)
    # 0.60% = 60 bps
    assert strip.predicted_gap_bps["NIFTY"] == pytest.approx(0.5 * 60.0)
    assert strip.predicted_gap_bps["BANKNIFTY"] == pytest.approx(0.6 * 60.0)


def test_predicted_gap_is_zero_when_gift_unavailable():
    ctx = _fake_global_context()
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=None)
    assert strip.predicted_gap_bps == {"NIFTY": 0.0, "BANKNIFTY": 0.0}


def test_missing_ticker_renders_em_dash_tile():
    """If global_context is missing one of the rows we expect, the tile shows —."""
    ctx = MagicMock()
    ctx.signals = []   # no signals at all
    strip = build_briefing(
        as_of=datetime(2026, 5, 5, 8, 45),
        context=ctx,
        gift_nifty=None,
    )
    for tile in strip.tiles:
        if tile.label != "GIFT Nifty":
            assert tile.value == "—"
            assert tile.change_pct == 0.0
            assert tile.sentiment == "neutral"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab/test_briefing.py -v`
Expected: ImportError for `build_briefing`.

- [ ] **Step 3: Implement `briefing.py`**

```python
"""Briefing-strip adapter.

Pure function: takes a GlobalContext (from signals/global_context.py) and an
optional GiftNiftyQuote, returns a BriefingStrip dataclass. No I/O. No mutation.

The 10 tiles are emitted in a fixed order matching the heatmap columns. If a
ticker is missing from the GlobalContext, the tile renders as '—' (neutral
sentiment, change_pct=0.0). The predicted_gap_bps formula is a placeholder
that uses 0.5x / 0.6x of the GIFT Nifty change in bps; the real model lands
in Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote
from india_quant.global_tab.types import BriefingStrip, BriefingTile

# (label, ticker) in render order. Ticker is None for GIFT Nifty (scraper).
_TILE_ORDER: list[tuple[str, str | None]] = [
    ("SPX", "^GSPC"),
    ("Nasdaq", "^IXIC"),
    ("GIFT Nifty", None),
    ("DXY", "DX-Y.NYB"),
    ("Brent", "BZ=F"),
    ("India VIX", "^INDIAVIX"),
    ("US 10Y", "^TNX"),
    ("Nikkei", "^N225"),
    ("Hang Seng", "^HSI"),
    ("FTSE", "^FTSE"),
]


def _format_value(price: float | None) -> str:
    if price is None:
        return "—"
    if price >= 1000:
        return f"{price:,.2f}"
    return f"{price:.2f}"


def _sentiment_for_pct(pct: float | None) -> Literal["bullish", "bearish", "neutral"]:
    if pct is None or pct == 0.0:
        return "neutral"
    return "bullish" if pct > 0.0 else "bearish"


def _gift_tile(quote: GiftNiftyQuote | None) -> BriefingTile:
    if quote is None:
        return BriefingTile(label="GIFT Nifty", value="—", change_pct=0.0, sentiment="neutral")
    return BriefingTile(
        label="GIFT Nifty",
        value=_format_value(quote.last_price),
        change_pct=quote.change_pct or 0.0,
        sentiment=_sentiment_for_pct(quote.change_pct),
    )


def _row_by_ticker(rows: Iterable, ticker: str):
    for r in rows:
        if getattr(r, "ticker", None) == ticker:
            return r
    return None


def build_briefing(
    *,
    as_of: datetime,
    context,
    gift_nifty: GiftNiftyQuote | None,
) -> BriefingStrip:
    rows = list(getattr(context, "signals", []) or [])

    tiles: list[BriefingTile] = []
    for label, ticker in _TILE_ORDER:
        if label == "GIFT Nifty":
            tiles.append(_gift_tile(gift_nifty))
            continue

        row = _row_by_ticker(rows, ticker)
        if row is None:
            tiles.append(BriefingTile(label=label, value="—", change_pct=0.0, sentiment="neutral"))
            continue

        pct = getattr(row, "pct_1d", None)
        tiles.append(
            BriefingTile(
                label=label,
                value=_format_value(getattr(row, "price", None)),
                change_pct=pct if pct is not None else 0.0,
                sentiment=_sentiment_for_pct(pct),
            )
        )

    if gift_nifty is not None and gift_nifty.change_pct is not None:
        gift_bps = gift_nifty.change_pct * 100.0  # 0.60% → 60 bps
        gap = {"NIFTY": 0.5 * gift_bps, "BANKNIFTY": 0.6 * gift_bps}
    else:
        gap = {"NIFTY": 0.0, "BANKNIFTY": 0.0}

    return BriefingStrip(as_of=as_of, tiles=tiles, predicted_gap_bps=gap)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab/test_briefing.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/pushkaraj/Documents/Trading && git add india_quant/global_tab/briefing.py tests/global_tab/test_briefing.py && git commit -m "feat(global_tab): add briefing strip adapter"
```

---

## Task 4: `correlation.py` — `load_history` + `build_heatmap`

**Why:** Build the 2×10 heatmap by querying historical EOD data for NIFTY, BANKNIFTY and the 10 column tickers, then computing rolling 20d/60d Pearson correlations.

**Files:**
- Create: `india_quant/global_tab/correlation.py`
- Create: `tests/global_tab/test_correlation.py`

**Design:**
- `load_history(session, *, lookback_days=120) -> pd.DataFrame` — queries `global_signals` and `price_data` (NIFTY, BANKNIFTY are in `price_data` per the existing schema), returns a wide frame indexed by date with columns `[NIFTY, BANKNIFTY, SPX, NASDAQ, DXY, BRENT, INDIA_VIX, US10Y, NIKKEI, HSI, FTSE]`. GIFT Nifty is NOT in this frame for Phase 2 — it has no historical EOD store yet. Document inline.
- `build_heatmap(history: pd.DataFrame, *, as_of: date) -> CorrelationHeatmap` — pure function. Computes rolling 20d and 60d correlations between each row asset (NIFTY, BANKNIFTY) and each column asset, using log returns (`log(P_t / P_{t-1})`). Returns a `CorrelationHeatmap` with one `CorrelationCell` per (row, col) pair.
- Skips a column if its history is too short (fewer than 30 valid observations) — that cell is omitted, not zeroed.

- [ ] **Step 1: Write failing tests**

`tests/global_tab/test_correlation.py`:

```python
"""Tests for the correlation heatmap builder."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from india_quant.global_tab.correlation import build_heatmap, COLUMN_TICKERS, ROW_TICKERS
from india_quant.global_tab.types import CorrelationCell, CorrelationHeatmap


def _synthetic_history(n_days: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=date(2026, 5, 4), periods=n_days, freq="B")
    cols = ROW_TICKERS + COLUMN_TICKERS
    # Random walks, with NIFTY and BANKNIFTY weakly correlated to SPX
    base_spx = np.cumsum(rng.normal(0, 0.01, n_days)) + 5500
    base_nifty = 0.6 * (base_spx - 5500) + np.cumsum(rng.normal(0, 0.01, n_days)) + 24000
    base_bn = 0.5 * (base_spx - 5500) + np.cumsum(rng.normal(0, 0.012, n_days)) + 51000
    data = {}
    for c in cols:
        if c == "SPX":
            data[c] = base_spx
        elif c == "NIFTY":
            data[c] = base_nifty
        elif c == "BANKNIFTY":
            data[c] = base_bn
        else:
            data[c] = np.cumsum(rng.normal(0, 0.01, n_days)) + 100
    return pd.DataFrame(data, index=idx)


def test_build_heatmap_returns_typed_object():
    hist = _synthetic_history()
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    assert isinstance(hm, CorrelationHeatmap)
    assert hm.as_of == date(2026, 5, 4)


def test_heatmap_has_two_rows_times_columns_minus_missing():
    hist = _synthetic_history()
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    expected_count = len(ROW_TICKERS) * len(COLUMN_TICKERS)
    assert len(hm.cells) == expected_count


def test_each_cell_has_valid_correlation_range():
    hist = _synthetic_history()
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    for cell in hm.cells:
        assert isinstance(cell, CorrelationCell)
        assert -1.0 <= cell.rho_20d <= 1.0
        assert -1.0 <= cell.rho_60d <= 1.0


def test_nifty_spx_correlation_is_positive_in_synthetic_data():
    hist = _synthetic_history(n_days=120)
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    nifty_spx = next(c for c in hm.cells if c.asset_a == "NIFTY" and c.asset_b == "SPX")
    assert nifty_spx.rho_60d > 0.2


def test_columns_with_insufficient_data_are_skipped():
    hist = _synthetic_history(n_days=120)
    # Wipe most of the FTSE column
    hist["FTSE"] = np.nan
    hist.iloc[:5, hist.columns.get_loc("FTSE")] = 100.0
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    ftse_cells = [c for c in hm.cells if c.asset_b == "FTSE"]
    # Either skipped entirely or rho is None — we choose skipped
    assert ftse_cells == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab/test_correlation.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `correlation.py`**

```python
"""Correlation heatmap builder.

Pure function: takes a wide history DataFrame indexed by date, returns a
CorrelationHeatmap with one CorrelationCell per (row_asset, col_asset) pair.
Computes Pearson correlation on log returns over rolling 20- and 60-day
windows, anchored at as_of.

GIFT Nifty is not in the heatmap (no historical EOD store yet — added in a
later phase). Columns with fewer than 30 valid observations are skipped.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from india_quant.global_tab.types import CorrelationCell, CorrelationHeatmap

ROW_TICKERS: list[str] = ["NIFTY", "BANKNIFTY"]
COLUMN_TICKERS: list[str] = [
    "SPX", "NASDAQ", "DXY", "BRENT", "INDIA_VIX",
    "US10Y", "NIKKEI", "HSI", "FTSE",
]
# Note: GIFT Nifty not yet in this list; tile-only for Phase 2.

_MIN_OBS = 30


def _log_returns(series: pd.Series) -> pd.Series:
    return np.log(series).diff()


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> float | None:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < _MIN_OBS or len(pair) < window:
        return None
    sliced = pair.iloc[-window:]
    if sliced.iloc[:, 0].std(ddof=0) == 0 or sliced.iloc[:, 1].std(ddof=0) == 0:
        return None
    rho = float(sliced.iloc[:, 0].corr(sliced.iloc[:, 1]))
    if np.isnan(rho):
        return None
    return rho


def build_heatmap(*, history: pd.DataFrame, as_of: date) -> CorrelationHeatmap:
    cells: list[CorrelationCell] = []

    returns = {col: _log_returns(history[col]) for col in history.columns if col in ROW_TICKERS + COLUMN_TICKERS}

    for row in ROW_TICKERS:
        if row not in returns:
            continue
        for col in COLUMN_TICKERS:
            if col not in returns:
                continue
            rho_20 = _rolling_corr(returns[row], returns[col], window=20)
            rho_60 = _rolling_corr(returns[row], returns[col], window=60)
            if rho_20 is None or rho_60 is None:
                # Skip cells where rolling correlation cannot be computed.
                continue
            cells.append(
                CorrelationCell(
                    asset_a=row,
                    asset_b=col,
                    rho_20d=rho_20,
                    rho_60d=rho_60,
                )
            )

    return CorrelationHeatmap(as_of=as_of, cells=cells)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab/test_correlation.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/pushkaraj/Documents/Trading && git add india_quant/global_tab/correlation.py tests/global_tab/test_correlation.py && git commit -m "feat(global_tab): add correlation heatmap builder"
```

---

## Task 5: `heatmap_view.py` — Plotly figure factory

**Why:** Convert `CorrelationHeatmap` into a Plotly HTML embed string for Jinja injection.

**Files:**
- Create: `india_quant/global_tab/heatmap_view.py`
- Create: `tests/global_tab/test_heatmap_view.py`

- [ ] **Step 1: Write failing tests**

`tests/global_tab/test_heatmap_view.py`:

```python
"""Tests for the Plotly heatmap renderer."""
from datetime import date

import pytest

from india_quant.global_tab.heatmap_view import render_heatmap_html
from india_quant.global_tab.types import CorrelationCell, CorrelationHeatmap


def _sample_heatmap() -> CorrelationHeatmap:
    cells = []
    for row in ["NIFTY", "BANKNIFTY"]:
        for col in ["SPX", "NASDAQ", "DXY"]:
            cells.append(
                CorrelationCell(asset_a=row, asset_b=col, rho_20d=0.4, rho_60d=0.5)
            )
    return CorrelationHeatmap(as_of=date(2026, 5, 4), cells=cells)


def test_renders_html_string():
    hm = _sample_heatmap()
    html = render_heatmap_html(hm)
    assert isinstance(html, str)
    assert "<div" in html  # plotly returns a div
    assert "plotly" in html.lower()


def test_renders_a_div_with_data_points_for_each_cell():
    hm = _sample_heatmap()
    html = render_heatmap_html(hm)
    # 6 cells; the asset names should appear in the rendered figure data.
    for ticker in ["NIFTY", "BANKNIFTY", "SPX", "NASDAQ", "DXY"]:
        assert ticker in html


def test_returns_empty_state_message_when_no_cells():
    empty = CorrelationHeatmap(as_of=date(2026, 5, 4), cells=[])
    html = render_heatmap_html(empty)
    assert "no correlation data" in html.lower() or "—" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab/test_heatmap_view.py -v`
Expected: ImportError or fixture error.

- [ ] **Step 3: Check Plotly availability**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -c "import plotly; print(plotly.__version__)"`

If Plotly is not installed (it ships transitively with `shap` but might not be reliable), add `plotly==5.24.1` to `requirements.txt` and `pip install plotly==5.24.1`.

- [ ] **Step 4: Implement `heatmap_view.py`**

```python
"""Plotly HTML renderer for correlation heatmap.

Returns an HTML embed string suitable for direct Jinja injection. The figure
shows a 20d-window heatmap with hover text revealing both 20d and 60d rho.
"""
from __future__ import annotations

import plotly.graph_objects as go

from india_quant.global_tab.types import CorrelationHeatmap


def render_heatmap_html(heatmap: CorrelationHeatmap) -> str:
    if not heatmap.cells:
        return '<div class="heatmap-empty">No correlation data available — —</div>'

    rows = sorted({c.asset_a for c in heatmap.cells})
    cols = sorted({c.asset_b for c in heatmap.cells})

    z_20 = [[None for _ in cols] for _ in rows]
    hover = [[None for _ in cols] for _ in rows]

    for cell in heatmap.cells:
        ri = rows.index(cell.asset_a)
        ci = cols.index(cell.asset_b)
        z_20[ri][ci] = cell.rho_20d
        hover[ri][ci] = (
            f"<b>{cell.asset_a} ↔ {cell.asset_b}</b><br>"
            f"ρ 20d: {cell.rho_20d:+.2f}<br>"
            f"ρ 60d: {cell.rho_60d:+.2f}"
        )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_20,
            x=cols,
            y=rows,
            text=hover,
            hoverinfo="text",
            colorscale="RdBu",
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="ρ 20d"),
        )
    )
    fig.update_layout(
        title=f"NIFTY / BANKNIFTY correlations · as of {heatmap.as_of.isoformat()}",
        margin=dict(l=80, r=20, t=60, b=40),
        height=300,
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="global-heatmap")
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab/test_heatmap_view.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/pushkaraj/Documents/Trading && git add india_quant/global_tab/heatmap_view.py tests/global_tab/test_heatmap_view.py requirements.txt && git commit -m "feat(global_tab): add Plotly heatmap renderer"
```

(Include `requirements.txt` only if Plotly was added.)

---

## Task 6: Rewrite `/global` Flask route + new Jinja template

**Why:** Wire briefing + heatmap into the dashboard at `localhost:5050/global`.

**Files:**
- Modify: `india_quant/dashboard/app.py:138` (the existing `global_context_page` route)
- Create: `india_quant/dashboard/templates/global_v2.html`
- Keep: `india_quant/dashboard/templates/global_context.html` (the OLD template) until Phase 7 cleanup.

**Strategy:** Replace the route's body to call our new pipeline. Render `global_v2.html`. The old template stays on disk but is no longer referenced — Phase 7 removes it.

- [ ] **Step 1: Audit the existing route to understand what it currently passes to the template**

Run: `cd /Users/pushkaraj/Documents/Trading && sed -n '135,200p' india_quant/dashboard/app.py`

Note any side effects in the existing route (e.g., logging, debug output). The new route preserves the URL `/global`; it does NOT preserve the old template's variables.

- [ ] **Step 2: Read the existing `global_context.html` to see if it has navigation/header partials we should reuse**

Look for `{% extends "base.html" %}` — we will reuse `base.html` so the dashboard's nav remains consistent.

- [ ] **Step 3: Create `global_v2.html`**

`india_quant/dashboard/templates/global_v2.html`:

```html
{% extends "base.html" %}
{% block content %}
<div class="global-tab">
  <h1>Global · {{ as_of.strftime("%a %d %b %Y · %H:%M IST") }}</h1>

  <section class="briefing-strip">
    {% for tile in briefing.tiles %}
    <div class="tile tile-{{ tile.sentiment }}">
      <div class="tile-label">{{ tile.label }}</div>
      <div class="tile-value">{{ tile.value }}</div>
      <div class="tile-change">
        {% if tile.change_pct > 0 %}+{% endif %}{{ "%.2f"|format(tile.change_pct) }}%
      </div>
    </div>
    {% endfor %}
  </section>

  <section class="predicted-gap">
    <span>Predicted gap:</span>
    <strong>NIFTY {{ "%+.0f"|format(briefing.predicted_gap_bps["NIFTY"]) }} bps</strong>
    ·
    <strong>BANKNIFTY {{ "%+.0f"|format(briefing.predicted_gap_bps["BANKNIFTY"]) }} bps</strong>
  </section>

  <section class="heatmap-section">
    <h2>Correlations · NIFTY & BANKNIFTY vs global</h2>
    {{ heatmap_html|safe }}
  </section>

  {% if data_warnings %}
  <section class="data-warnings">
    {% for w in data_warnings %}<p>⚠ {{ w }}</p>{% endfor %}
  </section>
  {% endif %}
</div>

<style>
.briefing-strip { display: flex; gap: 8px; flex-wrap: wrap; margin: 16px 0; }
.tile { padding: 8px 12px; border-radius: 6px; min-width: 100px; }
.tile-bullish { background: #1b5e20; color: white; }
.tile-bearish { background: #b71c1c; color: white; }
.tile-neutral { background: #424242; color: white; }
.tile-label { font-size: 11px; opacity: 0.8; }
.tile-value { font-size: 16px; font-weight: 600; }
.tile-change { font-size: 12px; }
.predicted-gap { padding: 8px; background: #263238; color: white; border-radius: 6px; margin-bottom: 16px; }
.heatmap-section { margin-top: 24px; }
.data-warnings p { color: #ff8a65; }
</style>
{% endblock %}
```

- [ ] **Step 4: Rewrite the route in `app.py`**

Replace the body of `global_context_page` (around `app.py:139`) with:

```python
@app.route("/global")
def global_context_page():
    from datetime import datetime, date
    from india_quant.signals.global_context import get_global_context
    from india_quant.data.fetchers.gift_nifty_fetcher import fetch_gift_nifty_quote
    from india_quant.global_tab.briefing import build_briefing
    from india_quant.global_tab.correlation import build_heatmap
    from india_quant.global_tab.heatmap_view import render_heatmap_html
    from india_quant.dashboard import data as ddata

    as_of = datetime.now()
    warnings: list[str] = []

    try:
        ctx = get_global_context()
    except Exception as exc:
        logger.warning("global_context fetch failed: {}", exc)
        ctx = type("EmptyCtx", (), {"signals": []})()
        warnings.append("Live global signals unavailable; tiles fall back to —")

    gift = fetch_gift_nifty_quote()
    if gift is None:
        warnings.append("GIFT Nifty source unreachable; tile shows —")

    briefing = build_briefing(as_of=as_of, context=ctx, gift_nifty=gift)

    try:
        history = ddata.load_global_history(lookback_days=120)  # see Step 5
        heatmap = build_heatmap(history=history, as_of=date.today())
        heatmap_html = render_heatmap_html(heatmap)
    except Exception as exc:
        logger.warning("heatmap build failed: {}", exc)
        heatmap_html = '<div class="heatmap-empty">Heatmap unavailable today.</div>'
        warnings.append("Correlation heatmap unavailable; check DB connectivity")

    return render_template(
        "global_v2.html",
        as_of=as_of,
        briefing=briefing,
        heatmap_html=heatmap_html,
        data_warnings=warnings,
    )
```

- [ ] **Step 5: Add `load_global_history` helper in `dashboard/data.py`**

Open `india_quant/dashboard/data.py`. Append a function:

```python
def load_global_history(lookback_days: int = 120):
    """Load wide history frame for correlation heatmap.

    Returns a DataFrame indexed by date with columns
    [NIFTY, BANKNIFTY, SPX, NASDAQ, DXY, BRENT, INDIA_VIX, US10Y, NIKKEI, HSI, FTSE].
    Missing tickers result in missing columns (the heatmap skips them).
    """
    import pandas as pd
    from datetime import date, timedelta
    from india_quant.data.db import get_session
    from india_quant.data.models import PriceData
    # global_signals model name varies; adjust import as needed after audit.
    from india_quant.signals.global_context import GROUPS

    end = date.today()
    start = end - timedelta(days=lookback_days * 2)  # buffer for weekends/holidays

    # Map our column names to source tickers.
    source_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SPX": "^GSPC",
        "NASDAQ": "^IXIC",
        "DXY": "DX-Y.NYB",
        "BRENT": "BZ=F",
        "INDIA_VIX": "^INDIAVIX",
        "US10Y": "^TNX",
        "NIKKEI": "^N225",
        "HSI": "^HSI",
        "FTSE": "^FTSE",
    }

    series: dict[str, pd.Series] = {}
    with get_session() as s:
        for col_name, ticker in source_map.items():
            rows = (
                s.query(PriceData.date, PriceData.close)
                .filter(PriceData.ticker == ticker)
                .filter(PriceData.date >= start)
                .order_by(PriceData.date)
                .all()
            )
            if not rows:
                continue
            idx = pd.DatetimeIndex([r.date for r in rows])
            vals = [float(r.close) for r in rows]
            series[col_name] = pd.Series(vals, index=idx)

    if not series:
        return pd.DataFrame()

    df = pd.concat(series, axis=1)
    df = df.sort_index()
    return df
```

If the actual model class for global signals is different (e.g., `GlobalSignal` not `PriceData`), the implementer adjusts the query. Schema audit step:

```bash
cd /Users/pushkaraj/Documents/Trading && grep -nE "^class.*Base|^class.*Mapped" india_quant/data/models.py | head -20
```

- [ ] **Step 6: Verify the route renders without exceptions**

Run:

```bash
cd /Users/pushkaraj/Documents/Trading && venv/bin/python -c "
from india_quant.dashboard.app import main as create_app
"
```

(Or whatever the actual entry point is — adjust based on `app.py`.)

If imports succeed, no syntax errors. Smoke test in Task 7 will exercise the route handler.

- [ ] **Step 7: Commit**

```bash
cd /Users/pushkaraj/Documents/Trading && git add india_quant/dashboard/app.py india_quant/dashboard/data.py india_quant/dashboard/templates/global_v2.html && git commit -m "feat(global_tab): rewrite /global route with briefing + heatmap"
```

---

## Task 7: Route smoke test

**Files:**
- Create: `tests/global_tab/test_route_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
"""Smoke test for the /global Flask route.

Uses the Flask test client with mocked external calls (DB, get_global_context,
GIFT Nifty fetcher) so the test is offline.
"""
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote


@pytest.fixture
def client():
    from india_quant.dashboard.app import build_app  # adjust if factory differs
    app = build_app()
    app.config["TESTING"] = True
    return app.test_client()


def _fake_global_ctx():
    rows = []
    ctx = MagicMock()
    ctx.signals = rows
    return ctx


def _fake_history():
    idx = pd.date_range(end=date(2026, 5, 4), periods=80, freq="B")
    cols = ["NIFTY", "BANKNIFTY", "SPX", "NASDAQ", "DXY", "BRENT", "INDIA_VIX", "US10Y", "NIKKEI", "HSI", "FTSE"]
    import numpy as np
    rng = np.random.default_rng(0)
    data = {c: 100 + np.cumsum(rng.normal(0, 0.5, len(idx))) for c in cols}
    return pd.DataFrame(data, index=idx)


def test_global_route_returns_200(client):
    with patch("india_quant.dashboard.app.get_global_context", return_value=_fake_global_ctx()), \
         patch("india_quant.dashboard.app.fetch_gift_nifty_quote", return_value=GiftNiftyQuote(24412.0, 0.6)), \
         patch("india_quant.dashboard.app.ddata.load_global_history", return_value=_fake_history()):
        resp = client.get("/global")
    assert resp.status_code == 200


def test_global_route_renders_tiles_and_heatmap(client):
    with patch("india_quant.dashboard.app.get_global_context", return_value=_fake_global_ctx()), \
         patch("india_quant.dashboard.app.fetch_gift_nifty_quote", return_value=GiftNiftyQuote(24412.0, 0.6)), \
         patch("india_quant.dashboard.app.ddata.load_global_history", return_value=_fake_history()):
        resp = client.get("/global")
    body = resp.data.decode()
    assert "GIFT Nifty" in body
    assert "Predicted gap" in body
    assert "global-heatmap" in body  # Plotly div id


def test_global_route_renders_when_gift_unavailable(client):
    with patch("india_quant.dashboard.app.get_global_context", return_value=_fake_global_ctx()), \
         patch("india_quant.dashboard.app.fetch_gift_nifty_quote", return_value=None), \
         patch("india_quant.dashboard.app.ddata.load_global_history", return_value=_fake_history()):
        resp = client.get("/global")
    body = resp.data.decode()
    assert resp.status_code == 200
    assert "—" in body  # em dash present somewhere (the GIFT Nifty tile)
    assert "GIFT Nifty source unreachable" in body
```

If the Flask app factory differs (no `build_app`), adjust the fixture. The implementer should:

```bash
cd /Users/pushkaraj/Documents/Trading && grep -nE "^def main|^def create_app|^def build_app|app = Flask" india_quant/dashboard/app.py
```

…and use whatever exists. If only `main(host, port)` exists (which calls `app.run()`), refactor to extract a `def build_app() -> Flask` function and have `main()` call `build_app().run(host, port)`. Tests need the bare `Flask` instance.

- [ ] **Step 2: Run smoke tests**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab/test_route_smoke.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
cd /Users/pushkaraj/Documents/Trading && git add tests/global_tab/test_route_smoke.py india_quant/dashboard/app.py && git commit -m "test(global_tab): smoke test for /global Flask route"
```

(Include `app.py` only if a `build_app()` refactor was needed.)

---

## Task 8: Phase 2 demo gate

**Files:** none modified. Run the suite, then run the dashboard manually.

- [ ] **Step 1: Full Phase 1 + Phase 2 test suite**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab -v`
Expected: 40 (Phase 1) + ~20 new (Phase 2) = ~60 passed.

- [ ] **Step 2: Coverage check**

Run: `cd /Users/pushkaraj/Documents/Trading && venv/bin/python -m pytest tests/global_tab --cov=india_quant.global_tab --cov-report=term-missing`
Expected: ≥95% on the four Phase-2 modules. The route helper in `dashboard/data.py` is exercised through the smoke test only — coverage there may be lower; fine.

- [ ] **Step 3: Run the dashboard live**

Open Docker / verify Postgres is up, ensure `.env` is filled, then:

```bash
cd /Users/pushkaraj/Documents/Trading && venv/bin/python main.py --pipeline   # if data not yet backfilled today
cd /Users/pushkaraj/Documents/Trading && venv/bin/python main.py --dashboard
```

Open `http://localhost:5050/global` in a browser. Confirm:
- 10 briefing tiles render in order: SPX, Nasdaq, GIFT Nifty, DXY, Brent, India VIX, US 10Y, Nikkei, Hang Seng, FTSE.
- Predicted gap line shows non-zero values (or "0 bps" if GIFT Nifty fetch failed).
- The Plotly heatmap renders with up to 18 cells (2 rows × 9 cols, GIFT Nifty deliberately excluded from the heatmap in this phase).
- Hover on a heatmap cell reveals 20d and 60d rho.
- If any data source is unavailable, a `data-warnings` block at the bottom names the issue.

- [ ] **Step 4: NO commit**

Phase 2 demo gate is a verification step only.

---

## Self-Review

**Spec coverage (Phase 2 scope, spec §11.2):**
- Audit `fetchers/` and `data/backfill_global.py` — Task 1 handles Brent gap, Task 2 handles GIFT Nifty gap. ✓
- `briefing.py` — Task 3 ✓
- `correlation.py` — Task 4 ✓
- "Top strip + heatmap fed by real artifacts" — Tasks 5–6 (rendered live from DB rather than parquet artifacts; documented deviation). ✓

**Out of scope (explicit deferral):**
- Trade-ticket cards (Phase 3).
- Backtest sub-tab (Phase 4).
- Intraday checkpoint scripts (Phase 5).
- LLM narrator (Phase 6).
- launchd scheduler config + parquet artifacts (Phase 7).

**Placeholder scan:** No `TBD` / `TODO` / `implement later`. The `predicted_gap_bps` formula is explicitly labelled placeholder in `briefing.py` docstring (correct — Phase 3 replaces it).

**Type consistency:** All Phase-2 modules consume Phase-1 types unchanged. New types: `GiftNiftyQuote` (in fetcher, frozen dataclass).

**Plan-level risks logged for the user:**

- **GIFT Nifty scrape** is fragile (regex against unstable HTML). If parsing fails, briefing tile shows `—` and the system continues. Plan acknowledges this by making the test contract loose (`> 20000 and < 30000`) and gating the fetcher behind `try/except` in the route. If the implementer hits a hard wall (no HTML source containing the price), they escalate as BLOCKED and we either pick a different source or defer.
- **Existing model schema** for `global_signals` is not fully audited — Task 6 Step 5 has a fallback grep instruction. Implementer may need to adjust `load_global_history` after seeing real model class names.
- **Plotly dependency**: present transitively; pinning explicitly is recommended in Task 5 Step 3.
