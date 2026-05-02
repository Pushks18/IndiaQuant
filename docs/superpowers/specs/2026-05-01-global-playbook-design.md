# Global Playbook Engine — Design Spec
**Date:** 2026-05-01
**Status:** Approved

---

## Goal

Convert the live cross-market signals already produced by `global_context.py`
into actionable, plain-English trade decisions. For each trading day, the
playbook engine answers four questions in order:

1. **Index call** — what is the Nifty / Bank Nifty / sector-index expected
   move at next open, with confidence?
2. **Sector tilt** — which Indian sectors do today's global signals favour
   (and which to avoid)?
3. **Stock picks** — top 3 stocks per favoured sector, picked from the F&O
   universe, with concrete entry / stop / T1 / T2 levels.
4. **Why** — plain-English rationale citing the leading global signal and
   the matching historical playbook rule.

All four are surfaced in three time horizons, rendered as separate tables:

- **Next-day open** — single-shot, exit by 10:00 IST
- **Intraday** — 5-min ORB style, exit by 15:30 IST
- **5-day swing** — positional

The playbook is layered on top of the existing `/global` dashboard page (new
sections at the top). The existing instrument trade levels and correlation
heatmap stay below as reference.

---

## Scope

In scope:

- `playbook_engine.py` core logic (KNN historical analog, rule matcher,
  conviction tiers, sector tilt, stock picker)
- `playbook_rules.yaml` curated rule set (12–15 rules)
- F&O ticker universe + sector mapping
- New sections in `/global` template + fix existing stale "today" date label
- Offline unit tests + one live integration test

Out of scope (v1):

- Telegram / email delivery of the playbook
- Multi-year backtest of the playbook itself (90-day track record only)
- Real-time intraday refresh (manual reload, TTL is 15 min)
- Position-sizing differentiation by horizon (reuses existing
  `_position_size()` for all three)

---

## Inputs

| Source | Contents | Cadence |
|---|---|---|
| `get_global_context()` | 25 live signals + regime + drivers | TTL-cached 15 min |
| `global_signals` table | 396 days of daily history per ticker | Updated by pre-market pipeline |
| `price_data` table | F&O stock daily bars (90 days each) | Updated by EOD pipeline |
| `playbook_rules.yaml` | Curated rule library | Hand-edited; reloaded each request |
| F&O universe with sector mapping | ~180 tickers × sector | Hand-curated YAML |

---

## Architecture

```
yfinance (live signals)         global_signals (DB, 396 days)
        │                               │
        └────────┬──────────────────────┘
                 ▼
        playbook_engine.py  ◄── playbook_rules.yaml  (curated)
        │
        ├── time_ordered_signal_vector()    weight by recency / independence
        ├── knn_lookup()                    n similar past days → Nifty open distribution
        ├── match_rules()                   text rules for current signal vector
        ├── reconcile_disagreement()        IQR width → CONVICTION level
        └── stock_picker()                  rank F&O stocks by sector_match × tech_score
                 │
                 ▼
        PlaybookCall objects ─► /global page (3 new sections, expand-on-click stats)
```

**Key boundary:** `playbook_engine.py` is pure logic. It reads from
`global_signals` and `price_data` but writes nothing. The dashboard route
calls it on every page load; aggressive caching of expensive pieces happens
inside `global_context.py` (already in place) and on the F&O daily-bar fetch
(new 24-hour memo).

---

## File Changes

| File | Action | Responsibility |
|---|---|---|
| `india_quant/signals/playbook_engine.py` | **Create** | Core engine: KNN, rule match, conviction, sector tilt, stock picker |
| `india_quant/signals/playbook_rules.yaml` | **Create** | 12–15 curated rules with thresholds + plain-English templates |
| `india_quant/data/fo_universe.py` | **Create** | F&O ticker list + sector mapping |
| `india_quant/signals/global_context.py` | **Modify** | Add `time_ordered_signal_vector()` helper |
| `india_quant/dashboard/app.py` | **Modify** | `/global` route calls `generate_playbook()` |
| `india_quant/dashboard/templates/global_context.html` | **Modify** | Include playbook sections at top, fix date label |
| `india_quant/dashboard/templates/_playbook_section.html` | **Create** | Partial template for index call + sector tilt + 3 horizon tables |
| `tests/test_integration.py` | **Modify** | Add ≥8 new offline tests + 1 live integration test |

---

## Data Flow per Request

```
GET /global  (or 08:00 IST scheduled run)
    │
    ▼
1. ctx = get_global_context()                    [TTL 15 min, fetched fresh from yfinance]
    │   25 live signals, regime, drivers, USDINR
    │
    ▼
2. signal_vec = time_ordered_signal_vector(ctx)
    │   {"Asia": +0.6, "US": +0.4, "Europe": -0.2, "FX": +0.1, "Comm": -0.3}
    │   weights: Asia 0.50, US 0.30, FX 0.15, Europe 0.05
    │
    ▼
3. knn = knn_lookup(signal_vec, k=8)             [reads global_signals + price_data]
    │   {n_similar: 8, median_open: +0.18%, iqr: [-0.12%, +0.42%], hit_rate_long: 0.62}
    │
    ▼
4. matched_rules = match_rules(ctx, signal_vec)  [reads playbook_rules.yaml]
    │   ["asia_lead_bullish_v1", "dxy_neutral_v2"]
    │   each carries plain-English template, sector tilt, hit-rate stat
    │
    ▼
5. conviction = reconcile(knn, matched_rules)
    │   HIGH / MEDIUM / LOW (MIXED)
    │
    ▼
6. sector_ranks = sector_tilt(matched_rules, ctx)
    │   [("BANK", +0.8), ("IT", +0.4), ("PHARMA", -0.3), ...]
    │
    ▼
7. for horizon in (open, intraday, swing):
    │   stock_picks = pick_stocks(top_sectors, fo_universe, horizon)
    │   for each pick: entry/stop/T1/T2 via existing _build_orb_levels()
    │
    ▼
8. PlaybookCall renders into /global template (3 new sections)
```

**Performance budget:** total flow < 3s when global signals are cached.
KNN over 396 days × 6 z-scored dimensions is trivial (<50 ms).

---

## Time-Ordered Signal Vector

Signals arrive in chronological order each session:

| Market | Close (IST) | Independence weight |
|---|---|---|
| London (FTSE, DAX) | 21:30 yesterday | 0.05 |
| US (S&P, Nasdaq) | 02:30 today | 0.30 |
| FX / DXY / USDINR | continuous → 09:00 today | 0.15 |
| Asia (Nikkei, HSI, KOSPI) | 12:30 today (Nikkei 11:30) | 0.50 |

Asia gets the highest weight because by the time Tokyo closes, it has
**already absorbed** London's previous close and the US session, plus its
own reaction. Weighting Asia higher is the engineering equivalent of
"the most recent independent signal dominates".

`time_ordered_signal_vector()` returns a 6-tuple z-scored against the
396-day history:

```python
(sp500_z, nikkei_z, dxy_z, usdinr_z, vix_z, crude_z)
```

Z-scoring removes scale differences across dimensions so KNN distances are
comparable.

---

## KNN Historical Analog

**Distance:** Euclidean over the 6-dimensional z-scored signed-signal vector.

**Neighborhood:** k = 8. Reject any neighbor with distance > 3.0 (≈ 3 std
dev across all dimensions); if fewer than 4 valid neighbors remain,
conviction forced to LOW.

**Target metric:** for each neighbor, look up the **next trading day's**
Nifty (^NSEI) open return — `gap = open / prev_close - 1`. Returns the
empirical conditional distribution.

**Output:**

```python
@dataclass
class KNNResult:
    n_similar:           int
    median_open_pct:     float       # % expected Nifty open move
    iqr_low:             float       # 25th percentile
    iqr_high:            float       # 75th percentile
    hit_rate_long:       float       # fraction with positive open
    representative_dates: list[date] # the 8 historical analog dates
```

`representative_dates` is surfaced in the UI so the user can sanity-check
("show me Apr 18, that day was similar").

---

## Rule Format

`playbook_rules.yaml`:

```yaml
- id: asia_lead_bullish_v1
  triggers:
    nikkei_pct_1d: { min: 0.5 }
    sp_pct_1d:     { min: 0.0 }
    vix_price:     { max: 18 }
  sector_tilt:
    BANK:   +0.8
    IT:     +0.6
    REALTY: +0.4
    PHARMA: -0.2
  template: >
    Asia rallied overnight (Nikkei {nikkei_pct_1d:+.1f}%, HSI {hsi_pct_1d:+.1f}%).
    US closed firm with VIX subdued. Historical lead: Bank Nifty opens
    {median_open_pct:+.2f}% on {n_similar}/{k} similar days. Favour
    HDFCBANK / ICICIBANK setups; avoid pharma defensives.
  refs: [Patel & Shah 2014, NSE VECM 2019]
```

A rule fires when **all** triggers match (AND semantics). Multiple rules
can fire simultaneously; sector tilts are summed across firing rules.

The shipped rule library covers ~12–15 conditions:

- `asia_lead_bullish` / `asia_lead_bearish`
- `us_strong_inr_weak` / `us_weak_inr_strong`
- `dxy_breakout_em_pressure`
- `crude_spike_inflation_risk`
- `vix_spike_risk_off`
- `vix_low_complacent_bullish`
- `mixed_signals_neutral`
- (others driven by historical evidence)

---

## Conviction Tiers

| Tier | Trigger | UI | Suggested size |
|---|---|---|---|
| HIGH | `\|median_open\| > 0.4%` AND `IQR width < 0.6%` AND `hit_rate > 0.6` | green badge | full |
| MEDIUM | `\|median_open\| > 0.2%` AND `IQR width < 1.0%` | yellow badge | half |
| LOW (MIXED) | else | grey badge | skip / individual setups only |

When LOW, the playbook section explicitly says "MIXED — no directional
play, look at individual setups only" and skips index / sector / stock
calls.

---

## Sector Tilt

Sum the `sector_tilt` map across all firing rules. Result:

```python
{"BANK": +1.2, "IT": +0.6, "PHARMA": -0.5, "REALTY": +0.4, ...}
```

Top 3 sectors by positive tilt → favoured (long bias). Bottom 2 by
negative tilt → avoid (or short bias if sector tilt < −0.5).

---

## Stock Picker

For each favoured sector:

1. Filter F&O universe to stocks in that sector.
2. Run existing `_score()` from `screener.py` (already produces
   long_score / short_score per stock).
3. Sort by score, take top 3.
4. Compute entry / stop / T1 / T2 via existing `_build_orb_levels()`.
5. Pull 90-day track record from a memoized cache: hit-rate, average
   return, Sharpe (last 90 days only for v1).

Three time horizons differ only in the levels:

| Horizon | Entry trigger | Stop | T1 | T2 | Holding window |
|---|---|---|---|---|---|
| Open | prev_high / prev_low ± 0.05% buffer | ATR × 0.5 | ATR × 1.0 | ATR × 1.8 | exit by 10:00 IST |
| Intraday | 5-min ORB high / low | ATR × 0.4 | ATR × 1.5 | ATR × 2.8 | exit by 15:30 IST |
| Swing (5d) | prev_close + 0.3% conf | ATR × 1.5 | ATR × 3.0 | ATR × 5.0 | exit at +5 trading days or T2 |

All three reuse existing `_build_orb_levels()` with different multipliers;
no new technical-analysis logic added.

---

## F&O Universe and Sector Mapping

`india_quant/data/fo_universe.py` exports:

```python
FO_TICKERS: list[str]        # ~180 NSE F&O eligible stocks (with .NS suffix)
TICKER_SECTOR: dict[str, str]  # {"HDFCBANK.NS": "BANK", ...}
SECTOR_INDEX_TICKER: dict[str, str]  # {"BANK": "^NSEBANK", "IT": "^CNXIT", ...}
```

Source: NSE F&O segment list as of last quarterly review. Sectors aligned
to existing screener `SECTOR_ETFS` keys plus extensions
(AUTO, METALS, FMCG, OILGAS, MEDIA, TELECOM).

The list is hand-curated YAML loaded at module import. Updates require a
PR — out-of-scope for the engine to fetch / parse NSE filings.

---

## UI Changes (`/global` page)

New sections inserted at the top, in this order, with the existing Day
Bias card moved to the second section:

1. **Index Call card** (single most prominent block at top of page)
   - Conviction badge (HIGH / MEDIUM / LOW)
   - Big number: expected Nifty open % move with IQR band
   - Plain-English rationale (template-rendered from matched rules)
   - "Show 8 historical analog dates" expand link

2. **Day Bias card** (existing, demoted from top)

3. **Sector Tilt strip** — horizontal bar chart, sectors ranked by tilt;
   green for favoured, red for avoid

4. **Three stock-pick tables** — one per horizon:
   - Columns: Ticker · Sector · Bias · Entry · Stop · T1 · T2 · R:R · Qty
     · Margin · Max Loss · T1 Profit · 90d hit-rate (expandable chip)
   - Top 3 stocks per top-2 favoured sectors (≤ 6 rows per table)
   - Bias coloring matches existing `/intraday` page conventions

5. **(existing) Global Instrument Trade Levels** — unchanged
6. **(existing) Nifty Stock Setups** — unchanged
7. **(existing) Correlation Heatmap** — unchanged

The "today" date label fixes to read `ctx.fetched_at.strftime('%Y-%m-%d')`
instead of `latest_trading_date()`, which lags behind when the EOD
pipeline hasn't run.

---

## Error Handling

- **Insufficient KNN history** (< 4 neighbors within distance 3): force
  conviction LOW, show "insufficient historical analog" message.
- **Rule YAML parse failure**: log warning, fall back to KNN-only output.
  Page renders.
- **F&O stock fetch failure**: skip that stock, fill with next candidate.
  Don't fail the pick list.
- **`global_signals` empty** (backfill not run): playbook section shows
  "run `python -m india_quant.data.backfill_global` to enable historical
  analog lookup". Rest of `/global` renders normally.
- **`_build_orb_levels()` failure per stock**: skip stock, log.
- **Stale signals** (TTL miss + yfinance down): fall back to last
  persisted `global_signals` row; show "stale" badge with last-fetch
  timestamp.
- **Holiday handling**: KNN ignores Indian holidays in neighbor lookup
  (Nifty open undefined those days).
- **Pre-market hours** (run before 09:15 IST): use yesterday's Nifty
  close for gap calc.

---

## Testing

**Offline unit tests** (no DB required):

- `test_knn_returns_neighbors` — synthetic 50-day history, confirm k=8 lookup
- `test_knn_rejects_far_neighbors` — distance > 3.0 excluded
- `test_rule_match_triggers` — rule with VIX < 18 matches when VIX = 17
- `test_rule_match_no_trigger` — same rule rejected when VIX = 19
- `test_conviction_high_medium_low` — three synthetic inputs hit each tier
- `test_time_ordered_vector_weights` — Asia weight > US weight
- `test_mixed_signal_returns_low_conviction` — opposing signals yield LOW
- `test_sector_tilt_aggregation` — two firing rules, summed correctly
- `test_playbook_renders_when_global_signals_empty` — graceful empty state
- `test_fo_universe_loads` — sector mapping covers all tickers

**Live integration test** (separate suite, requires DB):

- `test_full_playbook_today` — end-to-end against real `global_signals`
  data, asserts a `PlaybookCall` object is produced and contains all
  three horizon tables.

---

## Open Questions (resolved)

| Question | Decision |
|---|---|
| Output granularity | D — index call + sector tilt + ranked stock picks |
| Time horizons | Three: open / intraday / 5-day swing in separate tables |
| Lead-lag method | C — both statistical (KNN) and rule-based playbook |
| Page placement | A — extend `/global` (new sections at top) |
| Conflict reconciliation | D + E — KNN historical lookup + time-ordered weighting + disagreement-band IQR + MIXED flag below threshold |
| Stock universe | A — F&O eligible (~180 stocks) |
| Track record visibility | B — expandable chip per stock pick (90-day window) |

---

## Spec Coverage Checklist

| Requirement | Section |
|---|---|
| Three-layer output (index → sector → stock) | Goal, Stock Picker |
| Three time horizons in separate tables | Goal, Stock Picker |
| Statistical lead-lag via KNN | KNN Historical Analog |
| Rule-based plain-English playbook | Rule Format |
| Time-ordered signal weighting | Time-Ordered Signal Vector |
| Conviction tiers / disagreement band | Conviction Tiers |
| MIXED flag below threshold | Conviction Tiers |
| F&O universe with sector mapping | F&O Universe |
| 90-day expandable track record | UI Changes, Stock Picker |
| `/global` page extended (new sections at top) | UI Changes |
| Date label fixed to `ctx.fetched_at` | UI Changes |
| Error handling for empty `global_signals` | Error Handling |
| Offline unit tests + 1 live integration | Testing |
