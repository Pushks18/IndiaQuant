# Global Tab Revamp — Phase 3d (Feature Engineering) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift LightGBM forecaster magnitude q50 by ≥ 15% (the unmet Phase 3c target) and squeeze further direction logloss by adding Indian-market-specific features the current 11-column matrix doesn't capture: **sectoral breadth**, **cross-sectional advance/decline**, **factor-tilt aggregates** from `factor_scores`, and **relative-strength regimes**. No protocol changes — same `LightGBMArtifact` path, just a wider `FEATURE_COLUMNS` list and a richer `assemble_training_frame`.

**Spec reference:** `docs/superpowers/specs/2026-05-05-global-tab-revamp-design.md` §6.2. Picks up the Phase 3c plan §5b acceptance gap on magnitude pinball loss.

---

## 0. Why this is needed

Phase 3c hit the direction baseline (NIFTY 0.675 / BANKNIFTY 0.684 vs 0.693), but **magnitude q50 only dropped 8–12%** vs the 15% target. The 11-feature set is dominated by overnight global signals (SPX/NDX/DXY/Brent/IndiaVIX) and 2 NIFTY trend features. It carries no information about:

- **Where in India the move is concentrated** — IT vs Banks vs Pharma rotation, which signals macro vs idiosyncratic moves.
- **Breadth** — index gains driven by 5 megacaps look very different from broad-based rallies in terms of mean-reversion risk.
- **Cross-sectional dispersion** — high vol-of-vol days (factor regime shifts) systematically widen the realized return distribution.
- **Options-implied risk** — `iv_skew` / `vrp` on the basket level proxies what the options market expects.

These sit in tables we already populate (`factor_scores`, `price_data`, `global_signals`). No new fetchers needed.

---

## 1. Slim slice (additive, not replacing)

`FEATURE_COLUMNS` grows from 11 → ~22. The existing 11 stay byte-identical so any reproducibility gate passing before still passes. New features are additive — if a row has NaN on a new column, the existing imputation (`StubArtifact` median fallback at predict time, `dropna()` at train time) handles it.

---

## 2. File Structure

**Modified files:**
- `india_quant/global_tab/training_features.py` — add 11 new columns to `FEATURE_COLUMNS`, extend `assemble_training_frame` to compute them, extend `to_feature_store` if used.
- `india_quant/global_tab/forecaster.py` — `FeatureRow` gains 11 new optional fields (default `None` so Phase 3a/3b/3c call sites keep working).
- `india_quant/global_tab/lightgbm_artifact.py` — `_vectorize` reads the new fields and treats `None` as median-imputation 0.0 (same convention as Phase 3b).
- `india_quant/global_tab/orchestrator.py` — `_build_features` populates the new fields at serve time from the same DB queries used at train time.

**New files:**
- `tests/global_tab/test_feature_engineering.py` — unit tests for each new feature: shape, no-future-peek, sensible bounds, NaN handling.

**Files NOT touched:**
- `tuning.py` — Optuna ranges still apply; the new features just give Optuna more to choose from.
- `lightgbm_artifact.py`'s public protocol — only the `_vectorize` internal changes.
- Any template, briefing, or sizer code.

---

## 3. New features (the 11 additions)

| Column                          | Source                                 | Why it should help                                 |
|---------------------------------|----------------------------------------|----------------------------------------------------|
| `bank_vs_nifty_5d_relstr`       | `global_signals` `^NSEBANK` − `^NSEI` 5d | Bank-led rallies vs broad rallies behave differently |
| `it_vs_nifty_5d_relstr`         | `global_signals` `^CNXIT` − `^NSEI` 5d  | IT is USD-sensitive; signals DXY transmission       |
| `pharma_vs_nifty_5d_relstr`     | `global_signals` `^CNXPHARMA` − `^NSEI` 5d | Defensive rotation flag                             |
| `realty_vs_nifty_5d_relstr`     | `global_signals` `^CNXREALTY` − `^NSEI` 5d | Rate-sensitive sector — RBI proxy                   |
| `sector_dispersion_5d`          | std of all 6 sector pct_5d              | Rotation/dispersion regime                         |
| `pct_above_20dma`               | `price_data` Nifty-50 universe          | Classic breadth indicator                          |
| `pct_above_50dma`               | `price_data` Nifty-50 universe          | Trend breadth (slower)                             |
| `advance_decline_5d`            | `price_data` count(up) - count(down) over 5d | Breadth momentum                                  |
| `mean_realized_vol_universe`    | `factor_scores.realized_vol` mean       | Volatility regime                                  |
| `mean_iv_skew_universe`         | `factor_scores.iv_skew` mean            | Options market's downside-risk pricing             |
| `mean_oi_flow_universe`         | `factor_scores.oi_flow` mean            | Smart-money positioning proxy                      |

All as-of T-1 close (no future peek). Universe = Nifty-50 constituents present in `factor_scores` for that date.

---

## 4. Conventions

- Run pytest as `PYTHONPATH=. venv/bin/python -m pytest …`.
- New features: snake_case, lowercase, units in name where ambiguous (`_pct`, `_5d`, `_bps`).
- Imputation policy: NaN → 0.0 at predict time; `dropna()` at train time. Same as Phase 3b.

---

## 5. Tasks

### Task 1: Sector-relative-strength + dispersion (4 features)

**File:** `india_quant/global_tab/training_features.py`

- [ ] **Step 1:** Add `_SECTOR_TICKERS = {'NSEBANK': '^NSEBANK', 'CNXIT': '^CNXIT', 'CNXPHARMA': '^CNXPHARMA', 'CNXREALTY': '^CNXREALTY', 'CNXENERGY': '^CNXENERGY', 'CNXINFRA': '^CNXINFRA'}`. These are already in `global_signals` (verified during Phase 3c — counts 366–367 each, 2024-10-25 → 2026-05-05; backfill to 2021 via existing `backfill_global --days 1600` if not already done).
- [ ] **Step 2:** Pull `pct_5d` (already a stored column in `global_signals`) for the 6 sectors + `^NSEI` over the same lookback window as the existing global pull. Pivot wide on `date × ticker`.
- [ ] **Step 3:** Compute 4 new columns per session date:
  - `bank_vs_nifty_5d_relstr = pct_5d['^NSEBANK'] - pct_5d['^NSEI']`
  - `it_vs_nifty_5d_relstr = pct_5d['^CNXIT'] - pct_5d['^NSEI']`
  - `pharma_vs_nifty_5d_relstr`, `realty_vs_nifty_5d_relstr` similarly
  - `sector_dispersion_5d = std([pct_5d for all 6 sectors])`
- [ ] **Step 4:** Append all 4 to `FEATURE_COLUMNS` (preserving existing order — only append; never reorder).

### Task 2: Breadth from price_data (3 features)

**File:** `india_quant/global_tab/training_features.py`

- [ ] **Step 1:** Add a `_NIFTY50_TICKERS` constant — pull from `india_quant.data.fetchers.yfinance_fetcher.YFinanceFetcher.NIFTY_50` (the canonical list; already used for the broad universe). This is the "universe" for breadth.
- [ ] **Step 2:** New helper `_breadth_features(start, end, *, session_factory) -> pd.DataFrame`:
  - Single SQL: `SELECT ticker, datetime, close FROM price_data WHERE ticker IN (...) AND interval='1d' AND datetime BETWEEN :start-90d AND :end+2d`. Note the 90-day lookback for the 50dma window.
  - Pivot to `date × ticker` close matrix; tz-handle the same way Phase 3b's bugfix does (`utc=True` then `tz_convert("Asia/Kolkata")`).
  - For each session date `sd` and each ticker, compute `close > rolling_mean(20)` and `close > rolling_mean(50)`.
  - `pct_above_20dma = mean of the 50 booleans on sd` (range [0, 1]).
  - `pct_above_50dma` similarly.
  - `advance_decline_5d = sum over 5 sessions of sign(daily_return)` averaged across tickers.
- [ ] **Step 3:** Inner-join the breadth frame onto the existing per-session frame inside `assemble_training_frame`.
- [ ] **Step 4:** Append the 3 new columns to `FEATURE_COLUMNS`.

### Task 3: Factor-aggregate features from factor_scores (3 features)

**File:** `india_quant/global_tab/training_features.py`

- [ ] **Step 1:** New helper `_factor_aggregates(start, end, *, session_factory)`:
  - SQL: `SELECT date, AVG(realized_vol) AS mean_realized_vol_universe, AVG(iv_skew) AS mean_iv_skew_universe, AVG(oi_flow) AS mean_oi_flow_universe FROM factor_scores WHERE date BETWEEN :start AND :end AND ticker IN (... Nifty-50 ...) GROUP BY date`.
  - Returns a date-indexed DataFrame with 3 columns.
- [ ] **Step 2:** Left-join onto the per-session frame. Some early dates in the window may have missing factor_scores rows → NaN → dropped by the existing `dropna()` step. Log the drop count so Task 5's acceptance check can spot if the table is too sparse.
- [ ] **Step 3:** Append the 3 columns to `FEATURE_COLUMNS`.

### Task 4: Wire new fields through FeatureRow + LightGBMArtifact + orchestrator

- [ ] **Step 1:** `FeatureRow` (in `forecaster.py`) gains 10 new fields, all `float | None = None` so old call sites compile. (Sector RS×4 + dispersion + 3 breadth + 3 factor-agg = 11; one of them, `sector_dispersion_5d`, is already in scope from Task 1.)
- [ ] **Step 2:** `LightGBMArtifact._vectorize` builds the input vector in `FEATURE_COLUMNS` order, mapping each missing field to 0.0. Implementation is identical to Phase 3b — just a longer dict.
- [ ] **Step 3:** `orchestrator._build_features` populates the new fields at serve time. Most of them reuse the same DB pulls already happening for the briefing/heatmap modules — refactor opportunity, but keep separate for now to keep this PR focused.
- [ ] **Step 4:** Run Phase 3a/3b determinism tests; they must still pass byte-for-byte (the new fields default to None → 0.0, identical to predicting on the old vector padded with zeros).

### Task 5: Tests

**File:** `tests/global_tab/test_feature_engineering.py`

- [ ] **Step 1:** Synthetic-DB fixture extension: add 6 sector tickers and 50 Nifty-50 tickers to the `_seed_synth_db` helper used in `test_train_script.py`. Use a deterministic random walk so breadth metrics are computable.
- [ ] **Step 2:** Unit tests, one per feature:
  - `bank_vs_nifty_5d_relstr` matches a hand-computed expected diff on a known fixture row.
  - `sector_dispersion_5d` is finite and ≥ 0.
  - `pct_above_20dma` is in [0, 1].
  - `advance_decline_5d` flips sign correctly on a fixture where prices reverse.
  - `mean_realized_vol_universe` matches a hand-computed AVG over fixture rows.
- [ ] **Step 3:** No-future-peek: assert that for each session row, no underlying data point used had a timestamp > the session date.
- [ ] **Step 4:** Back-compat: `FeatureRow(gift_nifty_premium_bps=10.0, spx_overnight_pct=0.005, dxy_delta_pct=-0.001, india_vix_delta_pct=0.02, brent_overnight_pct=0.003)` still constructs (legacy 5-field call), and the artifact still predicts.

### Task 6: End-to-end live run + acceptance gate

- [ ] **Step 1:** `PYTHONPATH=. venv/bin/python -m pytest tests/global_tab/ -q` — all green, ≥ 150 tests (145 from 3c + ≥ 5 new feature-engineering tests).
- [ ] **Step 2:** Untuned baseline on the new feature set:
  ```
  PYTHONPATH=. venv/bin/python scripts/train_global_forecaster.py \
    --index NIFTY --start 2021-01-01 --end <today> --seed 42 --out models/global_tab/
  ```
  Compare logloss/pinball to Phase 3c untuned (NIFTY: 0.91 logloss / 30.7 pinball q50). Logloss may regress if the new features are noisy without tuning — that's expected; the value is in step 3.
- [ ] **Step 3:** Tuned run on the new feature set, **fresh Optuna study name** (not the `_42` from Phase 3c, because the search space changes effective dimensionality):
  ```
  PYTHONPATH=. venv/bin/python scripts/train_global_forecaster.py \
    --index NIFTY --start 2021-01-01 --end <today> --seed 42 \
    --tune --n-trials 40 --tune-storage sqlite:///optuna_global_tab_3d.db \
    --out models/global_tab/
  ```
  Bumped from 30 → 40 trials because TPE has a wider effective space (22 features vs 11).
  Repeat for `--index BANKNIFTY`.
- [ ] **Step 4:** Restart dashboard, hit `/global?capital=100000&mode=balanced`. Confirm artifact banner still reads `Forecast: lightgbm`. Run the orchestrator determinism test to confirm prediction byte-equality across two runs of the same `as-of` date.

---

## 6. Acceptance criteria

1. **Magnitude q50 drops ≥ 15%** vs Phase 3c on at least one of {NIFTY, BANKNIFTY}. (NIFTY 29.84 → ≤ 25.36; BANKNIFTY 35.38 → ≤ 30.07.)
2. **Direction logloss does not regress** from Phase 3c (NIFTY ≤ 0.685; BANKNIFTY ≤ 0.694) — adding features shouldn't make a tuned model worse, and if it does, the wrong features are in.
3. **`FEATURE_COLUMNS`** has exactly 22 entries, in the order `[<11 Phase 3b columns>, <4 sector-RS>, <3 breadth>, <3 factor-agg>, <1 dispersion>]`. Preserving the order matters because pickled boosters before this PR were trained on a specific column order.
4. **Test count ≥ 150**, all green.
5. **Back-compat**: a `FeatureRow` constructed with only the original 5 Phase-3a kwargs still works through the LightGBMArtifact predict path (verified by Task 5 step 4).
6. **Pickles + summary** updated under `models/global_tab/`. New `feature_columns` list in `training_summary.json` shows 22 entries.

---

## 7. Risks logged for the user

- **`factor_scores` may be sparse on early 2021 dates.** The factor pipeline started populating in 2019 but coverage thins for new tickers. If the `dropna()` log shows > 200 rows dropped (vs Phase 3c's ~50), the universe is too thin — narrow to "tickers with ≥ 60 days of factor history before the session" or accept a shorter training window. Documented; check before declaring acceptance.
- **`iv_skew` / `oi_flow` may be NULL for many rows.** Options factors require an option_chain populated; this DB has 0 option_chain rows (verified 2026-05-07), which means `iv_skew` was likely computed from a defunct fetcher and may be stale. Mitigation: log fraction of NaN per factor-aggregate column. If > 30%, drop that feature from the column set rather than imputing zero everywhere (zero imputation will let the model fit to "is the column populated?" instead of the actual signal).
- **Look-ahead from rolling windows.** `pct_above_20dma` on session `sd` must use closes through `sd-1`, not `sd`. The implementation MUST shift the rolling-mean by one session. Test step 3 (no-future-peek) catches this if the shift is missing.
- **Universe drift.** Nifty-50 constituents change semi-annually. The hardcoded `NIFTY_50` list in `yfinance_fetcher.py` is point-in-time. For a 2021-2026 training window, ~5 names changed. Acceptable for now (the breadth average is robust to small constituent swaps), but flag it as a Phase 3e concern alongside sentiment.
- **Optuna over a 22-feature space wants more trials.** Bumped to `--n-trials 40` (vs 30 in Phase 3c). Total tuning time roughly doubles per index (~10 min each). Median pruner still trims wisely.
- **Pickle compatibility with old artifacts.** Boosters trained in Phase 3c expect 11 features; after this PR they expect 22. The artifact must NOT silently load an old 11-feature pickle and predict on a 22-feature vector. Add a guard: at load time, assert `booster.n_features_in_ == len(FEATURE_COLUMNS)`. If the user has stale pickles around, retraining is a 5-minute operation.

---

## 8. Out of scope (Phase 3e candidates)

- **Sentiment features.** `sentiment_aggregate` has 46 rows total as of 2026-05-07 — not viable. Phase 3e: backfill sentiment via the FinBERT pipeline over 2021-2026 news, then add `mean_sentiment_universe`, `sentiment_dispersion`, `news_volume_z`.
- **Options-chain features.** `option_chain` has 0 rows — the NSE options fetcher hasn't been running historically. Phase 3e prerequisite: backfill option_chain (NSE only retains ~21 days of historical chains, so this is a forward-only problem — start logging now and accumulate).
- **Intraday vol surface.** Requires intraday `price_data` for ^NSEI/^NSEBANK at 5-min, which the current schema supports but isn't backfilled. Defer.
- **Term structure of US futures.** `YM=F` is in `global_signals`; could add `yield_curve_2_10` from `^TNX` and a 2y proxy. Marginal value given DXY already proxies rate differentials. Skip.
- **Per-stock-level inputs.** Aggregate-only here. A second model (`PerStockArtifact`) that predicts top-mover next-session direction per Nifty-50 name is a separate plan, not a feature-engineering pass on the index forecaster.
