# Global Tab Phase 7 — Make Trade Cards Actually Fire

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the dashboard currently emits NO_TRADE on essentially every card every day. This is *honest* (the strategies as designed have weak/negative edge given current data), but it's also useless. Phase 7 redesigns the strategy mix so cards fire on most days with positive expected value, and adds a backtester so we never again ship a strategy whose live behaviour is unknown.

---

## 0. Diagnosis — why NO_TRADE persists (verified live, 2026-05-07)

### Issue A: directional model has marginal edge

After all phases through 3c+5c, the LightGBM direction artifact gives:

```
NIFTY     proba_up = 0.515    (ratio to 0.5: +1.5pp)
BANKNIFTY proba_up = 0.498    (ratio to 0.5: -0.2pp)
```

OOS direction logloss across 5 walk-forward folds was **0.674** (NIFTY), **0.684** (BANKNIFTY) vs always-up baseline **0.693**. That's a tiny edge.

Mode dead zones:
- aggressive ±0.02 → trades when |proba-0.5| > 0.02
- balanced ±0.05 → trades when |proba-0.5| > 0.05
- conservative ±0.10 → trades when |proba-0.5| > 0.10

Empirically, |proba-0.5| > 0.05 on a small minority of sessions, so balanced fires rarely. The model is a probability calibrator, not a strong directional engine — overnight global signals (SPX/DXY/VIX/Brent) just don't carry enough information about NIFTY's next session.

**The fix is NOT to lower the threshold.** That converts NO_TRADE into negative-EV trades. The fix is to **trade more conservatively when the signal is weak — scale position size with confidence instead of going binary.**

### Issue B: long-volatility is structurally negative EV most days

Today: India VIX = 17.1%, realized 5-day vol forecast = 10.4%. The straddle gate `forecast > implied × 1.15` requires forecast > 19.7%. Forecast is barely 60% of that.

This isn't a bug. The **volatility risk premium (VRP)** is a documented anomaly: VIX > realized vol most of the time because (a) options sellers demand a premium for tail risk and (b) end-investors systematically overpay for downside protection. Across global markets, VIX averages ~3-5 vol points above realized.

**Buying volatility is structurally negative EV most days.** The right strategy here is **selling volatility** — capture the VRP — not buying it.

### Issue C: option_chain table is empty

Every directional card that *would* fire instead emits `data_gap` because the orchestrator can't find ATM strikes to size. NSE only retains ~21 days of historical chains, so this is a forward-only problem — start logging today, accumulate the buffer.

---

## 1. Strategy redesign

Phase 7 reframes the dashboard around **three structural edges that are real**:

| Phase | Strategy | Edge | Trigger | Risk |
|------|---------|------|---------|------|
| 7a | **Backtester** (cross-cutting) | n/a | n/a | n/a |
| 7b | **Iron condor / short straddle** | Volatility risk premium (sell vol) | Implied > forecast × buffer (inverse of Phase 6) | Defined via wings |
| 7c | **Dynamic-sizing directional** | Marginal directional information | Always fires; size scales with \|proba-0.5\| | Smaller premium per trade |
| 7d | **Multi-day swing** (separate model) | 5-day momentum + reversion | New direction model on 5-day labels | Larger premium, fewer trades |
| 7e | **Option chain ingestion** | (enabler, not strategy) | n/a | n/a |

Each phase is independently shippable. 7a (backtester) is sequenced first because **everything downstream needs it for acceptance**.

---

## 2. File Structure

**New files:**
- `india_quant/global_tab/backtester.py` — `Backtester` class. Replays `data/artifacts/global_tickets/*.jsonl` (the snapshot logger) against the actual realised next-session moves. Outputs hit-rate, win-loss, Sharpe, profit factor, max drawdown per strategy kind.
- `india_quant/global_tab/condor_strategy.py` — `build_iron_condor_ticket(...)` — sibling of `vol_strategy.py` but shorts vol. Sells ATM call+put, buys protective wings at ±X% from spot.
- `india_quant/global_tab/dynamic_sizer.py` — `dynamic_size_directional(forecast, capital, mode) -> int_lots`. Replaces the all-or-nothing path: lots = min_lots × clip((|proba-0.5| - threshold) × scale_factor + 1, 1, max_lots).
- `india_quant/global_tab/swing_artifact.py` — separate LightGBMArtifact-shaped class predicting 5-day-ahead returns. Different labels, same training infra.
- `scripts/train_swing_forecaster.py` — `--horizon-days 5` variant of `train_global_forecaster.py`.
- `scripts/backtest_global_tab.py` — CLI that runs `Backtester.replay(...)` against the audit trail and writes `data/artifacts/backtest_YYYY-MM-DD.json`.
- `scripts/option_chain_ingestor.py` — hourly snapshot of NSE option chain → `option_chain` table. Forward-only. Add to scheduler in 7e.

**Modified files:**
- `india_quant/global_tab/types.py` — add `kind: Literal[..., "iron_condor", "swing"]`. New dataclass `IronCondorLeg(short_call, short_put, long_call_wing, long_put_wing, ...)`.
- `india_quant/global_tab/orchestrator.py` — produce up to 4 cards per index per request: directional + straddle + condor + swing. Skip a card kind if its prerequisite (model, chain) isn't met.
- `india_quant/global_tab/options_sizer.py` — replace size_trade()'s binary lots-or-None with dynamic sizing.
- `india_quant/global_tab/narrator.py` — blurbs for condor and swing kinds.
- `india_quant/dashboard/templates/_card.html` — render condor (4 strikes) and swing (longer-dated) variants.
- `india_quant/scheduler.py` — wire `option_chain_ingestor.py` to run hourly during NSE hours.

---

## 3. Conventions

- Run pytest as `PYTHONPATH=. venv/bin/python -m pytest …`.
- All strategies share the per-mode `max_loss_fraction` (already in `MODE_CONFIGS`). New strategies must respect it.
- All new strategies must have a Backtester acceptance gate before live use (see §5).
- Card ordering in `view.cards`: directional → straddle → condor → swing per index. Stable so the JSON consumer can index by position.

---

## 4. Tasks

### Task 7a: Backtester (do this FIRST — it gates everything else)

**Files:** `india_quant/global_tab/backtester.py`, `scripts/backtest_global_tab.py`, `tests/global_tab/test_backtester.py`

- [ ] **Step 1:** `class Backtester` with constructor `(snapshot_dir: Path, price_data_session_factory)`. Loads JSONL snapshot files from `data/artifacts/global_tickets/`, joins each `(as_of, index)` to actual next-session NIFTY/BANKNIFTY close.
- [ ] **Step 2:** Per-card P&L computation:
  - Directional: realised P&L = lots × lot_size × (exit_premium − entry_premium). For now, model exit_premium with intrinsic-only (max(spot_at_close − strike, 0) for call). Add slippage later.
  - Straddle: realised P&L = total_premium_received − max(0, |spot_at_close − strike|) per unit, signed correctly for long.
  - Iron condor: + total_credit at entry, − max(0, breach × lot_size) at expiry.
  - Swing: same as directional but over 5-day exit.
- [ ] **Step 3:** Aggregate metrics: hit-rate (% positive P&L), avg win, avg loss, profit factor, Sharpe annualized, max drawdown. Per strategy kind, per mode, per index.
- [ ] **Step 4:** CLI `scripts/backtest_global_tab.py --since YYYY-MM-DD --kind directional --kind straddle` → writes JSON to `data/artifacts/backtest_<date>.json` and prints a Markdown table to stdout.
- [ ] **Step 5:** Tests: synthetic JSONL fixture with known card outcomes; assert Sharpe and hit-rate match hand-computed values.
- [ ] **Step 6:** **Acceptance gate:** the backtester must report metrics for *all 4* card kinds with both real and synthetic data before any 7b-7d code is shipped. No more flying blind.

### Task 7b: Iron condor / short straddle (the VRP play)

**Files:** `india_quant/global_tab/condor_strategy.py`, `india_quant/global_tab/types.py`, `india_quant/global_tab/orchestrator.py`, `tests/global_tab/test_condor_strategy.py`

- [ ] **Step 1:** New dataclass `IronCondorLeg(strike_short_call, strike_short_put, strike_long_call, strike_long_put, expiry, lot_size, lots, total_credit, max_loss, breakeven_high, breakeven_low, vol_forecast_pct, vol_implied_pct)`.
- [ ] **Step 2:** `build_iron_condor_ticket(*, index, spot, vol_forecast_pct, vol_implied_pct, mode, capital, expiry, as_of, chain=None) -> TradeTicket(kind="iron_condor")`.
  - Trigger: `vol_implied > vol_forecast × buffer` (inverse of straddle). Buffers: aggressive 1.10, balanced 1.20, conservative 1.40.
  - Strikes: short call/put at ATM ±0.5σ_implied×spot; long wings at ATM ±1.5σ_implied×spot. (Adjust to nearest strike step.)
  - Premiums: live mid from chain when available; else Black-Scholes ATM approx scaled by moneyness.
  - Total credit = (short_call_prem + short_put_prem) − (long_call_prem + long_put_prem).
  - Max loss = (wing distance × lot_size × lots) − total_credit.
  - Lots sized to keep max_loss ≤ capital × max_loss_fraction.
- [ ] **Step 3:** Orchestrator wiring: append condor card after straddle card per index. Card kind disambiguates the renderer.
- [ ] **Step 4:** Narrator: condor blurb shows the four strikes, total credit, max loss, breakeven range, σ ratio.
- [ ] **Step 5:** Tests: vol gate (above/below buffer), strike spacing, defined-loss invariant (max_loss > 0 and finite), Conservative analog hit-rate floor (analog hit-rate within breakeven range ≥ 60%).
- [ ] **Step 6:** **Backtest acceptance gate:** running `scripts/backtest_global_tab.py --kind iron_condor --since 2024-01-01` over the historical training-frame replay must show:
  - Hit-rate ≥ 55% in balanced mode
  - Profit factor ≥ 1.10
  - Max drawdown ≤ 25% of total premium collected
  Ship to live only if all three pass.

### Task 7c: Dynamic-sizing directional (no more binary NO_TRADE)

**Files:** `india_quant/global_tab/dynamic_sizer.py`, `india_quant/global_tab/options_sizer.py`, `india_quant/global_tab/orchestrator.py`

- [ ] **Step 1:** `dynamic_size_directional(proba_up: float, max_lots_for_mode: int, *, mode: Mode) -> int`:
  - confidence = abs(proba_up - 0.5) × 2 ∈ [0, 1]
  - Soft floor: `lots = round(max_lots × clip((confidence - 0.05) × 4, 0.25, 1.0))`. So even confidence=0.05 fires 0.25× max_lots.
  - mode caps the maximum: aggressive max_lots=10, balanced=5, conservative=2 (subject to capital cap).
- [ ] **Step 2:** Replace `forecast_index()`'s NO_TRADE branch — keep returning a Direction (LONG or SHORT based on whichever side proba leans), and pass confidence through. The "ambiguous" case (proba exactly = 0.5) is the only true NO_TRADE.
- [ ] **Step 3:** Orchestrator: drop the `no_overnight_catalyst` branch. Now directional cards always fire (modulo data_gap and chain availability).
- [ ] **Step 4:** Tests: confidence 0.0 → 25% size; 0.20 → ~80% size; 0.50 → 100%; mode caps respected.
- [ ] **Step 5:** **Backtest acceptance gate:** dynamic-sizing version must beat the binary version on *risk-adjusted* Sharpe over the 2024-2026 window. If Sharpe is worse, the binary path was correctly cautious — keep it. Document the result either way.

### Task 7d: Multi-day swing artifact

**Files:** `india_quant/global_tab/swing_artifact.py`, `scripts/train_swing_forecaster.py`, `india_quant/global_tab/training_features.py` (new label), `india_quant/global_tab/orchestrator.py`

- [ ] **Step 1:** Add new label to `assemble_training_frame`: `label_5d_return_bps = log(close[t+5]) − log(close[t])` × 10000 and `label_5d_direction = 1 if positive`. Drops trailing 5 rows.
- [ ] **Step 2:** `class SwingArtifact` mirrors `LightGBMArtifact` but loaded from `{INDEX}_swing_direction.pkl` etc. Quantile regressors at q10/q50/q90 on the 5-day return.
- [ ] **Step 3:** `scripts/train_swing_forecaster.py` reuses Phase 3c trainer + OptunaSweep wrapper, swapping in the 5-day labels. Different sqlite storage URI to avoid cross-contamination.
- [ ] **Step 4:** New strategy: `build_swing_ticket(...)` — long ITM call (delta ~0.7) for LONG, long ITM put for SHORT, expiring 5-7 sessions out. Holding period 5 sessions; status flips honor that horizon (extend `compute_status` to handle multi-day windows).
- [ ] **Step 5:** **Backtest acceptance gate:** swing artifact OOS direction logloss must beat 0.674 (NIFTY 1-day baseline). If swing labels carry more signal than 1-day, this should pass easily — 5-day moves are more predictable than 1-day.
- [ ] **Step 6:** Orchestrator: 4th card per index. Stays a NO_TRADE when artifact missing or chain gap.

### Task 7e: Option chain ingestion (the enabler)

**Files:** `scripts/option_chain_ingestor.py`, `india_quant/data/fetchers/nse_options_fetcher.py` (already exists — verify), `india_quant/scheduler.py`

- [ ] **Step 1:** Verify `nse_options_fetcher.py` works end-to-end against NSE's public option-chain JSON endpoint. Test with a sample index NIFTY ATM ±10 strikes.
- [ ] **Step 2:** `scripts/option_chain_ingestor.py` — fetches NIFTY + BANKNIFTY chains for the next 2 weekly expiries, upserts into `option_chain` table. Idempotent on (index, expiry, strike, option_type, snapshot_at).
- [ ] **Step 3:** Scheduler hook: run every 30 minutes during NSE hours (09:15-15:30 IST, Mon-Fri). Off-hours: skip.
- [ ] **Step 4:** Backfill: forward-only — start running today and accumulate. After 30 days the dashboard will have meaningful chain coverage for backtests.
- [ ] **Step 5:** Acceptance: `option_chain` table has > 0 rows after one full trading day. The directional card's `data_gap` reason should disappear within 24 hours of starting the ingestor.

---

## 5. Acceptance criteria (Phase 7 as a whole)

1. **Backtester ships first** — every subsequent strategy carries its OOS metrics in its commit message. No live deploy without a measured edge.
2. **Iron condor card fires on ≥ 60% of historical sessions in balanced mode** over the 2024-2026 window AND backtest hit-rate ≥ 55% AND profit factor ≥ 1.10.
3. **Dynamic-sizing directional** improves risk-adjusted Sharpe over the binary version, OR documents that the binary was correct and keeps the binary.
4. **Swing artifact** OOS direction logloss < 0.674 (1-day NIFTY baseline). If it doesn't, the 5-day signal isn't there either; document and skip 7d's live deploy.
5. **Option chain ingestor running** with > 1000 chain rows accumulated within a week of going live.
6. **Test count ≥ 230, all green** (200 baseline + ≥30 new).
7. **No commit mentions Claude as co-author** (per longstanding feedback in `~/.claude/projects/.../memory/feedback_no_git_commands.md`).
8. **`/global` page** typically shows ≥ 2 fired (non-NO_TRADE) cards per request, not 0/4.

---

## 6. Risks and honest caveats

- **Short vol is the most dangerous category in trading.** A single jump (e.g., 2008-style, 2024 yen-carry unwind, geopolitical shock) wipes out months of premium. The iron condor's defined wings are the only thing keeping this risk-bounded — no naked short straddles ever, even in aggressive mode.
- **VRP can collapse.** Recent regimes have seen India VIX trade *below* realized for periods — short-vol strategies underperform then. The mode buffer (1.10/1.20/1.40) widens the margin but doesn't prevent regime risk. Mitigation: backtest gate (hit-rate ≥ 55%, max drawdown ≤ 25% of premium) catches regime weakness in historical data.
- **Backtest survivorship bias.** Replaying audit-trail snapshots only captures decisions we *would have* made with current code. To audit the strategies *historically*, we have to back-fill features for past dates and run the orchestrator against each — implemented inside the Backtester (replay mode) rather than just JSONL playback. Plan calls this out.
- **Over-fitting risk on the swing artifact.** 5-day labels are smoother but the same OOS-CV pattern applies. Optuna sweep with `seed=42` and the Phase 3c reproducibility gate carry over. If swing OOS logloss < 0.5 on a held-out fold, that's *too good to be true* and likely a leak — investigate before shipping.
- **Option chain ingestion is rate-limited.** NSE blocks aggressive scraping. Use Scrapling with realistic headers (the project's documented default), and respect the 30-minute cadence. Don't try to backfill historical chains — they're not retained anyway.

---

## 7. Out of scope (Phase 8+ candidates)

- **Pair trading (NIFTY vs BANKNIFTY)** — different beta exposure, mean-reverting spread strategy. Needs cointegration analysis.
- **News/sentiment-triggered overrides** — when a Fed/RBI/big-corp event drops, override the model with a "sit out" signal. Requires news ingestion.
- **Live broker integration** — Angel SmartAPI exists but isn't wired to actual order entry. Big project; requires paper-trade verification first.
- **Per-stock cards** — currently global-tab is index-only. A "top movers" card model (e.g., long-short on factor signals) is a distinct surface area.
- **Greeks-based hedge** — once positions are live, dynamic delta hedging across the book. Way out.

---

## 8. Code-level pointers for the next session

The following invariants from previous phases must be preserved:

- `FEATURE_COLUMNS[:11]` order is byte-stable (Phase 3a-3c reproducibility). New columns append.
- `LightGBMArtifact._load_index` validates `n_features_in_` — any new artifact must do the same (already pattern-matched in `LightGBMVolArtifact`).
- `TradeTicket.kind` discriminates renderers; new kinds (`iron_condor`, `swing`) just need a new branch in `_card.html` and `narrator.py`.
- `AnalogIndex` exposes `lookup` (directional) and `lookup_breakeven` (vol). Add `lookup_in_range(features, low_bps, high_bps, k)` for iron condor.
- The snapshot logger writes every served view to `data/artifacts/global_tickets/YYYY-MM-DD.jsonl`. The Backtester replays this *plus* re-runs the orchestrator over historical dates for full coverage.
- Memory pointers worth preserving for the next session:
  - `~/.claude/projects/.../memory/feedback_no_git_commands.md` — never use Co-Authored-By: Claude
  - `~/.claude/projects/.../memory/feedback_python_scraping_scrapling.md` — use Scrapling for option_chain ingestion
- All run commands compiled in CLAUDE.md → "Layer 4b — Global Tab" section.

Suggested execution order:
```
Day 1: 7a (Backtester) — purely testable, no live data dependency
Day 2: 7e (Chain ingestor) — start the data clock running in parallel
Day 3: 7b (Iron condor) — once 7a is shipped you can measure
Day 4: 7c (Dynamic sizing) — refactor with backtest gate
Day 5: 7d (Swing artifact) — biggest model build
```

Each phase is a separate set of commits. Do not commingle.
