# Global Tab Revamp — Phase 3 (Forecaster + Sizer + Orchestrator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First end-to-end "decision cockpit" view of the Global tab. Per-index trade-ticket card (NIFTY, BANKNIFTY) with direction, options leg, R:R, timing window, and a deterministic-template blurb — or an explicit `NO_TRADE` with reason. No LLM blurb yet (Phase 6). No backtest sub-tab (Phase 4). No live status transitions (Phase 5).

**Spec reference:** `docs/superpowers/specs/2026-05-05-global-tab-revamp-design.md` §6.2 (forecaster, options_sizer, narrator, orchestrator), §8 (mode configs), §11.3 (Phase 3 demo state).

---

## 0. Carry-over deviations from spec

These were declared in the Phase 2 plan and continue in Phase 3:

1. **Flask, not Streamlit.** Phase 2 wired `/global` as a Flask route with a 60-second cache; Phase 3 extends the same route. The spec's `global_context_page` Streamlit shape stays a deferred refactor.
2. **Read from TimescaleDB at request time, no parquet artifacts.** The spec's nightly/preopen artifact pipeline (Flow A/B) is deferred to Phase 4 (when the backtester actually needs it). Phase 3 reads `OptionChain`, `global_signals`, and Nifty 50 EOD direct from DB through a thin loader.
3. **Capital + mode via query params**, not Streamlit widgets. `?capital=100000&mode=balanced` (defaults: 100000, balanced).

---

## 1. Slim slice first (Phase 3a → 3b)

**Phase 3a (this plan, Tasks 1–8):** ship a working tab end-to-end with a **stub forecaster**. The stub turns GIFT Nifty premium bps into a direction + confidence (premium > +20bps → LONG; < −20bps → SHORT; else NO_TRADE). Magnitude is a fixed table by mode. This is enough to drive the sizer, render cards, and prove the interfaces are correct.

**Phase 3b (Task 9, separate plan once 3a ships):** replace the stub with a real LightGBM direction classifier + quantile-regression magnitude model trained via the `PointInTimeFeatureStore`. Same `forecast_index()` signature; only the `ModelArtifact` changes.

**Why the slim slice:** the integration risk lives in the orchestrator → sizer → template → renderer chain, not in the model. Training plumbing is its own multi-day arc; doing it inside Phase 3 means no demoable tab for a week. Slim slice gives a working tab in 1–2 days, then 3b plugs in the real model with zero interface churn.

---

## 2. Scrapling assessment

Phase 3 introduces **no new scraping**. All data flows through the existing TimescaleDB tables + `signals.global_context.get_global_context()`. The two existing scrapers (`gift_nifty_fetcher.py` curl-cffi/regex, `nse_options_fetcher.py` curl-cffi/JSON) are not refactored:

- `gift_nifty_fetcher.py` already extracts the `__NEXT_DATA__` blob with curl-cffi + regex. Scrapling would parse the same HTML the same way — no speedup or robustness gain.
- `nse_options_fetcher.py` calls a JSON endpoint. Scrapling's parser advantage is moot; curl-cffi already gives TLS impersonation.

If a future feature needs a *new* scraper (FII/DII flows, RBI calendar, broker margin tables, etc.) Scrapling is the default per `~/Documents/CLAUDE.md`. Not in this phase.

---

## 3. File Structure

**New files:**
- `india_quant/global_tab/instruments.py` — `LOT_SIZES`, `is_weekly_expiry`, `next_weekly_expiry()`.
- `india_quant/global_tab/options_chain.py` — `OptionsChainSnapshot` dataclass + `load_chain_snapshot(index, as_of, expiry)`.
- `india_quant/global_tab/forecaster.py` — `IndexForecast`, `ModelArtifact` protocol, `forecast_index()`. Phase 3a ships `StubArtifact`.
- `india_quant/global_tab/options_sizer.py` — `size_trade()`.
- `india_quant/global_tab/narrator.py` — `blurb_for_ticket()` (template-only in 3a; LLM hook is `llm=None` no-op).
- `india_quant/global_tab/orchestrator.py` — `build_global_view()`.
- `india_quant/dashboard/templates/_card.html` — Jinja partial for one trade ticket.
- `tests/global_tab/test_options_sizer.py`
- `tests/global_tab/test_forecaster_stub.py`
- `tests/global_tab/test_orchestrator_determinism.py`
- `tests/global_tab/test_narrator_template.py`
- `tests/global_tab/test_route_smoke_phase3.py`
- `tests/global_tab/test_options_chain_loader.py`

**Modified files:**
- `india_quant/dashboard/app.py` — `/global` route reads capital/mode from query params, calls `build_global_view`, passes `cards` into template.
- `india_quant/dashboard/templates/global_v2.html` — adds a `{% include '_card.html' %}` loop below the heatmap.
- `india_quant/global_tab/__init__.py` — re-exports the new public names.

**Files NOT touched:**
- Phase 1/2 modules (`types.py`, `modes.py`, `feature_store.py`, `cost_model.py`, `briefing.py`, `correlation.py`, `heatmap_view.py`).
- The 7 existing fetchers.
- `signals/global_context.py`.

---

## 4. Conventions

- Run pytest as `venv/bin/python -m pytest …`.
- Loguru for logging.
- Currency stays as `float` rupees; bps stays as `int`.
- One commit per task, conventional-commits style. **User runs `git push` themselves; the agent does not run git write commands.**
- Imports: absolute (`from india_quant.global_tab.<module> import …`).

---

## 5. Tasks

### Task 1: `instruments.py` + `options_chain.py` (loader)

**Files:**
- New: `india_quant/global_tab/instruments.py`
- New: `india_quant/global_tab/options_chain.py`
- New: `tests/global_tab/test_options_chain_loader.py`

- [ ] **Step 1:** `instruments.py` — `LOT_SIZES = {"NIFTY": 25, "BANKNIFTY": 15}` (NSE current, lock-in date noted in module docstring); `is_weekly_expiry(d) -> bool` (Thursday); `next_weekly_expiry(from_date) -> date` (next Thursday strictly after `from_date`, skipping NSE holidays — Phase 3 uses calendar-only Thursday and a TODO for the holiday list).
- [ ] **Step 2:** `options_chain.py` — `OptionsChainSnapshot` frozen dataclass holding `index: str`, `as_of: datetime`, `expiry: date`, `underlying_spot: float`, `chain: list[OptionsChainRow]` where each row carries `strike, option_type (CE/PE), last_price, bid, ask, iv, oi`.
- [ ] **Step 3:** `load_chain_snapshot(index, as_of, expiry)` reads from existing `OptionChain` table. Returns `None` (not raise) if zero rows match — sizer treats `None` as `DATA_GAP`.
- [ ] **Step 4:** Tests: fixture inserts 5 strikes around spot for one expiry, asserts loader returns expected rows + filters by expiry correctly. Empty-DB case returns `None`.

### Task 2: `options_sizer.size_trade()`

**Files:** `india_quant/global_tab/options_sizer.py`, `tests/global_tab/test_options_sizer.py`

- [ ] **Step 1:** Strike selection helper: `pick_strike(spot, chain, rule)`:
  - `atm` → strike closest to `spot`.
  - `otm_1` → for LONG/CE: first strike strictly above ATM; for SHORT/PE: first strike strictly below ATM.
  - `itm_1` → for LONG/CE: first strike strictly below ATM; for SHORT/PE: first strike strictly above ATM.
- [ ] **Step 2:** `size_trade(forecast, capital, mode, chain) -> tuple[OptionsLeg, RiskReward, TimingWindow] | None`:
  - Return `None` if `forecast.direction == NO_TRADE` or chain is `None`.
  - Pick option_type from direction (LONG → CE, SHORT → PE).
  - Pick strike per mode; pick the row's mid-price as `premium_estimate` (fallback `last_price` if mid missing); set `premium_zone = (premium*0.97, premium*1.03)`.
  - `lots = floor((capital * MAX_LOSS_FRACTION) / (lot_size * premium))` where `MAX_LOSS_FRACTION = 0.02` for Aggressive, `0.015` Balanced, `0.01` Conservative (constants in `modes.py` — extend `ModeConfig`).
  - Targets/stops via `MODE_CONFIGS[mode]` multiples.
  - Underlying entry/T1/T2/stop triggers via spot ± expected_move_bps from forecast.
  - `RiskReward.win_probability = forecast.confidence`; `expected_value = p * target_t1_pnl - (1-p) * max_loss`; `risk_reward_ratio = target_t1_pnl / max_loss`.
  - Return `None` if EV < `mode.min_expected_value` or p < `mode.min_win_probability` (push the gate up to the orchestrator? — sizer enforces it locally and returns `None`; orchestrator marks card `NO_TRADE` with `no_trade_reason_code = "below_mode_threshold"`).
  - `TimingWindow`: hardcoded per mode (Aggressive 09:20–10:30 entry, 14:30–15:20 exit; Balanced 09:25–11:00 entry, 14:00–15:25 exit; Conservative 09:30–10:30 entry, 13:30–15:25 exit). `invalidation_time = 14:55`.
- [ ] **Step 3:** Tests: per-mode strike rule (3 cases × 2 directions); lots arithmetic with hand-checked numbers; EV gate rejects sub-threshold; chain=None → returns None; option_type matches direction.

### Task 3: `forecaster.py` (with stub artifact)

**Files:** `india_quant/global_tab/forecaster.py`, `tests/global_tab/test_forecaster_stub.py`

- [ ] **Step 1:** `IndexForecast` dataclass per spec §6.2.
- [ ] **Step 2:** `ModelArtifact` protocol: `predict_direction(features) -> tuple[Direction, float]`, `predict_magnitude(features) -> tuple[float, float, float]` (median, p10, p90 in bps).
- [ ] **Step 3:** `StubArtifact` (Phase 3a):
  - Direction from GIFT Nifty premium bps: `> +20` → LONG (confidence 0.6 + |premium|/200, capped 0.8); `< −20` → SHORT (mirror); else NO_TRADE (confidence 0).
  - Magnitude: per-mode fixed table (`AGGRESSIVE`: 80/40/120 bps median/p10/p90; `BALANCED`: 60/30/100; `CONSERVATIVE`: 50/25/85). Magnitude is independent of features in the stub; it becomes feature-driven in Phase 3b.
- [ ] **Step 4:** `forecast_index(index, as_of, mode, features, model_artifact) -> IndexForecast`:
  - If direction is NO_TRADE → `IndexForecast` with `no_trade_reason_code = "no_overnight_catalyst"`.
  - Else → populate fields from artifact; `feature_attributions` = top-3 features by `|value|` from the features dict.
- [ ] **Step 5:** `FeatureRow` dataclass: `gift_nifty_premium_bps: float | None, spx_overnight_pct: float | None, dxy_delta_pct: float | None, india_vix_delta_pct: float | None, brent_overnight_pct: float | None`. Phase 3b adds the rest.
- [ ] **Step 6:** Tests: stub direction-by-premium-sign (3 cases), magnitude-by-mode (3 cases), no-trade reason wiring, feature_attributions ordering.

### Task 4: `narrator.py` (template-only)

**Files:** `india_quant/global_tab/narrator.py`, `tests/global_tab/test_narrator_template.py`

- [ ] **Step 1:** `_render_template(ctx: ReasoningContext, direction: Direction, index: str, blurb_kind: Literal["trade", "no_trade"]) -> str`:
  - Trade template: `"{index} {dir_word}: top driver {drv0_name} ({drv0_val:+.0f}bps); {analog_count} analog sessions averaged {analog_winrate:.0%} win rate, ₹{analog_avg_pnl:,.0f} avg P&L."` — interpolation only, no arithmetic.
  - No-trade template: `"{index}: no trade. {reason_pretty}."` where `reason_pretty` is a small dict from `no_trade_reason_code` → English.
- [ ] **Step 2:** `blurb_for_ticket(ctx, direction, index, llm=None) -> str`:
  - Phase 3: `llm` arg accepted but ignored; always returns template.
  - Phase 6 will add: optional LLM rewrite + validator (length ≤ 400, no foreign digits, no foreign instrument names).
- [ ] **Step 3:** Tests: 3 reason codes render correct English; trade-template interpolation matches expected string; passing `llm=<mock>` does NOT call it in Phase 3.

### Task 5: `orchestrator.build_global_view()`

**Files:** `india_quant/global_tab/orchestrator.py`, `tests/global_tab/test_orchestrator_determinism.py`

- [ ] **Step 1:** Signature: `build_global_view(as_of: datetime, mode: Mode, capital: float, llm=None) -> GlobalTabView`.
- [ ] **Step 2:** Internal flow:
  1. Load global context + briefing strip (existing Phase 2 code path).
  2. Load correlation history + heatmap (existing Phase 2).
  3. Build `FeatureRow` from briefing + global context (GIFT premium = `(gift_nifty_last - prior_nifty_close) / prior_nifty_close * 10000`).
  4. For each index in `["NIFTY", "BANKNIFTY"]`:
     - Load `OptionsChainSnapshot` for next weekly expiry.
     - `forecast = forecast_index(index, as_of, mode, features, StubArtifact())`.
     - `sized = size_trade(forecast, capital, mode, chain)` → tuple or None.
     - Build `ReasoningContext` (analog_count/winrate/avg_pnl = 0/0.0/0.0 in Phase 3 — the analogs index is Phase 4).
     - `blurb = blurb_for_ticket(ctx, direction, index, llm=llm)` (llm always None in 3a).
     - Assemble `TradeTicket`. If `sized is None` → `direction=NO_TRADE`, leg/timing/risk_reward = None, `no_trade_reason_code` set.
     - Initial `LiveTicket`: `Status.WAITING`, `live_pnl=None`, `last_update=as_of`.
  5. Assemble `GlobalTabView` with `staleness` populated from briefing/correlation timestamps.
- [ ] **Step 3:** Determinism test: run `build_global_view` twice with frozen `as_of`, identical DB state, `llm=None`. Assert `dataclasses.asdict(view1) == dataclasses.asdict(view2)`. Property test (hypothesis) over random `(capital ∈ [10k, 10M], mode)` asserting determinism within each (capital, mode) pair.

### Task 6: Wire trade cards into `/global`

**Files:** `india_quant/dashboard/app.py`, `india_quant/dashboard/templates/global_v2.html`, `india_quant/dashboard/templates/_card.html`, `tests/global_tab/test_route_smoke_phase3.py`

- [ ] **Step 1:** `/global` route — read `capital` (default 100_000) and `mode` (default `balanced`) from `request.args`; validate (numeric > 0; one of three modes — else 400 with banner). Call `build_global_view(as_of=now_ist(), mode, capital)`. Pass `view.cards` to the template.
- [ ] **Step 2:** `_card.html` partial: header (index, direction badge, confidence pill), leg block (CE/PE / strike / lots × lot_size / premium zone), R:R block (max_loss, target_t1, target_t2, win_prob, EV, R:R), timing strip (entry/exit windows, invalidation), blurb at bottom. NO_TRADE renders only the header + reason sentence.
- [ ] **Step 3:** `global_v2.html`: add `{% for card in cards %}{% include "_card.html" %}{% endfor %}` block below the heatmap, inside a 2-column grid.
- [ ] **Step 4:** Capital + mode form: simple GET form at top of page (number input + dropdown + submit).
- [ ] **Step 5:** Smoke test (Flask test client): GET `/global?capital=100000&mode=balanced` → 200, response body contains `NIFTY` and `BANKNIFTY` and either a `Direction:` badge or a `No trade —` sentence per card. GET with invalid mode → 400.

### Task 7: Aggregate Phase 3 test run + commit

- [ ] **Step 1:** `venv/bin/python -m pytest tests/global_tab/ -v` — all green.
- [ ] **Step 2:** Manual demo: `python main.py --scheduler` (or whichever local-server entry) → open `http://localhost:5050/global?capital=100000&mode=balanced`, verify cards render.
- [ ] **Step 3:** Print suggested commit message; user runs `git add` + `git commit` themselves (per standing rule).

---

## 6. Phase 3b — real LightGBM training (separate plan after 3a ships)

Outline only; full plan written after Phase 3a is merged.

- `scripts/train_global_forecaster.py`: pull historical features via `PointInTimeFeatureStore`, walk-forward fit LightGBM direction classifier (binary up/down) + LightGBM quantile regressor (q=0.1, 0.5, 0.9 on next-day open-to-close return in bps). Pickle to `models/global_tab/{NIFTY,BANKNIFTY}_{direction,magnitude}.pkl`.
- Replace `StubArtifact` with `LightGBMArtifact` that pickle-loads on first use.
- Feature set v1: GIFT Nifty premium bps, SPX overnight return, DXY delta, India VIX delta, Brent overnight return, NIFTY 5d momentum, trailing 20d realized vol, day-of-week one-hot, weekly-expiry indicator, RBI policy proximity (days).
- Reproducibility gate: same train window + seed → identical model bytes.

---

## 7. Done criteria (Phase 3a)

1. `tests/global_tab/` fully green.
2. `http://localhost:5050/global?capital=100000&mode=balanced` shows briefing strip + heatmap + 2 trade-ticket cards (or NO_TRADE cards with reason).
3. Mode toggle re-renders cards with different strike rules + sizes.
4. `build_global_view(..., llm=None)` is byte-identical across two runs (determinism test passes).
5. Phase 1/2 functionality unchanged.
