# Global Tab Phase 6 — Daily Straddle Strategy (Long Volatility)

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-directional, volatility-driven trade card so the dashboard fires actionable cards even when the LightGBM direction model says NO_TRADE. Buy an ATM straddle (long call + long put) when forecast realized vol exceeds implied vol by a mode-specific buffer. Profits whenever NIFTY moves enough either way.

**Non-goal:** sell-vol strategies (short straddle / iron condor). Those need much tighter risk controls and an active hedge model — defer.

---

## 0. Why this exists

After the Phase 5c train/serve fix, the direction model honestly reports proba_up ≈ 0.5 on most days because daily index direction is genuinely hard to predict with global features alone. But NIFTY *moves* every day. A long-vol setup captures that movement without needing a directional bet.

The trigger is the gap between **forecast realized vol** and **implied vol**:
- σ_forecast > σ_implied → market underpriced movement → buy the straddle
- Otherwise NO_TRADE (negative carry from theta)

---

## 1. Slim slice (v0 — analytical forecast, no new ML model)

Phase 6a ships the strategy with an analytical HAR-RV-style vol forecast (no LightGBM training required). Phase 6b replaces it with a quantile regressor trained on the same 11-feature frame the direction model uses.

`σ_forecast = 0.4·σ_1d + 0.3·σ_5d + 0.3·σ_20d` (annualized %, where σ_kd = stdev of log returns over last k sessions × √252)

Implied vol:
- NIFTY: India VIX (already populated daily in `global_signals`).
- BANKNIFTY: India VIX × 1.20 (placeholder multiplier; refine in 6b).

Premium estimate (when `option_chain` empty): Black-Scholes ATM approximation per leg:
```
prem_atm ≈ spot × σ_impl × √(T/365) × 0.4
```
with T = days to weekly expiry. Total straddle premium = 2 × prem_atm.

When `option_chain` rows exist for the ATM strike, prefer the live mid for both legs.

---

## 2. File Structure

**New files:**
- `india_quant/global_tab/vol_strategy.py` — pure assembler `build_straddle_ticket(...)`. No DB, no network — takes spot, vol_forecast, vol_implied, mode, capital, expiry, optional chain snapshot.
- `india_quant/global_tab/vol_forecaster.py` — analytical `forecast_realized_vol(closes_log_ret_history) -> VolForecast` (Phase 6a). Phase 6b adds `LightGBMVolArtifact` mirroring the direction artifact.
- `tests/global_tab/test_vol_strategy.py`
- `tests/global_tab/test_vol_forecaster.py`

**Modified files:**
- `india_quant/global_tab/types.py` — add `StraddleLeg` (two `OptionsLeg`s) and `StraddleTicket` (sibling of `TradeTicket`). Or extend `TradeTicket` with `kind: Literal["directional","straddle"]`. Plan picks the second to keep the renderer simple.
- `india_quant/global_tab/orchestrator.py` — produce a straddle card *in addition to* the directional card per index, when the strategy fires.
- `india_quant/global_tab/narrator.py` — straddle blurb: "buy NIFTY 24500 straddle, breakeven 24380/24620, max loss ₹X. Forecast σ=14.2% > implied 12.0% (×1.18). 12 of 20 analog sessions exceeded breakeven."
- `india_quant/dashboard/templates/global_v2.html` (or `_card.html`) — render the straddle card variant.
- `india_quant/global_tab/analog_index.py` — add `lookup_breakeven(features, breakeven_bps, k=20) -> float` returning the fraction of analog sessions where |return_bps| > breakeven_bps.

---

## 3. Conventions

- Run pytest as `PYTHONPATH=. venv/bin/python -m pytest …`.
- Vol units: annualized % (as published by India VIX). Internal conversions via `σ_daily = σ_annual / √252` and `σ_per_T_days = σ_annual × √(T/365)`.
- Strike ladder: NIFTY 50-pt, BANKNIFTY 100-pt.
- Lot sizes already in `india_quant/global_tab/instruments.LOT_SIZES`.

---

## 4. Tasks

### Task 1: Vol forecaster (analytical)

**File:** `india_quant/global_tab/vol_forecaster.py`

- [ ] **Step 1:** `@dataclass VolForecast(annualized_pct: float, components: dict[str, float], n_obs: int)`.
- [ ] **Step 2:** `forecast_realized_vol(closes: list[float]) -> VolForecast | None`. Inputs are oldest-first NIFTY closes. Compute daily log returns; daily std for k ∈ {1, 5, 20}; annualize each by × √252; blend 0.4/0.3/0.3. Return None if fewer than 21 closes.
- [ ] **Step 3:** Tests: synthetic prices with known vol, assert blend matches hand-computed value within 1e-6. Edge cases: too-short input returns None; constant prices returns 0.0.

### Task 2: StraddleTicket types + sizer

**Files:** `india_quant/global_tab/types.py`, `india_quant/global_tab/vol_strategy.py`

- [ ] **Step 1:** New dataclass `StraddleLeg(call_strike, put_strike, expiry, lot_size, lots, call_premium, put_premium, total_premium)`. Sibling of `OptionsLeg`.
- [ ] **Step 2:** Extend `TradeTicket` with `kind: Literal["directional","straddle"] = "directional"` and `straddle: StraddleLeg | None = None`. Existing call sites unchanged because `kind` defaults.
- [ ] **Step 3:** `build_straddle_ticket(index, spot, vol_forecast, vol_implied, mode, capital, expiry, *, chain=None, analog_index=None) -> TradeTicket`:
  - Compute σ ratio = forecast / implied. If below mode threshold → emit no_trade ticket with reason `vol_below_threshold`.
  - Compute ATM strike (50 / 100 pt rounding).
  - If chain has the ATM strike row, use mid for both legs; else use Black-Scholes ATM approx with σ_implied.
  - Build `StraddleLeg`. Compute breakeven_high / breakeven_low / max_loss.
  - Lots = `floor(capital × max_loss_fraction / max_loss_per_lot)`. If 0 → no_trade with reason `below_mode_threshold`.
  - When `analog_index` is provided, call `analog_index.lookup_breakeven(features, breakeven_bps, k=20)`; if conservative mode and frac < 0.6 → no_trade with reason `vol_analog_low_hitrate`.
- [ ] **Step 4:** Tests: synthetic spot + vols pinned, assert breakeven, max_loss, lots, kind="straddle", and that the 3 mode buffers gate correctly.

### Task 3: AnalogIndex.lookup_breakeven

**File:** `india_quant/global_tab/analog_index.py`

- [ ] **Step 1:** Add `def lookup_breakeven(self, features, breakeven_bps: float, k: int = 20) -> float`. Reuse `_X @ q` similarity, take top-k, return `np.mean(np.abs(self._yr[idx]) >= breakeven_bps)`.
- [ ] **Step 2:** Test: synthetic frame where label_return_bps follows known distribution, assert returned fraction matches `np.mean(|y_top_k| >= bps)`.

### Task 4: Orchestrator wiring

**File:** `india_quant/global_tab/orchestrator.py`

- [ ] **Step 1:** After the directional card per index, also call `build_straddle_ticket(...)` and append to `cards`. Each index now contributes two cards.
- [ ] **Step 2:** New provider injection: `vol_implied_provider: Callable[[str], float | None]` (returns annualized % per index). Default: read India VIX from `context.signals` for NIFTY, ×1.20 for BANKNIFTY.
- [ ] **Step 3:** Existing tests must still pass — test_view_returns_two_cards updates to expect 4 cards (2 directional + 2 straddle), or asserts ≥ 2 cards with directional kind.

### Task 5: Narrator + template

- [ ] **Step 1:** Narrator branch on `ticket.kind`: straddle blurb shows breakeven prices, max_loss in ₹, forecast σ vs implied σ, and the analog hit-rate.
- [ ] **Step 2:** Template card variant for `kind=straddle` — show two strike rows (CE, PE), breakeven_high/low, max loss; reuse the analog section.
- [ ] **Step 3:** Existing directional card layout untouched.

### Task 6: Live smoke + acceptance gate

- [ ] **Step 1:** `PYTHONPATH=. venv/bin/python -m pytest tests/global_tab/ -q` — ≥ 195 tests, all green.
- [ ] **Step 2:** Hit `/global?capital=100000&mode=balanced`. Expect 4 cards. At least one of the straddle cards is *not* NO_TRADE on most days (the threshold is ×1.15, not "always".)
- [ ] **Step 3:** Verify `/api/global/cards.json` returns the new fields and the JSON serialiser handles `kind`, `straddle`, etc. (already enum-aware via `_coerce_json`).

---

## 5. Acceptance criteria (Phase 6a)

1. Each index emits a directional and a straddle card per request — 4 cards total.
2. Straddle card fires (non-NO_TRADE) on at least 30% of historical sessions in 2024-2026 when backtested with the analytical forecast (verify offline, not in CI). 30% is the floor; refine the buffer in 6b if it's higher.
3. When forecast σ < implied σ × buffer, straddle card emits no_trade with reason `vol_below_threshold` and the blurb explains the gap.
4. Conservative mode additionally requires AnalogStats hit-rate ≥ 60% — emits `vol_analog_low_hitrate` otherwise.
5. ≥ 195 tests, all green.

---

## 6. Phase 6b (deferred — model upgrade, not in this PR)

- Replace the analytical HAR-RV blend with a `LightGBMVolArtifact` quantile regressor (q50 = forecast, q10/q90 = uncertainty band).
- Train on the same `assemble_training_frame` output, label = next-session realized log-return std (annualized).
- Reuse `OptunaSweep`. Add `scripts/train_vol_forecaster.py` mirroring the direction trainer.
- Acceptance gate: OOS pinball loss on q50 vs the analytical baseline. Ship only if measurable lift.

---

## 7. Risks / honest caveats

- **Theta cost is real.** Even when the forecast > implied, daily theta on a 0-DTE straddle is brutal; the buffer (1.15× balanced) is a conservative cushion but doesn't guarantee positive EV in live markets. Backtest before live use.
- **India VIX is a proxy, not the actual ATM IV.** Different rounding, term structure, and skew effects mean the implied side is approximate. Once `option_chain` is populated, we can pull the real ATM IV per expiry per strike.
- **Black-Scholes ATM approximation** is rough — it ignores skew. The chain-mid-when-available path is the correct one; analytical fallback is a "show something" cushion.
- **Liquidity / slippage** on weekly expiries is fine for NIFTY ATM but tightens for BANKNIFTY OTM strikes. The strategy uses ATM only, which is the most liquid bucket.
- **0-DTE risk** on Thursday expiries: gamma blows up, theta evaporates fast. The strategy treats every day uniformly, which is suboptimal — Phase 6c could add an "avoid expiry day" gate.

---

## 8. Out of scope (Phase 6c+ candidates)

- Short-vol strategies (sell straddle, iron condor) — different risk profile, separate plan.
- Skew-aware leg selection (e.g., risk reversals based on put-call skew) — needs option_chain.
- Term-structure trades (calendar spreads) — needs multi-expiry chain data.
- Live exit logic (when to close the straddle mid-session) — Phase 5b's price-based status flips already model this for directional; straddle exits are different (manage delta, gamma, vega).
