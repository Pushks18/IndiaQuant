# Global Tab Revamp — Phase 3b (LightGBM Forecaster) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `StubArtifact` with a real LightGBM artifact behind the same `ModelArtifact` protocol introduced in Phase 3a. Direction is a binary classifier (up/down vs prior close); magnitude is a quantile regressor (q=0.10, 0.50, 0.90). Train walk-forward over `PointInTimeFeatureStore`; serve from a pickle. Zero changes to `forecaster.forecast_index()`, `options_sizer`, `orchestrator`, or templates — only the artifact swap.

**Spec reference:** `docs/superpowers/specs/2026-05-05-global-tab-revamp-design.md` §6.2 (default model family). Builds on the Phase 3a plan §6.

---

## 0. Prerequisites

- [x] **brew install libomp** — required by lightgbm on macOS (verified missing in env on 2026-05-05). Without this, `import lightgbm` raises `Library not loaded: @rpath/libomp.dylib`. User runs this once.
- [x] **TimescaleDB up** with at least 250 trading days of `global_signals` + Nifty/BankNifty `price_data`. The current DB has ~8.9k `global_signals` rows through 2026-05-05 — sufficient.
- [x] **lightgbm pin** — confirm `lightgbm` is in `requirements.txt`; pin to a version compatible with installed numpy 2.2 / scikit-learn (currently `lightgbm==4.5.0` works with numpy 2.x).

---

## 1. Slim slice (single LightGBMArtifact, both indices)

One artifact class loads four pickles (NIFTY direction, NIFTY magnitude, BANKNIFTY direction, BANKNIFTY magnitude). No per-index class hierarchy. The artifact picks the right pair at predict-time from `forecast.index`. Magnitude regressor outputs (p10, p50, p90) in bps of next-session open-to-close return; direction probability gates LONG/SHORT/NO_TRADE via per-mode thresholds shared with the stub.

---

## 2. File Structure

**New files:**
- `scripts/train_global_forecaster.py` — CLI: `--index NIFTY|BANKNIFTY --target direction|magnitude --start 2022-01-01 --end 2025-12-31 --seed 42 --out models/global_tab/`
- `india_quant/global_tab/training_features.py` — `assemble_training_frame(index, start, end) -> pd.DataFrame` — single source of truth for the feature matrix used by both the trainer and (a thin reuse path inside) `orchestrator._build_features` for live serving.
- `india_quant/global_tab/lightgbm_artifact.py` — `LightGBMArtifact` class implementing the `ModelArtifact` protocol; loads pickles lazily on first call.
- `models/global_tab/` — pickle output dir (gitignored except for a `.gitkeep`).
- `tests/global_tab/test_lightgbm_artifact.py` — protocol conformance test using a tiny in-memory model.
- `tests/global_tab/test_training_features.py` — feature-matrix shape, no-future-peek, label correctness.
- `tests/global_tab/test_train_script.py` — invoke training on synthetic data, assert pickle written + reproducibility gate.

**Modified files:**
- `india_quant/global_tab/forecaster.py` — no signature change; add a comment noting `LightGBMArtifact` is the production swap-in.
- `india_quant/global_tab/orchestrator.py` — accept `model_artifact` parameter (already does); the Flask route passes a memoized `LightGBMArtifact()` instead of relying on the default `StubArtifact()`.
- `india_quant/dashboard/app.py` — `/global` route constructs `LightGBMArtifact` once at app-init (module-level singleton with lazy load) and passes it into `build_global_view`. Falls back to `StubArtifact` if pickles missing — logged as a warning, never a 500.
- `requirements.txt` — confirm `lightgbm>=4.5,<5`.
- `.gitignore` — add `models/global_tab/*.pkl`.

**Files NOT touched:**
- `forecaster.IndexForecast`, `options_sizer`, `narrator`, `_card.html`, `global_v2.html`, `feature_store.py`, the seven existing fetchers, briefing/correlation/heatmap modules.

---

## 3. Conventions

- Run pytest as `venv/bin/python -m pytest …`.
- Loguru logging.
- One commit per task, conventional-commits style. **User runs `git push` themselves.**
- Imports: absolute.
- Pickles use `joblib.dump(..., compress=3)` — smaller than pickle and version-tolerant for sklearn/lightgbm.

---

## 4. Feature Set v1

Single feature row per session, anchored at India pre-open (08:30 IST). All features are causally available before NSE opens at 09:15.

| Feature | Source | Window |
|---|---|---|
| `gift_nifty_premium_bps` | `gift_nifty_fetcher` snapshot at 08:30 | spot only |
| `spx_overnight_pct` | `global_signals.pct_1d` for `^GSPC` | T-1 close → T close (US close = ~02:00 IST T) |
| `nasdaq_overnight_pct` | `global_signals.pct_1d` for `^IXIC` | same |
| `dxy_delta_pct` | `global_signals.pct_1d` for `DX-Y.NYB` | same |
| `india_vix_delta_pct` | `global_signals.pct_1d` for `^INDIAVIX` | T-1 close to last available |
| `brent_overnight_pct` | `global_signals.pct_1d` for `BZ=F` | same |
| `nifty_5d_momentum` | computed from `price_data` `^NSEI` close | last 5 trading days |
| `nifty_realized_vol_20d` | std of log returns from `price_data` `^NSEI` | last 20 trading days |
| `dow_one_hot_*` | derived from session date (Mon..Fri) | — |
| `is_expiry_week` | bool — current week contains a Thursday expiry | — |
| `days_to_rbi_policy` | from `Config.rbi_policy_dates` (already in repo) | — |

Total: 11 numeric features (4 dow one-hot collapsed to dow integer to keep it tight; the spec calls for one-hot but quantile-regressor handles ints fine — log this deviation in the trainer).

**Label (direction):** sign of next-session NIFTY (or BANKNIFTY) close-to-close log return; classes `{1: up, 0: down}`. Drop sessions with |return| < 5 bps to remove ambiguity (~3% of sessions).

**Label (magnitude):** next-session close-to-close return in bps. Trained as quantile regression with `objective='quantile', alpha=0.1/0.5/0.9` — three separate boosters.

---

## 5. Tasks

### Task 1: Feature assembly module + tests

**Files:** `india_quant/global_tab/training_features.py`, `tests/global_tab/test_training_features.py`

- [x] **Step 1:** Define `FEATURE_COLUMNS` list (the 11 names above) at module top so tests assert against it.
- [x] **Step 2:** `assemble_training_frame(index: str, start: date, end: date, *, session_factory) -> pd.DataFrame`. Returns a DataFrame indexed by session date with `FEATURE_COLUMNS + ['label_direction', 'label_return_bps']`.
  - Pull `global_signals` for tickers `[^GSPC, ^IXIC, DX-Y.NYB, ^INDIAVIX, BZ=F]` between `start` and `end`.
  - Pull `price_data` for `^NSEI` / `^NSEBANK` over the window plus 30 days of lookback (for momentum + realized vol).
  - Build features per session date; drop rows where any feature is NaN (log count).
  - Compute labels by shifting close one session forward; drop the trailing row.
- [x] **Step 3:** Wrap feature assembly in a `to_feature_store(frame) -> PointInTimeFeatureStore` helper so live serving can reuse the same column logic via `register(name, frame[name])`.
- [x] **Step 4:** Tests:
  - Synthetic DB fixture (extend `conftest.py`): 60 sessions of fake `global_signals` + Nifty prices.
  - Assert frame has `FEATURE_COLUMNS` columns and at least N-1 rows after label shift.
  - Assert no row has feature timestamp > its index date (no-future-peek).
  - Assert `label_direction` is in `{0, 1}` and matches sign of `label_return_bps`.

### Task 2: LightGBMArtifact + protocol-conformance test

**Files:** `india_quant/global_tab/lightgbm_artifact.py`, `tests/global_tab/test_lightgbm_artifact.py`

- [x] **Step 1:** `class LightGBMArtifact(ModelArtifact)` with `__init__(self, models_dir: Path = Path('models/global_tab'))`. Lazily loads `{INDEX}_direction.pkl` and `{INDEX}_magnitude_q{10,50,90}.pkl` on first predict for that index. Caches loaded boosters in a dict.
- [x] **Step 2:** `predict_direction(features, mode) -> (Direction, float)`:
  - Build feature vector from `FeatureRow` in the order LightGBM was trained on.
  - `proba_up = booster.predict(vec)[0]` → in [0, 1].
  - `direction = LONG if proba_up > 0.5 + threshold else SHORT if proba_up < 0.5 - threshold else NO_TRADE`. Mode-specific threshold (e.g. balanced 0.05, conservative 0.10, aggressive 0.02).
  - `confidence = abs(proba_up - 0.5) * 2`, clipped to [0.5, 0.85].
- [x] **Step 3:** `predict_magnitude(features, mode) -> (median, p10, p90)` in bps. Returns 0,0,0 if direction model is missing.
- [x] **Step 4:** Protocol-conformance test using a tiny `lightgbm.LGBMClassifier()` fit on 50 random rows and pickled to a tmp dir. Assert `predict_direction` returns valid `(Direction, confidence)` and `predict_magnitude` returns 3 floats with `p10 ≤ p50 ≤ p90`.
- [x] **Step 5:** Test the missing-pickles fallback: instantiate with a path that doesn't exist, assert `predict_direction` raises a clear error (not a generic FileNotFoundError) — Flask route catches and falls back to StubArtifact.

### Task 3: Training script

**File:** `scripts/train_global_forecaster.py`

- [x] **Step 1:** argparse: `--index`, `--target {direction,magnitude,both}` (default both), `--start`, `--end`, `--seed`, `--out`, `--n-splits` (default 5 walk-forward folds).
- [x] **Step 2:** Pull frame via `assemble_training_frame`. Time-series split (sklearn `TimeSeriesSplit`); for each fold log out-of-sample logloss (direction) / pinball loss (magnitude).
- [x] **Step 3:** Refit on full train window with best hyperparams (start with sensible defaults: 200 trees, lr=0.05, num_leaves=31, min_data=20). No tuning sweep in 3b — log a TODO for Optuna in Phase 3c if the OOS metrics warrant it.
- [x] **Step 4:** Pickle each booster via `joblib.dump(booster, out / f'{index}_{target}.pkl', compress=3)`. Write a `training_summary.json` next to it with seed, fold metrics, sample counts, feature list, lightgbm version.
- [x] **Step 5:** Reproducibility gate: re-run training with the same seed + window, assert `joblib.dumps(booster)` byte-identical (or feature_importances arrays equal, since lightgbm pickle bytes can differ across runs for the same model in some versions).

### Task 4: Wire LightGBMArtifact into the Flask route

**Files:** `india_quant/dashboard/app.py`

- [x] **Step 1:** At module scope (or inside `create_app` before route registration), construct `_DEFAULT_ARTIFACT = LightGBMArtifact()`. If the models dir doesn't exist or any pickle is unreadable, log a warning and set `_DEFAULT_ARTIFACT = StubArtifact()`.
- [x] **Step 2:** Pass `model_artifact=_DEFAULT_ARTIFACT` into `build_global_view` from the `/global` route handler.
- [x] **Step 3:** Add a small banner in `global_v2.html` when `view.staleness['artifact'] == 'stub'` so the user knows they're seeing stub output. Orchestrator populates `view.staleness['artifact'] = artifact.name` (add `name` property to both artifacts: `"stub"` / `"lightgbm@<git-sha>"`).

### Task 5: End-to-end demo + test gate

- [x] **Step 1:** `venv/bin/python -m pytest tests/global_tab/ -v` — all green (Phase 1 + 2 + 3a + 3b ≥ 130 tests).
- [ ] **Step 2:** `venv/bin/python scripts/train_global_forecaster.py --index NIFTY --start 2024-01-01 --end 2026-04-30 --seed 42 --out models/global_tab/`. Inspect `training_summary.json`: OOS log-loss should beat 0.69 (the always-up baseline = ~0.693). If not, log finding and document — Phase 3c tuning is a separate plan. **(awaiting live TimescaleDB run by user)**
- [ ] **Step 3:** Same for `--index BANKNIFTY`. **(awaiting live TimescaleDB run by user)**
- [ ] **Step 4:** Restart dashboard, hit `/global?capital=100000&mode=balanced`, confirm:
  - Cards no longer show `artifact=stub` banner.
  - Direction can be LONG, SHORT, or NO_TRADE depending on today's features (vs Phase 3a where direction was deterministic from GIFT premium).
  - Determinism property test from Phase 3a still passes (LightGBM is deterministic when seeded + no GPU).
  **(awaiting trained pickles from steps 2-3)**

---

## 6. Done criteria (Phase 3b)

1. `tests/global_tab/` ≥ 130 tests, all green.
2. Two pickled boosters per index live under `models/global_tab/`, with a `training_summary.json` recording seed, OOS metrics, feature list, and lightgbm version.
3. `/global` renders cards driven by LightGBM (artifact banner reads `lightgbm@<sha>`, not `stub`).
4. Reproducibility gate: re-running training with `--seed 42` over the same window produces identical feature_importances arrays.
5. Phase 3a determinism property test still passes after the artifact swap.

---

## 7. Risks logged for the user

- **Small training set.** ~250 sessions × ~3 years = ~750 rows; LightGBM with 200 trees can overfit. Mitigation: heavy early stopping (`early_stopping_rounds=20`) + small num_leaves (31). If OOS log-loss is below baseline, document and move on — the spec calls this out as a known constraint of the regime.
- **Class imbalance after the |return| < 5 bps drop.** Up days slightly outnumber down days in NIFTY. Mitigation: `class_weight='balanced'` in the classifier.
- **Pickle compatibility across lightgbm versions.** Pin `lightgbm` in requirements.txt; if a future bump breaks pickle load, retrain from scratch (cheap — ~30 seconds on this dataset).
- **Live feature timestamps vs training feature timestamps.** Training reads end-of-day `global_signals`; live serving reads pre-open. Mitigation: training assembles features at the *same* "as-of" point (08:30 IST T+1, after US close) by selecting `pct_1d` rows where `signal_date == prior_session_date` of NSE — codified in `assemble_training_frame`.
- **No tuning sweep.** Phase 3b ships sensible defaults. Phase 3c (separate plan) does Optuna over (num_leaves, lr, min_data, n_estimators) if OOS metrics justify it.
