# Global Tab Revamp — Phase 3c (LightGBM Tuning + Training-Window Expansion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the LightGBM forecaster's out-of-sample direction logloss below the always-up baseline (≤ 0.693) and tighten the magnitude pinball losses, without touching the `ModelArtifact` protocol or any callers. Two levers: (1) **expand the training window** by backfilling pre-2024-04-16 history for the five global tickers, and (2) **Optuna sweep** over `(num_leaves, learning_rate, min_data_in_leaf, n_estimators, feature_fraction, bagging_fraction)` with walk-forward CV. The deliverable is the same artifact path (`models/global_tab/{INDEX}_{target}.pkl`) — only the weights change.

**Spec reference:** `docs/superpowers/specs/2026-05-05-global-tab-revamp-design.md` §6.2. Builds on Phase 3b plan §7 ("No tuning sweep. Phase 3b ships sensible defaults. Phase 3c does Optuna…").

---

## 0. Why this is needed (OOS baseline from Phase 3b live run, 2026-05-07)

Trained 2024-04-16 → 2026-05-07, seed=42, 5-fold walk-forward.

| Index     | n_samples | OOS direction logloss (mean) | Always-up baseline | OOS pinball q50 (mean, bps) |
|-----------|-----------|------------------------------|--------------------|------------------------------|
| NIFTY     | 359       | **0.855**                    | 0.693              | 33.86                        |
| BANKNIFTY | 355       | **0.840**                    | 0.693              | 38.48                        |

Direction model loses to the trivial "always predict up" baseline by 23%/21% logloss. Two diagnostic patterns visible in the per-fold numbers:

- **Fold 4 outlier** — magnitude pinball blows out from ~10–15 bps to 30–60 bps, suggesting a regime shift in the most recent ~60 sessions that the model can't generalize to from earlier folds.
- **Folds 1–2 worst** — first ~120 sessions of training, model overfits the small sample and validates poorly.

Both patterns point to **insufficient data** (~360 rows is below the rule-of-thumb 10× features × n_leaves for tree ensembles) and **default hyperparameters** (200 trees + lr=0.05 + 31 leaves with no early stopping is a known overfit recipe on small panels).

---

## 1. Slim slice

**One PR, two commits:** (a) backfill pre-2024-04-16 data for `^INDIAVIX` + `BZ=F` so the training window can extend to 2022-01-01; (b) add an Optuna sweep mode to `scripts/train_global_forecaster.py` (gated behind `--tune`, default off so existing CLI keeps working). No protocol changes, no template changes, no test removals — only additions.

---

## 2. File Structure

**New files:**
- `scripts/backfill_global_history.py` — one-shot backfill that extends `global_signals` for `^INDIAVIX` + `BZ=F` back to 2022-01-01 (yfinance via the upgraded 1.3.0 client). Runs once; idempotent via `ON CONFLICT DO NOTHING`.
- `india_quant/global_tab/tuning.py` — `OptunaSweep` class wrapping the walk-forward CV objective. Used by the training script when `--tune` is passed.
- `tests/global_tab/test_tuning.py` — synthetic-data sweep with `n_trials=5` to assert the sweep wires up correctly, picks a non-default config, and the resulting booster still implements the `ModelArtifact` protocol.

**Modified files:**
- `scripts/train_global_forecaster.py` — add `--tune`, `--n-trials` (default 30), `--tune-storage` (default `sqlite:///optuna_global_tab.db` for resumability) flags. When `--tune` is set, run Optuna first, then refit with the best params and persist via the existing pickle path.
- `requirements.txt` — add `optuna>=3.5,<4`.
- `.gitignore` — add `optuna_global_tab.db`.

**Files NOT touched:**
- `lightgbm_artifact.py`, `forecaster.py`, `orchestrator.py`, `app.py`, `global_v2.html`, `training_features.py` (already correct after the 2026-05-07 tz fix).

---

## 3. Conventions

- Run pytest as `PYTHONPATH=. venv/bin/python -m pytest …`.
- All scripts assume `PYTHONPATH=.` from repo root (matches existing scripts/).
- Optuna study name = `global_tab_{index}_{target}_{seed}` for collision-free resumability.

---

## 4. Tasks

### Task 1: Extend training window via yfinance backfill

**File:** `scripts/backfill_global_history.py`

- [ ] **Step 1:** Use the existing `india_quant.data.backfill_global` machinery but parameterize the date window. Add `--start 2022-01-01 --end <today>` flags. Default behavior: skip dates already in `global_signals` (the existing `ON CONFLICT DO NOTHING` clause handles this).
- [ ] **Step 2:** Verify post-backfill coverage with `psql`:
  ```sql
  SELECT ticker, count(*), min(date), max(date)
  FROM global_signals
  WHERE ticker IN ('^GSPC','^IXIC','DX-Y.NYB','^INDIAVIX','BZ=F')
  GROUP BY ticker;
  ```
  Expect ≥ 800 rows per ticker (vs current 520–600).
- [ ] **Step 3:** Also backfill `price_data` for `^NSEI`/`^NSEBANK` from 2022-01-01 (yfinance returns these reliably under 1.3.0 — confirmed during Phase 3b live run).
- [ ] **Step 4:** Re-run the **un-tuned** training as a sanity check, same defaults as Phase 3b but `--start 2022-01-01`. Capture the new n_samples (~700+ expected) and OOS logloss in `training_summary.json`. If logloss already drops below 0.693 from data alone, log it and document — Optuna sweep becomes nice-to-have rather than necessary.

### Task 2: Optuna sweep wrapper

**File:** `india_quant/global_tab/tuning.py`

- [ ] **Step 1:** `class OptunaSweep` with constructor `(features: pd.DataFrame, labels: pd.Series, *, target: Literal['direction','magnitude'], quantile: float | None, n_splits: int, seed: int)`.
- [ ] **Step 2:** `def objective(self, trial) -> float:` — suggest hyperparameters per the table below, run `TimeSeriesSplit(n_splits)` walk-forward CV, return mean OOS metric (logloss for direction, pinball loss at the requested quantile for magnitude).

  | Param                | Range / choices                | Notes                                  |
  |----------------------|--------------------------------|----------------------------------------|
  | `num_leaves`         | int [7, 63] log               | small to fight overfit on ~700 rows    |
  | `learning_rate`      | float [0.01, 0.2] log         |                                        |
  | `min_data_in_leaf`   | int [10, 60]                  | floor at 10 — trees can't memorise     |
  | `n_estimators`       | int [100, 500]                | with `early_stopping_rounds=20`        |
  | `feature_fraction`   | float [0.6, 1.0]              |                                        |
  | `bagging_fraction`   | float [0.6, 1.0]              | with `bagging_freq=5`                  |
  | `min_gain_to_split`  | float [0.0, 0.1]              | regularizer                            |
  | `lambda_l2`          | float [0.0, 1.0] log+ε        | regularizer                            |

- [ ] **Step 3:** `def run(self, n_trials: int, storage: str | None) -> dict:` — return `study.best_params` and `study.best_value`. Use TPE sampler with `seed=self.seed`. Storage URI enables resumability across crashes.
- [ ] **Step 4:** Pruning — use `optuna.pruners.MedianPruner(n_warmup_steps=2)` reporting fold-level OOS metric so unpromising configs die after 2 of 5 folds.

### Task 3: Wire `--tune` into the training CLI

**File:** `scripts/train_global_forecaster.py`

- [ ] **Step 1:** Add argparse flags: `--tune` (store_true), `--n-trials` (default 30), `--tune-storage` (default `sqlite:///optuna_global_tab.db`).
- [ ] **Step 2:** When `--tune` is set, after `assemble_training_frame` returns:
  1. Build `OptunaSweep(target=direction, …)`, run for `n_trials`, capture best params.
  2. Repeat for each magnitude quantile (q10, q50, q90) — three separate studies.
  3. Refit the final boosters on the full window with the best params (existing refit path), pickle them.
  4. Stash `best_params` + `best_value` per target inside `training_summary.json` under a new `tuning` key. Existing fields stay where they are for back-compat.
- [ ] **Step 3:** When `--tune` is **not** set, behavior is byte-identical to Phase 3b (verified by re-running the seed-42 reproducibility gate after the patch). This is the back-compat guarantee.

### Task 4: Tests

**File:** `tests/global_tab/test_tuning.py`

- [ ] **Step 1:** Synthetic dataset fixture: 200 rows, 11 features (matching `FEATURE_COLUMNS`), random seed.
- [ ] **Step 2:** Run `OptunaSweep` with `n_trials=5`, `n_splits=3`. Assert: `best_params` is a dict with all 8 keys; `best_value` is finite; the suggested params lie inside the documented ranges.
- [ ] **Step 3:** Refit with `best_params`, wrap in `LightGBMArtifact`, assert protocol conformance (predict_direction returns `(Direction, float)`; predict_magnitude returns 3 floats with `p10 ≤ p50 ≤ p90`).
- [ ] **Step 4:** Tune-disabled path test: run the existing training script entry point with `--tune` absent on the synthetic fixture; assert the resulting `training_summary.json` matches the Phase 3b shape exactly (`tuning` key absent or empty).

### Task 5: End-to-end live run + acceptance gate

- [ ] **Step 1:** `PYTHONPATH=. venv/bin/python scripts/backfill_global_history.py --start 2022-01-01`.
- [ ] **Step 2:** Untuned baseline on the expanded window:
  ```
  PYTHONPATH=. venv/bin/python scripts/train_global_forecaster.py \
    --index NIFTY --start 2022-01-01 --end <today> --seed 42 --out models/global_tab/
  ```
  Record new OOS logloss. Compare to Phase 3b's 0.855.
- [ ] **Step 3:** Tuned run:
  ```
  PYTHONPATH=. venv/bin/python scripts/train_global_forecaster.py \
    --index NIFTY --start 2022-01-01 --end <today> --seed 42 \
    --tune --n-trials 30 --out models/global_tab/
  ```
  Same for `--index BANKNIFTY`. Two studies running ~20 minutes total on a laptop.
- [ ] **Step 4:** Restart dashboard, hit `/global?capital=100000&mode=balanced`. Confirm:
  - Banner still reads `Forecast: lightgbm` (artifact name didn't regress).
  - Determinism property test from Phase 3a still passes after the new pickles drop in.
- [ ] **Step 5:** `PYTHONPATH=. venv/bin/python -m pytest tests/global_tab/ -q` — all green, ≥ 142 tests (138 existing + ≥ 4 new).

---

## 5. Acceptance criteria

1. **Direction logloss** (mean across folds) drops below **0.693** (always-up baseline) for at least one of {NIFTY, BANKNIFTY}. If neither beats baseline after Optuna + window expansion, document as a model-capacity ceiling and write up Phase 3d (richer features: intraday vol surface, sectoral breadth, options OI flow) as the next plan.
2. **Magnitude pinball q50** (mean across folds) drops by ≥ 15% on both indices vs Phase 3b (NIFTY 33.86 → ≤ 28.78; BANKNIFTY 38.48 → ≤ 32.71).
3. **No regression** on the un-tuned path: running the script without `--tune` produces a `training_summary.json` byte-identical to Phase 3b's for the same seed + window.
4. **Test count ≥ 142**, all green.
5. **Reproducibility:** running `--tune` twice with the same `--seed` and the same Optuna storage URI converges to the same `best_params`. (Optuna TPE is seeded; storage gives resumability.)
6. **Pickles live** under `models/global_tab/` (gitignored), and `training_summary.json` carries a new `tuning` block recording the chosen hyperparameters per target.

---

## 6. Risks logged for the user

- **yfinance pre-2024 history thinning out.** `^INDIAVIX` daily closes from yfinance occasionally have gaps in 2022-2023. The `dropna()` step in `assemble_training_frame` will silently drop those rows; expect n_samples to be ~700 rather than the full ~1000 calendar trading days. Documented, not fatal.
- **Optuna TPE on small data is noisy.** With only ~700 rows split 5-fold, the same `best_params` can shift across re-seeded runs even if the seed is held. Mitigation: (a) run with `n_trials ≥ 30` so TPE has signal; (b) the reproducibility gate uses *the same Optuna storage URI* — TPE is deterministic given the same trial history, so resumed studies are bit-identical.
- **Logloss might not budge.** If global macro features alone can't beat the always-up baseline even with tuning, that's an information-content ceiling, not a model-capacity ceiling. The follow-up is Phase 3d (feature engineering), not deeper trees. The acceptance criterion §1 explicitly allows this outcome by routing to a documented Phase 3d.
- **Training time.** Optuna 30 trials × 4 targets × 5 folds × ~2s per fit ≈ 20 minutes on the dev laptop. Acceptable for a once-a-week training cron; flagged so the user doesn't expect 30s like Phase 3b's untuned run.
- **Phase 3a determinism property test.** Already passes in Phase 3b because LightGBM is deterministic on CPU when seeded. Optuna doesn't change that — but the `seed` param must be threaded through the booster's `random_state` so re-running training with the same seed produces byte-identical pickles. Verified in Task 4 step 4.

---

## 7. Out of scope (kept for future phases)

- **Optuna-suggested feature subset** (vs the fixed 11 in `FEATURE_COLUMNS`) — wait for Phase 3d feature additions before doing column selection.
- **Multi-seed ensembling** — train 5 boosters with different seeds and average. Cheap win if a single best-params config still has high variance, but adds artifact-storage complexity (5× pickles per target). Defer.
- **Conformal prediction wrapping** for the magnitude quantiles — gives calibrated p10/p90 instead of relying on quantile-regression's empirical coverage. Belongs in its own plan once we measure the calibration gap on live data.
- **Switch to CatBoost or XGBoost** as alternative model families — protocol allows it (the `ModelArtifact` interface doesn't care about the backing library), but no evidence yet that LightGBM is the bottleneck rather than the data.
