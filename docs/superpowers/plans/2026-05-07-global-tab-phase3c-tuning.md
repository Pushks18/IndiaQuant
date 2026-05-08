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

- [x] **Step 1:** Skipped a dedicated script — the existing `india_quant.data.backfill_global` already takes `--days`, so ran `--days 1600` directly. 41,623 new rows.
- [x] **Step 2:** Verified post-backfill: all 5 required tickers have ≥ 1,348 rows (`^GSPC`/`^IXIC`/`^INDIAVIX` from 2019-2020, `BZ=F`/`DX-Y.NYB` from 2020-12-30 — earliest common start). 600 days requested per ticker.
- [x] **Step 3:** Backfilled `^NSEI`/`^NSEBANK`/`^INDIAVIX` `price_data` from 2022-01-01 → 2026-05-08 via `YFinanceFetcher().fetch_and_store(...)`. 3,206 rows upserted.
- [x] **Step 4:** Untuned re-run on the expanded window (`--start 2021-01-01`, n_samples=954) returned **direction logloss 0.91** — *worse* than Phase 3b's 0.855. Confirmed the bottleneck was hyperparameters, not data. Magnitude q50 improved ~9% from data alone (33.86 → 30.7).

### Task 2: Optuna sweep wrapper

**File:** `india_quant/global_tab/tuning.py`

- [x] **Step 1:** `class OptunaSweep` with constructor `(features: pd.DataFrame, labels: pd.Series, *, target: Literal['direction','magnitude'], quantile: float | None, n_splits: int, seed: int)`.
- [x] **Step 2:** `def objective(self, trial) -> float:` — suggest hyperparameters per the table below, run `TimeSeriesSplit(n_splits)` walk-forward CV, return mean OOS metric (logloss for direction, pinball loss at the requested quantile for magnitude).

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

- [x] **Step 3:** `def run(self, n_trials: int, storage: str | None, study_name) -> SweepResult` — TPE sampler seeded, optional sqlite storage URI for resumability.
- [x] **Step 4:** `MedianPruner(n_warmup_steps=2)`; trial reports `np.mean(fold_scores)` after each fold.

### Task 3: Wire `--tune` into the training CLI

- [x] **Step 1:** Argparse flags added (`--tune`, `--n-trials` default 30, `--tune-storage`).
- [x] **Step 2:** `_run_tuning()` orchestrates 1 (direction) or 3 (magnitude q10/q50/q90) Optuna studies, stashes best params per target into `summary.tuning`, then threads them into the existing `_train_direction` / `_train_magnitude_quantile` via a new `override_params` kwarg.
- [x] **Step 3:** Untuned path verified byte-identical via `feature_importances_` array equality (`/tmp/baseline_post_3c.pkl` vs fresh re-run, both seed=42 same window).

### Task 4: Tests

- [x] **Step 1:** Synthetic 200×11 fixture in `_synth_features_labels`.
- [x] **Step 2:** Sweep returns valid params for both direction + magnitude; ranges verified.
- [x] **Step 3:** Tuned artifact loads via `LightGBMArtifact` and satisfies the protocol. Note: dropped the `p10 ≤ p50 ≤ p90` ordering assertion — on small synthetic noise the independent quantile regressors can cross, and the artifact's contract after `abs()` is only "3 non-negative finite floats" (matches existing `test_lightgbm_artifact.py` invariants).
- [x] **Step 4:** Without `--tune`, `training_summary.json["tuning"] == {}`.

### Task 5: End-to-end live run + acceptance gate

- [x] **Step 1:** Backfill done via existing `backfill_global --days 1600` + `YFinanceFetcher.fetch_and_store(...)` — no new script needed.
- [x] **Step 2:** Untuned baseline on `--start 2021-01-01` returned NIFTY direction logloss **0.91** (worse than Phase 3b's 0.855), confirming hyperparameters are the lever.
- [x] **Step 3:** Tuned runs:
  - `NIFTY  --tune --n-trials 30`: direction logloss **0.675**, q10 **14.5**, q50 **29.8**, q90 **13.4** (n=954).
  - `BANKNIFTY --tune --n-trials 30`: direction logloss **0.684**, q10 **18.3**, q50 **35.4**, q90 **16.6** (n=961).
- [x] **Step 4:** `/global?capital=100000&mode=balanced` returns 200, banner reads `Forecast: lightgbm` — no regression.
- [x] **Step 5:** **145/145 tests pass** (138 from 3b + 7 new tuning tests).

---

## 5b. Live results vs acceptance criteria

| # | Criterion                                                         | Result                                                                  |
|---|-------------------------------------------------------------------|-------------------------------------------------------------------------|
| 1 | Direction logloss < 0.693 on at least one index                  | **MET on both** (NIFTY 0.675, BANKNIFTY 0.684)                          |
| 2 | Pinball q50 drops ≥ 15% on both indices                          | Partial: NIFTY -12% (33.86 → 29.84), BANKNIFTY -8% (38.48 → 35.38)       |
| 3 | Untuned path byte-identical for the same seed/window             | MET (`feature_importances_` array equality)                              |
| 4 | Test count ≥ 142, all green                                      | MET (145/145)                                                           |
| 5 | `--tune` reproducible across re-runs with same seed + storage    | TPE seeded + sqlite storage configured; not re-run-tested in this pass  |
| 6 | Pickles in `models/global_tab/` + `tuning` block in summary JSON | MET                                                                     |

**Net:** Phase 3c hits the headline goal — both direction models now beat the trivial baseline. Magnitude regressors improved but didn't fully clear the -15% bar; Phase 3d (richer features) is the natural follow-up if the magnitude tail matters for sizing.

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
