# Global Tab Revamp — Design Spec

**Date:** 2026-05-05
**Project:** `india_quant` (Trading repo)
**Surface:** `global_context_page` in `india_quant/dashboard/app.py`
**Status:** Design — pending user review before plan-writing

---

## 1. Goal

Turn the existing read-only "Global" tab into a paper-trading **decision cockpit** for NIFTY 50 and BANKNIFTY weekly options. For every Indian session, the tab presents:

- A **pre-open briefing strip** (overnight global moves, GIFT Nifty premium, predicted gap).
- A **correlation heatmap** between NIFTY/BANKNIFTY and global drivers.
- One **trade-ticket card per index** with a fully specified options trade (or an explicit `NO_TRADE` with reason).
- A **Backtest sub-tab** with walk-forward equity curves per mode.

The system is decision-support. All trading decisions are produced by deterministic Python; the LLM is restricted to writing one prose blurb per card from a frozen `ReasoningContext`.

## 2. Non-Goals

- **No live broker integration.** Paper trading first; broker API is out of scope.
- **No real-time tick streaming.** Refresh is pre-open + scheduled intraday checkpoints (09:30, 11:00, 13:00, 14:30 IST).
- **No deterministic-second-and-paisa price tickets.** Outputs are probabilistic with explicit win-prob, EV, and max-loss.
- **No additional indices in v1.** FINNIFTY, MIDCPNIFTY, SENSEX deferred. The engine is parameterized so adding them later is a config change.
- **No futures recommendations in v1.** Options only (CE/PE, weekly expiry).
- **No ticket stacking.** One ticket per card per session.

## 3. User Inputs

- **Capital** (₹) — free-form numeric input at the top of the tab. Used for sizing per-card (each card sized as if the full capital were used for that trade).
- **Mode** — dropdown: `Aggressive` / `Balanced` / `Conservative`. Controls confidence gating, strike selection, and target/stop multipliers.

Both inputs are global to the tab. Switching either re-renders all cards from a pre-computed all-modes artifact (no recomputation).

## 4. Architecture

### 4.1 Approach

**Streamlit-direct, cached pure functions** with a single orchestrator. Heavy work runs in scheduled scripts that write parquet/sqlite artifacts; the request path reads artifacts and assembles the view.

```
                      ┌──────────────────────────────────────┐
                      │ Streamlit page: global_context_page  │
                      │ (india_quant/dashboard/app.py)       │
                      │                                       │
                      │ ┌──────────────────────────────┐     │
                      │ │ build_global_view(           │     │
                      │ │   capital, mode, as_of       │     │
                      │ │ ) → GlobalTabView            │     │
                      │ └──────────────┬───────────────┘     │
                      └──────────────────┼───────────────────┘
                                         │ pure call, st.cache_data
                                         ▼
        ┌────────────────────────────────────────────────────────────┐
        │ Orchestrator: india_quant/global_tab/orchestrator.py       │
        │ (deterministic, no LLM)                                    │
        └─┬───────────┬────────────┬────────────┬──────────┬─────────┘
          ▼           ▼            ▼            ▼          ▼
    ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ briefing │ │correlation │ │forecaster│ │ options  │ │ backtest │
    │   .py    │ │    .py     │ │   .py    │ │ sizer.py │ │   .py    │
    └────┬─────┘ └─────┬──────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘
         │             │              │            │            │
         └──────┬──────┴───────┬──────┘            │            │
                ▼              ▼                   ▼            ▼
       ┌──────────────┐  ┌────────────┐     ┌──────────┐  ┌──────────┐
       │ data layer   │  │ artifacts  │     │  chains  │  │ backtest │
       │ (existing    │  │ /parquet   │     │ snapshot │  │ artifact │
       │ fetchers +   │  │ nightly    │     │ (NSE/BSE)│  │ (parquet)│
       │ backfill)    │  │            │     │          │  │          │
       └──────────────┘  └────────────┘     └──────────┘  └──────────┘

           Narrative (LAST step, optional, never affects numbers):
           ┌──────────────────────────────────────────────────┐
           │ narrator.py: ReasoningContext (dict) → blurb str │
           │   - calls llm_client                             │
           │   - validates output (regex/length/no-numbers)   │
           │   - falls back to deterministic template on fail │
           └──────────────────────────────────────────────────┘
```

### 4.2 Determinism boundary

- **Deterministic pipeline (no LLM):** direction, strike, expiry, lots, lot size, premium estimate, entry/exit windows, T1, T2, stop-loss, underlying triggers, win probability, expected value, max loss, target P&L, R:R, confidence, mode gating, live status, backtest curves, correlation cells, briefing tile values, analog stats.
- **LLM-only:** card reasoning blurb (one paragraph) and "why no trade" sentence. Both produced from a frozen `ReasoningContext` dict whose numeric fields are interpolated, not generated.

The orchestrator never feeds raw prices into the LLM, never parses LLM JSON output, and never lets LLM output influence a numeric field.

### 4.3 LLM guardrails

The narrator runs after `GlobalTabView` is fully built:

1. Build deterministic template string from `ReasoningContext` (always works).
2. If `llm` is provided, attempt `llm.rewrite(template, ctx)`.
3. Validate output:
   - length ≤ 400 chars
   - contains no digit not present in `ctx`
   - contains no instrument name not present in `ctx`
4. On any validation failure → return template; log the rejected response to `data/artifacts/llm_rejected/`.

A unit test asserts: for fixed pipeline inputs, all numeric fields on the card are byte-identical across runs (LLM mocked to return junk vs. valid prose vs. None).

## 5. Data Flows

### 5.1 Flow A — Nightly batch (~22:00 IST)

`scripts/nightly_global.py`:

- Fetch global EOD: SPX, Nasdaq, FTSE, Nikkei, HSI, DXY, Brent, US10Y, VIX, India VIX.
- Fetch GIFT Nifty session.
- Refresh correlation artifact (windows 20d, 60d, 120d) → `data/artifacts/corr_YYYY-MM-DD.parquet`.
- Run walk-forward backtest for all three modes → `data/artifacts/backtest_YYYY-MM-DD.parquet`.
- Refresh historical-analogs index → `data/artifacts/analogs.sqlite`.

All writes are atomic (temp + rename). Idempotent.

### 5.2 Flow B — Pre-open snapshot (~08:45 IST)

`scripts/preopen_snapshot.py`:

- Fetch overnight recap (yesterday's US close levels).
- Fetch GIFT Nifty premium vs prior NIFTY close.
- Fetch options chain snapshot for NIFTY and BANKNIFTY.
- Run `orchestrator.compose(as_of=08:45, mode=ALL)` → `data/artifacts/view_YYYY-MM-DD_preopen.parquet` with one row per `(index, mode)`.

All three modes are pre-computed at this checkpoint so mode-switch in the UI is a parquet row lookup.

### 5.3 Flow C — Intraday checkpoints (09:30, 11:00, 13:00, 14:30 IST)

`scripts/intraday_checkpoint.py --t=HHMM`:

- Fetch intraday quotes for NIFTY/BANKNIFTY and options chain.
- `orchestrator.recompose(as_of=t, mode=ALL, prev_view=last_artifact)`:
  - Updates live status (`WAITING` → `ENTRY ZONE ACTIVE` → `IN POSITION` → exit states).
  - Recomputes live P&L for in-position cards.
  - Re-evaluates intraday entry/exit triggers.
- Writes `data/artifacts/view_YYYY-MM-DD_HHMM.parquet`.

Streamlit auto-refreshes every 60 s and reads the latest artifact. Render is parquet-read + format only — no network on the request path.

### 5.4 Streamlit request path

```
user opens tab
   │
   ├─ st.cache_data(ttl=60s) keyed on (date, mode, capital_bucket, latest_artifact_mtime)
   │     └─ load_view_artifact(date, latest) → GlobalTabView
   │            └─ resize_options_for_capital(capital)   # cheap re-sizing
   │
   └─ render_briefing_strip + render_heatmap + render_cards
        │
        └─ for each card: narrator.blurb(card.reasoning_context)
             # cached on (reasoning_context_hash); LLM only on cache miss
```

`capital_bucket = round(capital / 1000) * 1000` so 9,400 and 9,800 share a cache entry; lots are exact-sized after the cache hit via cheap arithmetic.

## 6. Components

New module: `india_quant/global_tab/`. Existing `signals/global_context.py` becomes a deprecated shim that re-exports `build_global_view` under its old name.

### 6.1 Types (`types.py`)

```python
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Literal

class Mode(str, Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"

class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"

class Status(str, Enum):
    WAITING = "waiting"
    ENTRY_ZONE_ACTIVE = "entry_zone_active"
    IN_POSITION = "in_position"
    TARGET_HIT = "target_hit"
    STOPPED_OUT = "stopped_out"
    EXPIRED_NO_ENTRY = "expired_no_entry"
    DATA_GAP = "data_gap"

@dataclass(frozen=True)
class BriefingTile:
    label: str
    value: str
    change_pct: float
    sentiment: Literal["bullish", "bearish", "neutral"]

@dataclass(frozen=True)
class BriefingStrip:
    as_of: datetime
    tiles: list[BriefingTile]
    predicted_gap_bps: dict[str, float]   # {"NIFTY": 35.0, "BANKNIFTY": 48.0}

@dataclass(frozen=True)
class CorrelationCell:
    asset_a: str
    asset_b: str
    rho_20d: float
    rho_60d: float

@dataclass(frozen=True)
class CorrelationHeatmap:
    as_of: date
    cells: list[CorrelationCell]

@dataclass(frozen=True)
class OptionsLeg:
    underlying: str
    strike: float
    option_type: Literal["CE", "PE"]
    expiry: date
    lot_size: int
    lots: int
    premium_estimate: float
    premium_zone: tuple[float, float]
    target_t1: float
    target_t2: float
    stop_loss: float
    underlying_entry_trigger: float
    underlying_target_t1: float
    underlying_target_t2: float
    underlying_stop_trigger: float

@dataclass(frozen=True)
class RiskReward:
    capital_deployed: float
    max_loss: float
    target_pnl_t1: float
    target_pnl_t2: float
    win_probability: float
    expected_value: float
    risk_reward_ratio: float

@dataclass(frozen=True)
class TimingWindow:
    entry_window_start: time
    entry_window_end: time
    exit_window_start: time
    exit_window_end: time
    invalidation_time: time

@dataclass(frozen=True)
class ReasoningContext:
    top_drivers: list[tuple[str, float]]
    analog_count: int
    analog_winrate: float
    analog_avg_pnl: float
    no_trade_reason_code: str | None

@dataclass(frozen=True)
class LiveTicket:
    status: Status
    live_pnl: float | None
    last_update: datetime

@dataclass(frozen=True)
class TradeTicket:
    index: str
    direction: Direction
    confidence: float
    leg: OptionsLeg | None
    timing: TimingWindow | None
    risk_reward: RiskReward | None
    reasoning: ReasoningContext
    live: LiveTicket
    blurb: str

@dataclass(frozen=True)
class GlobalTabView:
    as_of: datetime
    mode: Mode
    capital: float
    briefing: BriefingStrip
    heatmap: CorrelationHeatmap
    cards: list[TradeTicket]
    artifact_paths: dict[str, str]
    staleness: dict[str, datetime]
```

### 6.2 Module contracts

**`briefing.py`**
```python
def build_briefing(as_of: datetime, snapshot: GlobalSnapshot) -> BriefingStrip: ...
```
Pure. `GlobalSnapshot` is a dataclass holding fetched values; no I/O inside.

**`correlation.py`**
```python
def build_heatmap(as_of: date, history: pd.DataFrame) -> CorrelationHeatmap: ...
```
Pure. `history` is a pre-loaded EOD frame.

**`forecaster.py`**
```python
@dataclass(frozen=True)
class IndexForecast:
    index: str
    direction: Direction
    confidence: float
    expected_move_bps: float
    expected_move_low_bps: float
    expected_move_high_bps: float
    feature_attributions: list[tuple[str, float]]
    no_trade_reason_code: str | None

def forecast_index(
    index: str, as_of: datetime, mode: Mode,
    features: FeatureRow, model_artifact: ModelArtifact
) -> IndexForecast: ...
```
Pure given features + artifact. Default model family: LightGBM classifier (direction) + LightGBM quantile regressor (magnitude). Final hyperparameters and feature set are decided in the implementation plan; the *interface* is fixed by this spec.

**`options_sizer.py`**
```python
def size_trade(
    forecast: IndexForecast, capital: float, mode: Mode,
    chain_snapshot: OptionsChainSnapshot
) -> tuple[OptionsLeg, RiskReward, TimingWindow] | None: ...
```
Returns `None` when `forecast.direction == NO_TRADE`. Mode determines strike-selection rule:
- Aggressive → OTM-1
- Balanced → ATM
- Conservative → ITM-1

Lots = `floor((capital × max_loss_fraction) / (lot_size × premium))`. Exact formulas finalized in plan.

**`backtest.py`**
```python
def load_backtest_artifact(date: date) -> BacktestResult: ...
def run_walk_forward(start: date, end: date, mode: Mode) -> BacktestResult: ...
```
Walk-forward only. Uses `PointInTimeFeatureStore` so backtest and live share the feature-assembly code path (no train/serve skew).

**`narrator.py`**
```python
def blurb_for_ticket(ctx: ReasoningContext, llm: LLMClient | None) -> str: ...
```
1. Deterministic template from `ctx`.
2. Optional LLM rewrite.
3. Validator (length, no foreign digits, no foreign instrument names) → reject → template.

**`orchestrator.py`**
```python
def build_global_view(
    as_of: datetime, mode: Mode, capital: float,
    artifacts: ArtifactPaths, llm: LLMClient | None = None
) -> GlobalTabView: ...
```
Loads artifacts, calls each module in order, assembles `GlobalTabView`, calls narrator last.

### 6.3 Cross-cutting components

**`feature_store.py`**
```python
class PointInTimeFeatureStore:
    def get(self, feature: str, at: datetime) -> float:
        # Raises FuturePeekError if any data point used has timestamp > at.
```
Used by both live forecaster and backtest. Single source of truth for feature assembly.

**`cost_model.py`**
```python
def realized_pnl(leg: OptionsLeg, exit_premium: float) -> float:
    # Subtracts brokerage, STT (sell-side on options premium), exchange + SEBI
    # charges, GST, and a slippage estimate proportional to bid-ask of the
    # chosen strike.
```
Used for every trade in the backtest. No "before-cost" numbers are surfaced anywhere on the tab.

### 6.4 Streamlit page

`global_context_page` becomes a thin renderer (~80 lines):
- Reads capital input + mode dropdown.
- Calls `build_global_view`.
- Renders briefing strip, heatmap, cards, Backtest sub-tab.
- No business logic.

Testable by snapshot-rendering a fixture `GlobalTabView`.

## 7. Failure Semantics

Two-tier rule, enforced by types:

- **Numeric fields** (`float | None`) — missing/stale input → `None` → UI renders `—` with tooltip. Never substituted.
- **Narrative fields** (`str`) — silent fallback to template. Cosmetic.

### 7.1 Failure matrix

| Failure | Detection | UI behavior | Backend behavior |
|---|---|---|---|
| Nightly backtest job missed | `backtest_YYYY-MM-DD.parquet` absent | Backtest sub-tab banner: "Stats from <last available date>"; cards still render | Falls back to most recent within 7 days; raises `BacktestStaleError` if older |
| Pre-open snapshot missed | `view_*_preopen.parquet` absent | Card status `DATA_GAP`; ticket fields blank; briefing falls back to yesterday's close with `STALE` badge | Orchestrator returns view with empty `cards`; `staleness` populated |
| Intraday checkpoint missed | Latest artifact mtime > 90 min during market hours | Banner "Last updated HH:MM"; live status frozen | Auto-refresh keeps polling |
| Single global ticker fetch fails | Fetch returns `None` | Tile shows `—`; correlation cells `n/a`; forecaster proceeds; missing-feature flagged in `top_drivers` | Fetcher logs WARN; jobs continue |
| Options chain fetch fails | Empty snapshot for that index | `leg`, `risk_reward`, `timing` = `None`; status `DATA_GAP` | Pre-open job retries 3x with backoff |
| Forecaster model artifact missing/corrupt | Pickle load raises | Card direction `NO_TRADE`, `no_trade_reason_code = "model_unavailable"`; red banner | Logged ERROR; other indices continue |
| LLM narrator timeout / invalid output | Validator rejects | Template renders; `🤖 fallback` icon | Rejected response logged to `llm_rejected/` |
| Capital input invalid | Streamlit validator | Field red; previous view stays | No backend call |
| Mode switch | Cache miss | ~200 ms spinner | Parquet row lookup; no recompute |

### 7.2 Concurrency & atomicity

- All artifact writes use temp-file + atomic rename (`os.replace`).
- Page reads check `mtime`; mid-flight replacement → retry once.
- Scheduled jobs are independently runnable and idempotent.
- `scripts/lock.py` provides `fcntl.flock`-based job locks (NFS/iCloud-safe).

### 7.3 Logging & observability

- `structlog` per module. Every job logs `job_name`, `as_of`, `duration_ms`, `result_status`, `artifacts_written`.
- `data/artifacts/run_log.sqlite` records every job run; Backtest sub-tab shows a "Pipeline health" footer.
- LLM calls log prompt-hash, response-hash, tokens, latency, validator verdict. Rejections persisted under `data/artifacts/llm_rejected/`.

### 7.4 Hard invariants (test-enforced)

1. **Determinism.** `build_global_view` with identical artifacts and `llm=None` produces byte-identical `GlobalTabView` (excluding `as_of`).
2. **No-future-peek.** `PointInTimeFeatureStore.get(f, t)` raises if any required raw datapoint has timestamp > `t`. Property test fuzzes random `t` across the historical range.
3. **Cost-inclusive.** `BacktestResult.trades` carries `gross_pnl` and `net_pnl`; equity curve uses `net_pnl` only. Test asserts no UI surface reads `gross_pnl`.
4. **LLM-isolation.** 50 fixture views rendered with LLM mocked to return junk vs. mocked to return template — all numeric fields byte-identical.

## 8. Mode Configuration

Modes are pure-data configs in `modes.py`, not separate code paths.

| Mode | Min EV | Min win-prob | Strike rule | T1 multiple | T2 multiple | Stop multiple |
|---|---|---|---|---|---|---|
| Aggressive | EV > 0 | none | OTM-1 | 1.5× premium | 2.5× premium | 0.6× premium |
| Balanced | EV > 0 *and* p > 0.55 | 0.55 | ATM | 1.4× premium | 2.0× premium | 0.7× premium |
| Conservative | top-decile analog match | 0.62 | ITM-1 | 1.25× premium | 1.6× premium | 0.8× premium |

(Multiples are placeholders; finalized in plan via backtest sweep.)

When `direction = NO_TRADE`, the card always renders with the no-trade reason code surfaced in plain English ("GIFT Nifty premium within noise band; no overnight catalyst").

## 9. Testing Strategy

Five layers.

### 9.1 Unit tests

`tests/global_tab/test_<module>.py` for every pure function. Behavior-focused: exact strike + lots for fixed inputs; mode-by-mode strike rule; cost model verified field-by-field; narrator validator accept/reject cases; rolling correlations against numpy reference.

### 9.2 Property-based tests (hypothesis)

- No-future-peek over random `(feature, as_of)`.
- Deterministic orchestrator over random `(capital, mode)`.
- Capital monotonicity: lots non-decreasing in capital (mode + forecast fixed).
- Renderer never introduces arithmetic.

### 9.3 Integration tests

`tests/global_tab/integration/fixtures/2024-09-23/` holds golden artifacts for one historical event-day. `test_full_pipeline_2024_09_23.py` runs the orchestrator end-to-end and asserts the resulting `GlobalTabView` matches a checked-in snapshot JSON. `--update-snapshots` regenerates.

### 9.4 Backtest-as-test

- **Reproducibility gate:** `run_walk_forward(2018-01-01, 2024-12-31, mode=Balanced)` produces an `equity_curve` whose hash matches `tests/global_tab/golden/backtest_balanced_2018_2024.hash`. Code changes that alter historical decisions fail this loudly.
- **Cost-inclusive sanity:** after-cost Sharpe ≥ low floor (e.g., 0.3) on the full window across all modes.
- **Nightly reports** (not CI): per-mode equity-curve PDF + no-trade-rate report. The no-trade-rate report is the explicit antidote to the "silent every day" concern: Balanced mode's no-trade rate must sit between ~10% and ~80% of session-days for the gates to be considered well-tuned.

### 9.5 Streamlit smoke test

`tests/global_tab/test_page_smoke.py` uses `AppTest` to render the page against a fixture view. Asserts cards render, controls present, Backtest sub-tab loads, mocked LLM rejection still produces a fully-rendered card.

### 9.6 Explicitly not tested

- Forecast accuracy on unseen data — that's what walk-forward backtesting *is*; live paper-trading vs. backtest expectation will be the unseen-data test.
- LLM aesthetic quality — validator gates correctness; aesthetics monitored manually via `llm_rejected/`.
- Broker integration — out of scope.

## 10. Project Layout

```
Trading/
├── india_quant/
│   ├── global_tab/                       # NEW
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── modes.py
│   │   ├── feature_store.py
│   │   ├── cost_model.py
│   │   ├── briefing.py
│   │   ├── correlation.py
│   │   ├── forecaster.py
│   │   ├── options_sizer.py
│   │   ├── backtest.py
│   │   ├── narrator.py
│   │   └── orchestrator.py
│   │
│   ├── signals/
│   │   └── global_context.py             # SHIM, deprecated
│   │
│   └── dashboard/
│       └── app.py                        # global_context_page rewritten
│
├── scripts/
│   ├── nightly_global.py                 # NEW
│   ├── preopen_snapshot.py               # NEW
│   ├── intraday_checkpoint.py            # NEW
│   └── lock.py                           # NEW
│
├── data/
│   └── artifacts/                        # NEW
│       ├── corr_YYYY-MM-DD.parquet
│       ├── backtest_YYYY-MM-DD.parquet
│       ├── view_YYYY-MM-DD_preopen.parquet
│       ├── view_YYYY-MM-DD_HHMM.parquet
│       ├── analogs.sqlite
│       ├── run_log.sqlite
│       └── llm_rejected/
│
├── tests/
│   └── global_tab/                       # NEW
│       ├── test_briefing.py
│       ├── test_correlation.py
│       ├── test_forecaster.py
│       ├── test_options_sizer.py
│       ├── test_cost_model.py
│       ├── test_narrator.py
│       ├── test_feature_store.py
│       ├── test_orchestrator_determinism.py
│       ├── test_page_smoke.py
│       ├── integration/
│       │   ├── fixtures/2024-09-23/
│       │   └── test_full_pipeline_2024_09_23.py
│       └── golden/
│           └── backtest_balanced_2018_2024.hash
│
└── reports/
    ├── equity_curve_<mode>_YYYY-MM-DD.pdf
    └── no_trade_rate_YYYY-MM-DD.csv
```

## 11. Implementation Sequencing

Phases are demoable end-states. The implementation plan turns these into stepwise tasks.

1. **Phase 1 — Spine.** Types, mode configs, point-in-time feature store, cost model, unit + property tests. Demo: pytest output.
2. **Phase 2 — Data + briefing + heatmap.** Audit existing `fetchers/` and `data/backfill_global.py`; extend for GIFT Nifty / DXY / VIX as needed. Wire nightly + pre-open jobs. Demo: tab top-strip + heatmap fed by real artifacts.
3. **Phase 3 — Forecaster + sizer + orchestrator (no LLM).** First end-to-end view with cards. Template blurbs only. Snapshot test passes. Demo: full tab in browser.
4. **Phase 4 — Backtest + sub-tab + golden-hash gate.** Walk-forward in CI. Demo: equity curves + Backtest sub-tab.
5. **Phase 5 — Intraday checkpoints + live status transitions.** Demo: replay an intraday session; cards flip through statuses.
6. **Phase 6 — Narrator + validator + LLM-isolation test.** Demo: card with LLM blurb vs. template (toggle).
7. **Phase 7 — Failure-path hardening + observability + cleanup.** All failure-matrix cells exercised. Shim removed. Pipeline-health footer live.

Each phase ends in a working, tested system. Stopping mid-roadmap leaves no broken state.

## 12. Decisions Deferred to Plan

These are confirmed defaults; the implementation plan revisits each:

1. **Scheduler:** `launchd` LaunchAgent on macOS (cron-portable).
2. **Forecaster model family:** LightGBM classifier (direction) + LightGBM quantile regressor (magnitude). Hyperparameters and feature set finalized via Phase-3 backtest sweep.
3. **Data sources for global EOD / GIFT Nifty / options chain:** audit `fetchers/` and `data/backfill_global.py` in Phase 2; reuse where possible, extend where missing.
4. **Mode threshold values:** placeholders in §8; finalized via Phase-4 backtest sweep so Balanced no-trade-rate lands in [10%, 80%] of session-days.

## 13. Open Risks

- **Strategy edge may not survive costs.** The cost-inclusive Sharpe gate forces this to surface in CI rather than after live deployment. If the gate fails, the strategy is wrong, not the test.
- **GIFT Nifty data licensing.** SGX/NSE-IFSC redistribution terms vary by source. Phase 2 audit must confirm a license-clean source before relying on GIFT Nifty as a primary feature.
- **Walk-forward window shift.** As live paper-trading produces post-launch data, retraining must keep that window walled off from future training data — easy to forget in a year. Add a dated retraining checklist to `india_quant/global_tab/README.md` in Phase 7.
- **Streamlit cache invalidation correctness.** If `mtime` checks miss a file replacement, the page can serve stale data. Mitigated by atomic-rename + double-check on read; failure mode is staleness banner, not silent stale numbers.
