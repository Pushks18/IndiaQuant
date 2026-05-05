# Global Tab Revamp — Phase 1 (Spine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the type system, mode configuration, point-in-time feature store, and realistic cost model that the rest of the Global tab pipeline depends on. No UI, no fetchers, no orchestrator yet — just the deterministic spine, fully tested.

**Architecture:** New package `india_quant/global_tab/` with four pure-Python modules (`types.py`, `modes.py`, `feature_store.py`, `cost_model.py`) and a parallel `tests/global_tab/` test tree. Everything is pure functions or frozen dataclasses; no I/O, no globals. The feature store enforces the no-future-peek invariant structurally so backtest and live forecasting share one feature-assembly path.

**Tech Stack:** Python 3.11+, pandas 2.2, pydantic 2.10 (already in repo), pytest 8.3 (already), hypothesis (NEW — added to requirements), LightGBM/scikit-learn (already; not used in Phase 1 but reserved for Phase 3).

**Spec reference:** `docs/superpowers/specs/2026-05-05-global-tab-revamp-design.md` §6.1 (types), §8 (modes), §6.3 (feature store, cost model), §9.1–9.2 (testing).

**Phase end-state demo:** `pytest tests/global_tab -v` passes with property tests; nothing else changes (no UI, no scripts).

---

## File Structure

**New files (created in this phase):**
- `india_quant/global_tab/__init__.py` — empty package marker
- `india_quant/global_tab/types.py` — all `@dataclass(frozen=True)` types + `Mode`, `Direction`, `Status` enums
- `india_quant/global_tab/modes.py` — `ModeConfig` dataclass + `MODE_CONFIGS` dict
- `india_quant/global_tab/feature_store.py` — `FuturePeekError`, `PointInTimeFeatureStore`
- `india_quant/global_tab/cost_model.py` — `CostBreakdown`, `compute_costs()`, `realized_pnl()`
- `tests/global_tab/__init__.py` — empty
- `tests/global_tab/test_types.py`
- `tests/global_tab/test_modes.py`
- `tests/global_tab/test_feature_store.py`
- `tests/global_tab/test_cost_model.py`

**Modified files:**
- `requirements.txt` — add `hypothesis==6.122.3` and `pyarrow==18.1.0` (parquet I/O reserved for later phases; pin now to lock the version)

**Files NOT touched in Phase 1:**
- `india_quant/dashboard/app.py` — UI work is Phase 2+
- `india_quant/signals/global_context.py` — shim conversion is Phase 7
- Any fetcher / data layer file — Phase 2

---

## Conventions

- **Test file path:** `tests/global_tab/test_<module>.py` mirrors `india_quant/global_tab/<module>.py`. Run with `pytest tests/global_tab -v` from the repo root.
- **Imports** in tests use the absolute path: `from india_quant.global_tab.types import Mode`.
- **Logging:** existing repo uses `loguru`; this phase has no logging (pure functions only).
- **Currency:** all money values are `float` representing rupees (₹). Premium values are per-share rupees; lot-level money is computed by multiplying.
- **Dates / times:** `datetime.date` for trading dates, `datetime.datetime` (timezone-naive, IST is implicit) for intraday timestamps, `datetime.time` for time-of-day.
- **Commits:** one commit per task, conventional-commits style (`feat:`, `test:`, `chore:`). The user runs `git commit` themselves; the plan only suggests the message.

---

## Task 1: Set up package skeleton + add test dependencies

**Files:**
- Create: `india_quant/global_tab/__init__.py`
- Create: `tests/global_tab/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create the package directories and empty init files**

```bash
mkdir -p india_quant/global_tab tests/global_tab
touch india_quant/global_tab/__init__.py tests/global_tab/__init__.py
```

- [ ] **Step 2: Append the two new dependencies to `requirements.txt`**

Add at the end of the file (under existing `# Testing` and a new `# Storage` section):

```
hypothesis==6.122.3

# Storage (parquet)
pyarrow==18.1.0
```

- [ ] **Step 3: Install the new dependencies in the existing venv**

Run: `cd /Users/pushkaraj/Documents/Trading && source venv/bin/activate && pip install hypothesis==6.122.3 pyarrow==18.1.0`
Expected: `Successfully installed hypothesis-6.122.3 pyarrow-18.1.0` (and any sub-deps).

- [ ] **Step 4: Verify pytest can collect the empty test package**

Run: `pytest tests/global_tab -v`
Expected: `collected 0 items` (no tests yet, no errors).

- [ ] **Step 5: Suggest commit (user runs)**

```bash
git add india_quant/global_tab/__init__.py tests/global_tab/__init__.py requirements.txt
git commit -m "chore: scaffold global_tab package + add hypothesis/pyarrow"
```

---

## Task 2: Define enums in `types.py` (Mode, Direction, Status)

**Files:**
- Create: `india_quant/global_tab/types.py`
- Create: `tests/global_tab/test_types.py`

- [ ] **Step 1: Write the failing tests**

`tests/global_tab/test_types.py`:

```python
"""Tests for india_quant.global_tab.types."""
from india_quant.global_tab.types import Direction, Mode, Status


def test_mode_values():
    assert Mode.AGGRESSIVE.value == "aggressive"
    assert Mode.BALANCED.value == "balanced"
    assert Mode.CONSERVATIVE.value == "conservative"


def test_mode_membership():
    assert set(Mode) == {Mode.AGGRESSIVE, Mode.BALANCED, Mode.CONSERVATIVE}


def test_direction_values():
    assert Direction.LONG.value == "long"
    assert Direction.SHORT.value == "short"
    assert Direction.NO_TRADE.value == "no_trade"


def test_status_values():
    assert Status.WAITING.value == "waiting"
    assert Status.ENTRY_ZONE_ACTIVE.value == "entry_zone_active"
    assert Status.IN_POSITION.value == "in_position"
    assert Status.TARGET_HIT.value == "target_hit"
    assert Status.STOPPED_OUT.value == "stopped_out"
    assert Status.EXPIRED_NO_ENTRY.value == "expired_no_entry"
    assert Status.DATA_GAP.value == "data_gap"


def test_enums_are_str_subclasses():
    """Round-trip through json without a custom encoder."""
    import json
    payload = {"mode": Mode.BALANCED, "direction": Direction.LONG, "status": Status.WAITING}
    assert json.loads(json.dumps(payload)) == {"mode": "balanced", "direction": "long", "status": "waiting"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/global_tab/test_types.py -v`
Expected: `ImportError: cannot import name 'Direction' from 'india_quant.global_tab.types'` (file doesn't exist yet).

- [ ] **Step 3: Create `types.py` with the three enums**

`india_quant/global_tab/types.py`:

```python
"""Type definitions for the global_tab package.

This module is pure: only dataclasses and enums, no I/O, no logic.
Every dataclass is frozen so views are immutable once produced.
"""
from enum import Enum


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/global_tab/test_types.py -v`
Expected: 5 passed.

- [ ] **Step 5: Suggest commit**

```bash
git add india_quant/global_tab/types.py tests/global_tab/test_types.py
git commit -m "feat(global_tab): add Mode/Direction/Status enums"
```

---

## Task 3: Add dataclasses to `types.py` (BriefingTile through GlobalTabView)

**Files:**
- Modify: `india_quant/global_tab/types.py` (append)
- Modify: `tests/global_tab/test_types.py` (append)

- [ ] **Step 1: Append failing tests for the dataclasses**

Append to `tests/global_tab/test_types.py`:

```python
import dataclasses
from datetime import date, datetime, time

import pytest

from india_quant.global_tab.types import (
    BriefingStrip,
    BriefingTile,
    CorrelationCell,
    CorrelationHeatmap,
    GlobalTabView,
    LiveTicket,
    OptionsLeg,
    ReasoningContext,
    RiskReward,
    TimingWindow,
    TradeTicket,
)


def _sample_leg() -> OptionsLeg:
    return OptionsLeg(
        underlying="NIFTY",
        strike=24350.0,
        option_type="CE",
        expiry=date(2026, 5, 8),
        lot_size=75,
        lots=2,
        premium_estimate=142.0,
        premium_zone=(138.0, 148.0),
        target_t1=199.0,
        target_t2=284.0,
        stop_loss=99.0,
        underlying_entry_trigger=24310.0,
        underlying_target_t1=24420.0,
        underlying_target_t2=24500.0,
        underlying_stop_trigger=24220.0,
    )


def test_options_leg_is_frozen():
    leg = _sample_leg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        leg.lots = 99  # type: ignore[misc]


def test_options_leg_premium_zone_is_tuple():
    leg = _sample_leg()
    assert isinstance(leg.premium_zone, tuple)
    assert leg.premium_zone[0] < leg.premium_zone[1]


def test_risk_reward_round_trip():
    rr = RiskReward(
        capital_deployed=10000.0,
        max_loss=14850.0,
        target_pnl_t1=8550.0,
        target_pnl_t2=21300.0,
        win_probability=0.62,
        expected_value=4860.0,
        risk_reward_ratio=1.43,
    )
    assert dataclasses.asdict(rr)["win_probability"] == pytest.approx(0.62)


def test_timing_window_construction():
    tw = TimingWindow(
        entry_window_start=time(9, 18),
        entry_window_end=time(9, 25),
        exit_window_start=time(14, 30),
        exit_window_end=time(15, 15),
        invalidation_time=time(11, 0),
    )
    assert tw.entry_window_start < tw.entry_window_end


def test_reasoning_context_no_trade_optional():
    ctx = ReasoningContext(
        top_drivers=[("gift_nifty_premium_bps", 60.0)],
        analog_count=47,
        analog_winrate=0.64,
        analog_avg_pnl=3200.0,
        no_trade_reason_code=None,
    )
    assert ctx.no_trade_reason_code is None


def test_trade_ticket_no_trade_has_none_leg():
    ticket = TradeTicket(
        index="NIFTY",
        direction=Direction.NO_TRADE,
        confidence=12.0,
        leg=None,
        timing=None,
        risk_reward=None,
        reasoning=ReasoningContext(
            top_drivers=[],
            analog_count=0,
            analog_winrate=0.0,
            analog_avg_pnl=0.0,
            no_trade_reason_code="gift_premium_in_noise_band",
        ),
        live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=datetime(2026, 5, 5, 8, 45)),
        blurb="No trade today: GIFT Nifty premium is within the noise band.",
    )
    assert ticket.leg is None
    assert ticket.risk_reward is None


def test_global_tab_view_assembles():
    view = GlobalTabView(
        as_of=datetime(2026, 5, 5, 8, 45),
        mode=Mode.BALANCED,
        capital=10000.0,
        briefing=BriefingStrip(
            as_of=datetime(2026, 5, 5, 8, 45),
            tiles=[BriefingTile(label="SPX", value="5,612.40", change_pct=1.10, sentiment="bullish")],
            predicted_gap_bps={"NIFTY": 35.0, "BANKNIFTY": 48.0},
        ),
        heatmap=CorrelationHeatmap(
            as_of=date(2026, 5, 5),
            cells=[CorrelationCell(asset_a="NIFTY", asset_b="SPX", rho_20d=0.45, rho_60d=0.52)],
        ),
        cards=[],
        artifact_paths={},
        staleness={},
    )
    assert view.mode == Mode.BALANCED
    assert view.cards == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/global_tab/test_types.py -v`
Expected: ImportErrors for the new dataclass names.

- [ ] **Step 3: Append dataclass definitions to `types.py`**

Append to `india_quant/global_tab/types.py`:

```python
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal


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
    predicted_gap_bps: dict[str, float]


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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/global_tab/test_types.py -v`
Expected: 11 passed.

- [ ] **Step 5: Suggest commit**

```bash
git add india_quant/global_tab/types.py tests/global_tab/test_types.py
git commit -m "feat(global_tab): add frozen dataclasses for view/cards/legs"
```

---

## Task 4: `modes.py` — `ModeConfig` and `MODE_CONFIGS`

**Files:**
- Create: `india_quant/global_tab/modes.py`
- Create: `tests/global_tab/test_modes.py`

The mode config encodes the gating rules from spec §8 as data, not branching code. Strike rule values (`"otm_1"`, `"atm"`, `"itm_1"`) are interpreted by `options_sizer` in Phase 3 — Phase 1 only stores them.

- [ ] **Step 1: Write the failing tests**

`tests/global_tab/test_modes.py`:

```python
"""Tests for mode threshold configurations."""
import pytest

from india_quant.global_tab.modes import MODE_CONFIGS, ModeConfig
from india_quant.global_tab.types import Mode


def test_all_modes_have_config():
    assert set(MODE_CONFIGS) == {Mode.AGGRESSIVE, Mode.BALANCED, Mode.CONSERVATIVE}


def test_aggressive_has_loosest_gates():
    cfg = MODE_CONFIGS[Mode.AGGRESSIVE]
    assert cfg.min_expected_value == 0.0
    assert cfg.min_win_probability == 0.0
    assert cfg.strike_rule == "otm_1"


def test_balanced_thresholds():
    cfg = MODE_CONFIGS[Mode.BALANCED]
    assert cfg.min_expected_value == 0.0
    assert cfg.min_win_probability == pytest.approx(0.55)
    assert cfg.strike_rule == "atm"


def test_conservative_thresholds():
    cfg = MODE_CONFIGS[Mode.CONSERVATIVE]
    assert cfg.min_win_probability == pytest.approx(0.62)
    assert cfg.strike_rule == "itm_1"
    assert cfg.require_top_decile_analog is True


def test_target_stop_multiples_descend_with_caution():
    """Conservative mode = lower target multiple, higher stop floor."""
    a = MODE_CONFIGS[Mode.AGGRESSIVE]
    b = MODE_CONFIGS[Mode.BALANCED]
    c = MODE_CONFIGS[Mode.CONSERVATIVE]
    assert a.target_t1_multiple > b.target_t1_multiple > c.target_t1_multiple
    assert a.target_t2_multiple > b.target_t2_multiple > c.target_t2_multiple
    assert a.stop_loss_multiple < b.stop_loss_multiple < c.stop_loss_multiple


def test_mode_config_is_frozen():
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        MODE_CONFIGS[Mode.BALANCED].min_win_probability = 0.99  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/global_tab/test_modes.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `modes.py`**

`india_quant/global_tab/modes.py`:

```python
"""Mode threshold configurations.

Modes are pure data, not separate code paths. The forecaster, options sizer,
and backtester all read MODE_CONFIGS rather than branching on Mode.

Threshold values here are PLACEHOLDERS finalized via Phase-4 backtest sweep
(target: Balanced no-trade-rate in [10%, 80%] of session-days).
See spec §8 and §12.4.
"""
from dataclasses import dataclass
from typing import Literal

from india_quant.global_tab.types import Mode

StrikeRule = Literal["itm_1", "atm", "otm_1"]


@dataclass(frozen=True)
class ModeConfig:
    min_expected_value: float
    min_win_probability: float
    strike_rule: StrikeRule
    target_t1_multiple: float    # T1 = entry_premium * multiple
    target_t2_multiple: float
    stop_loss_multiple: float    # SL = entry_premium * multiple (multiple < 1)
    require_top_decile_analog: bool


MODE_CONFIGS: dict[Mode, ModeConfig] = {
    Mode.AGGRESSIVE: ModeConfig(
        min_expected_value=0.0,
        min_win_probability=0.0,
        strike_rule="otm_1",
        target_t1_multiple=1.5,
        target_t2_multiple=2.5,
        stop_loss_multiple=0.6,
        require_top_decile_analog=False,
    ),
    Mode.BALANCED: ModeConfig(
        min_expected_value=0.0,
        min_win_probability=0.55,
        strike_rule="atm",
        target_t1_multiple=1.4,
        target_t2_multiple=2.0,
        stop_loss_multiple=0.7,
        require_top_decile_analog=False,
    ),
    Mode.CONSERVATIVE: ModeConfig(
        min_expected_value=0.0,
        min_win_probability=0.62,
        strike_rule="itm_1",
        target_t1_multiple=1.25,
        target_t2_multiple=1.6,
        stop_loss_multiple=0.8,
        require_top_decile_analog=True,
    ),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/global_tab/test_modes.py -v`
Expected: 6 passed.

- [ ] **Step 5: Suggest commit**

```bash
git add india_quant/global_tab/modes.py tests/global_tab/test_modes.py
git commit -m "feat(global_tab): add ModeConfig and per-mode thresholds"
```

---

## Task 5: `feature_store.py` — `PointInTimeFeatureStore` (unit tests)

**Files:**
- Create: `india_quant/global_tab/feature_store.py`
- Create: `tests/global_tab/test_feature_store.py`

**Design:** the store holds `pd.Series` per feature, indexed by `pd.Timestamp`. `get(name, at)` returns the most recent value with `index <= at`. If no such value exists (only future data is registered), it raises `FuturePeekError` — that name is correct because semantically the only data available is in the future relative to `at`.

A separate `KeyError` (let it propagate from the dict) is raised when the feature name itself is unknown.

- [ ] **Step 1: Write the failing unit tests**

`tests/global_tab/test_feature_store.py`:

```python
"""Tests for PointInTimeFeatureStore."""
from datetime import datetime

import pandas as pd
import pytest

from india_quant.global_tab.feature_store import (
    FuturePeekError,
    PointInTimeFeatureStore,
)


def _series(*pairs):
    idx = pd.DatetimeIndex([ts for ts, _ in pairs])
    vals = [v for _, v in pairs]
    return pd.Series(vals, index=idx, dtype=float)


def test_get_returns_most_recent_value_at_or_before_time():
    store = PointInTimeFeatureStore()
    store.register(
        "spx_close",
        _series(
            (datetime(2026, 5, 1, 16, 0), 5600.0),
            (datetime(2026, 5, 2, 16, 0), 5612.0),
            (datetime(2026, 5, 3, 16, 0), 5630.0),
        ),
    )
    assert store.get("spx_close", datetime(2026, 5, 2, 23, 0)) == 5612.0


def test_get_at_exact_timestamp_returns_that_value():
    store = PointInTimeFeatureStore()
    store.register("dxy", _series((datetime(2026, 5, 2, 16, 0), 104.2)))
    assert store.get("dxy", datetime(2026, 5, 2, 16, 0)) == 104.2


def test_get_before_first_observation_raises_future_peek_error():
    store = PointInTimeFeatureStore()
    store.register("vix", _series((datetime(2026, 5, 5, 16, 0), 13.4)))
    with pytest.raises(FuturePeekError) as exc:
        store.get("vix", datetime(2026, 5, 1, 0, 0))
    assert "vix" in str(exc.value)
    assert "2026-05-01" in str(exc.value)


def test_get_unknown_feature_raises_key_error():
    store = PointInTimeFeatureStore()
    with pytest.raises(KeyError):
        store.get("never_registered", datetime(2026, 5, 5))


def test_register_rejects_non_datetime_index():
    store = PointInTimeFeatureStore()
    bad = pd.Series([1.0, 2.0], index=[0, 1], dtype=float)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        store.register("bad", bad)


def test_register_rejects_unsorted_index():
    store = PointInTimeFeatureStore()
    bad = _series(
        (datetime(2026, 5, 3), 1.0),
        (datetime(2026, 5, 1), 2.0),
    )
    with pytest.raises(ValueError, match="monotonically increasing"):
        store.register("bad", bad)


def test_register_overwrites_existing_feature():
    store = PointInTimeFeatureStore()
    store.register("x", _series((datetime(2026, 5, 1), 1.0)))
    store.register("x", _series((datetime(2026, 5, 1), 2.0)))
    assert store.get("x", datetime(2026, 5, 1)) == 2.0


def test_features_method_lists_registered_names():
    store = PointInTimeFeatureStore()
    store.register("a", _series((datetime(2026, 5, 1), 1.0)))
    store.register("b", _series((datetime(2026, 5, 1), 2.0)))
    assert sorted(store.features()) == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/global_tab/test_feature_store.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `feature_store.py`**

`india_quant/global_tab/feature_store.py`:

```python
"""Point-in-time feature store.

The store guarantees the no-future-peek invariant structurally: get(name, at)
can never return a value derived from data with timestamp > at, because every
feature is stored as a sorted pandas Series and lookup is asof <=.

This module is used by both the live forecaster and the walk-forward backtest
so feature assembly is identical across train and serve. Train/serve skew is
impossible by construction.
"""
from datetime import datetime

import pandas as pd


class FuturePeekError(LookupError):
    """Raised when no observation exists at-or-before the requested time.

    The semantics: from the caller's point in time, the only data available
    for this feature is in the future. Returning anything would be a peek.
    """


class PointInTimeFeatureStore:
    def __init__(self) -> None:
        self._series: dict[str, pd.Series] = {}

    def register(self, name: str, values: pd.Series) -> None:
        if not isinstance(values.index, pd.DatetimeIndex):
            raise TypeError(
                f"feature {name!r} must have a DatetimeIndex, got {type(values.index).__name__}"
            )
        if not values.index.is_monotonic_increasing:
            raise ValueError(f"feature {name!r} index must be monotonically increasing")
        self._series[name] = values

    def get(self, name: str, at: datetime) -> float:
        series = self._series[name]  # KeyError on unknown feature
        ts = pd.Timestamp(at)
        eligible = series.loc[series.index <= ts]
        if eligible.empty:
            raise FuturePeekError(
                f"feature {name!r} has no observation at or before {ts.isoformat()}"
            )
        return float(eligible.iloc[-1])

    def features(self) -> list[str]:
        return list(self._series)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/global_tab/test_feature_store.py -v`
Expected: 8 passed.

- [ ] **Step 5: Suggest commit**

```bash
git add india_quant/global_tab/feature_store.py tests/global_tab/test_feature_store.py
git commit -m "feat(global_tab): add PointInTimeFeatureStore with no-future-peek guard"
```

---

## Task 6: `feature_store.py` — property test for no-future-peek

**Files:**
- Modify: `tests/global_tab/test_feature_store.py` (append)

This is the bias-defense regression test. For random `(timestamps, query_at)` pairs, the value returned must always come from a timestamp `<= query_at`.

- [ ] **Step 1: Append the property test**

Append to `tests/global_tab/test_feature_store.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def _series_and_query(draw):
    n = draw(st.integers(min_value=1, max_value=40))
    base = datetime(2020, 1, 1)
    offsets = sorted(draw(st.lists(st.integers(min_value=0, max_value=3650), min_size=n, max_size=n, unique=True)))
    timestamps = [base + pd.Timedelta(days=o) for o in offsets]
    values = draw(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    series = pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float)
    query_offset = draw(st.integers(min_value=-50, max_value=4000))
    query_at = base + pd.Timedelta(days=query_offset)
    return series, query_at


@settings(max_examples=300, deadline=None)
@given(_series_and_query())
def test_property_no_future_peek(series_and_query):
    series, query_at = series_and_query
    store = PointInTimeFeatureStore()
    store.register("f", series)

    eligible = series.loc[series.index <= pd.Timestamp(query_at)]
    if eligible.empty:
        with pytest.raises(FuturePeekError):
            store.get("f", query_at)
        return

    value = store.get("f", query_at)
    # The returned value must equal the LATEST observation with index <= query_at.
    assert value == float(eligible.iloc[-1])
    # And the source timestamp of that observation must be <= query_at.
    source_ts = eligible.index[-1]
    assert source_ts <= pd.Timestamp(query_at)
```

- [ ] **Step 2: Run the property test**

Run: `pytest tests/global_tab/test_feature_store.py::test_property_no_future_peek -v`
Expected: 1 passed (after hypothesis explores 300 examples).

- [ ] **Step 3: Run the full feature-store test file to confirm nothing regressed**

Run: `pytest tests/global_tab/test_feature_store.py -v`
Expected: 9 passed.

- [ ] **Step 4: Suggest commit**

```bash
git add tests/global_tab/test_feature_store.py
git commit -m "test(global_tab): property test for no-future-peek invariant"
```

---

## Task 7: `cost_model.py` — Indian options realistic cost model

**Files:**
- Create: `india_quant/global_tab/cost_model.py`
- Create: `tests/global_tab/test_cost_model.py`

**Cost model** (Zerodha-style, options on NSE; values current as of 2025 — finalize broker-specific in Phase 4):

For one round-trip of `qty = lots * lot_size` units at premiums `entry_premium` (buy) and `exit_premium` (sell):

| Component | Buy side | Sell side |
|---|---|---|
| Brokerage | min(₹20, 0.03% × premium × qty) | min(₹20, 0.03% × premium × qty) |
| STT | 0 | 0.0625% × exit_premium × qty |
| Exchange transaction (NSE) | 0.03503% × entry_premium × qty | 0.03503% × exit_premium × qty |
| SEBI charges | ₹10 / 1 crore × entry_premium × qty | ₹10 / 1 crore × exit_premium × qty |
| Stamp duty | 0.003% × entry_premium × qty | 0 |
| GST (18%) | 18% × (brokerage_buy + exch_buy + sebi_buy) | 18% × (brokerage_sell + exch_sell + sebi_sell) |

Plus a slippage estimate: `slippage_per_side = bid_ask_spread * 0.5 * qty`. Phase 1 takes `bid_ask_spread` as an argument; Phase 2 will fetch it from the chain snapshot.

**Realized P&L:**
```
gross_pnl = (exit_premium - entry_premium) * qty   # for LONG; flipped for SHORT
total_costs = sum_of_all_components_above + 2 * slippage_per_side
net_pnl = gross_pnl - total_costs
```

- [ ] **Step 1: Write the failing tests, component-by-component**

`tests/global_tab/test_cost_model.py`:

```python
"""Tests for cost_model: Indian options realistic costs."""
import pytest

from india_quant.global_tab.cost_model import (
    CostBreakdown,
    compute_costs,
    realized_pnl,
)
from india_quant.global_tab.types import Direction


def test_brokerage_capped_at_20_per_side():
    # qty=150, premium=200 → 0.03% × 30000 = 9, well below cap; brokerage = 9 each side.
    cb = compute_costs(entry_premium=200.0, exit_premium=210.0, qty=150, bid_ask_spread=0.0)
    assert cb.brokerage == pytest.approx(2 * 9.0)


def test_brokerage_uses_cap_for_large_notional():
    # qty=1000, premium=200 → 0.03% × 200000 = 60 > 20, so brokerage = 20 each side.
    cb = compute_costs(entry_premium=200.0, exit_premium=210.0, qty=1000, bid_ask_spread=0.0)
    assert cb.brokerage == pytest.approx(2 * 20.0)


def test_stt_only_on_sell_side():
    # STT = 0.0625% × exit_premium × qty
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    assert cb.stt == pytest.approx(0.000625 * 120.0 * 75)


def test_exchange_charges_both_sides():
    # NSE = 0.03503% on each side
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    expected = 0.0003503 * 100.0 * 75 + 0.0003503 * 120.0 * 75
    assert cb.exchange == pytest.approx(expected)


def test_sebi_charges_both_sides():
    # SEBI = ₹10 per crore = 1e-6 of turnover
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    expected = 1e-6 * 100.0 * 75 + 1e-6 * 120.0 * 75
    assert cb.sebi == pytest.approx(expected)


def test_stamp_duty_only_on_buy_side():
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    assert cb.stamp_duty == pytest.approx(0.00003 * 100.0 * 75)


def test_gst_is_18_percent_of_brokerage_plus_exchange_plus_sebi():
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    expected = 0.18 * (cb.brokerage + cb.exchange + cb.sebi)
    assert cb.gst == pytest.approx(expected)


def test_slippage_is_half_spread_per_side_total_two_sides():
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.4)
    # 0.5 * 0.4 * 75 per side, 2 sides
    assert cb.slippage == pytest.approx(2 * 0.5 * 0.4 * 75)


def test_total_is_sum_of_all_components():
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.4)
    assert cb.total == pytest.approx(
        cb.brokerage + cb.stt + cb.exchange + cb.sebi + cb.stamp_duty + cb.gst + cb.slippage
    )


def test_realized_pnl_long_winning_trade():
    # qty=150, +20 per share = 3000 gross
    pnl = realized_pnl(
        direction=Direction.LONG,
        entry_premium=100.0,
        exit_premium=120.0,
        qty=150,
        bid_ask_spread=0.4,
    )
    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=150, bid_ask_spread=0.4)
    assert pnl == pytest.approx(20.0 * 150 - cb.total)


def test_realized_pnl_short_winning_trade():
    # SHORT: profit when exit < entry
    pnl = realized_pnl(
        direction=Direction.SHORT,
        entry_premium=120.0,
        exit_premium=100.0,
        qty=150,
        bid_ask_spread=0.4,
    )
    cb = compute_costs(entry_premium=120.0, exit_premium=100.0, qty=150, bid_ask_spread=0.4)
    assert pnl == pytest.approx(20.0 * 150 - cb.total)


def test_realized_pnl_no_trade_returns_zero():
    pnl = realized_pnl(
        direction=Direction.NO_TRADE,
        entry_premium=100.0,
        exit_premium=120.0,
        qty=150,
        bid_ask_spread=0.4,
    )
    assert pnl == 0.0


def test_cost_breakdown_is_frozen():
    import dataclasses

    cb = compute_costs(entry_premium=100.0, exit_premium=120.0, qty=75, bid_ask_spread=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cb.brokerage = 0.0  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/global_tab/test_cost_model.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `cost_model.py`**

`india_quant/global_tab/cost_model.py`:

```python
"""Realistic Indian options cost model.

All numbers calibrated to NSE F&O / Zerodha as of 2025. The backtester uses
this for every trade so equity curves reflect after-cost reality. No "before
cost" P&L is ever surfaced on the Global tab.

References:
- Brokerage: min(20, 0.03% of turnover) per executed order, each side.
- STT: 0.0625% of premium turnover, sell side only.
- Exchange transaction (NSE F&O): 0.03503% of premium, both sides.
- SEBI charges: ₹10 per crore of turnover (1e-6), both sides.
- Stamp duty: 0.003% of premium, buy side only (state-imposed; uniform 2020+).
- GST: 18% on (brokerage + exchange + SEBI).
- Slippage: estimated as half the bid-ask spread, applied per side.

Phase-1 takes bid_ask_spread as an argument; the chain snapshot will supply
it in Phase 2.
"""
from dataclasses import dataclass

from india_quant.global_tab.types import Direction

_BROKERAGE_RATE = 0.0003           # 0.03%
_BROKERAGE_CAP = 20.0
_STT_RATE = 0.000625               # 0.0625% sell-side
_EXCHANGE_RATE = 0.0003503         # 0.03503% both sides
_SEBI_RATE = 1e-6                  # ₹10 / crore
_STAMP_DUTY_RATE = 0.00003         # 0.003% buy-side
_GST_RATE = 0.18                   # 18%


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp_duty: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange
            + self.sebi
            + self.stamp_duty
            + self.gst
            + self.slippage
        )


def compute_costs(
    *,
    entry_premium: float,
    exit_premium: float,
    qty: int,
    bid_ask_spread: float,
) -> CostBreakdown:
    buy_turnover = entry_premium * qty
    sell_turnover = exit_premium * qty

    brokerage_buy = min(_BROKERAGE_CAP, _BROKERAGE_RATE * buy_turnover)
    brokerage_sell = min(_BROKERAGE_CAP, _BROKERAGE_RATE * sell_turnover)
    brokerage = brokerage_buy + brokerage_sell

    stt = _STT_RATE * sell_turnover  # sell-side only

    exch_buy = _EXCHANGE_RATE * buy_turnover
    exch_sell = _EXCHANGE_RATE * sell_turnover
    exchange = exch_buy + exch_sell

    sebi_buy = _SEBI_RATE * buy_turnover
    sebi_sell = _SEBI_RATE * sell_turnover
    sebi = sebi_buy + sebi_sell

    stamp_duty = _STAMP_DUTY_RATE * buy_turnover  # buy-side only

    gst = _GST_RATE * (brokerage + exchange + sebi)

    slippage_per_side = 0.5 * bid_ask_spread * qty
    slippage = 2 * slippage_per_side

    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange=exchange,
        sebi=sebi,
        stamp_duty=stamp_duty,
        gst=gst,
        slippage=slippage,
    )


def realized_pnl(
    *,
    direction: Direction,
    entry_premium: float,
    exit_premium: float,
    qty: int,
    bid_ask_spread: float,
) -> float:
    if direction == Direction.NO_TRADE:
        return 0.0

    if direction == Direction.LONG:
        gross = (exit_premium - entry_premium) * qty
    else:  # SHORT
        gross = (entry_premium - exit_premium) * qty

    costs = compute_costs(
        entry_premium=entry_premium,
        exit_premium=exit_premium,
        qty=qty,
        bid_ask_spread=bid_ask_spread,
    )
    return gross - costs.total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/global_tab/test_cost_model.py -v`
Expected: 13 passed.

- [ ] **Step 5: Suggest commit**

```bash
git add india_quant/global_tab/cost_model.py tests/global_tab/test_cost_model.py
git commit -m "feat(global_tab): add realistic Indian options cost model"
```

---

## Task 8: Phase-1 final demo run

**Files:** none modified — this is the demo gate.

- [ ] **Step 1: Run the entire Phase-1 test suite**

Run: `pytest tests/global_tab -v`
Expected output (summary line at end): `39 passed` (5 enum tests + 6 dataclass tests + 6 mode tests + 8 feature-store unit tests + 1 property test + 13 cost-model tests).

- [ ] **Step 2: Run with coverage to confirm no untested code**

Run: `pytest tests/global_tab --cov=india_quant.global_tab --cov-report=term-missing`
Expected: 100% coverage on `types.py`, `modes.py`, `feature_store.py`, `cost_model.py`. (`__init__.py` is empty; coverage will note it.)

If any line is uncovered, do not advance to Phase 2 until a test covers it.

- [ ] **Step 3: Confirm no regressions in the existing test suite**

Run: `pytest tests/ -v`
Expected: all tests pass, including the existing `tests/test_integration.py`. If anything in the existing suite breaks, it is unrelated to this phase (no Phase-1 module imports anything from outside `global_tab`); investigate before advancing.

- [ ] **Step 4: Phase-1 demo announcement**

Phase 1 is complete when this report is true:
- `india_quant/global_tab/` package contains `types.py`, `modes.py`, `feature_store.py`, `cost_model.py`.
- 39 tests in `tests/global_tab/` pass, including one hypothesis property test for no-future-peek.
- `requirements.txt` pins `hypothesis==6.122.3` and `pyarrow==18.1.0`.
- No UI, scheduler, or fetcher code was touched.

The plan for Phase 2 (Data + briefing + heatmap) will be written next, against the same spec.

---

## Self-Review (already performed)

**Spec coverage (Phase-1 scope, spec §11.1):**
- Types — Tasks 2, 3 ✓
- Mode configs — Task 4 ✓
- `PointInTimeFeatureStore` — Tasks 5, 6 ✓
- Cost model — Task 7 ✓
- Unit tests — Tasks 2, 3, 4, 5, 7 ✓
- Property tests — Task 6 ✓
- Demo: pytest output — Task 8 ✓

Phase-1 scope covered. Phases 2–7 are out of scope by design and get their own plans.

**Placeholder scan:** No `TBD`, `TODO`, `implement later`, or `add appropriate X` strings in the plan. Every code step contains the actual code. Mode threshold values are explicitly flagged as "placeholders finalized via Phase-4 backtest sweep" in the module docstring — that's a documented forward-reference, not a plan placeholder.

**Type consistency:** Names match across tasks: `Mode`, `Direction`, `Status`, `OptionsLeg`, `RiskReward`, `TimingWindow`, `ReasoningContext`, `LiveTicket`, `TradeTicket`, `GlobalTabView`, `BriefingTile`, `BriefingStrip`, `CorrelationCell`, `CorrelationHeatmap`, `ModeConfig`, `MODE_CONFIGS`, `FuturePeekError`, `PointInTimeFeatureStore`, `CostBreakdown`, `compute_costs`, `realized_pnl`. All defined in the task that introduces them; later tasks only reference defined names.
