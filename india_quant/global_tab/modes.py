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
    max_loss_fraction: float     # share of capital at risk per trade (Phase 3a)


MODE_CONFIGS: dict[Mode, ModeConfig] = {
    Mode.AGGRESSIVE: ModeConfig(
        min_expected_value=0.0,
        min_win_probability=0.0,
        strike_rule="otm_1",
        target_t1_multiple=1.5,
        target_t2_multiple=2.5,
        stop_loss_multiple=0.6,
        require_top_decile_analog=False,
        max_loss_fraction=0.02,
    ),
    Mode.BALANCED: ModeConfig(
        min_expected_value=0.0,
        min_win_probability=0.55,
        strike_rule="atm",
        target_t1_multiple=1.4,
        target_t2_multiple=2.0,
        stop_loss_multiple=0.7,
        require_top_decile_analog=False,
        max_loss_fraction=0.015,
    ),
    Mode.CONSERVATIVE: ModeConfig(
        min_expected_value=0.0,
        min_win_probability=0.62,
        strike_rule="itm_1",
        target_t1_multiple=1.25,
        target_t2_multiple=1.6,
        stop_loss_multiple=0.8,
        require_top_decile_analog=True,
        max_loss_fraction=0.01,
    ),
}
