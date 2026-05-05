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
