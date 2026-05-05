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
