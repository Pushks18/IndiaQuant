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
