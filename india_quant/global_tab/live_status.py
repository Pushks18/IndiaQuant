"""Pure status-transition logic for the global-tab live ticker.

Phase 3a stubbed every ticket's `live.status` to WAITING. Phase 5a
upgrades that to a time-based flip:

    before entry_window_start  →  WAITING
    in entry window            →  ENTRY_ZONE_ACTIVE
    after entry_window_end before exit_window_end → IN_POSITION
    past exit_window_end       →  EXPIRED_NO_ENTRY  (time-only model)
    past invalidation_time     →  EXPIRED_NO_ENTRY

Price-based flips (TARGET_HIT / STOPPED_OUT) require a live spot feed
and a per-ticket trigger ledger; deferred to Phase 5b. For now, the
post-window state is conservatively EXPIRED_NO_ENTRY rather than
guessing at IN_POSITION → exit outcomes.

Pure: same (ticket, now_ist) → same status. No DB or network access.
"""
from __future__ import annotations

from datetime import datetime, time

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover - fallback for envs without tzdata
    from datetime import timezone, timedelta
    _IST = timezone(timedelta(hours=5, minutes=30))

from india_quant.global_tab.types import Direction, Status, TimingWindow, TradeTicket


def _now_ist_time(now: datetime) -> time:
    """Return the time-of-day component of `now` in IST."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=_IST)
    return now.astimezone(_IST).time()


def compute_status(ticket: TradeTicket, now: datetime) -> Status:
    """Pure time-based status transition for a TradeTicket."""
    if ticket.direction == Direction.NO_TRADE:
        # No trade was issued — status stays WAITING regardless of time.
        return Status.WAITING

    timing = ticket.timing
    if not isinstance(timing, TimingWindow):
        return Status.WAITING

    t = _now_ist_time(now)
    if t < timing.entry_window_start:
        return Status.WAITING
    if t < timing.entry_window_end:
        return Status.ENTRY_ZONE_ACTIVE
    if t < timing.exit_window_end and t < timing.invalidation_time:
        # Past entry window but inside trade lifecycle — model as IN_POSITION.
        # Phase 5b will refine with price triggers.
        return Status.IN_POSITION
    return Status.EXPIRED_NO_ENTRY
