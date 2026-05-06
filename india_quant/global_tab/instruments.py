"""NSE index lot sizes and weekly-expiry calendar helpers.

LOT_SIZES current as of NSE circular dated 2024-10-25 (NIFTY 25, BANKNIFTY 15);
update on the next SEBI lot-size revision.

Phase 3 uses calendar-Thursday only. NSE holiday list (which can shift expiry to
Wednesday in expiry weeks) is a TODO — the rule will be: if Thursday is a
trading holiday, expiry rolls to the immediately preceding trading day.
"""
from __future__ import annotations

from datetime import date, timedelta

LOT_SIZES: dict[str, int] = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
}

_THURSDAY = 3  # Monday = 0


def is_weekly_expiry(d: date) -> bool:
    """True iff `d` is a Thursday (calendar rule, holiday-shift TODO)."""
    return d.weekday() == _THURSDAY


def next_weekly_expiry(from_date: date) -> date:
    """Next Thursday strictly after `from_date`."""
    days_ahead = (_THURSDAY - from_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return from_date + timedelta(days=days_ahead)
