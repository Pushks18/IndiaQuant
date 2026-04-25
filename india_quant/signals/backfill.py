"""Historical backfill driver: factor snapshots → signal labels → optional model train."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from loguru import logger
from sqlalchemy import text

from india_quant.data.db import get_session


def trading_dates_friday(start: date, end: date) -> list[str]:
    """Every Friday between start and end where price_data has at least some rows."""
    cur = start
    while cur.weekday() != 4:  # Friday = 4
        cur += timedelta(days=1)
    fridays = []
    while cur <= end:
        fridays.append(cur)
        cur += timedelta(days=7)
    if not fridays:
        return []
    # Filter to dates that actually have price data
    with get_session() as s:
        rows = s.execute(text("""
            SELECT DISTINCT datetime::date AS d FROM price_data
            WHERE interval = '1d' AND datetime::date = ANY(:days)
        """), {"days": fridays}).fetchall()
    return sorted(str(r[0]) for r in rows)


def backfill_factor_history(start: str = "2019-01-04", end: str | None = None) -> int:
    """Compute factor snapshots for every Friday with price data."""
    from india_quant.signals.factors import FactorEngine
    end_d = date.fromisoformat(end) if end else date.today()
    start_d = date.fromisoformat(start)
    fe = FactorEngine()

    dates = trading_dates_friday(start_d, end_d)
    logger.info(f"[Backfill] Factor history: {len(dates)} Fridays from {start_d} to {end_d}")

    n = 0
    for i, d in enumerate(dates):
        try:
            df = fe.compute_all(d)
            n += len(df) if df is not None and not df.empty else 0
            if i % 10 == 0 or i == len(dates) - 1:
                logger.info(f"[Backfill] {i+1}/{len(dates)} → {d} ({n} cumulative rows)")
        except Exception as e:
            logger.error(f"[Backfill] {d}: {e}")
    logger.info(f"[Backfill] Factor history complete: {n} rows total")
    return n


def backfill_signal_labels(start: str = "2019-01-01", end: str | None = None) -> int:
    """Compute forward-return labels for the same date range."""
    from india_quant.signals.labels import compute_labels
    return compute_labels(start_date=start, end_date=end)


def run_full_backfill(start: str = "2019-01-04") -> dict:
    """One-shot: factors → labels → return summary."""
    fac_rows = backfill_factor_history(start=start)
    label_rows = backfill_signal_labels(start=start)
    return {"factor_rows": fac_rows, "label_rows": label_rows}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-04")
    p.add_argument("--end", default=None)
    p.add_argument("--factors-only", action="store_true")
    p.add_argument("--labels-only", action="store_true")
    args = p.parse_args()

    if args.labels_only:
        print(backfill_signal_labels(args.start, args.end))
    elif args.factors_only:
        print(backfill_factor_history(args.start, args.end))
    else:
        print(run_full_backfill(args.start))
