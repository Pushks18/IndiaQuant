"""Rebuild the AnalogIndex pickle from the live TimescaleDB.

The dashboard loads `models/global_tab/analog_index.pkl` at boot and reuses
it for the lifetime of the process. As new sessions complete, the index
goes stale — this script refreshes it.

Usage
-----
    PYTHONPATH=. venv/bin/python scripts/rebuild_analog_index.py
    PYTHONPATH=. venv/bin/python scripts/rebuild_analog_index.py --start 2022-01-01

Outputs:
    models/global_tab/analog_index.pkl

Recommended cadence: nightly post-market (after the daily yfinance fetch
backfills today's NIFTY close). A LaunchAgent / cron job can call this
right before the dashboard restart.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from loguru import logger


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rebuild the global-tab AnalogIndex.")
    p.add_argument("--index", choices=["NIFTY", "BANKNIFTY"], default="NIFTY",
                   help="Underlying index to fit the analog space on.")
    p.add_argument("--start", default="2021-01-01", help="YYYY-MM-DD")
    p.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD")
    p.add_argument("--out", default="models/global_tab/analog_index.pkl",
                   help="Output pickle path.")
    args = p.parse_args(argv)

    from india_quant.data.db import get_session_factory
    from india_quant.global_tab.analog_index import AnalogIndex

    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()

    logger.info(
        "Rebuilding AnalogIndex: index={}, window={}..{} → {}",
        args.index, start, end, args.out,
    )
    idx = AnalogIndex.build_from_db(
        index=args.index, start=start, end=end,
        session_factory=get_session_factory(),
    )
    out = Path(args.out)
    idx.save(out)
    logger.info("Done. {} samples persisted to {}", idx.n_samples, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
