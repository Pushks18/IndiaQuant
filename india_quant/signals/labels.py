"""Signal label generator: forward returns at 1d/5d/21d horizons.

For each (ticker, date) in price_data, compute future_return = price[date+H] / price[date] - 1
and class_label = 1 if future_return > 0 else 0. Upsert into signal_labels.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from india_quant.data.db import get_session
from india_quant.data.models import SignalLabels

# Horizon → trading days forward
HORIZON_DAYS = {"1d": 1, "5d": 5, "21d": 21}


def compute_labels(
    horizons: tuple[str, ...] = ("1d", "5d", "21d"),
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    """Generate forward-return labels for every (ticker, date) in price_data.

    Returns the number of rows upserted.
    """
    with get_session() as session:
        sql = """
            SELECT ticker, datetime::date AS date, close
            FROM price_data
            WHERE interval = '1d'
        """
        params = {}
        if start_date:
            sql += " AND datetime >= :start"
            params["start"] = start_date
        if end_date:
            sql += " AND datetime <= :end"
            params["end"] = end_date
        sql += " ORDER BY ticker, datetime"
        rows = session.execute(text(sql), params).fetchall()

    if not rows:
        logger.warning("[Labels] No price data for label generation.")
        return 0

    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    df["close"] = df["close"].astype(float)

    upserted = 0
    BATCH = 5000
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        closes = grp["close"].values
        dates = grp["date"].values

        batch: list[dict] = []
        for h in horizons:
            k = HORIZON_DAYS[h]
            if len(closes) <= k:
                continue
            fwd = (closes[k:] / closes[:-k]) - 1
            for i, ret in enumerate(fwd):
                ret_f = float(ret)
                batch.append({
                    "ticker": ticker,
                    "date": dates[i],
                    "horizon": h,
                    "future_return": ret_f,
                    "class_label": 1 if ret_f > 0 else 0,
                })
                if len(batch) >= BATCH:
                    with get_session() as session:
                        stmt = insert(SignalLabels).values(batch).on_conflict_do_update(
                            index_elements=["ticker", "date", "horizon"],
                            set_={"future_return": insert(SignalLabels).excluded.future_return,
                                  "class_label": insert(SignalLabels).excluded.class_label},
                        )
                        session.execute(stmt)
                    upserted += len(batch)
                    batch = []
        if batch:
            with get_session() as session:
                stmt = insert(SignalLabels).values(batch).on_conflict_do_update(
                    index_elements=["ticker", "date", "horizon"],
                    set_={"future_return": insert(SignalLabels).excluded.future_return,
                          "class_label": insert(SignalLabels).excluded.class_label},
                )
                session.execute(stmt)
            upserted += len(batch)
        logger.info(f"[Labels] {ticker}: done")

    logger.info(f"[Labels] Total upserts: {upserted}")
    return upserted


if __name__ == "__main__":
    compute_labels()
