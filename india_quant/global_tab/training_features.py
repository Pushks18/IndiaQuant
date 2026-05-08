"""Walk-forward training feature assembly for the global-tab forecaster.

Single source of truth for the feature matrix used by both
`scripts/train_global_forecaster.py` (offline training) and the live
serving path through `orchestrator._build_features`. Train/serve skew is
prevented structurally: every feature here is computed at the same
"as-of" moment — 08:30 IST T+1 (after US close) — and the column list
`FEATURE_COLUMNS` is the only contract.

Deviation from spec §6.2:
- Day-of-week is encoded as `dow_int` (0=Mon..4=Fri) instead of 4 one-hot
  columns. LightGBM handles ordinals natively for tree splits, so this
  costs nothing and keeps the matrix tight.
- `gift_nifty_premium_bps` has no historical store (GIFT migrated to
  NSE-IFSC mid-2023; the live fetcher is best-effort). For training we
  proxy it as `spx_overnight_pct * 100` (bps). Live serving uses the
  real GIFT snapshot. This deviation is logged in `training_summary.json`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select

from india_quant.data.models import GlobalSignal, PriceData
from india_quant.global_tab.feature_store import PointInTimeFeatureStore


FEATURE_COLUMNS: list[str] = [
    "gift_nifty_premium_bps",
    "spx_overnight_pct",
    "nasdaq_overnight_pct",
    "dxy_delta_pct",
    "india_vix_delta_pct",
    "brent_overnight_pct",
    "nifty_5d_momentum",
    "nifty_realized_vol_20d",
    "dow_int",
    "is_expiry_week",
    "days_to_rbi_policy",
]

LABEL_COLUMNS: list[str] = ["label_direction", "label_return_bps"]

_GLOBAL_TICKERS: dict[str, str] = {
    "^GSPC":      "spx_overnight_pct",
    "^IXIC":      "nasdaq_overnight_pct",
    "DX-Y.NYB":   "dxy_delta_pct",
    "^INDIAVIX":  "india_vix_delta_pct",
    "BZ=F":       "brent_overnight_pct",
}

_INDEX_TICKER: dict[str, str] = {
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

# RBI policy dates — kept here (not via Config) to avoid forcing tests to
# load .env at import time. Add new dates as the calendar advances.
_RBI_POLICY_DATES: tuple[date, ...] = (
    date(2025, 2, 7),  date(2025, 4, 9),  date(2025, 6, 6),
    date(2025, 8, 8),  date(2025, 10, 8), date(2025, 12, 5),
    date(2026, 2, 6),  date(2026, 4, 8),  date(2026, 6, 5),
    date(2026, 8, 7),  date(2026, 10, 7), date(2026, 12, 4),
)

_DROP_AMBIGUOUS_THRESHOLD_BPS = 5.0


def _is_thursday_in_week(d: date) -> bool:
    monday = d - timedelta(days=d.weekday())
    thursday = monday + timedelta(days=3)
    # In India, weekly expiry is the Thursday of every week — every week
    # has one. The "expiry week" flag is true on every session, but the
    # interpretation we care about is whether the *current* session is
    # within the expiry-week (Mon-Thu of a week containing Thursday).
    # Practically: True iff the session is Mon..Thu and the Thursday
    # itself isn't a holiday (treat as True for all sessions Mon..Fri).
    return thursday.weekday() == 3  # always True; placeholder for holiday-aware logic


def _days_to_next_rbi(d: date) -> int:
    upcoming = [p for p in _RBI_POLICY_DATES if p >= d]
    if not upcoming:
        return 999
    return (upcoming[0] - d).days


def assemble_training_frame(
    index: str,
    start: date,
    end: date,
    *,
    session_factory: Callable,
) -> pd.DataFrame:
    """Build the (features + labels) frame for an index over [start, end].

    Returns a DataFrame indexed by session date (Mon..Fri) with columns
    `FEATURE_COLUMNS + LABEL_COLUMNS`. The trailing row (no T+1 close
    available for the label) is dropped; rows with any NaN feature are
    dropped and counted in a log line.
    """
    if index not in _INDEX_TICKER:
        raise ValueError(f"unknown index {index!r}; expected one of {list(_INDEX_TICKER)}")

    nifty_ticker = _INDEX_TICKER[index]
    lookback_start = start - timedelta(days=45)  # 30 trading-day buffer for momentum/vol

    Session = session_factory
    with Session() as s:
        # Global signals (T-1 closes, joinable on date)
        gs_rows = s.execute(
            select(
                GlobalSignal.date,
                GlobalSignal.ticker,
                GlobalSignal.pct_1d,
            ).where(
                GlobalSignal.ticker.in_(list(_GLOBAL_TICKERS)),
                GlobalSignal.date >= lookback_start,
                GlobalSignal.date <= end,
            )
        ).all()

        # Index price history (for 5d momentum + 20d realized vol + label)
        px_rows = s.execute(
            select(
                PriceData.datetime,
                PriceData.close,
            ).where(
                PriceData.ticker == nifty_ticker,
                PriceData.interval == "1d",
                PriceData.datetime >= pd.Timestamp(lookback_start, tz="UTC"),
                PriceData.datetime <= pd.Timestamp(end + timedelta(days=2), tz="UTC"),
            )
        ).all()

    if not px_rows:
        raise ValueError(
            f"no price_data for {nifty_ticker} between {lookback_start} and {end}; "
            f"backfill required before training"
        )

    # Pivot global_signals to wide: index=date, columns=ticker_alias
    if gs_rows:
        gs_df = pd.DataFrame(gs_rows, columns=["date", "ticker", "pct_1d"])
        gs_df["date"] = pd.to_datetime(gs_df["date"]).dt.date
        gs_wide = (
            gs_df.pivot(index="date", columns="ticker", values="pct_1d")
            .rename(columns=_GLOBAL_TICKERS)
        )
    else:
        gs_wide = pd.DataFrame(columns=list(_GLOBAL_TICKERS.values()))
    # Make sure every expected column exists
    for col in _GLOBAL_TICKERS.values():
        if col not in gs_wide.columns:
            gs_wide[col] = np.nan

    # Index closes
    px_df = pd.DataFrame(px_rows, columns=["datetime", "close"])
    px_df["date"] = pd.to_datetime(px_df["datetime"]).dt.date
    px_df = px_df.drop_duplicates("date").sort_values("date").set_index("date")
    px_df["log_close"] = np.log(px_df["close"].astype(float))
    px_df["log_ret"] = px_df["log_close"].diff()
    px_df["nifty_5d_momentum"] = px_df["log_close"].diff(5)
    px_df["nifty_realized_vol_20d"] = px_df["log_ret"].rolling(20).std()

    # Build the per-session row in [start, end]
    sessions = sorted(set(px_df.index) & set(pd.date_range(start, end).date))
    rows = []
    for sd in sessions:
        if pd.Timestamp(sd).weekday() >= 5:
            continue  # skip Sat/Sun

        # Global features: use GS row dated == sd (signal_date in repo convention
        # is the IST session date for which the US-overnight pct is "as-of T-1
        # close, available before NSE open"). If sd missing in gs_wide, NaN.
        if sd in gs_wide.index:
            gs_row = gs_wide.loc[sd]
        else:
            gs_row = pd.Series({c: np.nan for c in _GLOBAL_TICKERS.values()})

        spx = gs_row.get("spx_overnight_pct", np.nan)
        gift_proxy = float(spx) * 100.0 if pd.notna(spx) else np.nan

        feat = {
            "gift_nifty_premium_bps":  gift_proxy,
            "spx_overnight_pct":       gs_row.get("spx_overnight_pct", np.nan),
            "nasdaq_overnight_pct":    gs_row.get("nasdaq_overnight_pct", np.nan),
            "dxy_delta_pct":           gs_row.get("dxy_delta_pct", np.nan),
            "india_vix_delta_pct":     gs_row.get("india_vix_delta_pct", np.nan),
            "brent_overnight_pct":     gs_row.get("brent_overnight_pct", np.nan),
            "nifty_5d_momentum":       px_df.loc[sd, "nifty_5d_momentum"] if sd in px_df.index else np.nan,
            "nifty_realized_vol_20d":  px_df.loc[sd, "nifty_realized_vol_20d"] if sd in px_df.index else np.nan,
            "dow_int":                 pd.Timestamp(sd).weekday(),
            "is_expiry_week":          int(_is_thursday_in_week(sd)),
            "days_to_rbi_policy":      _days_to_next_rbi(sd),
        }

        # Labels: next session's close-to-close return on the same index
        future_dates = [d for d in px_df.index if d > sd]
        if not future_dates:
            continue  # no T+1 close → drop trailing row
        next_close = float(px_df.loc[future_dates[0], "close"])
        today_close = float(px_df.loc[sd, "close"])
        ret_bps = (np.log(next_close) - np.log(today_close)) * 10_000.0
        feat["label_return_bps"] = ret_bps
        feat["label_direction"] = 1 if ret_bps > 0 else 0
        feat["__session_date"] = sd
        rows.append(feat)

    if not rows:
        raise ValueError(f"assemble_training_frame produced 0 rows for {index} {start}..{end}")

    df = pd.DataFrame(rows).set_index("__session_date").rename_axis("date")
    df = df[FEATURE_COLUMNS + LABEL_COLUMNS]

    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    n_after = len(df)
    if n_before != n_after:
        logger.info(
            "training_features: dropped {} of {} rows due to NaN features",
            n_before - n_after, n_before,
        )

    # Drop rows where |return| < 5 bps (ambiguous direction labels)
    n_before_amb = len(df)
    df = df.loc[df["label_return_bps"].abs() >= _DROP_AMBIGUOUS_THRESHOLD_BPS]
    if n_before_amb != len(df):
        logger.info(
            "training_features: dropped {} of {} rows with |return| < {} bps",
            n_before_amb - len(df), n_before_amb, _DROP_AMBIGUOUS_THRESHOLD_BPS,
        )

    return df


def to_feature_store(frame: pd.DataFrame) -> PointInTimeFeatureStore:
    """Bridge to PointInTimeFeatureStore for live-serving feature lookup.

    Each `FEATURE_COLUMNS` column is registered as a separate Series so the
    serving path can call `.get(name, at)` with the same column names.
    """
    if frame.empty:
        raise ValueError("cannot build feature store from empty frame")
    store = PointInTimeFeatureStore()
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            raise ValueError(f"frame missing required feature column {col!r}")
        store.register(col, pd.Series(frame[col].to_numpy(dtype=float), index=idx))
    return store
