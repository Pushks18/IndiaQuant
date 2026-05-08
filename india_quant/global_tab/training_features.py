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
    # Phase 3a/3b — preserved order so old reproducibility gates still pass
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
# Phase 3d candidate columns — infrastructure (helpers, FeatureRow fields,
# artifact guard) is in place but empirical OOS test on 2021-2026 NIFTY/
# BANKNIFTY showed no improvement (slight regression). Kept here as the
# canonical list for Phase 3e to opt-in once a richer feature source
# (sentiment, intraday vol surface, options OI) makes the column set
# information-additive. Append to FEATURE_COLUMNS to activate.
PHASE3D_CANDIDATE_COLUMNS: list[str] = [
    "bank_vs_nifty_5d_relstr",
    "it_vs_nifty_5d_relstr",
    "pharma_vs_nifty_5d_relstr",
    "realty_vs_nifty_5d_relstr",
    "sector_dispersion_5d",
    "pct_above_20dma",
    "pct_above_50dma",
    "advance_decline_5d",
    "mean_realized_vol_universe",
]

_SECTOR_TICKERS: dict[str, str] = {
    "^NSEBANK":    "bank_vs_nifty_5d_relstr",
    "^CNXIT":      "it_vs_nifty_5d_relstr",
    "^CNXPHARMA":  "pharma_vs_nifty_5d_relstr",
    "^CNXREALTY":  "realty_vs_nifty_5d_relstr",
}
_DISPERSION_TICKERS: tuple[str, ...] = (
    "^NSEBANK", "^CNXIT", "^CNXPHARMA", "^CNXREALTY",
    "^CNXENERGY", "^CNXINFRA",
)

LABEL_COLUMNS: list[str] = ["label_direction", "label_return_bps", "label_realized_vol_5d_pct"]

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


def _sector_features(
    start: date, end: date, *, session_factory: Callable
) -> pd.DataFrame:
    """Build the sector-RS + dispersion frame, indexed by session date."""
    Session = session_factory
    needed_tickers = list({*_SECTOR_TICKERS, *_DISPERSION_TICKERS})
    with Session() as s:
        rows = s.execute(
            select(
                GlobalSignal.date, GlobalSignal.ticker, GlobalSignal.pct_5d,
            ).where(
                GlobalSignal.ticker.in_(needed_tickers),
                GlobalSignal.date >= start - timedelta(days=10),
                GlobalSignal.date <= end,
            )
        ).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "ticker", "pct_5d"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    wide = df.pivot(index="date", columns="ticker", values="pct_5d")

    # nifty_pct_5d will come from px_df in the main loop, not from global_signals.
    # Sector RS = sector_pct_5d - nifty_pct_5d, computed inside assemble.
    return wide  # caller subtracts nifty_pct_5d


def _breadth_features(
    start: date, end: date, *, session_factory: Callable, universe: list[str],
) -> pd.DataFrame:
    """Per-session breadth indicators over the Nifty-50 universe.

    Uses closes through *yesterday* (no future peek).
    Returns a date-indexed frame with [pct_above_20dma, pct_above_50dma,
    advance_decline_5d].
    """
    if not universe:
        return pd.DataFrame()
    Session = session_factory
    lookback_start = start - timedelta(days=90)  # buffer for 50dma
    with Session() as s:
        rows = s.execute(
            select(
                PriceData.ticker, PriceData.datetime, PriceData.close,
            ).where(
                PriceData.ticker.in_(universe),
                PriceData.interval == "1d",
                PriceData.datetime >= pd.Timestamp(lookback_start, tz="UTC"),
                PriceData.datetime <= pd.Timestamp(end + timedelta(days=2), tz="UTC"),
            )
        ).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ticker", "datetime", "close"])
    df["date"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    df = df.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    closes = df.pivot(index="date", columns="ticker", values="close")
    closes = closes.sort_index()

    # Shift by 1 so the rolling mean uses only sessions strictly before sd.
    ma20 = closes.shift(1).rolling(20, min_periods=10).mean()
    ma50 = closes.shift(1).rolling(50, min_periods=25).mean()
    above20 = (closes > ma20).astype(float)
    above50 = (closes > ma50).astype(float)
    pct_above_20dma = above20.mean(axis=1)
    pct_above_50dma = above50.mean(axis=1)

    rets = closes.pct_change()
    # Daily breadth = mean sign of return across universe; 5d = sum of last 5
    daily_breadth = np.sign(rets).mean(axis=1)
    advance_decline_5d = daily_breadth.rolling(5, min_periods=3).sum()

    out = pd.DataFrame({
        "pct_above_20dma":     pct_above_20dma,
        "pct_above_50dma":     pct_above_50dma,
        "advance_decline_5d":  advance_decline_5d,
    })
    out.index = pd.to_datetime(out.index).date
    return out


def _factor_aggregate_features(
    start: date, end: date, *, session_factory: Callable, universe: list[str],
) -> pd.DataFrame:
    """Per-date mean of factor_scores.realized_vol across the universe.

    iv_skew and oi_flow dropped — verified 100% NULL across 2021-2026 because
    option_chain has 0 rows. Pure realized_vol still useful as a regime feature.
    """
    if not universe:
        return pd.DataFrame()
    Session = session_factory
    from india_quant.data.models import FactorScores
    try:
        with Session() as s:
            rows = s.execute(
                select(FactorScores.date, FactorScores.realized_vol).where(
                    FactorScores.ticker.in_(universe),
                    FactorScores.date >= start,
                    FactorScores.date <= end,
                    FactorScores.realized_vol.is_not(None),
                )
            ).all()
    except Exception as exc:
        # factor_scores table absent (e.g. test sqlite fixture) — feature stays NaN
        logger.debug("factor_aggregates: skipped ({})", exc)
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "realized_vol"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    grouped = df.groupby("date")["realized_vol"].mean().to_frame(
        name="mean_realized_vol_universe"
    )
    return grouped


def _nifty50_universe() -> list[str]:
    """Nifty-50 .NS tickers from the canonical fetcher list — used for breadth + factor agg."""
    try:
        from india_quant.data.fetchers.yfinance_fetcher import YFinanceFetcher
        return list(getattr(YFinanceFetcher, "NIFTY_50", []))
    except Exception:
        return []


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
    px_df["date"] = pd.to_datetime(px_df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    px_df = px_df.drop_duplicates("date").sort_values("date").set_index("date")
    px_df["log_close"] = np.log(px_df["close"].astype(float))
    px_df["log_ret"] = px_df["log_close"].diff()
    px_df["nifty_5d_momentum"] = px_df["log_close"].diff(5)
    px_df["nifty_realized_vol_20d"] = px_df["log_ret"].rolling(20).std()

    # Phase 3d auxiliary feature frames
    sector_wide = _sector_features(start, end, session_factory=session_factory)
    universe = _nifty50_universe()
    breadth_df = _breadth_features(start, end, session_factory=session_factory, universe=universe)
    factor_df = _factor_aggregate_features(start, end, session_factory=session_factory, universe=universe)

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

        # Phase 3d: nifty_pct_5d derived from px_df (more reliable than global_signals for ^NSEI)
        nifty_5d_pct = np.nan
        if sd in px_df.index:
            five_back = [d for d in px_df.index if d <= sd]
            if len(five_back) >= 6:
                today_c = float(px_df.loc[sd, "close"])
                prev_c = float(px_df.loc[five_back[-6], "close"])
                if prev_c > 0:
                    nifty_5d_pct = (today_c / prev_c - 1.0) * 100.0  # %

        sector_row = sector_wide.loc[sd] if (not sector_wide.empty and sd in sector_wide.index) else None
        def _sector_rs(ticker: str) -> float:
            if sector_row is None or ticker not in sector_row.index:
                return np.nan
            v = sector_row[ticker]
            if pd.isna(v) or pd.isna(nifty_5d_pct):
                return np.nan
            return float(v) - float(nifty_5d_pct)

        sector_disp = np.nan
        if sector_row is not None:
            disp_vals = [sector_row.get(t, np.nan) for t in _DISPERSION_TICKERS]
            disp_clean = [float(v) for v in disp_vals if pd.notna(v)]
            if len(disp_clean) >= 4:
                sector_disp = float(np.std(disp_clean, ddof=0))

        breadth_row = breadth_df.loc[sd] if (not breadth_df.empty and sd in breadth_df.index) else None
        factor_row = factor_df.loc[sd] if (not factor_df.empty and sd in factor_df.index) else None

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
            "bank_vs_nifty_5d_relstr":   _sector_rs("^NSEBANK"),
            "it_vs_nifty_5d_relstr":     _sector_rs("^CNXIT"),
            "pharma_vs_nifty_5d_relstr": _sector_rs("^CNXPHARMA"),
            "realty_vs_nifty_5d_relstr": _sector_rs("^CNXREALTY"),
            "sector_dispersion_5d":      sector_disp,
            "pct_above_20dma":           float(breadth_row["pct_above_20dma"])    if breadth_row is not None else np.nan,
            "pct_above_50dma":           float(breadth_row["pct_above_50dma"])    if breadth_row is not None else np.nan,
            "advance_decline_5d":        float(breadth_row["advance_decline_5d"]) if breadth_row is not None else np.nan,
            "mean_realized_vol_universe": float(factor_row["mean_realized_vol_universe"]) if factor_row is not None else np.nan,
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

        # Phase 6b label: forward 5-day realized vol (annualized %).
        # std of next 5 log returns × √252 × 100. Drop the row if we don't
        # have 5 forward sessions available.
        if len(future_dates) >= 5:
            forward_closes = [today_close] + [
                float(px_df.loc[future_dates[i], "close"]) for i in range(5)
            ]
            forward_log_rets = [
                np.log(forward_closes[i + 1]) - np.log(forward_closes[i])
                for i in range(5)
            ]
            mean_r = sum(forward_log_rets) / len(forward_log_rets)
            var_r = sum((r - mean_r) ** 2 for r in forward_log_rets) / len(forward_log_rets)
            sigma_d = float(np.sqrt(var_r))
            feat["label_realized_vol_5d_pct"] = sigma_d * np.sqrt(252.0) * 100.0
        else:
            feat["label_realized_vol_5d_pct"] = np.nan

        feat["__session_date"] = sd
        rows.append(feat)

    if not rows:
        raise ValueError(f"assemble_training_frame produced 0 rows for {index} {start}..{end}")

    df = pd.DataFrame(rows).set_index("__session_date").rename_axis("date")
    df = df[FEATURE_COLUMNS + LABEL_COLUMNS]

    # Phase 3d: only the original 11 columns are required to be non-NaN.
    # New columns (sector RS, breadth, factor agg) zero-impute at both train
    # and serve time so a sparse table doesn't crater the row count.
    _REQUIRED_COLUMNS = [
        "gift_nifty_premium_bps", "spx_overnight_pct", "nasdaq_overnight_pct",
        "dxy_delta_pct", "india_vix_delta_pct", "brent_overnight_pct",
        "nifty_5d_momentum", "nifty_realized_vol_20d",
        "dow_int", "is_expiry_week", "days_to_rbi_policy",
    ]
    n_before = len(df)
    df = df.dropna(subset=_REQUIRED_COLUMNS)
    n_after = len(df)
    if n_before != n_after:
        logger.info(
            "training_features: dropped {} of {} rows due to NaN required features",
            n_before - n_after, n_before,
        )
    # Zero-impute the new (additive) Phase 3d columns so LightGBM doesn't see NaN.
    _ADDITIVE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _REQUIRED_COLUMNS]
    if _ADDITIVE_COLUMNS:
        df[_ADDITIVE_COLUMNS] = df[_ADDITIVE_COLUMNS].fillna(0.0)

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
