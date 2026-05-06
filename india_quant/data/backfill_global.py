"""
One-time backfill of global_signals table.
Run once before the first ML retrain so ReturnPredictor has historical features.

Usage:
    python -m india_quant.data.backfill_global            # 365 days
    python -m india_quant.data.backfill_global --days 730 # 2 years
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta, datetime
import pandas as pd
import yfinance as yf
from loguru import logger

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    from datetime import timezone
    IST = timezone.utc

_RAW_GROUPS = {
    "US":          ["^GSPC", "^IXIC", "YM=F"],
    "Europe":      ["^GDAXI", "^FTSE"],
    "Asia":        ["^N225", "^HSI", "^KS11", "^TWII", "^AXJO", "000001.SS",
                    "^CNXIT", "^NSEBANK", "^CNXINFRA", "^CNXPHARMA", "^CNXREALTY", "^CNXENERGY"],
    "FX":          ["USDINR=X", "DX-Y.NYB", "USDJPY=X", "^TNX", "^VIX", "^INDIAVIX"],
    "Commodities": ["CL=F", "BZ=F", "GC=F", "NG=F"],
}

ALL_TICKERS = [t for tickers in _RAW_GROUPS.values() for t in tickers]

TICKER_GROUP = {t: g for g, tickers in _RAW_GROUPS.items() for t in tickers}

from india_quant.signals.global_context import GROUPS as _GROUPS
TICKER_LABEL: dict[str, str] = {}
for g_tickers in _GROUPS.values():
    TICKER_LABEL.update(g_tickers)


def _flatten_to_series(obj) -> pd.Series:
    """yfinance sometimes returns a 1-column DataFrame; coerce to Series."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        return obj.squeeze()
    return obj


def _normalize_index(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    idx = s.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    try:
        idx = idx.normalize()
    except Exception:
        pass
    out = s.copy()
    out.index = idx
    return out


def _compute_corr_series(returns: pd.Series, nifty_ret: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation of returns vs nifty_ret."""
    a = _normalize_index(_flatten_to_series(returns))
    b = _normalize_index(_flatten_to_series(nifty_ret))
    combined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(combined) < window:
        return pd.Series(dtype=float)
    return combined.iloc[:, 0].rolling(window).corr(combined.iloc[:, 1])


def backfill(days: int = 365):
    from india_quant.data.models import GlobalSignal
    from india_quant.data.db import get_session
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    period = f"{days + 30}d"
    logger.info(f"Downloading {len(ALL_TICKERS)} tickers, period={period} ...")

    nifty_raw   = yf.download("^NSEI", period=period, auto_adjust=True, progress=False)
    nifty_close = _flatten_to_series(nifty_raw["Close"])
    nifty_ret   = _normalize_index(nifty_close.pct_change().dropna())

    df = yf.download(ALL_TICKERS, period=period, auto_adjust=True, progress=True, threads=True)
    if df.empty:
        logger.error("Download returned empty dataframe")
        return

    close_df = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]

    inserted = 0
    with get_session() as session:
        for ticker in ALL_TICKERS:
            try:
                if isinstance(close_df.columns, pd.Index) and ticker in close_df.columns:
                    close = close_df[ticker].dropna()
                else:
                    continue
                if len(close) < 32:
                    continue

                close   = _normalize_index(close)
                returns = close.pct_change().dropna()
                corr30  = _compute_corr_series(returns, nifty_ret, 30)
                corr90  = _compute_corr_series(returns, nifty_ret, 90)

                for dt in returns.index[-days:]:
                    trade_date = dt.date() if hasattr(dt, "date") else dt
                    pct_1d = round(float(returns.loc[dt]) * 100, 3)
                    pos = list(close.index).index(dt)
                    pct_5d = None
                    if pos >= 5:
                        prev5 = float(close.iloc[pos - 5])
                        pct_5d = round((float(close.loc[dt]) / prev5 - 1) * 100, 3) if prev5 else None

                    c30_raw = corr30.loc[dt] if dt in corr30.index else None
                    c90_raw = corr90.loc[dt] if dt in corr90.index else None
                    c30 = float(c30_raw) if c30_raw is not None and not pd.isna(c30_raw) else None
                    c90 = float(c90_raw) if c90_raw is not None and not pd.isna(c90_raw) else None

                    stmt = pg_insert(GlobalSignal).values(
                        date=trade_date,
                        ticker=ticker,
                        label=TICKER_LABEL.get(ticker, ticker),
                        group=TICKER_GROUP.get(ticker, "OTHER"),
                        pct_1d=pct_1d,
                        pct_5d=pct_5d,
                        corr_30d=round(c30, 3) if c30 is not None else None,
                        corr_90d=round(c90, 3) if c90 is not None else None,
                        regime="NEUTRAL",
                    ).on_conflict_do_nothing()
                    session.execute(stmt)
                    inserted += 1

            except Exception as e:
                logger.warning(f"Skipping {ticker}: {e}")
                continue

    logger.info(f"Backfill complete: {inserted} rows inserted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    backfill(days=args.days)
