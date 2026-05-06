"""Correlation heatmap builder.

Pure function: takes a wide history DataFrame indexed by date, returns a
CorrelationHeatmap with one CorrelationCell per (row_asset, col_asset) pair.
Computes Pearson correlation on log returns over rolling 20- and 60-day
windows, anchored at as_of.

A "W-day window" uses the W most recent price observations to derive W-1
log returns, which are then correlated. GIFT Nifty is not in the heatmap
(no historical EOD store yet — added in a later phase). Columns with fewer
than 30 valid observations are skipped.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from india_quant.global_tab.types import CorrelationCell, CorrelationHeatmap

ROW_TICKERS: list[str] = ["NIFTY", "BANKNIFTY"]
COLUMN_TICKERS: list[str] = [
    "SPX", "NASDAQ", "DXY", "BRENT", "INDIA_VIX",
    "US10Y", "NIKKEI", "HSI", "FTSE",
]

_MIN_OBS = 30


def _window_corr(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int,
) -> float | None:
    """Compute Pearson correlation of log returns over the last *window* prices.

    Takes the last ``window`` valid (non-NaN) price observations from each
    series (requiring at least ``_MIN_OBS`` total observations), computes
    log returns (``window - 1`` values), then correlates.
    Returns ``None`` when either series has insufficient observations or
    insufficient variance.
    """
    # Keep only rows where both series are valid; guard against sparse columns
    pair_prices = pd.concat([prices_a, prices_b], axis=1).dropna()
    if len(pair_prices) < _MIN_OBS or len(pair_prices) < window:
        return None

    sliced = pair_prices.iloc[-window:]
    log_ret = np.log(sliced).diff().dropna()

    if log_ret.iloc[:, 0].std(ddof=0) == 0 or log_ret.iloc[:, 1].std(ddof=0) == 0:
        return None

    rho = float(log_ret.iloc[:, 0].corr(log_ret.iloc[:, 1]))
    if np.isnan(rho):
        return None
    return rho


def build_heatmap(*, history: pd.DataFrame, as_of: date) -> CorrelationHeatmap:
    cells: list[CorrelationCell] = []

    all_tickers = ROW_TICKERS + COLUMN_TICKERS
    prices = {
        col: history[col]
        for col in history.columns
        if col in all_tickers
    }

    for row in ROW_TICKERS:
        if row not in prices:
            continue
        for col in COLUMN_TICKERS:
            if col not in prices:
                continue
            rho_20 = _window_corr(prices[row], prices[col], window=20)
            rho_60 = _window_corr(prices[row], prices[col], window=60)
            if rho_20 is None or rho_60 is None:
                continue
            cells.append(
                CorrelationCell(
                    asset_a=row,
                    asset_b=col,
                    rho_20d=rho_20,
                    rho_60d=rho_60,
                )
            )

    return CorrelationHeatmap(as_of=as_of, cells=cells)
