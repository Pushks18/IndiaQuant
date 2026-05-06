"""Tests for the correlation heatmap builder."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from india_quant.global_tab.correlation import (
    COLUMN_TICKERS,
    ROW_TICKERS,
    build_heatmap,
)
from india_quant.global_tab.types import CorrelationCell, CorrelationHeatmap


def _synthetic_history(n_days: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=date(2026, 5, 4), periods=n_days, freq="B")
    cols = ROW_TICKERS + COLUMN_TICKERS
    base_spx = np.cumsum(rng.normal(0, 0.01, n_days)) + 5500
    base_nifty = 0.6 * (base_spx - 5500) + np.cumsum(rng.normal(0, 0.01, n_days)) + 24000
    base_bn = 0.5 * (base_spx - 5500) + np.cumsum(rng.normal(0, 0.012, n_days)) + 51000
    data = {}
    for c in cols:
        if c == "SPX":
            data[c] = base_spx
        elif c == "NIFTY":
            data[c] = base_nifty
        elif c == "BANKNIFTY":
            data[c] = base_bn
        else:
            data[c] = np.cumsum(rng.normal(0, 0.01, n_days)) + 100
    return pd.DataFrame(data, index=idx)


def test_build_heatmap_returns_typed_object():
    hist = _synthetic_history()
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    assert isinstance(hm, CorrelationHeatmap)
    assert hm.as_of == date(2026, 5, 4)


def test_heatmap_has_two_rows_times_columns():
    hist = _synthetic_history()
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    expected_count = len(ROW_TICKERS) * len(COLUMN_TICKERS)
    assert len(hm.cells) == expected_count


def test_each_cell_has_valid_correlation_range():
    hist = _synthetic_history()
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    for cell in hm.cells:
        assert isinstance(cell, CorrelationCell)
        assert -1.0 <= cell.rho_20d <= 1.0
        assert -1.0 <= cell.rho_60d <= 1.0


def test_nifty_spx_correlation_is_positive_in_synthetic_data():
    hist = _synthetic_history(n_days=120)
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    nifty_spx = next(c for c in hm.cells if c.asset_a == "NIFTY" and c.asset_b == "SPX")
    assert nifty_spx.rho_60d > 0.2


def test_columns_with_insufficient_data_are_skipped():
    hist = _synthetic_history(n_days=120)
    hist["FTSE"] = np.nan
    hist.iloc[:5, hist.columns.get_loc("FTSE")] = 100.0
    hm = build_heatmap(history=hist, as_of=date(2026, 5, 4))
    ftse_cells = [c for c in hm.cells if c.asset_b == "FTSE"]
    assert ftse_cells == []
