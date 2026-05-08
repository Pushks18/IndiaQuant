"""Tests for vol_forecaster.forecast_realized_vol — pure analytical."""
from __future__ import annotations

import math

import pytest

from india_quant.global_tab.vol_forecaster import (
    VolForecast, forecast_realized_vol, _TRADING_DAYS_PER_YEAR,
)


def test_constant_prices_yield_zero_vol():
    closes = [100.0] * 30
    vf = forecast_realized_vol(closes)
    assert vf is not None
    assert vf.annualized_pct == pytest.approx(0.0, abs=1e-9)


def test_too_short_input_returns_none():
    assert forecast_realized_vol([]) is None
    assert forecast_realized_vol([100.0] * 5) is None
    assert forecast_realized_vol([100.0] * 20) is None  # need >=21


def test_known_vol_input_matches_blend():
    """Synthetic series: log returns alternate +0.01, -0.01 → daily std ≈ 0.01."""
    closes = [100.0]
    sign = 1
    for _ in range(40):
        closes.append(closes[-1] * math.exp(0.01 * sign))
        sign *= -1
    vf = forecast_realized_vol(closes)
    assert vf is not None
    sigma_5d = vf.components["sigma_5d"]
    sigma_20d = vf.components["sigma_20d"]
    # daily std ≈ 0.01 → annualized × √252 × 100% ≈ 15.87
    assert sigma_5d == pytest.approx(0.01 * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100, rel=0.05)
    assert sigma_20d == pytest.approx(0.01 * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100, rel=0.05)


def test_components_blend_to_annualized():
    closes = [100.0 * math.exp(0.005 * i) for i in range(30)]
    vf = forecast_realized_vol(closes)
    assert vf is not None
    expected = (
        0.4 * vf.components["sigma_1d"]
        + 0.3 * vf.components["sigma_5d"]
        + 0.3 * vf.components["sigma_20d"]
    )
    assert vf.annualized_pct == pytest.approx(expected, abs=1e-9)
