"""Analytical realized-vol forecaster (Phase 6a).

HAR-RV-style blend of three lookback windows:

    σ_forecast = 0.4 * σ_1d  +  0.3 * σ_5d  +  0.3 * σ_20d

where each σ_kd is the standard deviation of daily log returns over the
last k sessions, annualized by × √252.

This is a placeholder for Phase 6b's LightGBM quantile regressor on the
same 11-feature frame the direction model uses. The analytical version
ships in 6a so the straddle strategy is testable end-to-end without a
new training run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


_TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class VolForecast:
    """Annualized realized-vol forecast in % (e.g. 14.2 for 14.2%)."""
    annualized_pct: float
    components: dict[str, float]
    n_obs: int


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _annualize(daily_std: float) -> float:
    return daily_std * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100.0  # → %


def forecast_realized_vol(closes: list[float]) -> VolForecast | None:
    """Compute the HAR-RV blend from oldest-first closes.

    Requires at least 21 closes (so we have 20 log returns for σ_20d).
    Returns None if input is too short or any close is non-positive.
    """
    if not closes or len(closes) < 21:
        return None
    c = [float(x) for x in closes if x is not None and x > 0]
    if len(c) < 21:
        return None
    log_rets = [math.log(c[i]) - math.log(c[i - 1]) for i in range(1, len(c))]
    if len(log_rets) < 20:
        return None

    sigma_1d  = _annualize(abs(log_rets[-1]))                    # |last return| × √252 — degenerate case
    sigma_5d  = _annualize(_std(log_rets[-5:]))
    sigma_20d = _annualize(_std(log_rets[-20:]))

    blended = 0.4 * sigma_1d + 0.3 * sigma_5d + 0.3 * sigma_20d
    return VolForecast(
        annualized_pct=float(blended),
        components={"sigma_1d": sigma_1d, "sigma_5d": sigma_5d, "sigma_20d": sigma_20d},
        n_obs=len(log_rets),
    )
