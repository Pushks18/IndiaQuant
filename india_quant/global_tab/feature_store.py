"""Point-in-time feature store.

The store guarantees the no-future-peek invariant structurally: get(name, at)
can never return a value derived from data with timestamp > at, because every
feature is stored as a sorted pandas Series and lookup is asof <=.

This module is used by both the live forecaster and the walk-forward backtest
so feature assembly is identical across train and serve. Train/serve skew is
impossible by construction.
"""
from datetime import datetime

import pandas as pd


class FuturePeekError(LookupError):
    """Raised when no observation exists at-or-before the requested time.

    The semantics: from the caller's point in time, the only data available
    for this feature is in the future. Returning anything would be a peek.
    """


class PointInTimeFeatureStore:
    def __init__(self) -> None:
        self._series: dict[str, pd.Series] = {}

    def register(self, name: str, values: pd.Series) -> None:
        if not isinstance(values.index, pd.DatetimeIndex):
            raise TypeError(
                f"feature {name!r} must have a DatetimeIndex, got {type(values.index).__name__}"
            )
        if not values.index.is_monotonic_increasing:
            raise ValueError(f"feature {name!r} index must be monotonically increasing")
        self._series[name] = values

    def get(self, name: str, at: datetime) -> float:
        series = self._series[name]  # KeyError on unknown feature
        ts = pd.Timestamp(at)
        eligible = series.loc[series.index <= ts]
        if eligible.empty:
            raise FuturePeekError(
                f"feature {name!r} has no observation at or before {ts.isoformat()}"
            )
        return float(eligible.iloc[-1])

    def features(self) -> list[str]:
        return list(self._series)
