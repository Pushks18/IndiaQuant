"""Tests for PointInTimeFeatureStore."""
from datetime import datetime

import pandas as pd
import pytest

from india_quant.global_tab.feature_store import (
    FuturePeekError,
    PointInTimeFeatureStore,
)


def _series(*pairs):
    idx = pd.DatetimeIndex([ts for ts, _ in pairs])
    vals = [v for _, v in pairs]
    return pd.Series(vals, index=idx, dtype=float)


def test_get_returns_most_recent_value_at_or_before_time():
    store = PointInTimeFeatureStore()
    store.register(
        "spx_close",
        _series(
            (datetime(2026, 5, 1, 16, 0), 5600.0),
            (datetime(2026, 5, 2, 16, 0), 5612.0),
            (datetime(2026, 5, 3, 16, 0), 5630.0),
        ),
    )
    assert store.get("spx_close", datetime(2026, 5, 2, 23, 0)) == 5612.0


def test_get_at_exact_timestamp_returns_that_value():
    store = PointInTimeFeatureStore()
    store.register("dxy", _series((datetime(2026, 5, 2, 16, 0), 104.2)))
    assert store.get("dxy", datetime(2026, 5, 2, 16, 0)) == 104.2


def test_get_before_first_observation_raises_future_peek_error():
    store = PointInTimeFeatureStore()
    store.register("vix", _series((datetime(2026, 5, 5, 16, 0), 13.4)))
    with pytest.raises(FuturePeekError) as exc:
        store.get("vix", datetime(2026, 5, 1, 0, 0))
    assert "vix" in str(exc.value)
    assert "2026-05-01" in str(exc.value)


def test_get_unknown_feature_raises_key_error():
    store = PointInTimeFeatureStore()
    with pytest.raises(KeyError):
        store.get("never_registered", datetime(2026, 5, 5))


def test_register_rejects_non_datetime_index():
    store = PointInTimeFeatureStore()
    bad = pd.Series([1.0, 2.0], index=[0, 1], dtype=float)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        store.register("bad", bad)


def test_register_rejects_unsorted_index():
    store = PointInTimeFeatureStore()
    bad = _series(
        (datetime(2026, 5, 3), 1.0),
        (datetime(2026, 5, 1), 2.0),
    )
    with pytest.raises(ValueError, match="monotonically increasing"):
        store.register("bad", bad)


def test_register_overwrites_existing_feature():
    store = PointInTimeFeatureStore()
    store.register("x", _series((datetime(2026, 5, 1), 1.0)))
    store.register("x", _series((datetime(2026, 5, 1), 2.0)))
    assert store.get("x", datetime(2026, 5, 1)) == 2.0


def test_features_method_lists_registered_names():
    store = PointInTimeFeatureStore()
    store.register("a", _series((datetime(2026, 5, 1), 1.0)))
    store.register("b", _series((datetime(2026, 5, 1), 2.0)))
    assert sorted(store.features()) == ["a", "b"]
