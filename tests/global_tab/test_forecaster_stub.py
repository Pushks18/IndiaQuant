"""Tests for forecaster.py + StubArtifact (Phase 3a Task 3)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from india_quant.global_tab.forecaster import (
    FeatureRow,
    IndexForecast,
    StubArtifact,
    forecast_index,
)
from india_quant.global_tab.types import Direction, Mode


def _features(premium: float | None) -> FeatureRow:
    return FeatureRow(
        gift_nifty_premium_bps=premium,
        spx_overnight_pct=0.5,
        dxy_delta_pct=-0.1,
        india_vix_delta_pct=0.2,
        brent_overnight_pct=0.8,
    )


def test_long_when_gift_premium_strongly_positive():
    fc = forecast_index(
        "NIFTY", datetime(2026, 5, 5, tzinfo=timezone.utc),
        Mode.BALANCED, _features(45.0), StubArtifact(),
    )
    assert fc.direction == Direction.LONG
    assert fc.confidence > 0.6
    assert fc.expected_move_bps > 0


def test_short_when_gift_premium_strongly_negative():
    fc = forecast_index(
        "NIFTY", datetime(2026, 5, 5, tzinfo=timezone.utc),
        Mode.BALANCED, _features(-50.0), StubArtifact(),
    )
    assert fc.direction == Direction.SHORT
    assert fc.expected_move_bps < 0
    assert fc.expected_move_low_bps < 0


def test_no_trade_when_premium_within_band():
    fc = forecast_index(
        "NIFTY", datetime(2026, 5, 5, tzinfo=timezone.utc),
        Mode.BALANCED, _features(10.0), StubArtifact(),
    )
    assert fc.direction == Direction.NO_TRADE
    assert fc.no_trade_reason_code == "no_overnight_catalyst"
    assert fc.confidence == 0.0
    assert fc.expected_move_bps == 0.0


def test_no_trade_when_premium_missing():
    fc = forecast_index(
        "NIFTY", datetime(2026, 5, 5, tzinfo=timezone.utc),
        Mode.BALANCED, _features(None), StubArtifact(),
    )
    assert fc.direction == Direction.NO_TRADE


@pytest.mark.parametrize("mode,expected_median", [
    (Mode.AGGRESSIVE, 80.0),
    (Mode.BALANCED, 60.0),
    (Mode.CONSERVATIVE, 50.0),
])
def test_magnitude_table_by_mode(mode, expected_median):
    fc = forecast_index(
        "NIFTY", datetime(2026, 5, 5, tzinfo=timezone.utc),
        mode, _features(45.0), StubArtifact(),
    )
    assert abs(fc.expected_move_bps) == expected_median


def test_feature_attributions_top_3_by_abs_value():
    feats = FeatureRow(
        gift_nifty_premium_bps=45.0,
        spx_overnight_pct=0.5,
        dxy_delta_pct=-0.1,
        india_vix_delta_pct=10.0,
        brent_overnight_pct=0.8,
    )
    fc = forecast_index(
        "NIFTY", datetime(2026, 5, 5, tzinfo=timezone.utc),
        Mode.BALANCED, feats, StubArtifact(),
    )
    assert len(fc.feature_attributions) == 3
    names = [n for n, _ in fc.feature_attributions]
    # Top 3 by |value|: gift_nifty_premium_bps (45), india_vix_delta_pct (10), brent (0.8)
    assert names[0] == "gift_nifty_premium_bps"
    assert "india_vix_delta_pct" in names
    assert "brent_overnight_pct" in names


def test_confidence_capped_at_0_8():
    fc = forecast_index(
        "NIFTY", datetime(2026, 5, 5, tzinfo=timezone.utc),
        Mode.BALANCED, _features(500.0), StubArtifact(),
    )
    assert fc.confidence == pytest.approx(0.8)


def test_index_forecast_is_frozen():
    fc = forecast_index(
        "NIFTY", datetime(2026, 5, 5, tzinfo=timezone.utc),
        Mode.BALANCED, _features(45.0), StubArtifact(),
    )
    with pytest.raises(Exception):
        fc.confidence = 0.0  # type: ignore[misc]
