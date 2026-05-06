"""Direction + magnitude forecaster.

Phase 3a ships a deterministic StubArtifact driven by GIFT Nifty premium bps —
enough to drive the sizer and prove the orchestrator wiring. Phase 3b replaces
the stub with a LightGBM artifact behind the same `ModelArtifact` protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from india_quant.global_tab.types import Direction, Mode


@dataclass(frozen=True)
class FeatureRow:
    """Phase 3a feature set. Phase 3b extends; the dataclass is additive."""
    gift_nifty_premium_bps: float | None
    spx_overnight_pct: float | None
    dxy_delta_pct: float | None
    india_vix_delta_pct: float | None
    brent_overnight_pct: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "gift_nifty_premium_bps": self.gift_nifty_premium_bps,
            "spx_overnight_pct": self.spx_overnight_pct,
            "dxy_delta_pct": self.dxy_delta_pct,
            "india_vix_delta_pct": self.india_vix_delta_pct,
            "brent_overnight_pct": self.brent_overnight_pct,
        }


@dataclass(frozen=True)
class IndexForecast:
    index: str
    direction: Direction
    confidence: float
    expected_move_bps: float
    expected_move_low_bps: float
    expected_move_high_bps: float
    feature_attributions: list[tuple[str, float]] = field(default_factory=list)
    no_trade_reason_code: str | None = None


class ModelArtifact(Protocol):
    def predict_direction(
        self, features: FeatureRow, mode: Mode
    ) -> tuple[Direction, float]:
        ...

    def predict_magnitude(
        self, features: FeatureRow, mode: Mode
    ) -> tuple[float, float, float]:
        """Return (median, p10, p90) in bps."""
        ...


# Phase 3a magnitude table per mode (bps). Independent of features in the stub;
# Phase 3b makes magnitude feature-driven via quantile regression.
_STUB_MAGNITUDE: dict[Mode, tuple[float, float, float]] = {
    Mode.AGGRESSIVE:   (80.0, 40.0, 120.0),
    Mode.BALANCED:     (60.0, 30.0, 100.0),
    Mode.CONSERVATIVE: (50.0, 25.0,  85.0),
}

_PREMIUM_THRESHOLD_BPS = 20.0


class StubArtifact:
    """Premium-bps driven direction; per-mode fixed magnitude.

    Direction rule:
      premium > +20 bps → LONG, confidence = clip(0.6 + |p|/200, 0.6, 0.8)
      premium < −20 bps → SHORT, mirror
      otherwise         → NO_TRADE, confidence = 0.0
    """

    def predict_direction(
        self, features: FeatureRow, mode: Mode
    ) -> tuple[Direction, float]:
        p = features.gift_nifty_premium_bps
        if p is None or abs(p) <= _PREMIUM_THRESHOLD_BPS:
            return Direction.NO_TRADE, 0.0
        confidence = min(0.8, 0.6 + abs(p) / 200.0)
        return (Direction.LONG if p > 0 else Direction.SHORT, confidence)

    def predict_magnitude(
        self, features: FeatureRow, mode: Mode
    ) -> tuple[float, float, float]:
        return _STUB_MAGNITUDE[mode]


def _top_attributions(features: FeatureRow, k: int = 3) -> list[tuple[str, float]]:
    items = [(name, val) for name, val in features.as_dict().items() if val is not None]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return items[:k]


def forecast_index(
    index: str,
    as_of: datetime,
    mode: Mode,
    features: FeatureRow,
    model_artifact: ModelArtifact,
) -> IndexForecast:
    direction, confidence = model_artifact.predict_direction(features, mode)
    if direction == Direction.NO_TRADE:
        return IndexForecast(
            index=index,
            direction=direction,
            confidence=0.0,
            expected_move_bps=0.0,
            expected_move_low_bps=0.0,
            expected_move_high_bps=0.0,
            feature_attributions=_top_attributions(features),
            no_trade_reason_code="no_overnight_catalyst",
        )

    median, p10, p90 = model_artifact.predict_magnitude(features, mode)
    # Sign the magnitude by direction so the sizer can simply use spot ± expected_move.
    sign = 1.0 if direction == Direction.LONG else -1.0
    return IndexForecast(
        index=index,
        direction=direction,
        confidence=confidence,
        expected_move_bps=sign * median,
        expected_move_low_bps=sign * p10,
        expected_move_high_bps=sign * p90,
        feature_attributions=_top_attributions(features),
        no_trade_reason_code=None,
    )
