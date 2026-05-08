"""LightGBM realized-vol artifact for Phase 6b.

Replaces the analytical HAR-RV blend in vol_forecaster with a learned
quantile regressor. One pickle per index:

    models/global_tab/{INDEX}_vol_q50.pkl

Loaded lazily on first call. Falls back gracefully (returns None) when
the pickle is missing — orchestrator then uses the analytical forecast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger

from india_quant.global_tab.forecaster import FeatureRow
from india_quant.global_tab.training_features import FEATURE_COLUMNS


class LightGBMVolArtifact:
    """Lazy-loading LightGBM vol artifact. One instance covers all indices."""

    name = "lightgbm-vol"

    def __init__(self, models_dir: str | Path = "models/global_tab") -> None:
        self._dir = Path(models_dir)
        self._cache: dict[str, Any] = {}

    def _load_index(self, index: str):
        if index in self._cache:
            return self._cache[index]
        path = self._dir / f"{index}_vol_q50.pkl"
        if not path.exists():
            self._cache[index] = None
            return None
        try:
            booster = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LightGBMVolArtifact: load failed for {} ({})", path, exc)
            self._cache[index] = None
            return None
        n_in = getattr(booster, "n_features_in_", None)
        if n_in is not None and n_in != len(FEATURE_COLUMNS):
            logger.warning(
                "LightGBMVolArtifact: {} expects {} features but FEATURE_COLUMNS has {}; "
                "falling back to analytical forecast",
                path, n_in, len(FEATURE_COLUMNS),
            )
            self._cache[index] = None
            return None
        self._cache[index] = booster
        return booster

    @staticmethod
    def _vectorize(features: FeatureRow) -> np.ndarray:
        d = features.as_dict()
        row = [
            float(d[col]) if d.get(col) is not None else 0.0
            for col in FEATURE_COLUMNS
        ]
        return np.asarray(row, dtype=np.float64).reshape(1, -1)

    def predict_vol(self, features: FeatureRow, *, index: str = "NIFTY") -> float | None:
        """Annualized realized-vol forecast in %; None if pickle missing or invalid.

        Floors at 0.0 (vol can't be negative). The model's q50 estimate is
        the median forecast, which the straddle strategy treats as a point
        prediction.
        """
        booster = self._load_index(index)
        if booster is None:
            return None
        vec = self._vectorize(features)
        try:
            y = booster.predict(vec)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LightGBMVolArtifact: predict failed for {} ({})", index, exc)
            return None
        out = float(np.atleast_1d(y)[0])
        return max(out, 0.0)
