"""Production LightGBM artifact implementing the `ModelArtifact` protocol.

Drop-in replacement for `StubArtifact`. Pickles produced by
`scripts/train_global_forecaster.py` are loaded lazily on the first
predict call for a given index. Four pickles per index:

    {INDEX}_direction.pkl              → LGBMClassifier (binary up vs down)
    {INDEX}_magnitude_q10.pkl          → LGBMRegressor  (alpha=0.10)
    {INDEX}_magnitude_q50.pkl          → LGBMRegressor  (alpha=0.50)
    {INDEX}_magnitude_q90.pkl          → LGBMRegressor  (alpha=0.90)

If any pickle is missing, `predict_direction` raises `ArtifactMissingError`
so the Flask route can catch it and fall back to `StubArtifact`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from india_quant.global_tab.forecaster import FeatureRow
from india_quant.global_tab.training_features import FEATURE_COLUMNS
from india_quant.global_tab.types import Direction, Mode


class ArtifactMissingError(FileNotFoundError):
    """Raised when a required LightGBM pickle is absent or unreadable."""


# Per-mode probability margin around 0.5 for LONG/SHORT vs NO_TRADE.
# Aggressive: trade more often (smaller dead zone).
# Conservative: only trade when the model is decisive.
_DIRECTION_THRESHOLD: dict[Mode, float] = {
    Mode.AGGRESSIVE:   0.02,
    Mode.BALANCED:     0.05,
    Mode.CONSERVATIVE: 0.10,
}

_CONFIDENCE_FLOOR = 0.5
_CONFIDENCE_CEIL  = 0.85


class LightGBMArtifact:
    """Lazy-loading LightGBM artifact. One instance covers all indices."""

    name = "lightgbm"

    def __init__(self, models_dir: str | Path = "models/global_tab") -> None:
        self._dir = Path(models_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    # ---- internal --------------------------------------------------------

    def _load_index(self, index: str) -> dict[str, Any]:
        if index in self._cache:
            return self._cache[index]
        files = {
            "direction":  self._dir / f"{index}_direction.pkl",
            "q10":        self._dir / f"{index}_magnitude_q10.pkl",
            "q50":        self._dir / f"{index}_magnitude_q50.pkl",
            "q90":        self._dir / f"{index}_magnitude_q90.pkl",
        }
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise ArtifactMissingError(
                f"LightGBMArtifact: missing pickles for {index}: {missing}"
            )
        try:
            booster = {k: joblib.load(p) for k, p in files.items()}
        except Exception as exc:  # pickle/version mismatch
            raise ArtifactMissingError(
                f"LightGBMArtifact: failed to load pickles for {index}: {exc}"
            ) from exc

        # Guard against stale pickles trained on a smaller FEATURE_COLUMNS list.
        expected = len(FEATURE_COLUMNS)
        for kind, model in booster.items():
            n_in = getattr(model, "n_features_in_", None)
            if n_in is not None and n_in != expected:
                raise ArtifactMissingError(
                    f"LightGBMArtifact: {index}_{kind} pickle expects {n_in} features "
                    f"but FEATURE_COLUMNS has {expected}; retrain required"
                )
        self._cache[index] = booster
        return booster

    @staticmethod
    def _vectorize(features: FeatureRow) -> np.ndarray:
        d = features.as_dict()
        # Median imputation (None → 0.0). Keep the order in lockstep with
        # FEATURE_COLUMNS so live serving matches train.
        row = [
            float(d[col]) if d.get(col) is not None else 0.0
            for col in FEATURE_COLUMNS
        ]
        return np.asarray(row, dtype=np.float64).reshape(1, -1)

    # ---- protocol API ----------------------------------------------------

    def predict_direction(
        self, features: FeatureRow, mode: Mode, *, index: str = "NIFTY"
    ) -> tuple[Direction, float]:
        bundle = self._load_index(index)
        clf = bundle["direction"]
        vec = self._vectorize(features)

        # Both LGBMClassifier (sklearn API) and lightgbm.Booster expose predict;
        # for the classifier we need predict_proba; for raw Booster, predict.
        if hasattr(clf, "predict_proba"):
            proba_up = float(clf.predict_proba(vec)[0, 1])
        else:
            proba_up = float(np.atleast_1d(clf.predict(vec))[0])

        thresh = _DIRECTION_THRESHOLD[mode]
        if proba_up > 0.5 + thresh:
            direction = Direction.LONG
        elif proba_up < 0.5 - thresh:
            direction = Direction.SHORT
        else:
            return Direction.NO_TRADE, 0.0

        confidence = abs(proba_up - 0.5) * 2.0
        confidence = min(_CONFIDENCE_CEIL, max(_CONFIDENCE_FLOOR, confidence))
        return direction, confidence

    def predict_magnitude(
        self, features: FeatureRow, mode: Mode, *, index: str = "NIFTY"
    ) -> tuple[float, float, float]:
        try:
            bundle = self._load_index(index)
        except ArtifactMissingError:
            return 0.0, 0.0, 0.0
        vec = self._vectorize(features)

        def _predict(model) -> float:
            if hasattr(model, "predict"):
                y = model.predict(vec)
                return float(np.atleast_1d(y)[0])
            raise TypeError(f"unrecognised model object: {type(model)}")

        median = _predict(bundle["q50"])
        p10    = _predict(bundle["q10"])
        p90    = _predict(bundle["q90"])

        # Guard against pathological miscalibration: enforce p10 ≤ p50 ≤ p90.
        lo, hi = min(p10, p90), max(p10, p90)
        median = max(lo, min(hi, median))
        return abs(median), abs(lo), abs(hi)
