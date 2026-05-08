"""Historical analog index for the global-tab forecaster.

Given today's feature vector, find the K most-similar past sessions and
compute (count, win-rate, avg P&L bps). Phase 3a stubbed these to zeros,
which made the narrator print "0 analog sessions averaged 0% win rate"
on every card. This module replaces that with a real lookup.

Distance metric: cosine over z-scored feature vectors. Cheap, robust to
units, and matches the spec's "top-decile analog" gate without
requiring a learned embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from india_quant.global_tab.forecaster import FeatureRow
from india_quant.global_tab.training_features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    assemble_training_frame,
)
from india_quant.global_tab.types import Direction


@dataclass(frozen=True)
class AnalogStats:
    count: int
    winrate: float            # in [0, 1]; share of analogs where realised direction matched the predicted one
    avg_return_bps: float     # mean signed return across the K analogs (signed so SHORTs flip sign)
    top_decile_match: bool    # True iff the closest analog's similarity is in the top decile of the index distribution


_SIMILARITY_TOP_DECILE_KEY = "_top_decile_threshold"


class AnalogIndex:
    """In-memory analog lookup. Build once, query many."""

    def __init__(
        self,
        feature_matrix: np.ndarray,    # (N, F) z-scored
        labels_dir: np.ndarray,        # (N,) in {0, 1}
        labels_ret_bps: np.ndarray,    # (N,) signed bps
        feature_means: np.ndarray,     # (F,)
        feature_stds: np.ndarray,      # (F,)
        top_decile_threshold: float,   # similarity threshold for "top decile"
    ) -> None:
        self._X = feature_matrix
        self._yd = labels_dir
        self._yr = labels_ret_bps
        self._mu = feature_means
        self._sigma = feature_stds
        self._top_decile = top_decile_threshold

    @property
    def n_samples(self) -> int:
        return int(self._X.shape[0])

    @classmethod
    def build_from_db(
        cls,
        *,
        index: str,
        start,
        end,
        session_factory: Callable,
    ) -> "AnalogIndex":
        df = assemble_training_frame(
            index=index, start=start, end=end, session_factory=session_factory,
        )
        return cls.build_from_frame(df)

    @classmethod
    def build_from_frame(cls, df: pd.DataFrame) -> "AnalogIndex":
        if df.empty:
            raise ValueError("AnalogIndex.build_from_frame: empty frame")
        X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        Xz = (X - mu) / sigma
        # Pre-normalize rows for cosine similarity
        norms = np.linalg.norm(Xz, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        Xz = Xz / norms

        yd = df["label_direction"].to_numpy(dtype=np.int64)
        yr = df["label_return_bps"].to_numpy(dtype=np.float64)

        # Top-decile similarity = 90th percentile of pairwise nearest-neighbor sims.
        # Computed once at build time so .lookup() is O(N) per call.
        nn_sims = []
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for i in range(min(len(Xz), 500)):  # cap at 500 for build speed
                sims = Xz @ Xz[i]
                sims = np.nan_to_num(sims, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
                sims[i] = -np.inf
                nn_sims.append(float(sims.max()))
        threshold = float(np.percentile(nn_sims, 90)) if nn_sims else 1.0

        return cls(
            feature_matrix=Xz,
            labels_dir=yd,
            labels_ret_bps=yr,
            feature_means=mu,
            feature_stds=sigma,
            top_decile_threshold=threshold,
        )

    def _vectorize_query(self, features: FeatureRow) -> np.ndarray:
        d = features.as_dict()
        x = np.array(
            [float(d.get(c, 0.0) or 0.0) for c in FEATURE_COLUMNS],
            dtype=np.float64,
        )
        xz = (x - self._mu) / self._sigma
        # Clip to prevent overflow when query is far outside the training distribution
        # (cosine similarity is scale-invariant after the L2-normalize anyway).
        xz = np.clip(xz, -1e6, 1e6)
        n = float(np.linalg.norm(xz))
        if n < 1e-9 or not np.isfinite(n):
            return np.zeros_like(xz)
        return xz / n

    def lookup(
        self,
        features: FeatureRow,
        predicted_direction: Direction,
        k: int = 20,
    ) -> AnalogStats:
        """Find K nearest analogs and report aggregate outcome."""
        if self.n_samples == 0:
            return AnalogStats(0, 0.0, 0.0, False)
        q = self._vectorize_query(features)
        # BLAS occasionally raises FPE flags on cosine sims even when the
        # result is finite; the np.errstate keeps the noise out of logs.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sims = self._X @ q  # (N,)
            sims = np.nan_to_num(sims, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        top_k = min(k, self.n_samples)
        top_idx = np.argsort(-sims)[:top_k]

        winrate = float(self._winrate(top_idx, predicted_direction))
        avg_ret = float(self._avg_signed_return(top_idx, predicted_direction))
        top_decile = bool(sims[top_idx[0]] >= self._top_decile)
        return AnalogStats(
            count=int(top_k),
            winrate=winrate,
            avg_return_bps=avg_ret,
            top_decile_match=top_decile,
        )

    def _winrate(self, idx: np.ndarray, predicted: Direction) -> float:
        if predicted == Direction.NO_TRADE:
            # Neutral baseline: report fraction of UP days for context only.
            return float(np.mean(self._yd[idx]))
        target = 1 if predicted == Direction.LONG else 0
        return float(np.mean(self._yd[idx] == target))

    def _avg_signed_return(self, idx: np.ndarray, predicted: Direction) -> float:
        rets = self._yr[idx]
        if predicted == Direction.SHORT:
            rets = -rets
        return float(np.mean(rets))

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "X":         self._X,
            "yd":        self._yd,
            "yr":        self._yr,
            "mu":        self._mu,
            "sigma":     self._sigma,
            "top_decile": self._top_decile,
        }, path, compress=3)
        logger.info("AnalogIndex: wrote {} ({} rows)", path, self.n_samples)

    @classmethod
    def load(cls, path: str | Path) -> "AnalogIndex":
        d = joblib.load(Path(path))
        return cls(
            feature_matrix=d["X"], labels_dir=d["yd"], labels_ret_bps=d["yr"],
            feature_means=d["mu"], feature_stds=d["sigma"],
            top_decile_threshold=d["top_decile"],
        )
