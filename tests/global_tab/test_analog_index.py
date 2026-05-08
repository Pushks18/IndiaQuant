"""Tests for AnalogIndex — purely synthetic, no DB required."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from india_quant.global_tab.analog_index import AnalogIndex, AnalogStats
from india_quant.global_tab.forecaster import FeatureRow
from india_quant.global_tab.training_features import FEATURE_COLUMNS, LABEL_COLUMNS
from india_quant.global_tab.types import Direction


def _synth_frame(n: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        feats = rng.normal(0, 1, size=len(FEATURE_COLUMNS))
        # Synthetic edge: spx_overnight_pct (index 1) drives next-session direction
        ret_bps = float(50.0 * feats[1] + rng.normal(0, 20))
        row = {c: float(feats[j]) for j, c in enumerate(FEATURE_COLUMNS)}
        row["label_return_bps"] = ret_bps
        row["label_direction"] = 1 if ret_bps > 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_from_frame_produces_index():
    idx = AnalogIndex.build_from_frame(_synth_frame(80))
    assert idx.n_samples == 80


def test_lookup_returns_well_typed_stats():
    idx = AnalogIndex.build_from_frame(_synth_frame(80))
    feats = FeatureRow(
        gift_nifty_premium_bps=0.0, spx_overnight_pct=0.5,
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
    )
    stats = idx.lookup(feats, predicted_direction=Direction.LONG, k=10)
    assert isinstance(stats, AnalogStats)
    assert 0 < stats.count <= 10
    assert 0.0 <= stats.winrate <= 1.0
    assert np.isfinite(stats.avg_return_bps)


def test_lookup_short_flips_signed_return():
    """A SHORT prediction's avg_return_bps should be the negation of the LONG case
    on the same feature vector + same K analogs."""
    idx = AnalogIndex.build_from_frame(_synth_frame(80))
    feats = FeatureRow(
        gift_nifty_premium_bps=0.0, spx_overnight_pct=0.5,
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
    )
    long_stats = idx.lookup(feats, predicted_direction=Direction.LONG, k=10)
    short_stats = idx.lookup(feats, predicted_direction=Direction.SHORT, k=10)
    assert short_stats.avg_return_bps == pytest.approx(-long_stats.avg_return_bps, abs=1e-9)


def test_lookup_winrate_tracks_synthetic_edge():
    """Where SPX is strongly positive, the synthetic edge predicts UP — LONG winrate
    should beat SHORT winrate on the same query."""
    idx = AnalogIndex.build_from_frame(_synth_frame(200))
    feats = FeatureRow(
        gift_nifty_premium_bps=0.0, spx_overnight_pct=2.0,  # strong positive
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
    )
    long_stats = idx.lookup(feats, predicted_direction=Direction.LONG, k=20)
    short_stats = idx.lookup(feats, predicted_direction=Direction.SHORT, k=20)
    assert long_stats.winrate > short_stats.winrate


def test_persistence_roundtrip(tmp_path):
    idx = AnalogIndex.build_from_frame(_synth_frame(60))
    path = tmp_path / "ai.pkl"
    idx.save(path)
    loaded = AnalogIndex.load(path)
    assert loaded.n_samples == idx.n_samples
    feats = FeatureRow(
        gift_nifty_premium_bps=0.0, spx_overnight_pct=0.3,
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
    )
    a = idx.lookup(feats, Direction.LONG, k=10)
    b = loaded.lookup(feats, Direction.LONG, k=10)
    assert a == b


def test_no_trade_query_returns_baseline_winrate():
    """For NO_TRADE, winrate is the share of UP days in the K-NN — neutral context."""
    idx = AnalogIndex.build_from_frame(_synth_frame(80))
    feats = FeatureRow(
        gift_nifty_premium_bps=0.0, spx_overnight_pct=0.0,
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
    )
    stats = idx.lookup(feats, predicted_direction=Direction.NO_TRADE, k=20)
    assert 0.0 <= stats.winrate <= 1.0


def test_empty_index_is_safe():
    """Constructing AnalogIndex on an empty frame raises; calling .lookup on a
    zero-row index returns zero stats without crashing."""
    with pytest.raises(ValueError):
        AnalogIndex.build_from_frame(pd.DataFrame())

    # Zero-row direct construction
    F = len(FEATURE_COLUMNS)
    empty = AnalogIndex(
        feature_matrix=np.zeros((0, F)),
        labels_dir=np.zeros(0, dtype=np.int64),
        labels_ret_bps=np.zeros(0),
        feature_means=np.zeros(F),
        feature_stds=np.ones(F),
        top_decile_threshold=1.0,
    )
    feats = FeatureRow(
        gift_nifty_premium_bps=0.0, spx_overnight_pct=0.0,
        dxy_delta_pct=0.0, india_vix_delta_pct=0.0, brent_overnight_pct=0.0,
    )
    stats = empty.lookup(feats, Direction.LONG, k=10)
    assert stats == AnalogStats(0, 0.0, 0.0, False)
