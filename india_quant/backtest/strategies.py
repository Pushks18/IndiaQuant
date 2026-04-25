"""Concrete long/short strategy selectors.

Each selector takes (rebalance_date, history_prices) and returns
{"long": [tickers], "short": [tickers]}, using only data up to rebalance_date.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from india_quant.data.db import get_session


# ─── Pure price-based ────────────────────────────────────────────────────────

def momentum_12_1(n: int = 5):
    """Jegadeesh-Titman: 12-month return skipping last month. Long winners, short losers."""
    def select(rd: date, prices: pd.DataFrame) -> dict:
        scores = {}
        for tkr, grp in prices.groupby("ticker"):
            grp = grp.sort_values("date")
            if len(grp) < 252:
                continue
            close = grp["close"].values
            mom = close[-22] / close[-252] - 1
            scores[tkr] = mom
        if len(scores) < 2 * n:
            return {"long": [], "short": []}
        s = pd.Series(scores).sort_values()
        return {"long": s.tail(n).index.tolist(), "short": s.head(n).index.tolist()}
    return select


def short_term_reversal(n: int = 5):
    """Long oversold (worst last 21d), short overbought. Tests the negative-IC effect."""
    def select(rd: date, prices: pd.DataFrame) -> dict:
        scores = {}
        for tkr, grp in prices.groupby("ticker"):
            grp = grp.sort_values("date")
            if len(grp) < 22:
                continue
            close = grp["close"].values
            mom_1 = close[-1] / close[-22] - 1
            scores[tkr] = mom_1
        if len(scores) < 2 * n:
            return {"long": [], "short": []}
        s = pd.Series(scores).sort_values()
        # Long the WORST last-month performers (expect mean reversion up)
        return {"long": s.head(n).index.tolist(), "short": s.tail(n).index.tolist()}
    return select


def low_vol(n: int = 5):
    """Long lowest realized vol, short highest. The classic low-vol anomaly."""
    def select(rd: date, prices: pd.DataFrame) -> dict:
        scores = {}
        for tkr, grp in prices.groupby("ticker"):
            grp = grp.sort_values("date")
            if len(grp) < 22:
                continue
            rets = pd.Series(grp["close"].values).pct_change().dropna()
            if len(rets) < 21:
                continue
            scores[tkr] = float(rets.iloc[-21:].std())
        if len(scores) < 2 * n:
            return {"long": [], "short": []}
        s = pd.Series(scores).sort_values()
        return {"long": s.head(n).index.tolist(), "short": s.tail(n).index.tolist()}
    return select


# ─── Factor-driven (uses precomputed factor_scores in DB) ───────────────────

def factor_score(factor_col: str, n: int = 5, ascending_for_long: bool = False):
    """Generic: long top-n by factor (or bottom if ascending_for_long=True)."""
    def select(rd: date, prices: pd.DataFrame) -> dict:
        with get_session() as s:
            row = s.execute(text(f"""
                WITH ranked AS (
                  SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                  FROM factor_scores WHERE date <= :rd
                )
                SELECT ticker, {factor_col} AS sc FROM ranked WHERE rn = 1
                  AND {factor_col} IS NOT NULL
            """), {"rd": rd}).fetchall()
        if not row or len(row) < 2 * n:
            return {"long": [], "short": []}
        df = pd.DataFrame(row, columns=["ticker", "sc"])
        df = df.dropna().sort_values("sc")
        if ascending_for_long:
            return {"long": df["ticker"].head(n).tolist(),
                    "short": df["ticker"].tail(n).tolist()}
        return {"long": df["ticker"].tail(n).tolist(),
                "short": df["ticker"].head(n).tolist()}
    return select


# ─── ML-driven (uses retrained model, predicts at rebalance date) ───────────

_ml_predictor = None


def _get_predictor():
    global _ml_predictor
    if _ml_predictor is None:
        import joblib
        from pathlib import Path
        models_dir = Path(__file__).parent.parent.parent / "models"
        try:
            xgb = joblib.load(models_dir / "xgb_1d.pkl")
            lgb = joblib.load(models_dir / "lgb_1d.pkl")
            _ml_predictor = (xgb, lgb)
        except FileNotFoundError:
            _ml_predictor = None
    return _ml_predictor


def ml_signal(n: int = 5, contrarian: bool = False):
    """Long top-n by model-predicted return (or bottom if contrarian=True)."""
    from india_quant.signals.ml_models import ReturnPredictor
    predictor = ReturnPredictor()

    def select(rd: date, prices: pd.DataFrame) -> dict:
        models = _get_predictor()
        if models is None:
            return {"long": [], "short": []}
        xgb, lgb = models

        with get_session() as s:
            rows = s.execute(text("""
                WITH ranked AS (
                  SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                  FROM factor_scores WHERE date <= :rd
                )
                SELECT * FROM ranked WHERE rn = 1
            """), {"rd": rd}).mappings().all()
        if not rows or len(rows) < 2 * n:
            return {"long": [], "short": []}

        df = pd.DataFrame([dict(r) for r in rows])
        train_features = list(getattr(xgb, "feature_names_in_", predictor.FEATURE_COLS))
        for c in train_features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
            else:
                df[c] = 0.0
        X = df[train_features]
        preds = (xgb.predict(X) + lgb.predict(X)) / 2
        df["pred"] = preds
        df = df.sort_values("pred")
        if contrarian:
            return {"long": df["ticker"].head(n).tolist(),
                    "short": df["ticker"].tail(n).tolist()}
        return {"long": df["ticker"].tail(n).tolist(),
                "short": df["ticker"].head(n).tolist()}
    return select


# ─── Long-only variants for risk-averse traders ─────────────────────────────

def long_only(base_selector):
    """Wrap any selector to drop the short basket."""
    def select(rd: date, prices: pd.DataFrame) -> dict:
        out = base_selector(rd, prices)
        return {"long": out.get("long", []), "short": []}
    return select
