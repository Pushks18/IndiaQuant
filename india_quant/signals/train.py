"""Training driver: walk-forward validation + final model save.

Reads factor_scores + signal_labels from DB, trains XGBoost + LightGBM
ensemble per horizon, saves pickles to models/.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
from loguru import logger

from india_quant.signals.ml_models import ReturnPredictor, MODEL_DIR


def train_and_validate(
    horizon: str = "1d",
    start: str = "2019-01-01",
    end: str | None = None,
    train_window: int = 252,
    test_window: int = 63,
) -> dict:
    """Run walk-forward, then retrain on full history and persist."""
    end = end or date.today().isoformat()
    rp = ReturnPredictor()

    logger.info(f"[Train] Walk-forward validation: {start} → {end} (horizon={horizon})")
    wf = rp.walk_forward_validate(
        start_date=start,
        end_date=end,
        train_window=train_window,
        test_window=test_window,
        horizon=horizon,
    )
    logger.info(f"[Train] Walk-forward result: {wf}")

    if "error" in wf:
        return wf

    logger.info("[Train] Retraining on full history for production model")
    rp.retrain_weekly(horizon=horizon)

    summary = {
        "horizon": horizon,
        "walk_forward": wf,
        "models_saved": [str(MODEL_DIR / f"xgb_{horizon}.pkl"),
                         str(MODEL_DIR / f"lgb_{horizon}.pkl")],
    }
    out = MODEL_DIR / f"training_summary_{horizon}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"[Train] Summary written to {out}")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", default="1d", choices=["1d", "5d", "21d"])
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=63)
    args = p.parse_args()
    print(json.dumps(train_and_validate(
        horizon=args.horizon, start=args.start, end=args.end,
        train_window=args.train_window, test_window=args.test_window,
    ), indent=2, default=str))
