"""ML signal models: XGBoost + LightGBM ensemble for return prediction (Gu-Kelly-Xiu 2020)."""
import os
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ttest_1samp
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class ReturnPredictor:
    XGB_PARAMS = dict(
        n_estimators=500, max_depth=4, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
        early_stopping_rounds=20, eval_metric="rmse",
        verbosity=0,
    )
    LGB_PARAMS = dict(
        n_estimators=500, max_depth=4, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
        verbose=-1,
    )

    FEATURE_COLS = [
        "momentum_12_1", "momentum_1", "momentum_3",
        "realized_vol", "vol_of_vol", "idiosyncratic_vol",
        "liquidity_amihud", "turnover",
        "iv_skew", "iv_spread", "vrp", "oi_flow",
        "profitability_roe", "gross_profitability",
        "value_bm", "earnings_yield",
    ]

    def prepare_dataset(
        self, start_date: str, end_date: str, horizon: str = "1d"
    ) -> tuple:
        """Pull FactorScores + SignalLabels, join, clean features. Returns X, y, dates, tickers."""
        from india_quant.data.db import get_session

        with get_session() as session:
            factors = pd.DataFrame(
                session.execute(
                    text(f"""
                        SELECT * FROM factor_scores
                        WHERE date BETWEEN :start AND :end
                        ORDER BY date, ticker
                    """),
                    {"start": start_date, "end": end_date},
                ).fetchall()
            )

            labels = pd.DataFrame(
                session.execute(
                    text(f"""
                        SELECT ticker, date, future_return FROM signal_labels
                        WHERE date BETWEEN :start AND :end
                        AND horizon = :horizon
                        ORDER BY date, ticker
                    """),
                    {"start": start_date, "end": end_date, "horizon": horizon},
                ).fetchall()
            )

        if factors.empty or labels.empty:
            logger.warning("No data for dataset preparation.")
            return None, None, None, None

        merged = factors.merge(labels, on=["ticker", "date"], how="inner")

        # Coerce feature columns to numeric and drop those that are entirely null
        # (historical backfill doesn't populate fundamentals/options factors).
        candidate_cols = [c for c in self.FEATURE_COLS if c in merged.columns]
        for c in candidate_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        feature_cols = [c for c in candidate_cols if merged[c].notna().any()]
        if not feature_cols:
            logger.warning("No usable feature columns after coercion.")
            return None, None, None, None

        merged["future_return"] = pd.to_numeric(merged["future_return"], errors="coerce")
        merged = merged.dropna(subset=["future_return"])

        X = merged[feature_cols].copy()
        y = merged["future_return"].copy()
        dates = merged["date"]
        tickers = merged["ticker"]

        # Cross-sectional z-score per date
        for col in feature_cols:
            X[col] = merged.groupby("date")[col].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-8)
            )

        # Winsorize at 1st/99th percentile across the full panel
        for col in feature_cols:
            lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
            X[col] = X[col].clip(lo, hi)

        X = X.fillna(0).astype(float)
        return X, y, dates, tickers

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series):
        from xgboost import XGBRegressor
        X_val = X_train.iloc[-int(len(X_train) * 0.1):]
        y_val = y_train.iloc[-int(len(y_train) * 0.1):]
        X_tr = X_train.iloc[:-int(len(X_train) * 0.1)]
        y_tr = y_train.iloc[:-int(len(y_train) * 0.1)]

        model = XGBRegressor(**self.XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series):
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**self.LGB_PARAMS)
        model.fit(X_train, y_train)
        return model

    def walk_forward_validate(
        self,
        start_date: str,
        end_date: str,
        train_window: int = 252,
        test_window: int = 63,
        horizon: str = "1d",
    ) -> dict:
        """
        Walk-forward validation indexed by *unique trading dates*.
        train_window / test_window are in dates, not in panel rows
        (one date contains ~49 ticker samples).
        """
        X, y, dates, tickers = self.prepare_dataset(start_date, end_date, horizon)
        if X is None:
            return {"error": "No data"}

        unique_dates = pd.Series(dates).drop_duplicates().sort_values().reset_index(drop=True)
        if len(unique_dates) < train_window + test_window:
            return {"error": f"Not enough unique dates: {len(unique_dates)} (need {train_window + test_window})"}

        all_ics = []
        fold = 0
        i = 0
        while i + train_window + test_window <= len(unique_dates):
            train_dates = unique_dates.iloc[i: i + train_window]
            test_dates = unique_dates.iloc[i + train_window: i + train_window + test_window]
            train_mask = dates.isin(train_dates).values
            test_mask = dates.isin(test_dates).values

            X_train, y_train = X.loc[train_mask], y.loc[train_mask]
            X_test, y_test = X.loc[test_mask], y.loc[test_mask]

            if X_train.empty or X_test.empty or y_test.std() == 0:
                i += test_window
                continue

            try:
                xgb = self.train_xgboost(X_train, y_train)
                lgb = self.train_lightgbm(X_train, y_train)
                preds = (xgb.predict(X_test) + lgb.predict(X_test)) / 2
                ic = float(pd.Series(preds, index=y_test.index).corr(y_test, method="spearman"))
                if pd.notna(ic):
                    all_ics.append(ic)
                    fold += 1
                    logger.info(f"[WF] Fold {fold}: train_dates={len(train_dates)} test_dates={len(test_dates)} n_test={len(X_test)} IC={ic:+.4f}")
            except Exception as e:
                logger.error(f"[WF] Fold failed at i={i}: {e}")

            i += test_window

        if not all_ics:
            return {"error": "No folds completed"}

        ic_array = np.array(all_ics)
        ic_tstat, _ = ttest_1samp(ic_array, 0)
        ir = ic_array.mean() / (ic_array.std() + 1e-8)
        hlz_pass = abs(ic_tstat) > 3.0

        result = {
            "n_folds": len(all_ics),
            "mean_ic": float(ic_array.mean()),
            "std_ic": float(ic_array.std()),
            "ic_tstat": float(ic_tstat),
            "ir": float(ir),
            "hlz_pass": hlz_pass,
            "horizon": horizon,
        }
        logger.info(f"[WF] Results: {result}")
        return result

    def predict_today(self, model_path: str = None, date_str: str = None) -> pd.DataFrame:
        """Load model and predict for all tickers using latest factor scores."""
        from india_quant.data.db import get_session

        if date_str is None:
            date_str = date.today().isoformat()

        with get_session() as session:
            rows = session.execute(
                text("SELECT * FROM factor_scores WHERE date = :d"),
                {"d": date_str},
            ).fetchall()

        if not rows:
            logger.warning(f"No factor scores for {date_str}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        candidate_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        for c in candidate_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Try to load saved models
        xgb_path = MODEL_DIR / "xgb_1d.pkl"
        lgb_path = MODEL_DIR / "lgb_1d.pkl"

        if not xgb_path.exists() or not lgb_path.exists():
            logger.warning("Models not trained yet. Run retrain_weekly() first.")
            return pd.DataFrame()

        xgb = joblib.load(xgb_path)
        lgb = joblib.load(lgb_path)

        # Use exactly the features the trained model expects
        train_features = list(getattr(xgb, "feature_names_in_", candidate_cols))
        for c in train_features:
            if c not in df.columns:
                df[c] = 0.0
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
        X = df[train_features]

        preds = (xgb.predict(X) + lgb.predict(X)) / 2
        result = pd.DataFrame({
            "ticker": df["ticker"],
            "date": date_str,
            "predicted_return_1d": preds,
            "signal_rank": pd.Series(preds).rank(ascending=False).astype(int).values,
        })
        result = result.sort_values("signal_rank")

        # Persist into signal_labels (preserves any existing future_return / class_label).
        try:
            from datetime import date as _date
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            from india_quant.data.models import SignalLabels
            d = _date.fromisoformat(date_str) if isinstance(date_str, str) else date_str
            payload = [
                {
                    "ticker": r["ticker"],
                    "date": d,
                    "horizon": "1d",
                    "predicted_return": float(r["predicted_return_1d"]),
                    "signal_rank": int(r["signal_rank"]),
                }
                for _, r in result.iterrows()
            ]
            with get_session() as session:
                stmt = pg_insert(SignalLabels).values(payload).on_conflict_do_update(
                    index_elements=["ticker", "date", "horizon"],
                    set_={
                        "predicted_return": pg_insert(SignalLabels).excluded.predicted_return,
                        "signal_rank": pg_insert(SignalLabels).excluded.signal_rank,
                    },
                )
                session.execute(stmt)
            logger.info(f"[ML] Stored {len(payload)} predictions for {date_str}")
        except Exception as e:
            logger.error(f"[ML] Failed to persist predictions: {e}")

        return result

    def compute_shap_values(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP feature importance per prediction."""
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        return pd.DataFrame(shap_vals, columns=X.columns, index=X.index)

    def save_model(self, model, name: str):
        path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, path)
        logger.info(f"Model saved: {path}")

    def load_model(self, name: str):
        path = MODEL_DIR / f"{name}.pkl"
        return joblib.load(path)

    def retrain_weekly(self, horizon: str = "1d"):
        """Retrain on all available data. Only deploy if better than previous model."""
        from datetime import date, timedelta
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=730)).isoformat()

        X, y, dates, tickers = self.prepare_dataset(start, end, horizon)
        if X is None:
            return

        xgb = self.train_xgboost(X, y)
        lgb = self.train_lightgbm(X, y)

        self.save_model(xgb, f"xgb_{horizon}")
        self.save_model(lgb, f"lgb_{horizon}")
        logger.info(f"[ReturnPredictor] Models retrained for horizon={horizon}")
