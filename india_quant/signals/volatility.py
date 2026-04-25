"""Volatility engine: HAR-RV (Corsi 2009) + Heston calibration (Carr-Madan FFT)."""
import warnings
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from india_quant.data.db import get_session
from india_quant.data.models import VolatilityData

warnings.filterwarnings("ignore")


class VolatilityEngine:
    def compute_realized_vol(
        self, prices_1h: pd.DataFrame, window_days: int = 1
    ) -> pd.Series:
        """
        Realized variance from intraday log-returns.
        Uses simple realized variance (sum of squared intraday log-returns).
        TSRV correction would require tick data — using 1h as approximation.
        Returns annualized RV as a pd.Series indexed by date.
        """
        prices_1h = prices_1h.sort_index()
        log_rets = np.log(prices_1h / prices_1h.shift(1)).dropna()

        rv_daily = []
        for day, grp in log_rets.groupby(log_rets.index.date):
            rv = np.sqrt(np.sum(grp ** 2) * 252)
            rv_daily.append((day, rv))

        if not rv_daily:
            return pd.Series(dtype=float)

        dates, vals = zip(*rv_daily)
        return pd.Series(vals, index=pd.to_datetime(dates), name="realized_vol")

    def fit_har_rv(self, rv_series: pd.Series) -> dict:
        """
        HAR-RV (Corsi 2009): OLS regression
        RV_t = c + b_d*RV_{t-1} + b_w*RV_{t-1:t-5} + b_m*RV_{t-1:t-22} + epsilon
        Returns coefficients and 1-day-ahead forecast.
        """
        rv = rv_series.dropna()
        if len(rv) < 30:
            return {"error": "insufficient data"}

        rv_d = rv.shift(1)
        rv_w = rv.shift(1).rolling(5).mean()
        rv_m = rv.shift(1).rolling(22).mean()

        X = pd.concat([pd.Series(np.ones(len(rv)), index=rv.index), rv_d, rv_w, rv_m], axis=1)
        X.columns = ["const", "rv_d", "rv_w", "rv_m"]
        y = rv

        df = pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(df) < 10:
            return {"error": "insufficient aligned data"}

        X_mat = df[["const", "rv_d", "rv_w", "rv_m"]].values
        y_vec = df["y"].values

        # OLS: beta = (X'X)^{-1} X'y
        try:
            beta = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
        except Exception as e:
            return {"error": str(e)}

        c, b_d, b_w, b_m = beta
        last_rv = rv.iloc[-1]
        last_rv_w = rv.iloc[-5:].mean()
        last_rv_m = rv.iloc[-22:].mean()
        forecast = c + b_d * last_rv + b_w * last_rv_w + b_m * last_rv_m

        return {
            "c": c, "b_d": b_d, "b_w": b_w, "b_m": b_m,
            "forecast_1d": max(0.0, forecast),
            "last_rv": last_rv,
        }

    def forecast_vol(self, har_model: dict, rv_series: pd.Series, horizon: int = 1) -> float:
        """Multi-step HAR-RV forecast using fitted coefficients."""
        if "error" in har_model:
            return np.nan
        c, b_d, b_w, b_m = har_model["c"], har_model["b_d"], har_model["b_w"], har_model["b_m"]
        rv = rv_series.dropna()
        forecast = c + b_d * rv.iloc[-1] + b_w * rv.iloc[-5:].mean() + b_m * rv.iloc[-22:].mean()
        return max(0.0, forecast)

    def calibrate_heston(
        self, options_df: pd.DataFrame, spot: float, r: float = 0.07
    ) -> dict:
        """
        Calibrate Heston model to NIFTY options IV surface.
        Minimizes (model_IV - market_IV)^2 using scipy differential_evolution.
        Uses Carr-Madan FFT characteristic function.
        """
        if options_df.empty:
            return {}

        # Filter valid strikes with IV data
        opts = options_df.dropna(subset=["iv", "strike"]).copy()
        if len(opts) < 5:
            return {"error": "insufficient options for calibration"}

        opts = opts[opts["iv"] > 0]
        market_ivs = opts["iv"].values / 100.0  # convert from % to decimal
        strikes = opts["strike"].values
        expiries = opts.get("expiry", pd.Series([30] * len(opts))).values

        # DTE in years
        if hasattr(expiries[0], "days"):
            T_arr = np.array([max(e.days, 1) / 252.0 for e in expiries])
        else:
            T_arr = np.array([30 / 252.0] * len(opts))

        def objective(params):
            kappa, theta, sigma, rho, v0 = params
            errors = []
            for iv_mkt, K, T in zip(market_ivs, strikes, T_arr):
                try:
                    iv_model = self._heston_iv(kappa, theta, sigma, rho, v0, spot, K, T, r)
                    errors.append((iv_model - iv_mkt) ** 2)
                except Exception:
                    errors.append(1.0)
            return np.mean(errors)

        # Parameter bounds: kappa, theta, sigma (vol-of-vol), rho, v0
        bounds = [
            (0.1, 10.0),   # kappa: mean-reversion speed
            (0.01, 1.0),   # theta: long-run variance
            (0.01, 2.0),   # sigma: vol of vol
            (-0.99, 0.0),  # rho: typically negative for equities
            (0.01, 1.0),   # v0: initial variance
        ]
        try:
            res = differential_evolution(objective, bounds, maxiter=200,
                                         tol=1e-6, seed=42, workers=1)
            kappa, theta, sigma, rho, v0 = res.x
            return {
                "kappa": kappa, "theta": theta, "sigma": sigma,
                "rho": rho, "v0": v0,
                "calibration_error": float(res.fun),
                "success": res.success,
            }
        except Exception as e:
            logger.error(f"Heston calibration failed: {e}")
            return {"error": str(e)}

    def _heston_iv(
        self, kappa: float, theta: float, sigma: float, rho: float, v0: float,
        S: float, K: float, T: float, r: float
    ) -> float:
        """
        Heston model IV via Carr-Madan FFT.
        Simplified: uses the closed-form characteristic function and numeric integration.
        """
        # Characteristic function integrand (simplified Heston)
        def heston_cf(u):
            xi = kappa - sigma * rho * 1j * u
            d = np.sqrt(xi ** 2 + (u ** 2 + 1j * u) * sigma ** 2)
            g = (xi - d) / (xi + d)
            C = kappa * (
                (xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
            ) / sigma ** 2
            D = (xi - d) * (1 - np.exp(-d * T)) / (sigma ** 2 * (1 - g * np.exp(-d * T)))
            return np.exp(C * theta + D * v0 + 1j * u * np.log(S * np.exp(r * T)))

        # Numerical integration for call price
        N = 128
        eta = 0.25
        b = N * eta / 2
        v = np.arange(1, N + 1) * eta
        k_log = -b + 2 * b / N * np.arange(N)
        K_grid = np.exp(k_log)
        log_K = np.log(K)

        simpson = (3 + (-1) ** np.arange(1, N + 1)) / 3
        simpson[0] = 1.0 / 3.0

        integrand = (
            np.exp(-1j * v * (b - log_K)) *
            heston_cf(v - 0.5j) /
            (v ** 2 + 0.25) *
            simpson * eta
        )
        call_price = np.exp(-log_K) / np.pi * np.real(np.sum(integrand)) * np.exp(-r * T)
        call_price = max(call_price, 1e-6)

        # Convert call price to implied vol (Newton-Raphson)
        return self._bs_iv(S, K, T, r, call_price)

    @staticmethod
    def _bs_iv(S: float, K: float, T: float, r: float, price: float) -> float:
        """Black-Scholes implied vol via bisection."""
        lo, hi = 0.001, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2
            d1 = (np.log(S / K) + (r + 0.5 * mid ** 2) * T) / (mid * np.sqrt(T))
            d2 = d1 - mid * np.sqrt(T)
            bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            if bs_price > price:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    def compute_garch(self, returns: pd.Series) -> dict:
        """Fit GARCH(1,1) using arch library."""
        try:
            from arch import arch_model
            model = arch_model(returns * 100, vol="Garch", p=1, q=1)
            res = model.fit(disp="off")
            forecast = res.forecast(horizon=1)
            var_forecast = forecast.variance.iloc[-1, 0] / 10000
            vol_forecast = np.sqrt(var_forecast * 252)
            return {
                "omega": float(res.params["omega"]),
                "alpha": float(res.params["alpha[1]"]),
                "beta": float(res.params["beta[1]"]),
                "forecast_1d_vol": float(vol_forecast),
            }
        except Exception as e:
            logger.error(f"GARCH failed: {e}")
            return {"error": str(e)}

    def heston_iv_surface(
        self, params: dict, strikes: np.ndarray, maturities: np.ndarray,
        spot: float, r: float
    ) -> np.ndarray:
        """Compute model IV surface for given strikes and maturities."""
        surface = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                try:
                    surface[i, j] = self._heston_iv(
                        params["kappa"], params["theta"], params["sigma"],
                        params["rho"], params["v0"], spot, K, T, r
                    )
                except Exception:
                    surface[i, j] = np.nan
        return surface

    def daily_vol_update(self, ticker: str, as_of_date: str):
        """Full pipeline: fetch 1h prices → compute RV → HAR forecast → store."""
        from india_quant.data.db import get_session

        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT datetime, close FROM price_data
                    WHERE ticker = :t AND interval = '1h'
                    AND datetime >= NOW() - INTERVAL '60 days'
                    ORDER BY datetime
                """),
                {"t": ticker},
            ).fetchall()

        if not rows:
            logger.warning(f"No 1h data for {ticker}")
            return

        prices = pd.Series(
            [r[1] for r in rows],
            index=pd.to_datetime([r[0] for r in rows]),
        )
        rv = self.compute_realized_vol(prices)
        har = self.fit_har_rv(rv)
        garch = self.compute_garch(prices.pct_change().dropna())

        factor_date = date.fromisoformat(as_of_date)
        with get_session() as session:
            stmt = insert(VolatilityData).values(
                ticker=ticker,
                date=factor_date,
                realized_vol_1d=float(rv.iloc[-1]) if len(rv) else None,
                har_forecast_1d=har.get("forecast_1d"),
                garch_forecast_1d=garch.get("forecast_1d_vol"),
            ).on_conflict_do_update(
                index_elements=["ticker", "date"],
                set_={"har_forecast_1d": har.get("forecast_1d"),
                      "garch_forecast_1d": garch.get("forecast_1d_vol")},
            )
            session.execute(stmt)
        logger.info(f"Vol updated for {ticker} on {as_of_date}: HAR={har.get('forecast_1d', 'N/A'):.4f}")
