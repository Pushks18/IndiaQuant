"""Factor engine: 20 cross-sectional alpha characteristics (Gu-Kelly-Xiu 2020 recipe for India)."""
import warnings
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from india_quant.data.db import get_session
from india_quant.data.models import FactorScores

warnings.filterwarnings("ignore")


class FactorEngine:
    def compute_momentum_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        mom_12_1: 12-month return skipping last month (Jegadeesh-Titman 1993)
        mom_1: last month return (short-term reversal)
        mom_3: 3-month return
        All cross-sectionally ranked 0-1.
        """
        prices = prices.sort_values(["ticker", "datetime"])
        result = {}

        for ticker, grp in prices.groupby("ticker"):
            close = grp.set_index("datetime")["close"].sort_index()
            if len(close) < 252:
                continue
            mom_12_1 = close.iloc[-22] / close.iloc[-252] - 1 if len(close) >= 252 else np.nan
            mom_1 = close.iloc[-1] / close.iloc[-22] - 1 if len(close) >= 22 else np.nan
            mom_3 = close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else np.nan
            result[ticker] = {
                "momentum_12_1": mom_12_1,
                "momentum_1": mom_1,
                "momentum_3": mom_3,
            }

        df = pd.DataFrame(result).T
        return self._cross_sectional_rank(df)

    def compute_value_factors(self, prices: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
        """
        value_bm: book-to-market (note: negative expected return post-2010 in India)
        earnings_yield: EPS / price
        """
        result = {}
        for ticker, info in fundamentals.items():
            close = self._latest_close(prices, ticker)
            if close is None or close == 0:
                continue
            pe = info.get("trailingPE") or info.get("forwardPE")
            eps = info.get("trailingEps")
            bv = info.get("bookValue")
            result[ticker] = {
                "value_bm": (bv / close) if bv and close else np.nan,
                "earnings_yield": (1 / pe) if pe and pe > 0 else np.nan,
            }
        df = pd.DataFrame(result).T
        return self._cross_sectional_rank(df)

    def compute_quality_factors(self, fundamentals: dict) -> pd.DataFrame:
        """ROE, gross profitability, investment growth (Fama-French 2015 CMA)."""
        result = {}
        for ticker, info in fundamentals.items():
            roe = info.get("returnOnEquity")
            gross_profit = info.get("grossProfits")
            total_assets = info.get("totalAssets")
            result[ticker] = {
                "profitability_roe": roe if roe else np.nan,
                "gross_profitability": (gross_profit / total_assets)
                if gross_profit and total_assets and total_assets > 0 else np.nan,
                "investment_ag": np.nan,  # YoY asset growth — needs two periods
            }
        df = pd.DataFrame(result).T
        return self._cross_sectional_rank(df)

    def compute_volatility_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Idiosyncratic vol (21d), realized vol (21d), vol-of-vol."""
        result = {}
        for ticker, grp in prices.groupby("ticker"):
            close = grp.set_index("datetime")["close"].sort_index()
            if len(close) < 22:
                continue
            rets = close.pct_change().dropna()
            rv_21 = rets.iloc[-21:].std() * np.sqrt(252)
            vol_of_vol = rets.rolling(21).std().dropna().std() * np.sqrt(252)
            result[ticker] = {
                "realized_vol": rv_21,
                "vol_of_vol": float(vol_of_vol) if not np.isnan(vol_of_vol) else np.nan,
                "idiosyncratic_vol": rv_21,  # simplified; full version uses FF3 residuals
            }
        df = pd.DataFrame(result).T
        return self._cross_sectional_rank(df)

    def compute_liquidity_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Amihud illiquidity ratio and turnover."""
        result = {}
        for ticker, grp in prices.groupby("ticker"):
            grp = grp.set_index("datetime").sort_index()
            if len(grp) < 22:
                continue
            rets = grp["close"].pct_change().abs()
            vol = grp["volume"]
            # Amihud: mean(|ret| / volume) * 1e6 for scaling
            amihud = (rets / vol.replace(0, np.nan)).iloc[-21:].mean() * 1e6
            result[ticker] = {
                "liquidity_amihud": float(amihud) if not np.isnan(amihud) else np.nan,
                "turnover": float(vol.iloc[-21:].mean()),
            }
        df = pd.DataFrame(result).T
        return self._cross_sectional_rank(df)

    def compute_options_factors(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """IV spread, IV skew, VRP, OI flow from options chain."""
        if options_df.empty:
            return pd.DataFrame()

        result = {}
        for underlying, grp in options_df.groupby("underlying"):
            ce = grp[grp["option_type"] == "CE"]
            pe = grp[grp["option_type"] == "PE"]
            if ce.empty or pe.empty:
                continue

            # ATM strike
            mid_strike = ce["strike"].median()
            atm_ce = ce[ce["strike"] == ce["strike"].sub(mid_strike).abs().idxmin()]["iv"]
            atm_pe = pe[pe["strike"] == pe["strike"].sub(mid_strike).abs().idxmin()]["iv"]

            atm_ce_iv = float(atm_ce.values[0]) if len(atm_ce) else np.nan
            atm_pe_iv = float(atm_pe.values[0]) if len(atm_pe) else np.nan

            # IV spread: call - put (Cremers-Weinbaum 2010)
            iv_spread = atm_ce_iv - atm_pe_iv if not (np.isnan(atm_ce_iv) or np.isnan(atm_pe_iv)) else np.nan

            # IV skew: OTM put IV - ATM call IV (Xing-Zhang-Zhao 2010)
            otm_puts = pe[pe["strike"] < mid_strike]
            if not otm_puts.empty and not np.isnan(atm_ce_iv):
                otm_iv = float(otm_puts.nlargest(1, "strike")["iv"].values[0])
                iv_skew = otm_iv - atm_ce_iv
            else:
                iv_skew = np.nan

            # OI flow: (call OI change - put OI change) / total OI
            call_oi_chg = ce["oi_change"].sum() if "oi_change" in ce.columns else 0
            put_oi_chg = pe["oi_change"].sum() if "oi_change" in pe.columns else 0
            total_oi = (ce["open_interest"].sum() + pe["open_interest"].sum()) or 1
            oi_flow = (call_oi_chg - put_oi_chg) / total_oi

            result[underlying] = {
                "iv_spread": iv_spread,
                "iv_skew": iv_skew,
                "oi_flow": float(oi_flow),
                "vrp": np.nan,  # filled by volatility engine
            }

        return pd.DataFrame(result).T

    def compute_all(self, as_of_date: str) -> pd.DataFrame:
        """Compute factors AS OF as_of_date — uses only price data on or before that date.
        This is critical for backfill: training data must avoid look-ahead bias."""
        logger.info(f"[FactorEngine] Computing all factors for {as_of_date}")

        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT ticker, datetime, open, high, low, close, volume
                    FROM price_data
                    WHERE interval = '1d'
                    AND datetime <= :as_of
                    AND datetime >= (:as_of)::timestamp - INTERVAL '400 days'
                    ORDER BY ticker, datetime
                """),
                {"as_of": as_of_date},
            ).fetchall()

        if not rows:
            logger.warning("No price data found. Run yfinance fetcher first.")
            return pd.DataFrame()

        prices = pd.DataFrame(rows, columns=["ticker", "datetime", "open", "high", "low", "close", "volume"])

        mom = self.compute_momentum_factors(prices)
        vol_f = self.compute_volatility_factors(prices)
        liq_f = self.compute_liquidity_factors(prices)

        all_factors = pd.concat([mom, vol_f, liq_f], axis=1)
        all_factors = all_factors[~all_factors.index.duplicated(keep="last")]

        factor_date = date.fromisoformat(as_of_date)
        upserted = 0
        with get_session() as session:
            for ticker, row in all_factors.iterrows():
                vals = {k: (None if np.isnan(v) else float(v)) for k, v in row.items()}
                vals["ticker"] = ticker
                vals["date"] = factor_date
                stmt = insert(FactorScores).values(**vals).on_conflict_do_update(
                    index_elements=["ticker", "date"],
                    set_=vals,
                )
                session.execute(stmt)
                upserted += 1

        logger.info(f"[FactorEngine] Upserted {upserted} factor rows for {as_of_date}")

        # Log cross-sectional stats
        for col in all_factors.columns:
            valid = all_factors[col].dropna()
            if len(valid) > 5:
                logger.info(f"  {col}: mean={valid.mean():.4f}, std={valid.std():.4f}, n={len(valid)}")

        return all_factors

    def compute_information_coefficient(
        self, factor_col: pd.Series, forward_return_col: pd.Series
    ) -> float:
        """Rank IC: Spearman correlation between factor ranks and future returns."""
        df = pd.DataFrame({"factor": factor_col, "return": forward_return_col}).dropna()
        if len(df) < 10:
            return np.nan
        ic, _ = spearmanr(df["factor"], df["return"])
        return float(ic)

    @staticmethod
    def _latest_close(prices: pd.DataFrame, ticker: str) -> float | None:
        grp = prices[prices["ticker"] == ticker]
        if grp.empty:
            return None
        return float(grp.sort_values("datetime").iloc[-1]["close"])

    @staticmethod
    def _cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank each column to [0, 1]."""
        return df.rank(pct=True)
