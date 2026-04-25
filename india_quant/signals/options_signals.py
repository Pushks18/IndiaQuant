"""Options signal engine: IV spread, IV skew, VRP, OI flow, PCR, Max Pain."""
import math

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm


class OptionsSignalEngine:
    def compute_iv_spread(self, options_df: pd.DataFrame) -> pd.Series:
        """
        IV_spread = ATM_call_IV - ATM_put_IV (Cremers-Weinbaum 2010)
        Positive spread → call more expensive → bullish signal.
        Uses nearest expiry with at least 5 DTE.
        Returns cross-sectionally ranked series.
        """
        results = {}
        for underlying, grp in options_df.groupby("underlying"):
            today = pd.Timestamp.now().normalize()
            grp = grp.copy()
            grp["dte"] = (pd.to_datetime(grp["expiry"]) - today).dt.days
            near_expiry = grp[grp["dte"] >= 5].sort_values("dte")
            if near_expiry.empty:
                continue
            exp = near_expiry["expiry"].iloc[0]
            chain = grp[grp["expiry"] == exp]

            spot_proxy = chain["strike"].median()
            ce = chain[chain["option_type"] == "CE"]
            pe = chain[chain["option_type"] == "PE"]

            atm_ce = ce.iloc[(ce["strike"] - spot_proxy).abs().argsort()[:1]]
            atm_pe = pe.iloc[(pe["strike"] - spot_proxy).abs().argsort()[:1]]

            if atm_ce.empty or atm_pe.empty:
                continue
            ce_iv = atm_ce["iv"].values[0]
            pe_iv = atm_pe["iv"].values[0]
            if not (np.isnan(ce_iv) or np.isnan(pe_iv)):
                results[underlying] = ce_iv - pe_iv

        s = pd.Series(results)
        return s.rank(pct=True)

    def compute_iv_skew(self, options_df: pd.DataFrame) -> pd.Series:
        """
        IV_skew = IV of 25-delta put - IV of 25-delta call (Xing-Zhang-Zhao 2010)
        High skew = expensive downside protection = bearish signal.
        """
        results = {}
        for underlying, grp in options_df.groupby("underlying"):
            today = pd.Timestamp.now().normalize()
            grp = grp.copy()
            grp["dte"] = (pd.to_datetime(grp["expiry"]) - today).dt.days
            near_expiry = grp[grp["dte"] >= 5].sort_values("dte")
            if near_expiry.empty:
                continue
            exp = near_expiry["expiry"].iloc[0]
            T = near_expiry["dte"].iloc[0] / 252.0
            chain = grp[grp["expiry"] == exp]

            spot = chain["strike"].median()
            r = 0.07

            ce = chain[chain["option_type"] == "CE"].dropna(subset=["iv"])
            pe = chain[chain["option_type"] == "PE"].dropna(subset=["iv"])

            # Find 25-delta put and call
            target_delta_put = -0.25
            target_delta_call = 0.25
            best_put_iv, best_call_iv = None, None
            best_put_err, best_call_err = 999, 999

            for _, row in pe.iterrows():
                if row["iv"] <= 0:
                    continue
                delta = self._bs_delta(spot, row["strike"], T, r, row["iv"] / 100, "PE")
                err = abs(delta - target_delta_put)
                if err < best_put_err:
                    best_put_err, best_put_iv = err, row["iv"]

            for _, row in ce.iterrows():
                if row["iv"] <= 0:
                    continue
                delta = self._bs_delta(spot, row["strike"], T, r, row["iv"] / 100, "CE")
                err = abs(delta - target_delta_call)
                if err < best_call_err:
                    best_call_err, best_call_iv = err, row["iv"]

            if best_put_iv is not None and best_call_iv is not None:
                results[underlying] = best_put_iv - best_call_iv

        s = pd.Series(results)
        return s.rank(pct=True)

    def compute_vrp(
        self, realized_vol: pd.Series, india_vix: pd.Series
    ) -> pd.Series:
        """
        VRP = India VIX (implied vol) - realized vol over past 30 days (Carr-Wu 2009)
        Positive VRP = options expensive vs realized → sell premium opportunity.
        """
        aligned = pd.concat([india_vix.rename("vix"), realized_vol.rename("rv")], axis=1).dropna()
        vrp = aligned["vix"] - aligned["rv"] * 100
        return vrp

    def compute_oi_flow(self, options_df: pd.DataFrame) -> pd.Series:
        """
        OI_flow = (call OI chg - put OI chg) / total OI
        Weighted by proximity to ATM.
        Positive → institutional bullish positioning.
        """
        results = {}
        for underlying, grp in options_df.groupby("underlying"):
            spot = grp["strike"].median()
            grp = grp.copy()
            grp["weight"] = 1 / (abs(grp["strike"] - spot) + 1)

            ce = grp[grp["option_type"] == "CE"]
            pe = grp[grp["option_type"] == "PE"]

            call_chg = (ce["oi_change"].fillna(0) * ce["weight"]).sum()
            put_chg = (pe["oi_change"].fillna(0) * pe["weight"]).sum()
            total_oi = (grp["open_interest"].fillna(0)).sum() or 1

            results[underlying] = (call_chg - put_chg) / total_oi

        return pd.Series(results)

    def compute_pcr(self, options_df: pd.DataFrame) -> dict:
        """
        Put-Call Ratio: total put OI / total call OI.
        Contrarian: PCR > 1.3 → oversold → bullish (NSE-specific threshold).
        """
        pcr = {}
        for underlying, grp in options_df.groupby("underlying"):
            ce_oi = grp[grp["option_type"] == "CE"]["open_interest"].sum()
            pe_oi = grp[grp["option_type"] == "PE"]["open_interest"].sum()
            if ce_oi and ce_oi > 0:
                ratio = pe_oi / ce_oi
                pcr[underlying] = {
                    "pcr": round(ratio, 3),
                    "signal": "bullish" if ratio > 1.3 else "bearish" if ratio < 0.7 else "neutral",
                }
        return pcr

    def compute_max_pain(self, options_df: pd.DataFrame) -> dict:
        """
        Max pain: strike where option sellers lose the least on expiry.
        Price tends to gravitate toward max pain on expiry day.
        """
        max_pain = {}
        for underlying, grp in options_df.groupby("underlying"):
            today = pd.Timestamp.now().normalize()
            grp = grp.copy()
            grp["dte"] = (pd.to_datetime(grp["expiry"]) - today).dt.days
            near = grp[grp["dte"] >= 0].sort_values("dte")
            if near.empty:
                continue
            exp = near["expiry"].iloc[0]
            chain = grp[grp["expiry"] == exp]

            strikes = sorted(chain["strike"].unique())
            min_pain = float("inf")
            pain_strike = strikes[0]

            for test_strike in strikes:
                ce = chain[chain["option_type"] == "CE"]
                pe = chain[chain["option_type"] == "PE"]
                # Loss to call sellers if price = test_strike
                call_pain = ce.apply(
                    lambda r: max(0, test_strike - r["strike"]) * (r["open_interest"] or 0), axis=1
                ).sum()
                put_pain = pe.apply(
                    lambda r: max(0, r["strike"] - test_strike) * (r["open_interest"] or 0), axis=1
                ).sum()
                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    pain_strike = test_strike

            max_pain[underlying] = {"max_pain_strike": pain_strike, "total_pain": min_pain}

        return max_pain

    def run_all(self, as_of_date: str) -> dict:
        """Compute all options signals for NIFTY and BANKNIFTY."""
        from india_quant.data.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT underlying, trade_date, expiry, strike, option_type,
                           last_price, bid, ask, iv, open_interest, oi_change, volume
                    FROM option_chain
                    WHERE trade_date = :d
                """),
                {"d": as_of_date},
            ).fetchall()

        if not rows:
            logger.warning(f"No options data for {as_of_date}")
            return {}

        df = pd.DataFrame(rows, columns=[
            "underlying", "trade_date", "expiry", "strike", "option_type",
            "last_price", "bid", "ask", "iv", "open_interest", "oi_change", "volume"
        ])

        iv_spread = self.compute_iv_spread(df)
        iv_skew = self.compute_iv_skew(df)
        oi_flow = self.compute_oi_flow(df)
        pcr = self.compute_pcr(df)
        max_pain = self.compute_max_pain(df)

        logger.info(f"[OptionsSignals] IV spread: {iv_spread.to_dict()}")
        logger.info(f"[OptionsSignals] PCR: {pcr}")
        logger.info(f"[OptionsSignals] Max Pain: {max_pain}")

        return {
            "iv_spread": iv_spread.to_dict(),
            "iv_skew": iv_skew.to_dict(),
            "oi_flow": oi_flow.to_dict(),
            "pcr": pcr,
            "max_pain": max_pain,
        }

    @staticmethod
    def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Black-Scholes delta."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if option_type == "CE":
            return norm.cdf(d1)
        return norm.cdf(d1) - 1.0
