"""Risk agent: Kelly sizing + portfolio limits + Indian-market-specific checks."""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RiskReview:
    status: Literal["APPROVED", "REJECTED", "MODIFIED"]
    reason: str
    adjusted_position_pct: float | None = None
    adjusted_stop: float | None = None
    adjusted_instrument: str | None = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class RiskAgent:
    HARD_LIMITS = {
        "max_position_pct": 0.05,      # 5% per position
        "max_sector_pct": 0.20,         # 20% per sector
        "max_portfolio_heat": 0.15,     # max 15% total at risk
        "max_daily_loss_pct": 0.02,     # pause all if portfolio drops 2% in a day
        "max_drawdown_pct": 0.15,       # pause if drawdown hits 15% from peak
        "min_risk_reward": 1.5,         # reject trades with R:R < 1.5
        "min_liquidity_crores": 50,     # reject if avg daily turnover < 50 Cr
    }

    def compute_kelly_size(
        self, win_prob: float, avg_win_pct: float, avg_loss_pct: float
    ) -> float:
        """
        Kelly fraction = (p*b - q) / b where b = avg_win/avg_loss
        Apply 0.25x (quarter-Kelly) for Indian market volatility.
        Cap at max_position_pct.
        """
        if avg_loss_pct == 0 or avg_win_pct == 0:
            return 0.01
        b = avg_win_pct / avg_loss_pct
        q = 1 - win_prob
        kelly = (win_prob * b - q) / b
        quarter_kelly = kelly * 0.25
        return min(max(quarter_kelly, 0.0), self.HARD_LIMITS["max_position_pct"])

    def check_fo_ban(self, ticker: str) -> bool:
        """Check NSE F&O ban list."""
        import requests
        try:
            resp = requests.get(
                "https://www.nseindia.com/api/fo-ban-list",
                timeout=5,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            data = resp.json().get("data", [])
            clean = ticker.replace(".NS", "").replace("-EQ", "").upper()
            return any(clean in str(item) for item in data)
        except Exception:
            return False

    def check_circuit_proximity(self, ticker: str, entry_price: float) -> dict:
        """Flag if entry is within 3% of circuit limits."""
        try:
            from india_quant.data.db import get_session
            from sqlalchemy import text
            with get_session() as session:
                row = session.execute(
                    text("SELECT close FROM price_data WHERE ticker=:t AND interval='1d' ORDER BY datetime DESC LIMIT 1"),
                    {"t": ticker},
                ).fetchone()
            if not row:
                return {"near_circuit": False}
            prev_close = float(row[0])
            lower = prev_close * 0.80
            upper = prev_close * 1.20
            near_lower = entry_price <= lower * 1.03
            near_upper = entry_price >= upper * 0.97
            return {
                "near_circuit": near_lower or near_upper,
                "near_lower": near_lower,
                "near_upper": near_upper,
                "lower_circuit": round(lower, 2),
                "upper_circuit": round(upper, 2),
                "distance_to_lower_pct": round((entry_price - lower) / lower * 100, 2),
                "distance_to_upper_pct": round((upper - entry_price) / upper * 100, 2),
            }
        except Exception as e:
            return {"near_circuit": False, "error": str(e)}

    def check_sebi_window(self, ticker: str) -> bool:
        """True if in SEBI insider trading blackout window (~2 weeks before results)."""
        today = date.today()
        month = today.month
        result_months = {7, 8, 10, 11, 1, 2, 4, 5}
        return month in result_months and today.day >= 15

    def check_portfolio_heat(self, existing_positions: list[dict]) -> float:
        """
        Current % of portfolio at risk.
        Each position's heat = size_pct * distance_to_stop_pct
        """
        total_heat = 0.0
        for pos in existing_positions:
            size = pos.get("position_size_pct", 0)
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            if entry and stop and entry > 0:
                downside_pct = abs(entry - stop) / entry
                total_heat += size * downside_pct
        return total_heat

    def check_expiry_risk(self, trade: dict) -> str | None:
        """
        NIFTY/BANKNIFTY weekly expiry = Thursday.
        Monthly expiry = last Thursday.
        """
        today = date.today()
        warnings = []

        # Weekly expiry warning (Wednesday/Thursday)
        if today.weekday() in [2, 3]:  # Wed=2, Thu=3
            if trade.get("ticker") in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                warnings.append(f"Weekly F&O expiry on Thursday — elevated vol and pin risk")

        # Monthly expiry: last Thursday of month
        last_thu = self._last_thursday_of_month(today)
        days_to_monthly = (last_thu - today).days
        if 0 <= days_to_monthly <= 3:
            warnings.append(f"Monthly expiry in {days_to_monthly} days — avoid large positions")

        return "; ".join(warnings) if warnings else None

    @staticmethod
    def _last_thursday_of_month(d: date) -> date:
        """Return the last Thursday of the month containing d."""
        import calendar
        last_day = calendar.monthrange(d.year, d.month)[1]
        last = date(d.year, d.month, last_day)
        # Walk back to Thursday
        while last.weekday() != 3:
            last -= timedelta(days=1)
        return last

    def check_liquidity(self, ticker: str) -> dict:
        """Check if avg daily turnover >= 50 Cr."""
        try:
            from india_quant.data.db import get_session
            from sqlalchemy import text
            with get_session() as session:
                row = session.execute(
                    text("""
                        SELECT AVG(close * volume) / 10000000 AS turnover_crores
                        FROM price_data
                        WHERE ticker = :t AND interval = '1d'
                        AND datetime >= NOW() - INTERVAL '30 days'
                    """),
                    {"t": ticker},
                ).fetchone()
            if row and row[0]:
                turnover = float(row[0])
                return {
                    "turnover_crores_30d": round(turnover, 2),
                    "liquid": turnover >= self.HARD_LIMITS["min_liquidity_crores"],
                }
        except Exception:
            pass
        return {"turnover_crores_30d": None, "liquid": True}

    def review_trade(self, trade: dict, portfolio_state: dict = None) -> RiskReview:
        """
        Run all checks. Returns APPROVED / REJECTED / MODIFIED.
        trade: dict from TraderAgent.propose_trade()
        portfolio_state: {existing_positions: [...], daily_pnl_pct: float, peak_value: float, current_value: float}
        """
        if portfolio_state is None:
            portfolio_state = {}
        warnings = []

        ticker = trade.get("ticker", "")
        instrument = trade.get("instrument", "equity")
        entry = trade.get("entry_price", 0)
        stop = trade.get("stop_loss", 0)
        target = trade.get("target_1", 0)
        size_pct = trade.get("position_size_pct", 0.02)

        # 1. Risk/reward check
        if entry and stop and target and entry != stop:
            rr = abs(target - entry) / abs(entry - stop)
            if rr < self.HARD_LIMITS["min_risk_reward"]:
                return RiskReview(
                    status="REJECTED",
                    reason=f"R:R {rr:.2f} < minimum {self.HARD_LIMITS['min_risk_reward']}",
                )

        # 2. F&O ban list
        in_ban = self.check_fo_ban(ticker)
        if in_ban and instrument != "equity":
            warnings.append(f"{ticker} is in F&O ban list — downgraded to equity only")
            return RiskReview(
                status="MODIFIED",
                reason="F&O ban list: only equity allowed",
                adjusted_instrument="equity",
                warnings=warnings,
                adjusted_position_pct=min(size_pct, self.HARD_LIMITS["max_position_pct"]),
            )

        # 3. Circuit proximity
        circuit = self.check_circuit_proximity(ticker, entry)
        if circuit.get("near_circuit"):
            if circuit.get("near_lower"):
                return RiskReview(
                    status="REJECTED",
                    reason=f"Entry within 3% of lower circuit ({circuit.get('lower_circuit')})",
                )
            warnings.append(f"Entry within 3% of upper circuit ({circuit.get('upper_circuit')})")

        # 4. SEBI window
        if self.check_sebi_window(ticker):
            warnings.append("SEBI insider blackout window active — consider waiting")

        # 5. Expiry risk
        expiry_warning = self.check_expiry_risk(trade)
        if expiry_warning:
            warnings.append(expiry_warning)

        # 6. Portfolio heat
        existing = portfolio_state.get("existing_positions", [])
        heat = self.check_portfolio_heat(existing)
        if heat + size_pct > self.HARD_LIMITS["max_portfolio_heat"]:
            new_size = max(0.01, self.HARD_LIMITS["max_portfolio_heat"] - heat)
            warnings.append(f"Portfolio heat {heat:.1%} → reducing size to {new_size:.1%}")
            return RiskReview(
                status="MODIFIED",
                reason="Portfolio heat limit reached",
                adjusted_position_pct=new_size,
                warnings=warnings,
            )

        # 7. Liquidity
        liquidity = self.check_liquidity(ticker)
        if not liquidity.get("liquid", True):
            warnings.append(f"Low liquidity: {liquidity.get('turnover_crores_30d')} Cr < 50 Cr")

        # 8. Daily loss limit
        daily_pnl = portfolio_state.get("daily_pnl_pct", 0)
        if daily_pnl < -self.HARD_LIMITS["max_daily_loss_pct"]:
            return RiskReview(
                status="REJECTED",
                reason=f"Daily loss limit hit: portfolio down {daily_pnl:.1%} today — no new trades",
            )

        # Cap position size
        capped_size = min(size_pct, self.HARD_LIMITS["max_position_pct"])

        return RiskReview(
            status="APPROVED",
            reason="All risk checks passed",
            adjusted_position_pct=capped_size,
            warnings=warnings,
        )

    def compute_var(self, portfolio_returns: pd.Series, confidence: float = 0.99, horizon: int = 1) -> float:
        """Historical VaR at given confidence level for a given horizon."""
        if portfolio_returns.empty:
            return 0.0
        scaled = portfolio_returns * np.sqrt(horizon)
        var = float(np.percentile(scaled, (1 - confidence) * 100))
        return abs(var)
