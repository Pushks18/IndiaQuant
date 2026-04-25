"""Fundamental analyst — deterministic scoring from yfinance fundamentals."""
from datetime import date, timedelta

from india_quant.agents.base import BaseAnalystAgent

SECTOR_PE = {
    "Technology": 28.0,
    "Financial Services": 18.0,
    "Consumer Cyclical": 35.0,
    "Consumer Defensive": 40.0,
    "Healthcare": 32.0,
    "Energy": 12.0,
    "Basic Materials": 15.0,
    "Industrials": 22.0,
    "Communication Services": 25.0,
    "Utilities": 18.0,
    "Real Estate": 30.0,
}


def _get_fundamentals(ticker: str) -> dict:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return {
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "eps": info.get("trailingEps"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "book_value": info.get("bookValue"),
            "dividend_yield": info.get("dividendYield"),
        }
    except Exception as e:
        return {"error": str(e)}


def _get_sector_median_pe(sector: str) -> float:
    return SECTOR_PE.get(sector, 25.0)


def _get_sebi_insider_window() -> bool:
    """Approx: ~2 weeks before result months."""
    today = date.today()
    return today.month in {1, 2, 4, 5, 7, 8, 10, 11} and today.day >= 15


class FundamentalAnalystAgent(BaseAnalystAgent):
    def __init__(self):
        super().__init__("fundamental_analyst")

    def _tool_registry(self) -> dict:
        return {
            "get_fundamentals": _get_fundamentals,
            "get_sector_median_pe": _get_sector_median_pe,
        }

    def analyze(self, context: dict, date_str: str) -> dict:
        ticker = context.get("ticker", "")
        f = _get_fundamentals(ticker)
        if "error" in f:
            return {
                "ticker": ticker, "date": date_str,
                "verdict": "neutral",
                "red_flags": ["fundamentals unavailable"],
                "error": f.get("error"),
            }

        pe = f.get("pe")
        sector = f.get("sector") or ""
        sector_pe = _get_sector_median_pe(sector)
        roe = f.get("roe")
        de = f.get("debt_to_equity")
        eps_growth = f.get("earnings_growth")
        rev_growth = f.get("revenue_growth")

        score = 0
        red_flags = []

        if pe is not None and pe > 0:
            if pe < sector_pe * 0.85:
                score += 2
            elif pe > sector_pe * 1.5:
                score -= 2
                red_flags.append(f"P/E {pe:.1f} > 1.5x sector median ({sector_pe})")

        if roe is not None:
            roe_pct = roe * 100 if abs(roe) < 1 else roe
            if roe_pct >= 18:
                score += 2
            elif roe_pct >= 12:
                score += 1
            elif roe_pct < 5:
                score -= 2
                red_flags.append(f"Low ROE: {roe_pct:.1f}%")

        if de is not None:
            de_val = de / 100 if de > 5 else de
            if de_val > 2.0 and (sector or "").lower() != "financial services":
                score -= 2
                red_flags.append(f"High D/E: {de_val:.2f}")
            elif de_val > 3.0:
                score -= 1

        if eps_growth is not None:
            if eps_growth > 0.15:
                score += 2
            elif eps_growth < -0.10:
                score -= 1
                red_flags.append(f"EPS growth negative: {eps_growth*100:.1f}%")

        if rev_growth is not None and rev_growth < -0.05:
            red_flags.append(f"Revenue declining: {rev_growth*100:.1f}%")

        if _get_sebi_insider_window():
            red_flags.append("SEBI insider blackout window may be active")

        if score >= 3:
            verdict = "bullish"
        elif score <= -2:
            verdict = "bearish"
        else:
            verdict = "neutral"

        return {
            "ticker": ticker,
            "date": date_str,
            "pe": round(pe, 2) if isinstance(pe, (int, float)) else None,
            "pb": round(f["pb"], 2) if isinstance(f.get("pb"), (int, float)) else None,
            "roe": round(roe * 100, 2) if isinstance(roe, (int, float)) and abs(roe) < 1 else (round(roe, 2) if isinstance(roe, (int, float)) else None),
            "eps_growth": round(eps_growth * 100, 2) if isinstance(eps_growth, (int, float)) else None,
            "promoter_pct": None,
            "fii_pct": None,
            "verdict": verdict,
            "red_flags": red_flags,
            "sector": sector,
            "sector_median_pe": sector_pe,
            "score": score,
        }
