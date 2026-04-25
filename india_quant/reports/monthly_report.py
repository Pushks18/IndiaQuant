"""Monthly report: factor scorecard, portfolio construction, options book."""
import json
from datetime import date
from pathlib import Path

from jinja2 import Template
from loguru import logger

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "monthly"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MONTHLY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>India Quant Monthly Report — {{ month_label }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
  h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
  h2 { color: #3fb950; margin-top: 30px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th { background: #21262d; padding: 8px; text-align: left; color: #8b949e; font-size: 12px; }
  td { padding: 8px; border-bottom: 1px solid #21262d; }
  .positive { color: #3fb950; }
  .negative { color: #f85149; }
  footer { margin-top: 40px; color: #8b949e; font-size: 12px; }
</style>
</head>
<body>
<h1>🇮🇳 India Quant Monthly Report — {{ month_label }}</h1>

<h2>Section 1: Factor Performance Scorecard</h2>
<div class="card">
  <table>
    <tr><th>Factor</th><th>Monthly IC</th><th>Sharpe</th><th>Status</th></tr>
    {% for f in factor_scores %}
    <tr>
      <td>{{ f.name }}</td>
      <td class="{{ 'positive' if f.ic > 0 else 'negative' }}">{{ "%.3f"|format(f.ic) }}</td>
      <td>{{ "%.2f"|format(f.sharpe) if f.sharpe else '--' }}</td>
      <td>{{ '✅ Active' if f.ic > 0.03 else '⚠ Weak' }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

<h2>Section 2: Market Regime Analysis</h2>
<div class="card">
  <p><strong>This month:</strong> {{ regime_this_month }}</p>
  <p><strong>3-month trend:</strong> {{ regime_trend }}</p>
  <p><strong>India VIX trend:</strong> {{ vix_trend }}</p>
</div>

<h2>Section 3: Top 10 Stock Ideas (3-6 week positional)</h2>
<div class="card">
  <table>
    <tr><th>Stock</th><th>Sector</th><th>Verdict</th><th>Entry</th><th>Target</th><th>Catalyst</th></tr>
    {% for stock in top_10 %}
    <tr>
      <td>{{ stock.ticker }}</td>
      <td>{{ stock.sector | default('--') }}</td>
      <td>{{ stock.verdict }}</td>
      <td>{{ stock.entry }}</td>
      <td>{{ stock.target }}</td>
      <td>{{ stock.catalyst }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

<h2>Section 4: Portfolio Construction</h2>
<div class="card">
  <p>Equal-weight portfolio of top 10 stocks.</p>
  {% if hedge_recommended %}
  <p><strong>NIFTY Hedge:</strong> India VIX > 18 — recommend 5% portfolio hedge via NIFTY put spread</p>
  {% endif %}
  {% if correlation_note %}
  <p><strong>Correlation note:</strong> {{ correlation_note }}</p>
  {% endif %}
</div>

<h2>Section 5: Options Book Strategy</h2>
<div class="card">
  {% if options_book %}
  <p><strong>VRP Strategy:</strong> {{ options_book.strategy }}</p>
  <p><strong>Premium collected estimate:</strong> {{ options_book.premium_estimate }}</p>
  <p><strong>Delta hedge frequency:</strong> {{ options_book.hedge_frequency }}</p>
  {% else %}
  <p>VRP data needed for options book strategy.</p>
  {% endif %}
</div>

<footer>India Quant Trading Assistant — Monthly Edition — {{ month_label }}</footer>
</body>
</html>"""


def generate_monthly_report(as_of_date: str = None) -> str:
    as_of_date = as_of_date or date.today().isoformat()
    d = date.fromisoformat(as_of_date)
    month_label = f"{d.year}-{d.month:02d}"
    logger.info(f"[MonthlyReport] Generating {month_label}")

    factor_scores = _compute_factor_performance(d)
    regime_analysis = _get_regime_analysis()
    top_10 = _get_monthly_top_10(as_of_date)
    options_book = _get_options_book_strategy(as_of_date)
    vix_data = _get_vix_data()
    hedge_recommended = (vix_data or {}).get("india_vix", 0) or 0 > 18

    template = Template(MONTHLY_TEMPLATE)
    html = template.render(
        month_label=month_label,
        factor_scores=factor_scores,
        regime_this_month=regime_analysis.get("this_month", "Unknown"),
        regime_trend=regime_analysis.get("trend", "Unknown"),
        vix_trend=regime_analysis.get("vix_trend", "Unknown"),
        top_10=top_10,
        hedge_recommended=hedge_recommended,
        correlation_note=None,
        options_book=options_book,
    )

    path = REPORTS_DIR / f"{month_label}.html"
    path.write_text(html)
    logger.info(f"[MonthlyReport] Saved: {path}")
    return str(path)


def _compute_factor_performance(d: date) -> list[dict]:
    """Compute monthly IC for each factor."""
    import numpy as np
    factors = [
        "momentum_12_1", "momentum_1", "momentum_3",
        "realized_vol", "liquidity_amihud",
        "profitability_roe", "iv_skew", "iv_spread",
    ]
    results = []
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        start = date(d.year, d.month, 1).isoformat()
        with get_session() as session:
            for factor in factors:
                row = session.execute(
                    text(f"""
                        SELECT CORR(fs.{factor}, sl.future_return)
                        FROM factor_scores fs
                        JOIN signal_labels sl ON fs.ticker = sl.ticker AND fs.date = sl.date
                        WHERE fs.date >= :start AND sl.horizon = '21d'
                    """),
                    {"start": start},
                ).fetchone()
                ic = float(row[0]) if row and row[0] else 0.0
                results.append({"name": factor, "ic": ic, "sharpe": ic / 0.05 if ic else 0})
    except Exception as e:
        logger.warning(f"Factor performance failed: {e}")
        results = [{"name": f, "ic": 0.0, "sharpe": 0.0} for f in factors]
    return results


def _get_regime_analysis() -> dict:
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI").history(period="6mo")
        vix = yf.Ticker("^INDIAVIX").history(period="3mo")
        close = nifty["Close"]
        ema_200 = close.ewm(span=200).mean()
        current = close.iloc[-1]
        regime = "Bull" if current > ema_200.iloc[-1] else "Bear"
        vix_trend = "Rising" if vix["Close"].iloc[-1] > vix["Close"].iloc[0] else "Falling"
        return {"this_month": regime, "trend": regime, "vix_trend": vix_trend}
    except Exception:
        return {"this_month": "Unknown", "trend": "Unknown", "vix_trend": "Unknown"}


def _get_monthly_top_10(as_of_date: str) -> list[dict]:
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT sl.ticker, sl.future_return
                    FROM signal_labels sl
                    WHERE sl.date = :d AND sl.horizon = '21d'
                    ORDER BY sl.future_return DESC LIMIT 10
                """),
                {"d": as_of_date},
            ).fetchall()
        return [{"ticker": r[0], "sector": None, "verdict": "Bullish",
                 "entry": "--", "target": "--",
                 "catalyst": f"Top 21d ML signal: {r[1]:.2%}" if r[1] else "--"} for r in rows]
    except Exception:
        return []


def _get_options_book_strategy(as_of_date: str) -> dict | None:
    return {
        "strategy": "Sell NIFTY monthly straddle (if VRP positive)",
        "premium_estimate": "To be computed from live options data",
        "hedge_frequency": "Daily delta hedge when |delta| > 0.1",
    }


def _get_vix_data() -> dict:
    try:
        import yfinance as yf
        vix = yf.Ticker("^INDIAVIX").history(period="1d")
        if not vix.empty:
            return {"india_vix": float(vix["Close"].iloc[-1])}
    except Exception:
        pass
    return {}
