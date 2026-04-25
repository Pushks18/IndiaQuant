"""Weekly report: swing trades, options strategies, sector rotation."""
import json
from datetime import date, timedelta
from pathlib import Path

from jinja2 import Template
from loguru import logger

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "weekly"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WEEKLY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>India Quant Weekly Report — {{ week_label }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
  h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
  h2 { color: #3fb950; margin-top: 30px; }
  h3 { color: #d2a8ff; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .long { border-left: 4px solid #3fb950; }
  .short { border-left: 4px solid #f85149; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th { background: #21262d; padding: 8px; text-align: left; color: #8b949e; font-size: 12px; }
  td { padding: 8px; border-bottom: 1px solid #21262d; }
  .verdict-Bullish { color: #3fb950; font-weight: bold; }
  .verdict-Bearish { color: #f85149; font-weight: bold; }
  .verdict-Neutral, .verdict-Mixed { color: #e3b341; font-weight: bold; }
  footer { margin-top: 40px; color: #8b949e; font-size: 12px; }
</style>
</head>
<body>
<h1>🇮🇳 India Quant Weekly Report — {{ week_label }}</h1>

<h2>Section 1: Week-Ahead Context</h2>
<div class="card">
  <p><strong>Key Events:</strong> {{ key_events | join(' | ') }}</p>
  <p><strong>NIFTY Key Levels:</strong> Support {{ nifty_support }}, Resistance {{ nifty_resistance }}</p>
  <p><strong>F&O Expiry:</strong> {{ expiry_date }}</p>
</div>

<h2>Section 2: Top 5 Swing Trade Ideas (3-7 day holding)</h2>
{% for trade in swing_trades %}
<div class="card {{ trade.direction }}">
  <h3>{{ trade.ticker }} — <span class="verdict-{{ trade.verdict }}">{{ trade.verdict }}</span></h3>
  <p><strong>Bull case:</strong> {{ trade.bull_summary }}</p>
  <p><strong>Bear case:</strong> {{ trade.bear_summary }}</p>
  <table>
    <tr><th>Entry Zone</th><th>Stop</th><th>Target 1</th><th>Time Stop</th><th>R:R</th></tr>
    <tr>
      <td>{{ trade.entry_zone }}</td>
      <td>{{ trade.stop_loss }}</td>
      <td>{{ trade.target_1 }}</td>
      <td>5 trading days</td>
      <td>{{ trade.risk_reward }}</td>
    </tr>
  </table>
  <p><strong>Catalyst:</strong> {{ trade.catalyst }}</p>
</div>
{% endfor %}

<h2>Section 3: NIFTY/BANKNIFTY Options Strategy</h2>
<div class="card">
  {% if options_strategy %}
  <p><strong>Strategy:</strong> {{ options_strategy.name }}</p>
  <p><strong>Expiry:</strong> {{ options_strategy.expiry }} | <strong>Strikes:</strong> {{ options_strategy.strikes }}</p>
  <p><strong>Max Profit:</strong> {{ options_strategy.max_profit }} | <strong>Max Loss:</strong> {{ options_strategy.max_loss }}</p>
  <p><strong>Breakeven:</strong> {{ options_strategy.breakevens }}</p>
  <p><strong>Rationale:</strong> {{ options_strategy.rationale }}</p>
  {% else %}
  <p>No options strategy this week (VRP insufficient or high IV).</p>
  {% endif %}
</div>

<h2>Section 4: Sector Rotation Dashboard</h2>
<div class="card">
  <table>
    <tr><th>Sector</th><th>FII Flow</th><th>Top Stock</th><th>Setup</th></tr>
    {% for s in sectors %}
    <tr>
      <td>{{ s.sector }}</td>
      <td>{{ s.fii_flow | default('--') }}</td>
      <td>{{ s.top_stock }}</td>
      <td>{{ s.setup }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

<h2>Section 5: Previous Week Review</h2>
<div class="card">
  {% if last_week %}
  <p><strong>Hit Rate:</strong> {{ last_week.hit_rate }}% of trades hit Target 1</p>
  <p><strong>Lessons:</strong> {{ last_week.lesson }}</p>
  {% else %}
  <p>No previous week data yet.</p>
  {% endif %}
</div>

<footer>India Quant Trading Assistant — Weekly Edition — {{ week_label }}</footer>
</body>
</html>"""


def generate_weekly_report(as_of_date: str = None) -> str:
    as_of_date = as_of_date or date.today().isoformat()
    d = date.fromisoformat(as_of_date)
    # ISO week label
    year, week, _ = d.isocalendar()
    week_label = f"{year}-W{week:02d}"
    logger.info(f"[WeeklyReport] Generating {week_label}")

    # Fetch swing trade candidates from debate results
    swing_trades = _fetch_swing_trades(as_of_date)

    # Key events for the coming week
    key_events = _get_week_events(d)

    # Next F&O expiry (Thursday)
    days_to_thu = (3 - d.weekday()) % 7 or 7
    expiry_date = (d + timedelta(days=days_to_thu)).isoformat()

    # Options strategy
    options_strategy = _get_weekly_options_strategy(as_of_date)

    # Sectors
    sectors = _get_sector_summary()

    # Last week performance
    last_week = _get_last_week_performance(d)

    # NIFTY levels
    nifty_levels = _get_nifty_levels()

    template = Template(WEEKLY_TEMPLATE)
    html = template.render(
        week_label=week_label,
        key_events=key_events,
        expiry_date=expiry_date,
        swing_trades=swing_trades,
        options_strategy=options_strategy,
        sectors=sectors,
        last_week=last_week,
        nifty_support=nifty_levels.get("support", "--"),
        nifty_resistance=nifty_levels.get("resistance", "--"),
    )

    path = REPORTS_DIR / f"{week_label}.html"
    path.write_text(html)
    logger.info(f"[WeeklyReport] Saved: {path}")
    return str(path)


def _fetch_swing_trades(as_of_date: str) -> list[dict]:
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT dr.ticker, dr.judge_verdict, dr.bull_report, dr.bear_report,
                           tp.direction, tp.entry_price, tp.stop_loss, tp.target_1, tp.risk_reward
                    FROM debate_result dr
                    LEFT JOIN trade_proposal tp ON dr.ticker = tp.ticker AND dr.date = tp.date
                    WHERE dr.date = :d
                    ORDER BY tp.risk_reward DESC LIMIT 5
                """),
                {"d": as_of_date},
            ).fetchall()
        trades = []
        for r in rows:
            import json as _j
            verdict_data = _j.loads(r[1]) if r[1] else {}
            bull = _j.loads(r[2]) if r[2] else {}
            bear = _j.loads(r[3]) if r[3] else {}
            trades.append({
                "ticker": r[0],
                "verdict": verdict_data.get("verdict", "Neutral"),
                "bull_summary": bull.get("bull_case", "")[:150],
                "bear_summary": bear.get("bear_case", "")[:150],
                "direction": r[4] or "long",
                "entry_zone": f"{r[5]*0.99:.1f}–{r[5]*1.01:.1f}" if r[5] else "--",
                "stop_loss": r[6],
                "target_1": r[7],
                "risk_reward": round(r[8], 2) if r[8] else None,
                "catalyst": verdict_data.get("watchout", ""),
            })
        return trades
    except Exception as e:
        logger.warning(f"Swing trades fetch failed: {e}")
        return []


def _get_week_events(d: date) -> list[str]:
    from india_quant.config import cfg
    events = []
    for i in range(7):
        day = d + timedelta(days=i)
        if day.isoformat() in cfg.rbi_policy_dates:
            events.append(f"RBI Policy on {day.isoformat()}")
    # Monthly expiry
    from india_quant.agents.risk_agent import RiskAgent
    last_thu = RiskAgent._last_thursday_of_month(d)
    if d <= last_thu <= d + timedelta(days=7):
        events.append(f"Monthly F&O Expiry: {last_thu.isoformat()}")
    events.append("Weekly NIFTY Expiry: Thursday")
    return events or ["No major events"]


def _get_weekly_options_strategy(as_of_date: str) -> dict | None:
    return None  # Populated after VRP data is available


def _get_sector_summary() -> list[dict]:
    return [
        {"sector": "Technology", "fii_flow": None, "top_stock": "TCS.NS", "setup": "Above 200 EMA"},
        {"sector": "Banking", "fii_flow": None, "top_stock": "HDFCBANK.NS", "setup": "Range bound"},
        {"sector": "Pharma", "fii_flow": None, "top_stock": "SUNPHARMA.NS", "setup": "Breakout watch"},
    ]


def _get_last_week_performance(d: date) -> dict | None:
    return None


def _get_nifty_levels() -> dict:
    try:
        import yfinance as yf
        data = yf.Ticker("^NSEI").history(period="1mo")
        if not data.empty:
            return {
                "support": round(float(data["Low"].tail(20).min()), 2),
                "resistance": round(float(data["High"].tail(20).max()), 2),
            }
    except Exception:
        pass
    return {}
