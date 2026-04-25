"""Daily data quality checks for the India Quant pipeline."""
from datetime import date, datetime, timedelta

from loguru import logger


def run_daily_quality_checks() -> list[str]:
    """
    Run all data quality checks. Returns list of alert strings.
    Empty list = all good.
    """
    alerts = []
    today = date.today()

    alerts.extend(_check_price_data_completeness(today))
    alerts.extend(_check_price_anomalies(today))
    alerts.extend(_check_options_freshness(today))
    alerts.extend(_check_factor_scores(today))
    alerts.extend(_check_sentiment_scores(today))

    if alerts:
        logger.warning(f"[QualityMonitor] {len(alerts)} alerts found")
        for a in alerts:
            logger.warning(f"  {a}")
    else:
        logger.info("[QualityMonitor] All data quality checks passed.")

    return alerts


def _check_price_data_completeness(today: date) -> list[str]:
    """Check: no missing dates in PriceData for NIFTY-50 for past 5 trading days."""
    alerts = []
    try:
        from india_quant.data.db import get_session
        from india_quant.data.fetchers.yfinance_fetcher import YFinanceFetcher
        from sqlalchemy import text

        with get_session() as session:
            row = session.execute(
                text("""
                    SELECT COUNT(DISTINCT ticker), MAX(datetime)::date
                    FROM price_data
                    WHERE interval = '1d'
                    AND datetime >= NOW() - INTERVAL '7 days'
                """)
            ).fetchone()

        n_tickers, last_date = row[0] or 0, row[1]
        expected_tickers = len(YFinanceFetcher.NIFTY_50)

        if n_tickers < expected_tickers * 0.8:
            alerts.append(f"Price data: only {n_tickers}/{expected_tickers} tickers in past 7 days")

        if last_date and last_date < today - timedelta(days=3):
            alerts.append(f"Price data stale: last record {last_date} (today={today})")

    except Exception as e:
        alerts.append(f"Price data check failed: {e}")

    return alerts


def _check_price_anomalies(today: date) -> list[str]:
    """Check: no price jumps > 20% (likely unadjusted corporate action)."""
    alerts = []
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            rows = session.execute(
                text("""
                    WITH daily AS (
                        SELECT ticker, datetime, close,
                               LAG(close) OVER (PARTITION BY ticker ORDER BY datetime) AS prev_close
                        FROM price_data
                        WHERE interval = '1d'
                        AND datetime >= NOW() - INTERVAL '5 days'
                    )
                    SELECT ticker, datetime, close, prev_close,
                           ABS(close - prev_close) / NULLIF(prev_close, 0) AS pct_chg
                    FROM daily
                    WHERE ABS(close - prev_close) / NULLIF(prev_close, 0) > 0.20
                """)
            ).fetchall()

        for r in rows:
            alerts.append(
                f"Price anomaly: {r[0]} moved {r[4]:.1%} on {r[1]} "
                f"(close={r[2]}, prev={r[3]}) — check for unadjusted corporate action"
            )

    except Exception as e:
        alerts.append(f"Price anomaly check failed: {e}")

    return alerts


def _check_options_freshness(today: date) -> list[str]:
    """Check: options chain fetched within last 2 hours."""
    alerts = []
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            row = session.execute(
                text("""
                    SELECT MAX(timestamp)
                    FROM option_chain
                    WHERE trade_date = :d
                """),
                {"d": today},
            ).fetchone()

        if not row or not row[0]:
            alerts.append("Options chain: no data for today")
        else:
            last_ts = row[0]
            if datetime.now(tz=last_ts.tzinfo) - last_ts > timedelta(hours=2):
                staleness = datetime.now(tz=last_ts.tzinfo) - last_ts
                alerts.append(f"Options chain stale: last fetched {staleness} ago")

    except Exception as e:
        alerts.append(f"Options freshness check failed: {e}")

    return alerts


def _check_factor_scores(today: date) -> list[str]:
    """Check: FactorScores computed for today."""
    alerts = []
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            row = session.execute(
                text("SELECT COUNT(*) FROM factor_scores WHERE date = :d"),
                {"d": today},
            ).fetchone()

        count = row[0] if row else 0
        if count < 30:
            alerts.append(f"Factor scores: only {count} tickers computed for today (expected 50+)")

    except Exception as e:
        alerts.append(f"Factor scores check failed: {e}")

    return alerts


def _check_sentiment_scores(today: date) -> list[str]:
    """Check: sentiment scores computed for top tickers."""
    alerts = []
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            row = session.execute(
                text("""
                    SELECT COUNT(DISTINCT ticker) FROM sentiment_aggregate
                    WHERE date >= :d - INTERVAL '1 day'
                """),
                {"d": today},
            ).fetchone()

        count = row[0] if row else 0
        if count < 5:
            alerts.append(f"Sentiment scores: only {count} tickers have sentiment data today")

    except Exception as e:
        alerts.append(f"Sentiment check failed: {e}")

    return alerts
