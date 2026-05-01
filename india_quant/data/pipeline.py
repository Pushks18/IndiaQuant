"""Data pipeline orchestrator — wires all fetchers into a daily schedule."""
from datetime import date

from loguru import logger


class DataPipeline:
    @staticmethod
    def run_pre_market(run_date: str = None):
        """
        08:00 IST: Fetch previous day's final prices, options snapshot, overnight news.
        run_date: 'YYYY-MM-DD' string (defaults to today)
        """
        run_date = run_date or date.today().isoformat()
        logger.info(f"[Pipeline] Pre-market run for {run_date}")

        # 1. Fetch EOD prices via yfinance
        try:
            from india_quant.data.fetchers.yfinance_fetcher import YFinanceFetcher
            fetcher = YFinanceFetcher()
            rows = fetcher.update_all()
            logger.info(f"[Pipeline] yfinance: {rows} rows updated")
        except Exception as e:
            logger.error(f"[Pipeline] yfinance failed: {e}")

        # 2. Fetch NSE option chain snapshot
        try:
            from india_quant.data.fetchers.nse_options_fetcher import NSEOptionsFetcher
            of = NSEOptionsFetcher()
            rows = of.fetch_and_store(["NIFTY", "BANKNIFTY"])
            logger.info(f"[Pipeline] Options: {rows} rows updated")
        except Exception as e:
            logger.error(f"[Pipeline] Options fetch failed: {e}")

        # 3. Fetch overnight news + score sentiment
        try:
            from india_quant.data.fetchers.news_fetcher import NewsFetcher
            from india_quant.data.fetchers.yfinance_fetcher import YFinanceFetcher
            nf = NewsFetcher()
            nifty_50 = YFinanceFetcher.NIFTY_50[:10]  # top 10 for speed
            rows = nf.fetch_and_store(nifty_50)
            logger.info(f"[Pipeline] News: {rows} articles stored")
        except Exception as e:
            logger.error(f"[Pipeline] News fetch failed: {e}")

        # 4. Fetch and store global cross-market signals
        try:
            n = DataPipeline.fetch_global_signals(run_date)
            logger.info(f"[Pipeline] GlobalSignals: {n} rows updated")
        except Exception as e:
            logger.error(f"[Pipeline] GlobalSignals fetch failed: {e}")

        logger.info(f"[Pipeline] Pre-market run complete for {run_date}")

    @staticmethod
    def fetch_global_signals(trade_date: str = None):
        """Fetch global context and upsert all signal rows to global_signals table."""
        from datetime import date as date_cls
        from india_quant.signals.global_context import get_global_context
        from india_quant.data.models import GlobalSignal
        from india_quant.data.db import get_session
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        if trade_date is None:
            trade_date = date_cls.today().isoformat()

        ctx = get_global_context()
        trade_date_obj = date_cls.fromisoformat(trade_date)

        with get_session() as session:
            for sig in ctx.signals:
                stmt = pg_insert(GlobalSignal).values(
                    date=trade_date_obj,
                    ticker=sig.ticker,
                    label=sig.label,
                    group=sig.group,
                    pct_1d=sig.pct_1d,
                    pct_5d=sig.pct_5d,
                    corr_30d=sig.corr_30d,
                    corr_90d=sig.corr_90d,
                    regime=ctx.regime,
                ).on_conflict_do_update(
                    index_elements=["date", "ticker"],
                    set_={
                        "pct_1d":   sig.pct_1d,
                        "pct_5d":   sig.pct_5d,
                        "corr_30d": sig.corr_30d,
                        "corr_90d": sig.corr_90d,
                        "regime":   ctx.regime,
                    },
                )
                session.execute(stmt)
        logger.info(f"[Pipeline] GlobalSignals: {len(ctx.signals)} rows upserted for {trade_date}")
        return len(ctx.signals)

    @staticmethod
    def run_intraday():
        """
        Every 5 min 09:15-15:30: Angel SmartAPI live prices.
        Every 30 min: Refresh options chain.
        Every 60 min: Latest news.
        """
        logger.info("[Pipeline] Intraday update running...")

        try:
            from india_quant.data.fetchers.shoonya_fetcher import ShoonyaFetcher
            af = ShoonyaFetcher()
            # Tokens resolved automatically via searchscrip cache on first call
            logger.info("[Pipeline] Shoonya live prices: configure symbol tokens first")
        except Exception as e:
            logger.error(f"[Pipeline] Shoonya live fetch failed: {e}")

    @staticmethod
    def run_post_market(run_date: str = None):
        """
        16:00 IST: Final EOD prices.
        16:30 IST: Compute factor scores.
        17:00 IST: Compute signal labels.
        17:30 IST: Run signal predictions.
        18:00 IST: Trigger report generation.
        """
        run_date = run_date or date.today().isoformat()
        logger.info(f"[Pipeline] Post-market run for {run_date}")

        try:
            from india_quant.data.fetchers.yfinance_fetcher import YFinanceFetcher
            fetcher = YFinanceFetcher()
            rows = fetcher.update_all()
            logger.info(f"[Pipeline] EOD prices: {rows} rows")
        except Exception as e:
            logger.error(f"[Pipeline] EOD fetch failed: {e}")

        try:
            from india_quant.signals.factors import FactorEngine
            fe = FactorEngine()
            fe.compute_all(run_date)
            logger.info("[Pipeline] Factor scores computed.")
        except Exception as e:
            logger.error(f"[Pipeline] Factor compute failed: {e}")

        try:
            from india_quant.signals.ml_models import ReturnPredictor
            rp = ReturnPredictor()
            predictions = rp.predict_today()
            logger.info(f"[Pipeline] ML predictions: {len(predictions)} tickers")
        except Exception as e:
            logger.error(f"[Pipeline] ML prediction failed: {e}")

        logger.info(f"[Pipeline] Post-market run complete for {run_date}")

    @staticmethod
    def run_weekly_maintenance():
        """Sunday 22:00: Retrain ML, data quality checks."""
        logger.info("[Pipeline] Weekly maintenance running...")

        try:
            from india_quant.signals.ml_models import ReturnPredictor
            rp = ReturnPredictor()
            rp.retrain_weekly()
            logger.info("[Pipeline] ML models retrained.")
        except Exception as e:
            logger.error(f"[Pipeline] Weekly retrain failed: {e}")

        try:
            from india_quant.data.quality_monitor import run_daily_quality_checks
            alerts = run_daily_quality_checks()
            if alerts:
                for a in alerts:
                    logger.warning(f"[Quality] {a}")
            else:
                logger.info("[Pipeline] Data quality: all checks passed.")
        except Exception as e:
            logger.error(f"[Pipeline] Quality check failed: {e}")
