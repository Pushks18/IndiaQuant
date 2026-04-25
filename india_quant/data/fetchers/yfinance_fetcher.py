"""yfinance historical OHLCV fetcher for NSE/BSE stocks."""
import argparse
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from india_quant.data.db import get_session
from india_quant.data.models import PriceData


class YFinanceFetcher:
    NIFTY_50 = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "NTPC.NS",
        "POWERGRID.NS", "HCLTECH.NS", "TECHM.NS", "BAJAJFINSV.NS", "NESTLEIND.NS",
        "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "EICHERMOT.NS",
        "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "JSWSTEEL.NS", "M&M.NS",
        "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS", "APOLLOHOSP.NS", "BRITANNIA.NS",
        "TATACONSUM.NS", "INDUSINDBK.NS", "UPL.NS", "SHREECEM.NS", "BAJAJ-AUTO.NS",
    ]

    NIFTY_INDICES = ["^NSEI", "^BSESN", "^NSEBANK", "^INDIAVIX", "^NSMIDCP"]

    # Nifty Alpha 50 — actual NSE index constituents (as of 2025 rebalance).
    # Refresh from https://niftyindices.com/Methodology/Method_NIFTY_Equity_Indices.pdf
    NIFTY_ALPHA_50 = [
        # Industrials & defence
        "BEL.NS", "BHEL.NS", "CUMMINSIND.NS", "CGPOWER.NS", "MAZDOCK.NS",
        "BDL.NS", "POLYCAB.NS", "VOLTAS.NS", "DIXON.NS", "KEI.NS",
        "HITACHIENGY.NS",
        # IT
        "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS",
        # Pharma / chemicals
        "LUPIN.NS", "MANKIND.NS", "NAVINFLUOR.NS", "AUROPHARMA.NS",
        # PSU / energy
        "PFC.NS", "RECLTD.NS", "OIL.NS", "NMDC.NS", "SUZLON.NS",
        "JSWENERGY.NS", "ADANIPOWER.NS", "ADANIGREEN.NS", "ADANIENT.NS",
        "NHPC.NS", "TATAPOWER.NS",
        # Financials
        "HDFCAMC.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS",
        "AUBANK.NS", "FEDERALBNK.NS", "RBLBANK.NS", "INDUSTOWER.NS",
        # Realty
        "OBEROIRLTY.NS", "PRESTIGE.NS", "DLF.NS", "GODREJPROP.NS",
        # Consumer / new-age / health
        "TRENT.NS", "MARUTI.NS", "VBL.NS", "FORTIS.NS", "MAXHEALTH.NS",
        "ZOMATO.NS", "POLICYBZR.NS", "NYKAA.NS", "DELHIVERY.NS",
        # Logistics / aviation
        "INDIGO.NS", "CONCOR.NS",
    ]

    # Combined universe for intraday scanning
    @classmethod
    def universe(cls) -> list[str]:
        return list(dict.fromkeys(cls.NIFTY_50 + cls.NIFTY_ALPHA_50))

    def fetch_daily(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Download adjusted daily OHLCV. Handles corporate actions and tickers
        that yfinance has stale/delisted entries for (e.g. TATAMOTORS post-demerger)."""
        logger.info(f"Fetching daily data for {len(tickers)} tickers: {start_date} → {end_date}")
        rows = []
        skipped = []
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    actions=True,
                    progress=False,
                )
                if df.empty:
                    skipped.append(ticker)
                    continue
                df = df.reset_index()
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df["ticker"] = ticker
                df["interval"] = "1d"
                df["source"] = "yfinance"
                df = df.rename(columns={"date": "datetime"})
                rows.append(df[["ticker", "datetime", "open", "high", "low", "close", "volume", "interval", "source"]])
            except Exception as e:
                logger.warning(f"{ticker}: fetch skipped — {e}")
                skipped.append(ticker)
        if skipped:
            logger.info(f"yfinance: {len(skipped)} tickers skipped (no data): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def fetch_intraday(
        self, tickers: list[str], interval: str = "1h", period: str = "60d"
    ) -> pd.DataFrame:
        """Download intraday OHLCV. For 1m, chunks 7-day windows to get more history."""
        rows = []
        if interval == "1m":
            # yfinance 1m limited to 7 days; fetch in 7-day chunks
            end = date.today()
            chunk_days = 7
            periods_back = 8  # ~56 days
            for i in range(periods_back):
                chunk_end = end - timedelta(days=i * chunk_days)
                chunk_start = chunk_end - timedelta(days=chunk_days)
                for ticker in tickers:
                    try:
                        df = yf.download(
                            ticker,
                            start=chunk_start.isoformat(),
                            end=chunk_end.isoformat(),
                            interval="1m",
                            auto_adjust=True,
                            progress=False,
                        )
                        if not df.empty:
                            df = self._normalize_intraday(df, ticker, "1m")
                            rows.append(df)
                    except Exception as e:
                        logger.error(f"{ticker} 1m chunk {chunk_start}: {e}")
        else:
            for ticker in tickers:
                try:
                    df = yf.download(ticker, period=period, interval=interval,
                                     auto_adjust=True, progress=False)
                    if not df.empty:
                        rows.append(self._normalize_intraday(df, ticker, interval))
                except Exception as e:
                    logger.error(f"{ticker} intraday {interval}: {e}")
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _normalize_intraday(self, df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = df.rename(columns={"datetime": "datetime", "date": "datetime", "index": "datetime"})
        if "datetime" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        df["ticker"] = ticker
        df["interval"] = interval
        df["source"] = "yfinance"
        cols = ["ticker", "datetime", "open", "high", "low", "close", "volume", "interval", "source"]
        return df[[c for c in cols if c in df.columns]]

    def fetch_and_store(
        self, tickers: list[str], start_date: str, end_date: str, interval: str = "1d"
    ) -> int:
        """Fetch + upsert into PriceData. Returns number of rows upserted."""
        if interval == "1d":
            df = self.fetch_daily(tickers, start_date, end_date)
        else:
            df = self.fetch_intraday(tickers, interval=interval)

        if df.empty:
            logger.warning("No data to store.")
            return 0

        rows = df.to_dict(orient="records")
        upserted = 0
        with get_session() as session:
            for row in rows:
                stmt = insert(PriceData).values(**row).on_conflict_do_update(
                    index_elements=["ticker", "datetime", "interval"],
                    set_={k: row[k] for k in ["open", "high", "low", "close", "volume", "source"]},
                )
                session.execute(stmt)
                upserted += 1

        logger.info(f"Upserted {upserted} rows into price_data.")
        return upserted

    def update_all(self) -> int:
        """Fetch latest data for all NIFTY_50 tickers since last stored date."""
        from india_quant.data.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            result = session.execute(
                text("SELECT MAX(datetime) FROM price_data WHERE interval = '1d'")
            ).scalar()

        if result:
            start = result.date().isoformat()
        else:
            start = "2018-01-01"

        end = date.today().isoformat()
        logger.info(f"Updating all NIFTY_50 from {start} to {end}")
        return self.fetch_and_store(self.NIFTY_50, start, end, interval="1d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="all")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()

    fetcher = YFinanceFetcher()
    tickers = YFinanceFetcher.NIFTY_50 if args.tickers == "all" else args.tickers.split(",")
    fetcher.fetch_and_store(tickers, args.start, args.end)
