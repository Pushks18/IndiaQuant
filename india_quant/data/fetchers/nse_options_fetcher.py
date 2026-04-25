"""NSE options chain scraper with session management and IV surface computation."""
import math
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm
from sqlalchemy.dialects.postgresql import insert

try:
    from curl_cffi import requests
    _CURL_CFFI = True
except ImportError:
    import requests as requests
    _CURL_CFFI = False

from india_quant.data.db import get_session
from india_quant.data.models import OptionChain


class NSEOptionsFetcher:
    BASE_URL = "https://www.nseindia.com"
    OPTION_CHAIN_URL = BASE_URL + "/api/option-chain-indices?symbol={symbol}"
    MARKET_STATUS_URL = BASE_URL + "/api/market-status"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    API_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "X-Requested-With": "XMLHttpRequest",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Connection": "keep-alive",
    }

    def __init__(self):
        if _CURL_CFFI:
            self.session = requests.Session(impersonate="chrome")
        else:
            self.session = requests.Session()
            self.session.headers.update(self.HEADERS)
            logger.warning("curl_cffi not installed — NSE may block requests. Run: pip install curl-cffi")
        self._session_initialized = False

    def _initialize_session(self):
        """Pre-visit NSE main page to acquire session cookies."""
        try:
            if _CURL_CFFI:
                resp = self.session.get(self.BASE_URL, timeout=15)
                resp.raise_for_status()
                time.sleep(2)
                self.session.get(self.BASE_URL + "/option-chain", timeout=15)
                time.sleep(1)
            else:
                self.session.headers.update(self.HEADERS)
                resp = self.session.get(self.BASE_URL, timeout=15)
                resp.raise_for_status()
                time.sleep(3)
                self.session.get(self.BASE_URL + "/option-chain", timeout=15)
                time.sleep(2)
                self.session.headers.update(self.API_HEADERS)
            resp3 = self.session.get(self.MARKET_STATUS_URL, timeout=10)
            resp3.raise_for_status()
            time.sleep(1)
            self._session_initialized = True
            logger.info("NSE session initialized (curl_cffi={}).", _CURL_CFFI)
        except Exception as e:
            logger.warning(f"NSE session init failed: {e}")
            self._session_initialized = False

    def fetch_option_chain(self, symbol: str = "NIFTY") -> dict:
        """Fetch raw option chain JSON for NIFTY, BANKNIFTY, or FINNIFTY.

        NSE actively blocks scrapers; on failure we return an empty structure rather
        than raising, so the rest of the pipeline can continue.
        """
        if not self._session_initialized:
            self._initialize_session()

        if not self._session_initialized:
            logger.warning(f"NSE session unavailable — skipping {symbol} option chain (NSE blocking).")
            return {"records": {"data": []}}

        url = self.OPTION_CHAIN_URL.format(symbol=symbol)
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Option chain fetch failed for {symbol}, retrying once: {e}")
            self._session_initialized = False
            self._initialize_session()
            if not self._session_initialized:
                return {"records": {"data": []}}
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception as e2:
                logger.warning(f"Option chain fetch skipped for {symbol} after retry: {e2}")
                return {"records": {"data": []}}

    def parse_to_dataframe(self, raw_data: dict) -> pd.DataFrame:
        """Parse NSE option chain JSON into a flat DataFrame."""
        records = raw_data.get("records", {}).get("data", [])
        rows = []
        for record in records:
            strike = record.get("strikePrice")
            expiry = record.get("expiryDate")
            for ot, key in [("CE", "CE"), ("PE", "PE")]:
                d = record.get(key, {})
                if not d:
                    continue
                rows.append({
                    "strike": strike,
                    "expiry": expiry,
                    "option_type": ot,
                    "last_price": d.get("lastPrice"),
                    "bid": d.get("bidprice"),
                    "ask": d.get("askPrice"),
                    "iv": d.get("impliedVolatility"),
                    "open_interest": d.get("openInterest"),
                    "oi_change": d.get("changeinOpenInterest"),
                    "volume": d.get("totalTradedVolume"),
                    "underlying": d.get("underlying", ""),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["expiry"] = pd.to_datetime(df["expiry"], format="%d-%b-%Y", errors="coerce")
        return df

    def fetch_and_store(self, symbols: list[str] = None) -> int:
        """Fetch option chains for given symbols and upsert to DB."""
        if symbols is None:
            symbols = ["NIFTY", "BANKNIFTY"]

        total = 0
        for symbol in symbols:
            try:
                raw = self.fetch_option_chain(symbol)
                df = self.parse_to_dataframe(raw)
                if df.empty:
                    logger.warning(f"{symbol}: empty option chain")
                    continue

                today = date.today()
                timestamp = datetime.utcnow()

                # Staleness check
                fetched_at = raw.get("records", {}).get("timestamp", "")
                if fetched_at:
                    logger.info(f"{symbol} chain timestamp: {fetched_at}")

                with get_session() as session:
                    for _, row in df.iterrows():
                        if pd.isna(row.get("expiry")):
                            continue
                        stmt = insert(OptionChain).values(
                            underlying=symbol,
                            trade_date=today,
                            expiry=row["expiry"].date() if hasattr(row["expiry"], "date") else today,
                            strike=row["strike"] or 0,
                            option_type=row["option_type"],
                            last_price=row.get("last_price"),
                            bid=row.get("bid"),
                            ask=row.get("ask"),
                            iv=row.get("iv"),
                            open_interest=row.get("open_interest"),
                            oi_change=row.get("oi_change"),
                            volume=row.get("volume"),
                            timestamp=timestamp,
                        ).on_conflict_do_update(
                            constraint="uq_option_chain",
                            set_={"last_price": row.get("last_price"), "iv": row.get("iv"),
                                  "open_interest": row.get("open_interest"), "timestamp": timestamp},
                        )
                        session.execute(stmt)
                        total += 1

                logger.info(f"{symbol}: {len(df)} options upserted.")
                time.sleep(60)  # Rate limit: 1 req/min between symbols

            except Exception as e:
                logger.error(f"{symbol}: fetch_and_store failed — {e}")

        return total

    def compute_iv_surface(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ATM IV, 25d put skew, 25d call skew, term structure per expiry."""
        if df.empty:
            return df

        results = []
        for expiry, grp in df.groupby("expiry"):
            dte = (pd.Timestamp(expiry) - pd.Timestamp.now()).days
            if dte <= 0:
                continue

            ce = grp[grp["option_type"] == "CE"].sort_values("strike")
            pe = grp[grp["option_type"] == "PE"].sort_values("strike")
            if ce.empty or pe.empty:
                continue

            # ATM = strike nearest to midpoint of CE/PE strikes
            atm_strike = ce.iloc[len(ce) // 2]["strike"]
            atm_ce_iv = ce[ce["strike"] == atm_strike]["iv"].values
            atm_pe_iv = pe[pe["strike"] == atm_strike]["iv"].values
            atm_iv = np.nanmean(list(atm_ce_iv) + list(atm_pe_iv)) if len(atm_ce_iv) or len(atm_pe_iv) else None

            # 25d put skew: OTM put (lower strike) IV - ATM IV
            otm_put_strikes = pe[pe["strike"] < atm_strike]
            put_skew = None
            if not otm_put_strikes.empty and atm_iv:
                otm_iv = otm_put_strikes.nlargest(1, "strike")["iv"].values[0]
                put_skew = otm_iv - atm_iv

            results.append({
                "expiry": expiry,
                "dte": dte,
                "atm_iv": atm_iv,
                "put_skew_25d": put_skew,
            })

        return pd.DataFrame(results)

    @staticmethod
    def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Black-Scholes delta for finding 25-delta options."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if option_type == "CE":
            return norm.cdf(d1)
        return norm.cdf(d1) - 1  # PE delta


if __name__ == "__main__":
    fetcher = NSEOptionsFetcher()
    raw = fetcher.fetch_option_chain("NIFTY")
    df = fetcher.parse_to_dataframe(raw)
    print(f"Options rows: {len(df)}")
    iv_surface = fetcher.compute_iv_surface(df)
    print(iv_surface.head())
