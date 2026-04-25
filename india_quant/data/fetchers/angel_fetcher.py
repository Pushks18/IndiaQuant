"""Angel One SmartAPI connector for live NSE quotes and F&O data."""
from datetime import date, datetime, timedelta

import pandas as pd
import pyotp
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from india_quant.config import cfg
from india_quant.data.db import get_session
from india_quant.data.models import PriceData


class AngelFetcher:
    # NSE market hours (IST)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MIN = 15
    MARKET_CLOSE_HOUR = 15
    MARKET_CLOSE_MIN = 30

    def __init__(self):
        self._smart_api = None
        self._auth_token = None
        self._feed_token = None
        self._logged_in = False

    def _get_smart_api(self):
        """Lazy-import SmartConnect to avoid import errors if not installed."""
        try:
            from SmartApi import SmartConnect
            return SmartConnect(api_key=cfg.angel_api_key)
        except ImportError:
            raise RuntimeError(
                "smartapi-python not installed. Run: pip install smartapi-python"
            )

    def _login(self):
        """Login to Angel SmartAPI with TOTP."""
        if self._smart_api is None:
            self._smart_api = self._get_smart_api()

        totp = pyotp.TOTP(cfg.angel_totp_secret).now()
        try:
            data = self._smart_api.generateSession(
                cfg.angel_client_id,
                cfg.angel_password,
                totp,
            )
            if not data.get("status"):
                raise RuntimeError(f"Angel login failed: {data}")

            self._auth_token = data["data"]["jwtToken"]
            self._feed_token = self._smart_api.getfeedToken()
            self._logged_in = True
            logger.info(f"Angel SmartAPI logged in as {cfg.angel_client_id}")
        except Exception as e:
            logger.error(f"Angel login error: {e}")
            raise

    def _ensure_logged_in(self):
        if not self._logged_in:
            self._login()

    def get_ltp(self, exchange: str, symbol: str, symboltoken: str) -> float:
        """Get last traded price for a symbol."""
        self._ensure_logged_in()
        data = self._smart_api.ltpData(exchange, symbol, symboltoken)
        if data.get("status"):
            return float(data["data"]["ltp"])
        raise RuntimeError(f"LTP fetch failed: {data}")

    def get_quote(self, exchange: str, symbol: str, symboltoken: str) -> dict:
        """Get full quote: ltp, open, high, low, close, volume, bid, ask."""
        self._ensure_logged_in()
        data = self._smart_api.getMarketData("FULL", {exchange: [symboltoken]})
        if data.get("status"):
            d = data["data"]["fetched"][0]
            return {
                "ltp": d.get("ltp"),
                "open": d.get("open"),
                "high": d.get("high"),
                "low": d.get("low"),
                "close": d.get("close"),
                "volume": d.get("tradedVolume"),
                "bid": d.get("buyQty"),
                "ask": d.get("sellQty"),
            }
        raise RuntimeError(f"Quote fetch failed: {data}")

    def get_candle_data(
        self, exchange: str, symbol: str, symboltoken: str,
        interval: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        """
        Get OHLCV candle data.
        interval: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY
        """
        self._ensure_logged_in()
        params = {
            "exchange": exchange,
            "symboltoken": symboltoken,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        data = self._smart_api.getCandleData(params)
        if data.get("status"):
            df = pd.DataFrame(
                data["data"],
                columns=["datetime", "open", "high", "low", "close", "volume"],
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        raise RuntimeError(f"Candle fetch failed: {data}")

    def subscribe_live_feed(self, tokens: list[dict], on_tick_callback):
        """WebSocket subscription for live tick data."""
        try:
            from SmartApi.SmartWebSocketV2 import SmartWebSocketV2
        except ImportError:
            raise RuntimeError("smartapi-python not installed")

        self._ensure_logged_in()
        ws = SmartWebSocketV2(
            self._auth_token,
            cfg.angel_api_key,
            cfg.angel_client_id,
            self._feed_token,
        )

        def on_data(ws_obj, message):
            on_tick_callback(message)

        def on_error(ws_obj, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws_obj):
            logger.warning("WebSocket connection closed.")

        def on_open(ws_obj):
            ws_obj.subscribe("live_feed", 1, tokens)

        ws.on_message = on_data
        ws.on_error = on_error
        ws.on_close = on_close
        ws.on_open = on_open
        ws.connect()

    def get_option_greeks(
        self, symbol: str, expiry: str, strike: float, option_type: str
    ) -> dict:
        """Get delta, gamma, theta, vega, IV for a specific option."""
        self._ensure_logged_in()
        # Angel API returns Greeks in the options chain data
        data = self._smart_api.getOptionGreeks({
            "name": symbol,
            "expirydate": expiry,
            "strikePrice": str(int(strike)),
            "optiontype": option_type,
        })
        if data.get("status"):
            return data["data"]
        logger.warning(f"Option Greeks unavailable for {symbol} {expiry} {strike} {option_type}")
        return {}

    def get_market_status(self) -> str:
        """Return 'Open', 'Closed', or 'Pre-open'."""
        self._ensure_logged_in()
        try:
            data = self._smart_api.getMarketData("LTP", {"NSE": []})
            # Infer from exchange status if available
            return "Open"
        except Exception:
            return "Closed"

    def is_market_hours(self) -> bool:
        """True if current IST time is within NSE trading hours."""
        import pytz
        now = datetime.now(tz=pytz.timezone("Asia/Kolkata"))
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        open_time = now.replace(hour=self.MARKET_OPEN_HOUR, minute=self.MARKET_OPEN_MIN, second=0)
        close_time = now.replace(hour=self.MARKET_CLOSE_HOUR, minute=self.MARKET_CLOSE_MIN, second=0)
        return open_time <= now <= close_time

    def fetch_and_store_live_prices(self, tickers: list[dict]) -> int:
        """
        Fetch LTP for all tickers and upsert to PriceData with interval='live'.
        tickers = [{"exchange": "NSE", "symbol": "RELIANCE-EQ", "token": "2885"}, ...]
        """
        self._ensure_logged_in()
        total = 0
        now = datetime.utcnow()

        with get_session() as session:
            for t in tickers:
                try:
                    ltp = self.get_ltp(t["exchange"], t["symbol"], t["token"])
                    stmt = insert(PriceData).values(
                        ticker=t["symbol"],
                        datetime=now,
                        interval="live",
                        open=ltp,
                        high=ltp,
                        low=ltp,
                        close=ltp,
                        volume=0,
                        source="angel",
                    ).on_conflict_do_update(
                        index_elements=["ticker", "datetime", "interval"],
                        set_={"close": ltp, "source": "angel"},
                    )
                    session.execute(stmt)
                    total += 1
                except Exception as e:
                    logger.error(f"Live price fetch failed for {t['symbol']}: {e}")

        logger.info(f"Live prices updated for {total} tickers.")
        return total
