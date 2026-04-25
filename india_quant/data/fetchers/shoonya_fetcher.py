"""Shoonya (Finvasia) API connector — live NSE quotes, candles, and F&O greeks."""
from datetime import datetime

import pandas as pd
import pyotp
import pytz
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from india_quant.config import cfg
from india_quant.data.db import get_session
from india_quant.data.models import PriceData

IST = pytz.timezone("Asia/Kolkata")


class ShoonyaFetcher:
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MIN = 15
    MARKET_CLOSE_HOUR = 15
    MARKET_CLOSE_MIN = 30

    # Maps Angel-style interval names → Shoonya interval strings (minutes)
    INTERVAL_MAP = {
        "ONE_MINUTE": "1",
        "FIVE_MINUTE": "5",
        "FIFTEEN_MINUTE": "15",
        "THIRTY_MINUTE": "30",
        "ONE_HOUR": "60",
        "TWO_HOUR": "120",
        "FOUR_HOUR": "240",
    }

    def __init__(self):
        self._api = None
        self._logged_in = False
        self._token_cache: dict[str, str] = {}  # "EXCHANGE:SYMBOL" → token string

    def _get_api(self):
        try:
            from api_helper import ShoonyaApiPy
            return ShoonyaApiPy()
        except ImportError:
            raise RuntimeError(
                "ShoonyaApi-py not installed. Run: "
                "pip install git+https://github.com/Shoonya-Dev/ShoonyaApi-py.git"
            )

    def _login(self):
        if self._api is None:
            self._api = self._get_api()
        totp = pyotp.TOTP(cfg.shoonya_totp_secret).now()
        ret = self._api.login(
            userid=cfg.shoonya_user_id,
            password=cfg.shoonya_password,
            twoFA=totp,
            vendor_code=cfg.shoonya_vendor_code,
            api_secret=cfg.shoonya_api_key,
            imei=cfg.shoonya_imei,
        )
        if ret is None or ret.get("stat") != "Ok":
            raise RuntimeError(f"Shoonya login failed: {ret}")
        self._logged_in = True
        logger.info(f"Shoonya logged in as {cfg.shoonya_user_id}")

    def _ensure_logged_in(self):
        if not self._logged_in:
            self._login()

    def _resolve_token(self, exchange: str, symbol: str) -> str:
        """Resolve NSE symbol (e.g. 'RELIANCE-EQ') to numeric exchange token."""
        key = f"{exchange}:{symbol}"
        if key in self._token_cache:
            return self._token_cache[key]
        self._ensure_logged_in()
        ret = self._api.searchscrip(exchange=exchange, searchtext=symbol)
        if ret and ret.get("stat") == "Ok":
            for v in ret.get("values", []):
                if v.get("tsym") == symbol:
                    self._token_cache[key] = v["token"]
                    return v["token"]
        raise RuntimeError(f"Token not found for {exchange}:{symbol} — check symbol format (e.g. RELIANCE-EQ)")

    # ── Public interface (mirrors AngelFetcher) ───────────────────────────────

    def get_ltp(self, exchange: str, symbol: str, symboltoken: str = None) -> float:
        """Get last traded price. symboltoken auto-resolved if omitted."""
        self._ensure_logged_in()
        token = symboltoken or self._resolve_token(exchange, symbol)
        ret = self._api.get_quotes(exchange=exchange, token=token)
        if ret and ret.get("stat") == "Ok":
            return float(ret["lp"])
        raise RuntimeError(f"LTP fetch failed for {symbol}: {ret}")

    def get_quote(self, exchange: str, symbol: str, symboltoken: str = None) -> dict:
        """Get full quote: ltp, open, high, low, close, volume, bid, ask, circuits."""
        self._ensure_logged_in()
        token = symboltoken or self._resolve_token(exchange, symbol)
        ret = self._api.get_quotes(exchange=exchange, token=token)
        if ret and ret.get("stat") == "Ok":
            return {
                "ltp": float(ret.get("lp", 0)),
                "open": float(ret.get("o", 0)),
                "high": float(ret.get("h", 0)),
                "low": float(ret.get("l", 0)),
                "close": float(ret.get("c", 0)),
                "volume": int(ret.get("v", 0)),
                "bid": float(ret.get("bp1", 0)),
                "ask": float(ret.get("sp1", 0)),
                "upper_circuit": float(ret.get("uc", 0)),
                "lower_circuit": float(ret.get("lc", 0)),
            }
        raise RuntimeError(f"Quote fetch failed for {symbol}: {ret}")

    def get_candle_data(
        self,
        exchange: str,
        symbol: str,
        symboltoken: str = None,
        interval: str = "FIVE_MINUTE",
        from_date: str = None,
        to_date: str = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV candle data.
        interval: ONE_MINUTE | FIVE_MINUTE | FIFTEEN_MINUTE | THIRTY_MINUTE | ONE_HOUR
        from_date / to_date: 'dd-mm-yyyy HH:MM:SS'  (Shoonya format)
        """
        self._ensure_logged_in()
        token = symboltoken or self._resolve_token(exchange, symbol)
        shoonya_interval = self.INTERVAL_MAP.get(interval, "5")

        def _to_ts(s: str) -> int:
            return int(datetime.strptime(s, "%d-%m-%Y %H:%M:%S").timestamp())

        now_ist = datetime.now(IST)
        start_ts = _to_ts(from_date) if from_date else int(
            now_ist.replace(hour=9, minute=15, second=0, microsecond=0).timestamp()
        )
        end_ts = _to_ts(to_date) if to_date else int(now_ist.timestamp())

        ret = self._api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=start_ts,
            endtime=end_ts,
            interval=shoonya_interval,
        )
        if not ret:
            return pd.DataFrame()

        rows = []
        for bar in ret:
            if isinstance(bar, dict) and bar.get("stat") == "Ok":
                rows.append({
                    "datetime": datetime.strptime(bar["time"], "%d-%m-%Y %H:%M:%S"),
                    "open": float(bar["into"]),
                    "high": float(bar["inth"]),
                    "low": float(bar["intl"]),
                    "close": float(bar["intc"]),
                    "volume": int(float(bar.get("intv", 0))),
                })
        return pd.DataFrame(rows)

    def subscribe_live_feed(self, tokens: list[dict], on_tick_callback):
        """
        WebSocket subscription for live tick data.
        tokens: [{"exchange": "NSE", "symbol": "RELIANCE-EQ", "token": "2885"}, ...]
        Tick updates delivered to on_tick_callback(tick_dict).
        """
        self._ensure_logged_in()
        instruments = [f"{t['exchange']}|{t['token']}" for t in tokens]

        def _on_open():
            self._api.subscribe(instruments)
            logger.info(f"Shoonya WebSocket: subscribed to {len(instruments)} instruments.")

        self._api.start_websocket(
            subscribe_callback=lambda tick: on_tick_callback(tick),
            order_update_callback=lambda _: None,
            socket_open_callback=_on_open,
            socket_close_callback=lambda: logger.warning("Shoonya WebSocket closed."),
        )

    def get_option_greeks(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        option_type: str,
        spot_price: float = None,
        volatility: float = 20.0,
        rate: float = 6.5,
    ) -> dict:
        """
        Compute option greeks via Shoonya calculator.
        expiry: 'DD-MMM-YYYY'  e.g. '28-NOV-2024'
        option_type: 'CE' or 'PE'
        volatility: annualised IV % (default 20)
        rate: risk-free rate % (default 6.5 = RBI repo rate approx)
        """
        self._ensure_logged_in()
        if spot_price is None:
            spot_price = self.get_ltp("NSE", f"{symbol}-EQ")
        ret = self._api.get_option_greek(
            expiredate=expiry,
            StrikePrice=str(int(strike)),
            SpotPrice=str(spot_price),
            InitRate=str(rate),
            Volatility=str(volatility),
            OptionType=option_type,
        )
        if ret and ret.get("stat", "").upper() == "OK":
            p = "cal" if option_type == "CE" else "put"
            return {
                "delta": float(ret.get(f"{p}_delta", 0)),
                "gamma": float(ret.get(f"{p}_gamma", 0)),
                "theta": float(ret.get(f"{p}_theta", 0)),
                "vega": float(ret.get(f"{p}_vego", 0)),
                "rho": float(ret.get(f"{p}_rho", 0)),
                "price": float(ret.get(f"{p}_price", 0)),
            }
        logger.warning(f"Option greeks unavailable for {symbol} {expiry} {strike}{option_type}: {ret}")
        return {}

    def get_market_status(self) -> str:
        """Return 'Open', 'Closed', or 'Pre-open' based on current IST time."""
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return "Closed"
        t = now.hour * 60 + now.minute
        open_t = self.MARKET_OPEN_HOUR * 60 + self.MARKET_OPEN_MIN
        close_t = self.MARKET_CLOSE_HOUR * 60 + self.MARKET_CLOSE_MIN
        if t < 9 * 60:
            return "Closed"
        if t < open_t:
            return "Pre-open"
        if t > close_t:
            return "Closed"
        return "Open"

    def is_market_hours(self) -> bool:
        return self.get_market_status() == "Open"

    def fetch_and_store_live_prices(self, tickers: list[dict]) -> int:
        """
        Fetch LTP for all tickers and upsert to price_data with interval='live'.
        tickers: [{"exchange": "NSE", "symbol": "RELIANCE-EQ", "token": "2885"}, ...]
        """
        self._ensure_logged_in()
        total = 0
        now = datetime.utcnow()

        with get_session() as session:
            for t in tickers:
                try:
                    ltp = self.get_ltp(t["exchange"], t["symbol"], t.get("token"))
                    stmt = insert(PriceData).values(
                        ticker=t["symbol"],
                        datetime=now,
                        interval="live",
                        open=ltp, high=ltp, low=ltp, close=ltp,
                        volume=0,
                        source="shoonya",
                    ).on_conflict_do_update(
                        index_elements=["ticker", "datetime", "interval"],
                        set_={"close": ltp, "source": "shoonya"},
                    )
                    session.execute(stmt)
                    total += 1
                except Exception as e:
                    logger.error(f"Live price fetch failed for {t['symbol']}: {e}")

        logger.info(f"Shoonya live prices updated for {total} tickers.")
        return total
