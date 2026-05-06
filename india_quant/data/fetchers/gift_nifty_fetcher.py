"""GIFT Nifty fetcher via investing.com public website.

GIFT Nifty migrated from SGX to NSE-IFSC in mid-2023. There is no clean
yfinance ticker for it. NSE IFSC's own website (nseifsc.com) is inaccessible:
the www subdomain is NXDOMAIN and the bare domain returns HTTP 400.

This module scrapes the investing.com GIFT Nifty 50 continuous front-month
page using curl-cffi (Chrome TLS impersonation) and extracts the price from
the embedded __NEXT_DATA__ JSON blob. Returns a GiftNiftyQuote, or None on
any failure. Callers must handle None gracefully — the briefing strip will
render '—' for that tile.

Source URL: https://www.investing.com/indices/gift-nifty-50-c1-futures
Instrument ID: 1209756 (parent / continuous front-month, isParent=True)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from curl_cffi import requests
from loguru import logger

_INVESTING_URL = "https://www.investing.com/indices/gift-nifty-50-c1-futures"
_TIMEOUT = 15

# Regex to extract the __NEXT_DATA__ JSON blob embedded by Next.js
_NEXT_DATA_RE = re.compile(
    r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
    re.DOTALL,
)


@dataclass(frozen=True)
class GiftNiftyQuote:
    last_price: float
    change_pct: float | None  # None if not available on the page


def parse_gift_nifty_page(html: str) -> GiftNiftyQuote | None:
    """Extract GiftNiftyQuote from an investing.com HTML page.

    The page embeds a Next.js __NEXT_DATA__ JSON blob containing the full
    instrument state, including live (or last-close) price data under:
        props.pageProps.state.indexStore.instrument.price
    """
    match = _NEXT_DATA_RE.search(html)
    if match is None:
        logger.debug("gift_nifty: __NEXT_DATA__ not found in page")
        return None

    try:
        data = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("gift_nifty: JSON parse error: {}", exc)
        return None

    try:
        price_block = (
            data["props"]["pageProps"]["state"]["indexStore"]["instrument"]["price"]
        )
    except (KeyError, TypeError) as exc:
        logger.debug("gift_nifty: price path missing: {}", exc)
        return None

    raw_last = price_block.get("last")
    if raw_last is None:
        logger.debug("gift_nifty: 'last' field missing in price block")
        return None

    try:
        last_price = float(raw_last)
    except (TypeError, ValueError) as exc:
        logger.debug("gift_nifty: could not parse 'last' as float: {}", exc)
        return None

    # changePcr is the percentage change; may be 0 when market is closed
    raw_pct = price_block.get("changePcr")
    change_pct: float | None = None
    if raw_pct is not None:
        try:
            change_pct = float(raw_pct)
        except (TypeError, ValueError):
            change_pct = None

    return GiftNiftyQuote(last_price=last_price, change_pct=change_pct)


def fetch_gift_nifty_quote() -> GiftNiftyQuote | None:
    """Fetch the GIFT Nifty front-month quote from investing.com.

    Returns None on any network, HTTP, or parse failure.
    """
    try:
        resp = requests.get(_INVESTING_URL, impersonate="chrome120", timeout=_TIMEOUT)
    except Exception as exc:
        logger.warning("gift_nifty fetch failed: {}", exc)
        return None

    if resp.status_code != 200:
        logger.warning("gift_nifty bad status: {}", resp.status_code)
        return None

    return parse_gift_nifty_page(resp.text)
