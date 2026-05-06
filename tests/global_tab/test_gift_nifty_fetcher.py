"""Tests for the GIFT Nifty NSE-IFSC scraper."""
from pathlib import Path
from unittest.mock import patch

import pytest

from india_quant.data.fetchers.gift_nifty_fetcher import (
    GiftNiftyQuote,
    fetch_gift_nifty_quote,
    parse_gift_nifty_page,
)


FIXTURE = Path(__file__).parent / "fixtures" / "gift_nifty_sample.html"


def test_parse_extracts_price_from_fixture():
    html = FIXTURE.read_text()
    quote = parse_gift_nifty_page(html)
    assert isinstance(quote, GiftNiftyQuote)
    assert 20000 < quote.last_price < 30000


def test_parse_returns_none_on_unparseable_html():
    assert parse_gift_nifty_page("<html><body>nothing here</body></html>") is None


def test_fetch_returns_none_on_http_error():
    class FakeResp:
        status_code = 503
        text = ""

    with patch("india_quant.data.fetchers.gift_nifty_fetcher.requests.get", return_value=FakeResp()):
        assert fetch_gift_nifty_quote() is None


def test_fetch_returns_none_on_network_exception():
    with patch(
        "india_quant.data.fetchers.gift_nifty_fetcher.requests.get",
        side_effect=Exception("network down"),
    ):
        assert fetch_gift_nifty_quote() is None
