"""Tests for the briefing-strip adapter."""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote
from india_quant.global_tab.briefing import build_briefing
from india_quant.global_tab.types import BriefingStrip, BriefingTile


def _fake_signal_row(ticker, label, group, pct_1d, price):
    obj = MagicMock()
    obj.ticker = ticker
    obj.label = label
    obj.group = group
    obj.pct_1d = pct_1d
    obj.price = price
    return obj


def _fake_global_context():
    rows = [
        _fake_signal_row("^GSPC", "S&P 500", "US", 1.10, 5612.40),
        _fake_signal_row("^IXIC", "Nasdaq", "US", 1.40, 17800.0),
        _fake_signal_row("DX-Y.NYB", "DXY", "FX", -0.30, 104.20),
        _fake_signal_row("BZ=F", "Brent", "Commodities", 0.85, 82.10),
        _fake_signal_row("^INDIAVIX", "India VIX", "FX", -2.10, 13.40),
        _fake_signal_row("^TNX", "US 10Y Yield", "FX", 0.50, 4.45),
        _fake_signal_row("^N225", "Nikkei 225", "Asia", 0.70, 39200.0),
        _fake_signal_row("^HSI", "Hang Seng", "Asia", -0.20, 18800.0),
        _fake_signal_row("^FTSE", "FTSE 100", "Europe", 0.30, 8200.0),
    ]
    ctx = MagicMock()
    ctx.signals = rows
    return ctx


def test_build_briefing_returns_typed_strip():
    ctx = _fake_global_context()
    gift = GiftNiftyQuote(last_price=24412.0, change_pct=0.60)
    as_of = datetime(2026, 5, 5, 8, 45)
    strip = build_briefing(as_of=as_of, context=ctx, gift_nifty=gift)
    assert isinstance(strip, BriefingStrip)
    assert strip.as_of == as_of


def test_build_briefing_has_all_ten_tiles_in_order():
    ctx = _fake_global_context()
    gift = GiftNiftyQuote(last_price=24412.0, change_pct=0.60)
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=gift)
    labels = [t.label for t in strip.tiles]
    assert labels == [
        "SPX", "Nasdaq", "GIFT Nifty", "DXY", "Brent",
        "India VIX", "US 10Y", "Nikkei", "Hang Seng", "FTSE",
    ]


def test_build_briefing_gift_tile_uses_scraper_value():
    ctx = _fake_global_context()
    gift = GiftNiftyQuote(last_price=24412.0, change_pct=0.60)
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=gift)
    gift_tile = next(t for t in strip.tiles if t.label == "GIFT Nifty")
    assert gift_tile.change_pct == pytest.approx(0.60)
    assert "24412" in gift_tile.value or "24,412" in gift_tile.value


def test_build_briefing_gift_tile_renders_em_dash_when_unavailable():
    ctx = _fake_global_context()
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=None)
    gift_tile = next(t for t in strip.tiles if t.label == "GIFT Nifty")
    assert gift_tile.value == "—"
    assert gift_tile.change_pct == 0.0
    assert gift_tile.sentiment == "neutral"


def test_build_briefing_sentiment_follows_pct_sign():
    ctx = _fake_global_context()
    strip = build_briefing(
        as_of=datetime(2026, 5, 5, 8, 45),
        context=ctx,
        gift_nifty=GiftNiftyQuote(last_price=24412.0, change_pct=0.60),
    )
    spx = next(t for t in strip.tiles if t.label == "SPX")
    dxy = next(t for t in strip.tiles if t.label == "DXY")
    vix = next(t for t in strip.tiles if t.label == "India VIX")
    assert spx.sentiment == "bullish"
    assert dxy.sentiment == "bearish"
    assert vix.sentiment == "bearish"


def test_predicted_gap_uses_gift_change():
    ctx = _fake_global_context()
    gift = GiftNiftyQuote(last_price=24412.0, change_pct=0.60)
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=gift)
    assert strip.predicted_gap_bps["NIFTY"] == pytest.approx(0.5 * 60.0)
    assert strip.predicted_gap_bps["BANKNIFTY"] == pytest.approx(0.6 * 60.0)


def test_predicted_gap_is_zero_when_gift_unavailable():
    ctx = _fake_global_context()
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=None)
    assert strip.predicted_gap_bps == {"NIFTY": 0.0, "BANKNIFTY": 0.0}


def test_missing_ticker_renders_em_dash_tile():
    ctx = MagicMock()
    ctx.signals = []
    strip = build_briefing(as_of=datetime(2026, 5, 5, 8, 45), context=ctx, gift_nifty=None)
    for tile in strip.tiles:
        if tile.label != "GIFT Nifty":
            assert tile.value == "—"
            assert tile.change_pct == 0.0
            assert tile.sentiment == "neutral"
