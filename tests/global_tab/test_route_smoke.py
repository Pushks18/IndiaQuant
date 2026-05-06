"""Smoke test for the /global Flask route.

Mocks get_global_context, fetch_gift_nifty_quote, and load_global_history so
the test is offline (no DB, no internet).
"""
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote


@pytest.fixture
def client():
    from india_quant.dashboard.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _fake_global_ctx():
    ctx = MagicMock()
    ctx.signals = []
    return ctx


def _fake_history():
    idx = pd.date_range(end=date(2026, 5, 4), periods=80, freq="B")
    cols = ["NIFTY", "BANKNIFTY", "SPX", "NASDAQ", "DXY", "BRENT", "INDIA_VIX", "US10Y", "NIKKEI", "HSI", "FTSE"]
    rng = np.random.default_rng(0)
    data = {c: 100 + np.cumsum(rng.normal(0, 0.5, len(idx))) for c in cols}
    return pd.DataFrame(data, index=idx)


# Patch targets must point at the symbols as looked up inside app.py at request
# time. Since app.py imports them inside the route function, patches target the
# source modules directly.
_PATCH_GET_CTX = "india_quant.signals.global_context.get_global_context"
_PATCH_GIFT = "india_quant.data.fetchers.gift_nifty_fetcher.fetch_gift_nifty_quote"
_PATCH_HISTORY = "india_quant.dashboard.data.load_global_history"


def test_global_route_returns_200(client):
    with patch(_PATCH_GET_CTX, return_value=_fake_global_ctx()), \
         patch(_PATCH_GIFT, return_value=GiftNiftyQuote(24412.0, 0.6)), \
         patch(_PATCH_HISTORY, return_value=_fake_history()):
        resp = client.get("/global")
    assert resp.status_code == 200, resp.data[:500]


def test_global_route_renders_tiles_and_heatmap(client):
    with patch(_PATCH_GET_CTX, return_value=_fake_global_ctx()), \
         patch(_PATCH_GIFT, return_value=GiftNiftyQuote(24412.0, 0.6)), \
         patch(_PATCH_HISTORY, return_value=_fake_history()):
        resp = client.get("/global")
    body = resp.data.decode()
    assert "GIFT Nifty" in body
    assert "Predicted gap" in body
    # Plotly div appears when heatmap renders successfully:
    assert "global-heatmap" in body or "heatmap-empty" in body


def test_global_route_renders_when_gift_unavailable(client):
    with patch(_PATCH_GET_CTX, return_value=_fake_global_ctx()), \
         patch(_PATCH_GIFT, return_value=None), \
         patch(_PATCH_HISTORY, return_value=_fake_history()):
        resp = client.get("/global")
    body = resp.data.decode()
    assert resp.status_code == 200
    assert "GIFT Nifty source unreachable" in body
