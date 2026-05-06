"""Phase 3a route smoke test for /global with capital + mode query params."""
from __future__ import annotations

import pytest

from india_quant.dashboard.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_default_query_returns_200(client):
    resp = client.get("/global")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "NIFTY" in body
    assert "BANKNIFTY" in body


def test_explicit_balanced_mode_returns_200(client):
    resp = client.get("/global?capital=100000&mode=balanced")
    assert resp.status_code == 200


def test_aggressive_mode_returns_200(client):
    resp = client.get("/global?capital=500000&mode=aggressive")
    assert resp.status_code == 200


def test_bad_mode_returns_400(client):
    resp = client.get("/global?mode=insane")
    assert resp.status_code == 400


def test_bad_capital_returns_400(client):
    resp = client.get("/global?capital=-1")
    assert resp.status_code == 400


def test_card_or_no_trade_rendered(client):
    """Body should contain either a direction badge or a 'No trade' sentence per index."""
    resp = client.get("/global?capital=100000&mode=balanced")
    body = resp.get_data(as_text=True)
    # one of these must appear (direction badges, no-trade label, or NO TRADE pill)
    assert any(tok in body for tok in ("LONG", "SHORT", "NO TRADE", "No trade"))
