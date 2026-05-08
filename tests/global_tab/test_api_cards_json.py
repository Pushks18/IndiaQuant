"""Tests for /api/global/cards.json — minimal smoke + shape assertions."""
from __future__ import annotations

import json

import pytest

from india_quant.dashboard.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return app.test_client()


def test_returns_200_with_default_params(client):
    r = client.get("/api/global/cards.json")
    assert r.status_code == 200
    assert r.is_json


def test_response_shape(client):
    r = client.get("/api/global/cards.json?capital=100000&mode=balanced")
    payload = r.get_json()
    assert "as_of" in payload
    assert "mode" in payload
    assert "capital" in payload
    assert "artifact" in payload
    assert "cards" in payload and isinstance(payload["cards"], list)
    # Should always emit one card per index
    assert len(payload["cards"]) == 2
    indices = {c["index"] for c in payload["cards"]}
    assert indices == {"NIFTY", "BANKNIFTY"}


def test_card_serialization_includes_required_keys(client):
    r = client.get("/api/global/cards.json")
    payload = r.get_json()
    for card in payload["cards"]:
        assert {"index", "direction", "confidence", "leg", "timing",
                "risk_reward", "reasoning", "live", "blurb"}.issubset(card.keys())
        # Enum values are serialised as strings
        assert isinstance(card["direction"], str)
        assert card["direction"] in {"long", "short", "no_trade"}
        assert isinstance(card["live"]["status"], str)


def test_bad_mode_returns_400(client):
    r = client.get("/api/global/cards.json?mode=ultra")
    assert r.status_code == 400
    assert r.get_json()["error"] == "bad mode"


def test_bad_capital_returns_400(client):
    r = client.get("/api/global/cards.json?capital=-1")
    assert r.status_code == 400
    assert r.get_json()["error"] == "bad capital"
