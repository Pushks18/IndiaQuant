"""Tests for snapshot.write_view_snapshot — append-only, fire-and-forget."""
from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, time, timezone

from india_quant.global_tab.snapshot import write_view_snapshot
from india_quant.global_tab.types import (
    Direction, GlobalTabView, LiveTicket, Mode, ReasoningContext, Status,
    TradeTicket,
)


def _minimal_view(as_of: datetime, mode: Mode = Mode.BALANCED) -> GlobalTabView:
    ctx = ReasoningContext(
        top_drivers=[("spx", 0.5)], analog_count=20, analog_winrate=0.6,
        analog_avg_pnl=12.0, no_trade_reason_code=None,
    )
    card = TradeTicket(
        index="NIFTY", direction=Direction.LONG, confidence=0.7,
        leg=None, timing=None, risk_reward=None, reasoning=ctx,
        live=LiveTicket(status=Status.WAITING, live_pnl=None, last_update=as_of),
        blurb="t",
    )
    # GlobalTabView signature varies; build via construct kwargs that align
    # with what orchestrator emits in tests.
    return GlobalTabView(
        as_of=as_of, mode=mode, capital=100_000.0,
        briefing=type("B", (), {"as_of": as_of, "tiles": []})(),
        heatmap=type("H", (), {"cells": []})(),
        cards=[card],
        artifact_paths={"name": "lightgbm"},
        staleness={},
    )


def test_writes_one_jsonl_line_per_call(tmp_path):
    view = _minimal_view(datetime(2026, 5, 7, 9, 30, tzinfo=timezone.utc))
    out = write_view_snapshot(view, root=tmp_path)
    assert out is not None and out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["mode"] == "balanced"
    assert rec["artifact"] == "lightgbm"
    assert rec["cards"][0]["index"] == "NIFTY"
    assert rec["cards"][0]["direction"] == "long"


def test_appends_subsequent_calls(tmp_path):
    view = _minimal_view(datetime(2026, 5, 7, 9, 30, tzinfo=timezone.utc))
    write_view_snapshot(view, root=tmp_path)
    write_view_snapshot(view, root=tmp_path)
    write_view_snapshot(view, root=tmp_path)
    fname = list(tmp_path.glob("*.jsonl"))[0]
    assert len(fname.read_text().strip().splitlines()) == 3


def test_per_day_file_naming(tmp_path):
    v1 = _minimal_view(datetime(2026, 5, 7, 9, 30))
    v2 = _minimal_view(datetime(2026, 5, 8, 9, 30))
    write_view_snapshot(v1, root=tmp_path)
    write_view_snapshot(v2, root=tmp_path)
    files = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert files == ["2026-05-07.jsonl", "2026-05-08.jsonl"]


def test_failure_is_silent(tmp_path):
    """Bad inputs must NOT raise — snapshot is fire-and-forget."""
    out = write_view_snapshot(object(), root=tmp_path)  # garbage view
    assert out is None  # no exception, just None
