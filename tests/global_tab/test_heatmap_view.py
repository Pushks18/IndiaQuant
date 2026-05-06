"""Tests for the Plotly heatmap renderer."""
from datetime import date

import pytest

from india_quant.global_tab.heatmap_view import render_heatmap_html
from india_quant.global_tab.types import CorrelationCell, CorrelationHeatmap


def _sample_heatmap() -> CorrelationHeatmap:
    cells = []
    for row in ["NIFTY", "BANKNIFTY"]:
        for col in ["SPX", "NASDAQ", "DXY"]:
            cells.append(
                CorrelationCell(asset_a=row, asset_b=col, rho_20d=0.4, rho_60d=0.5)
            )
    return CorrelationHeatmap(as_of=date(2026, 5, 4), cells=cells)


def test_renders_html_string():
    hm = _sample_heatmap()
    html = render_heatmap_html(hm)
    assert isinstance(html, str)
    assert "<div" in html
    assert "plotly" in html.lower()


def test_renders_a_div_with_data_points_for_each_cell():
    hm = _sample_heatmap()
    html = render_heatmap_html(hm)
    for ticker in ["NIFTY", "BANKNIFTY", "SPX", "NASDAQ", "DXY"]:
        assert ticker in html


def test_returns_empty_state_message_when_no_cells():
    empty = CorrelationHeatmap(as_of=date(2026, 5, 4), cells=[])
    html = render_heatmap_html(empty)
    assert "no correlation data" in html.lower() or "—" in html
