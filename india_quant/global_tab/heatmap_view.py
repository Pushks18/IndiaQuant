"""Plotly HTML renderer for correlation heatmap.

Returns an HTML embed string suitable for direct Jinja injection. The figure
shows a 20d-window heatmap with hover text revealing both 20d and 60d rho.
"""
from __future__ import annotations

import plotly.graph_objects as go

from india_quant.global_tab.types import CorrelationHeatmap


def render_heatmap_html(heatmap: CorrelationHeatmap) -> str:
    if not heatmap.cells:
        return '<div class="heatmap-empty">No correlation data available — —</div>'

    rows = sorted({c.asset_a for c in heatmap.cells})
    cols = sorted({c.asset_b for c in heatmap.cells})

    z_20: list[list[float | None]] = [[None for _ in cols] for _ in rows]
    hover: list[list[str | None]] = [[None for _ in cols] for _ in rows]

    for cell in heatmap.cells:
        ri = rows.index(cell.asset_a)
        ci = cols.index(cell.asset_b)
        z_20[ri][ci] = cell.rho_20d
        hover[ri][ci] = (
            f"<b>{cell.asset_a} ↔ {cell.asset_b}</b><br>"
            f"ρ 20d: {cell.rho_20d:+.2f}<br>"
            f"ρ 60d: {cell.rho_60d:+.2f}"
        )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_20,
            x=cols,
            y=rows,
            text=hover,
            hoverinfo="text",
            colorscale="RdBu",
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="ρ 20d"),
        )
    )
    fig.update_layout(
        title=f"NIFTY / BANKNIFTY correlations · as of {heatmap.as_of.isoformat()}",
        margin=dict(l=80, r=20, t=60, b=40),
        height=300,
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="global-heatmap")
