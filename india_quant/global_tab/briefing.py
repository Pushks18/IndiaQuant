"""Briefing-strip adapter.

Pure function: takes a GlobalContext (from signals/global_context.py) and an
optional GiftNiftyQuote, returns a BriefingStrip dataclass. No I/O. No mutation.

The 10 tiles are emitted in a fixed order matching the heatmap columns. If a
ticker is missing from the GlobalContext, the tile renders as '—' (neutral
sentiment, change_pct=0.0). The predicted_gap_bps formula is a placeholder
that uses 0.5x / 0.6x of the GIFT Nifty change in bps; the real model lands
in Phase 3.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Literal

from india_quant.data.fetchers.gift_nifty_fetcher import GiftNiftyQuote
from india_quant.global_tab.types import BriefingStrip, BriefingTile

# (label, ticker) in render order. Ticker is None for GIFT Nifty (separate fetcher).
_TILE_ORDER: list[tuple[str, str | None]] = [
    ("SPX", "^GSPC"),
    ("Nasdaq", "^IXIC"),
    ("GIFT Nifty", None),
    ("DXY", "DX-Y.NYB"),
    ("Brent", "BZ=F"),
    ("India VIX", "^INDIAVIX"),
    ("US 10Y", "^TNX"),
    ("Nikkei", "^N225"),
    ("Hang Seng", "^HSI"),
    ("FTSE", "^FTSE"),
]


def _format_value(price: float | None) -> str:
    if price is None:
        return "—"
    if price >= 1000:
        return f"{price:,.2f}"
    return f"{price:.2f}"


def _sentiment_for_pct(pct: float | None) -> Literal["bullish", "bearish", "neutral"]:
    if pct is None or pct == 0.0:
        return "neutral"
    return "bullish" if pct > 0.0 else "bearish"


def _gift_tile(quote: GiftNiftyQuote | None) -> BriefingTile:
    if quote is None:
        return BriefingTile(label="GIFT Nifty", value="—", change_pct=0.0, sentiment="neutral")
    return BriefingTile(
        label="GIFT Nifty",
        value=_format_value(quote.last_price),
        change_pct=quote.change_pct or 0.0,
        sentiment=_sentiment_for_pct(quote.change_pct),
    )


def _row_by_ticker(rows: Iterable, ticker: str):
    for r in rows:
        if getattr(r, "ticker", None) == ticker:
            return r
    return None


def build_briefing(
    *,
    as_of: datetime,
    context,
    gift_nifty: GiftNiftyQuote | None,
) -> BriefingStrip:
    rows = list(getattr(context, "signals", []) or [])

    tiles: list[BriefingTile] = []
    for label, ticker in _TILE_ORDER:
        if label == "GIFT Nifty":
            tiles.append(_gift_tile(gift_nifty))
            continue

        row = _row_by_ticker(rows, ticker)
        if row is None:
            tiles.append(BriefingTile(label=label, value="—", change_pct=0.0, sentiment="neutral"))
            continue

        pct = getattr(row, "pct_1d", None)
        tiles.append(
            BriefingTile(
                label=label,
                value=_format_value(getattr(row, "price", None)),
                change_pct=pct if pct is not None else 0.0,
                sentiment=_sentiment_for_pct(pct),
            )
        )

    if gift_nifty is not None and gift_nifty.change_pct is not None:
        gift_bps = gift_nifty.change_pct * 100.0  # 0.60% → 60 bps
        gap = {"NIFTY": 0.5 * gift_bps, "BANKNIFTY": 0.6 * gift_bps}
    else:
        gap = {"NIFTY": 0.0, "BANKNIFTY": 0.0}

    return BriefingStrip(as_of=as_of, tiles=tiles, predicted_gap_bps=gap)
