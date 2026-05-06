"""OptionsChainSnapshot loader — reads `option_chain` table for a given (index, expiry).

Returns `None` (not raise) when no rows match; the sizer treats that as a
DATA_GAP and the orchestrator emits a NO_TRADE card with the appropriate
reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from india_quant.data.models import OptionChain, PriceData


@dataclass(frozen=True)
class OptionsChainRow:
    strike: float
    option_type: str  # "CE" | "PE"
    last_price: float
    bid: float | None
    ask: float | None
    iv: float | None
    oi: float | None


@dataclass(frozen=True)
class OptionsChainSnapshot:
    index: str
    as_of: datetime
    expiry: date
    underlying_spot: float
    chain: list[OptionsChainRow]


# Map index name → underlying spot ticker in the price_data table.
_SPOT_TICKER = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}


def _load_spot(session, index: str, as_of: datetime) -> float | None:
    ticker = _SPOT_TICKER.get(index)
    if ticker is None:
        return None
    row = session.execute(
        select(PriceData)
        .where(
            PriceData.ticker == ticker,
            PriceData.interval == "1d",
            PriceData.datetime <= as_of,
        )
        .order_by(PriceData.datetime.desc())
        .limit(1)
    ).scalar_one_or_none()
    return float(row.close) if row is not None else None


def load_chain_snapshot(
    index: str,
    as_of: datetime,
    expiry: date,
    session_factory: Callable | sessionmaker,
) -> Optional[OptionsChainSnapshot]:
    """Load the latest available chain snapshot for (index, expiry).

    Trade-date used: the most recent `trade_date <= as_of.date()` that has rows
    for the requested (underlying, expiry).
    """
    with session_factory() as s:
        latest_trade_date = s.execute(
            select(OptionChain.trade_date)
            .where(
                OptionChain.underlying == index,
                OptionChain.expiry == expiry,
                OptionChain.trade_date <= as_of.date(),
            )
            .order_by(OptionChain.trade_date.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest_trade_date is None:
            return None

        rows = s.execute(
            select(OptionChain).where(
                OptionChain.underlying == index,
                OptionChain.expiry == expiry,
                OptionChain.trade_date == latest_trade_date,
            )
        ).scalars().all()

        if not rows:
            return None

        chain_rows = [
            OptionsChainRow(
                strike=float(r.strike),
                option_type=r.option_type,
                last_price=float(r.last_price) if r.last_price is not None else 0.0,
                bid=float(r.bid) if r.bid is not None else None,
                ask=float(r.ask) if r.ask is not None else None,
                iv=float(r.iv) if r.iv is not None else None,
                oi=float(r.open_interest) if r.open_interest is not None else None,
            )
            for r in rows
        ]
        chain_rows.sort(key=lambda r: (r.strike, r.option_type))

        spot = _load_spot(s, index, as_of)
        if spot is None:
            # Fallback: ATM-implied spot from the strike with smallest CE−PE last_price gap.
            by_strike: dict[float, dict[str, float]] = {}
            for r in chain_rows:
                by_strike.setdefault(r.strike, {})[r.option_type] = r.last_price
            paired = [(k, abs(v.get("CE", 0) - v.get("PE", 0))) for k, v in by_strike.items()
                      if "CE" in v and "PE" in v]
            spot = min(paired, key=lambda kv: kv[1])[0] if paired else chain_rows[0].strike

        return OptionsChainSnapshot(
            index=index,
            as_of=as_of,
            expiry=expiry,
            underlying_spot=float(spot),
            chain=chain_rows,
        )
