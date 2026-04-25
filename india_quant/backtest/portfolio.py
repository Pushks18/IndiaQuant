"""Cross-sectional long/short portfolio backtester.

Models a real swing trader's reality:
- weekly rebalance (Friday close)
- equal-weight long + equal-weight short basket
- N-day hold (full Indian cost model on entry + exit)
- benchmark = NIFTY 50 buy-and-hold over same period

Returns the same `BacktestResult` shape as the existing engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from india_quant.backtest.engine import IndiaBacktestEngine, BacktestResult
from india_quant.data.db import get_session


@dataclass
class PortfolioConfig:
    initial_capital: float = 1_000_000          # ₹10 lakh
    n_long: int = 5                              # top names
    n_short: int = 5                             # bottom names (set 0 for long-only)
    rebalance_weekday: int = 4                   # 0=Mon ... 4=Fri
    hold_days: int = 5                           # bars between rebalance
    leverage: float = 1.0                        # 1.0 = no leverage; 100% gross
    trade_type: str = "equity_delivery"
    name: str = "strategy"


def _load_prices(start: str, end: str) -> pd.DataFrame:
    with get_session() as s:
        rows = s.execute(text("""
            SELECT ticker, datetime::date AS date, close FROM price_data
            WHERE interval = '1d' AND datetime BETWEEN :s AND :e
            ORDER BY ticker, datetime
        """), {"s": start, "e": end}).fetchall()
    df = pd.DataFrame(rows, columns=["ticker", "date", "close"])
    df["close"] = df["close"].astype(float)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _next_trading_day(dates: list, target: date) -> date | None:
    """First trading date >= target."""
    for d in dates:
        if d >= target:
            return d
    return None


def run_portfolio(
    selector: Callable[[date, pd.DataFrame], dict[str, list[str]]],
    cfg: PortfolioConfig,
    start: str,
    end: str,
) -> BacktestResult:
    """
    selector(rebalance_date, prices_so_far) -> {"long": [tickers], "short": [tickers]}
    The selector must NOT look beyond rebalance_date (no look-ahead).
    """
    prices = _load_prices(start, end)
    if prices.empty:
        return _empty(cfg, start, end)
    prices = prices.sort_values(["ticker", "date"])
    pivot = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    all_dates = list(pivot.index)

    engine = IndiaBacktestEngine()
    cost_pct = lambda v: engine.compute_transaction_cost(v, cfg.trade_type) / max(v, 1.0)

    capital = cfg.initial_capital
    equity_curve: list[tuple[date, float]] = []
    trades: list[dict] = []
    positions: dict[str, dict] = {}   # ticker → {direction, entry_px, value}

    # Rebalance cadence = hold_days (so positions are held the full N days, then rolled).
    # Snap to the requested weekday after that gap.
    fridays = [d for d in all_dates if d.weekday() == cfg.rebalance_weekday]
    if not fridays:
        return _empty(cfg, start, end)
    step = max(1, round(cfg.hold_days / 5))  # weeks between rebalances
    rebalance_dates = fridays[::step]

    for i, rd in enumerate(rebalance_dates):
        # Mark-to-market existing positions on rd's close, then close them
        if positions:
            for tkr, p in positions.items():
                if tkr not in pivot.columns:
                    continue
                exit_px = pivot.loc[rd, tkr]
                if pd.isna(exit_px):
                    # try previous day
                    prev = pivot.loc[:rd, tkr].dropna()
                    if prev.empty:
                        continue
                    exit_px = float(prev.iloc[-1])
                gross = (exit_px / p["entry_px"] - 1) * (1 if p["direction"] == "long" else -1)
                gross_pnl = p["value"] * gross
                # Round-trip cost on the position value
                exit_cost = engine.compute_transaction_cost(p["value"], cfg.trade_type)
                net = gross_pnl - exit_cost
                capital += net
                trades.append({
                    "entry_date": p["entry_date"], "exit_date": rd, "ticker": tkr,
                    "direction": p["direction"], "entry_px": p["entry_px"], "exit_px": float(exit_px),
                    "gross_ret": gross, "net_pnl": net, "value": p["value"],
                })
            positions = {}

        # Stop after final rebalance — no new positions
        if i >= len(rebalance_dates) - 1:
            equity_curve.append((rd, capital))
            break

        # Selector must use only history up to and including rd
        history = prices[prices["date"] <= rd]
        sel = selector(rd, history)
        longs = sel.get("long", [])[: cfg.n_long]
        shorts = sel.get("short", [])[: cfg.n_short]
        n = len(longs) + len(shorts)
        if n == 0:
            equity_curve.append((rd, capital))
            continue

        per_name_value = (capital * cfg.leverage) / n
        for tkr in longs + shorts:
            if tkr not in pivot.columns:
                continue
            entry_px = pivot.loc[rd, tkr]
            if pd.isna(entry_px):
                continue
            positions[tkr] = {
                "direction": "long" if tkr in longs else "short",
                "entry_px": float(entry_px),
                "entry_date": rd,
                "value": per_name_value,
            }
        equity_curve.append((rd, capital))

    if len(equity_curve) < 2:
        return _empty(cfg, start, end)

    eq_df = pd.DataFrame(equity_curve, columns=["date", "portfolio"]).set_index("date")["portfolio"]
    eq_df.index = pd.to_datetime(eq_df.index)
    returns = eq_df.pct_change().dropna()
    trades_df = pd.DataFrame(trades)

    return BacktestResult(
        strategy_name=cfg.name,
        start_date=start, end_date=end,
        initial_capital=cfg.initial_capital,
        final_capital=float(eq_df.iloc[-1]),
        total_return=float(eq_df.iloc[-1] / cfg.initial_capital - 1),
        cagr=engine._compute_cagr(cfg.initial_capital, float(eq_df.iloc[-1]), start, end),
        sharpe=engine._sharpe(returns),
        sortino=engine._sortino(returns),
        max_drawdown=float(engine._max_drawdown(eq_df)),
        max_drawdown_duration_days=engine._max_drawdown_duration(eq_df),
        hit_rate=float((trades_df["net_pnl"] > 0).mean()) if len(trades_df) else 0.0,
        win_loss_ratio=engine._win_loss_ratio(trades_df) if len(trades_df) else 0.0,
        expectancy=engine._expectancy(trades_df) if len(trades_df) else 0.0,
        calmar=engine._calmar(
            engine._compute_cagr(cfg.initial_capital, float(eq_df.iloc[-1]), start, end),
            float(engine._max_drawdown(eq_df)),
        ),
        n_trades=len(trades_df),
        portfolio_returns=returns,
    )


def _empty(cfg: PortfolioConfig, start: str, end: str) -> BacktestResult:
    return BacktestResult(
        strategy_name=cfg.name, start_date=start, end_date=end,
        initial_capital=cfg.initial_capital, final_capital=cfg.initial_capital,
        total_return=0, cagr=0, sharpe=0, sortino=0, max_drawdown=0,
        max_drawdown_duration_days=0, hit_rate=0, win_loss_ratio=0, expectancy=0,
        calmar=0, n_trades=0,
    )


def nifty_benchmark(start: str, end: str) -> BacktestResult:
    """NIFTY 50 buy-and-hold over the same period."""
    try:
        import yfinance as yf
        df = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return _empty(PortfolioConfig(name="NIFTY 50 buy-hold"), start, end)
        close = df["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rets = close.pct_change().dropna()
        engine = IndiaBacktestEngine()
        eq = (1 + rets).cumprod() * 1_000_000
        return BacktestResult(
            strategy_name="NIFTY 50 buy-hold",
            start_date=start, end_date=end,
            initial_capital=1_000_000,
            final_capital=float(eq.iloc[-1]),
            total_return=float(eq.iloc[-1] / 1_000_000 - 1),
            cagr=engine._compute_cagr(1_000_000, float(eq.iloc[-1]), start, end),
            sharpe=engine._sharpe(rets),
            sortino=engine._sortino(rets),
            max_drawdown=float(engine._max_drawdown(eq)),
            max_drawdown_duration_days=engine._max_drawdown_duration(eq),
            hit_rate=float((rets > 0).mean()),
            win_loss_ratio=0,
            expectancy=0,
            calmar=engine._calmar(
                engine._compute_cagr(1_000_000, float(eq.iloc[-1]), start, end),
                float(engine._max_drawdown(eq)),
            ),
            n_trades=0,
            portfolio_returns=rets,
        )
    except Exception as e:
        logger.error(f"NIFTY benchmark failed: {e}")
        return _empty(PortfolioConfig(name="NIFTY 50 buy-hold"), start, end)
