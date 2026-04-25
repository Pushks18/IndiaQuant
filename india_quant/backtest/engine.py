"""Backtest engine with full Indian cost model (STT, GST, STT, slippage)."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "backtest"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration_days: int
    hit_rate: float
    win_loss_ratio: float
    expectancy: float
    calmar: float
    n_trades: int
    portfolio_returns: pd.Series = None


class IndiaBacktestEngine:
    """
    Backtest engine with accurate Indian market cost model.
    Reference: NSE circular on STT, GST, SEBI charges.
    """

    COST_MODEL = {
        "equity_delivery": {
            "stt_buy": 0.001,       # 0.1% on buy
            "stt_sell": 0.001,      # 0.1% on sell
            "exchange_nse": 0.0000325,
            "sebi": 0.000001,
            "gst_on_charges_rate": 0.18,
            "brokerage_flat_rs": 20,
            "slippage_liquid": 0.0005,
            "slippage_midcap": 0.0015,
        },
        "equity_intraday": {
            "stt_sell_only": 0.00025,
            "exchange_nse": 0.0000325,
            "sebi": 0.000001,
            "gst_on_charges_rate": 0.18,
            "brokerage_flat_rs": 20,
        },
        "futures": {
            "stt_sell": 0.0001,
            "exchange_nfo": 0.0000190,
            "sebi": 0.000001,
            "gst_on_charges_rate": 0.18,
            "brokerage_flat_rs": 20,
        },
        "options_buy": {
            "stt_sell": 0.0005,
            "exchange_nfo": 0.00053,
            "sebi": 0.000001,
            "gst_on_charges_rate": 0.18,
            "brokerage_flat_rs": 20,
        },
    }

    def compute_transaction_cost(
        self, trade_value: float, trade_type: str = "equity_delivery", is_liquid: bool = True
    ) -> float:
        """Returns total round-trip transaction cost in INR."""
        cm = self.COST_MODEL.get(trade_type, self.COST_MODEL["equity_delivery"])
        cost = 0.0

        if trade_type == "equity_delivery":
            stt = trade_value * (cm["stt_buy"] + cm["stt_sell"])
            exchange = trade_value * cm["exchange_nse"] * 2
            sebi = trade_value * cm["sebi"] * 2
            slippage = trade_value * (cm["slippage_liquid"] if is_liquid else cm["slippage_midcap"]) * 2
            brokerage = cm["brokerage_flat_rs"] * 2
            charges = exchange + sebi + brokerage
            gst = charges * cm["gst_on_charges_rate"]
            cost = stt + exchange + sebi + brokerage + gst + slippage

        elif trade_type == "equity_intraday":
            stt = trade_value * cm["stt_sell_only"]
            exchange = trade_value * cm["exchange_nse"] * 2
            sebi = trade_value * cm["sebi"] * 2
            brokerage = cm["brokerage_flat_rs"] * 2
            charges = exchange + sebi + brokerage
            gst = charges * cm["gst_on_charges_rate"]
            cost = stt + exchange + sebi + brokerage + gst

        elif trade_type == "futures":
            stt = trade_value * cm["stt_sell"]
            exchange = trade_value * cm["exchange_nfo"] * 2
            sebi = trade_value * cm["sebi"] * 2
            brokerage = cm["brokerage_flat_rs"] * 2
            charges = exchange + sebi + brokerage
            gst = charges * cm["gst_on_charges_rate"]
            cost = stt + exchange + sebi + brokerage + gst

        elif trade_type == "options_buy":
            stt = trade_value * cm["stt_sell"]
            exchange = trade_value * cm["exchange_nfo"] * 2
            sebi = trade_value * cm["sebi"] * 2
            brokerage = cm["brokerage_flat_rs"] * 2
            charges = exchange + sebi + brokerage
            gst = charges * cm["gst_on_charges_rate"]
            cost = stt + exchange + sebi + brokerage + gst

        return cost

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000,
        trade_type: str = "equity_delivery",
        strategy_name: str = "India Quant",
    ) -> BacktestResult:
        """
        signals: columns [ticker, date, direction, size_pct]
        prices: columns [ticker, datetime, close]
        Execute at next bar open. Apply full Indian cost model.
        """
        logger.info(f"[Backtest] Running {strategy_name} from {start_date} to {end_date}")

        portfolio = initial_capital
        portfolio_history = []
        trade_log = []

        signals = signals.sort_values("date")
        prices_idx = prices.set_index(["ticker", "datetime"])["close"]

        dates = sorted(signals["date"].unique())
        for i, signal_date in enumerate(dates):
            if signal_date < start_date or signal_date > end_date:
                continue

            day_signals = signals[signals["date"] == signal_date]
            day_portfolio = portfolio

            for _, sig in day_signals.iterrows():
                ticker = sig["ticker"]
                direction = sig["direction"]
                size_pct = sig["size_pct"]

                # Entry: next available price after signal
                entry_prices = prices_idx.get(ticker)
                if entry_prices is None:
                    continue

                future_prices = entry_prices[entry_prices.index > pd.Timestamp(signal_date)]
                if len(future_prices) < 2:
                    continue

                entry_px = float(future_prices.iloc[0])   # buy next open (approx = close)
                exit_px = float(future_prices.iloc[1])    # hold 1 day

                trade_value = portfolio * size_pct
                cost = self.compute_transaction_cost(trade_value, trade_type)

                gross_ret = (exit_px / entry_px - 1) * (1 if direction == "long" else -1)
                net_pnl = trade_value * gross_ret - cost
                portfolio += net_pnl

                trade_log.append({
                    "date": signal_date,
                    "ticker": ticker,
                    "direction": direction,
                    "entry": entry_px,
                    "exit": exit_px,
                    "gross_ret": gross_ret,
                    "net_pnl": net_pnl,
                    "cost": cost,
                })

            portfolio_history.append({"date": signal_date, "portfolio": portfolio})

        if not portfolio_history:
            logger.warning("[Backtest] No trades executed.")
            return self._empty_result(strategy_name, start_date, end_date, initial_capital)

        port_df = pd.DataFrame(portfolio_history).set_index("date")["portfolio"]
        returns = port_df.pct_change().dropna()
        trades_df = pd.DataFrame(trade_log)

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=float(port_df.iloc[-1]),
            total_return=float(port_df.iloc[-1] / initial_capital - 1),
            cagr=self._compute_cagr(initial_capital, float(port_df.iloc[-1]), start_date, end_date),
            sharpe=self._sharpe(returns),
            sortino=self._sortino(returns),
            max_drawdown=float(self._max_drawdown(port_df)),
            max_drawdown_duration_days=self._max_drawdown_duration(port_df),
            hit_rate=float((trades_df["net_pnl"] > 0).mean()) if len(trades_df) else 0.0,
            win_loss_ratio=self._win_loss_ratio(trades_df),
            expectancy=self._expectancy(trades_df),
            calmar=self._calmar(self._compute_cagr(initial_capital, float(port_df.iloc[-1]), start_date, end_date),
                                float(self._max_drawdown(port_df))),
            n_trades=len(trades_df),
            portfolio_returns=returns,
        )

    def compute_metrics(self, portfolio_returns: pd.Series) -> dict:
        rbi_rf = 0.065 / 252  # 6.5% RBI rate daily
        excess = portfolio_returns - rbi_rf
        sharpe = float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(252))
        dd = self._max_drawdown(portfolio_returns.add(1).cumprod())
        return {
            "sharpe": sharpe,
            "sortino": self._sortino(portfolio_returns),
            "max_drawdown": float(dd),
            "hit_rate": float((portfolio_returns > 0).mean()),
        }

    def walk_forward_backtest(
        self, strategy_fn, start_date: str, end_date: str,
        train_window: int = 504, test_window: int = 63
    ) -> list[BacktestResult]:
        """Roll walk-forward windows and collect results."""
        from india_quant.data.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT ticker, datetime, close FROM price_data
                    WHERE interval='1d' AND datetime BETWEEN :s AND :e
                    ORDER BY ticker, datetime
                """),
                {"s": start_date, "e": end_date},
            ).fetchall()

        prices = pd.DataFrame(rows, columns=["ticker", "datetime", "close"])
        dates = sorted(prices["datetime"].dt.date.unique())
        results = []
        i = 0
        while i + train_window + test_window <= len(dates):
            train_end = dates[i + train_window - 1]
            test_start = dates[i + train_window]
            test_end = dates[i + train_window + test_window - 1]

            signals = strategy_fn(
                train_start=dates[i].isoformat(),
                train_end=train_end.isoformat(),
            )
            if signals is not None and not signals.empty:
                result = self.run_backtest(
                    signals, prices, test_start.isoformat(), test_end.isoformat()
                )
                results.append(result)

            i += test_window

        return results

    def plot_results(self, result: BacktestResult):
        """Generate equity curve and drawdown chart as HTML."""
        try:
            import json
            data = result.portfolio_returns.reset_index()
            data.columns = ["date", "return"]
            report = {
                "strategy": result.strategy_name,
                "metrics": {
                    "total_return": f"{result.total_return:.2%}",
                    "sharpe": f"{result.sharpe:.2f}",
                    "max_drawdown": f"{result.max_drawdown:.2%}",
                    "n_trades": result.n_trades,
                }
            }
            path = REPORTS_DIR / f"{result.strategy_name.replace(' ', '_')}_report.json"
            path.write_text(json.dumps(report, default=str))
            logger.info(f"[Backtest] Results saved to {path}")
        except Exception as e:
            logger.error(f"Plot results failed: {e}")

    @staticmethod
    def _sharpe(returns: pd.Series, risk_free_daily: float = 0.065 / 252) -> float:
        excess = returns - risk_free_daily
        return float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(252))

    @staticmethod
    def _sortino(returns: pd.Series, risk_free_daily: float = 0.065 / 252) -> float:
        excess = returns - risk_free_daily
        downside = excess[excess < 0].std()
        return float(excess.mean() / (downside + 1e-8) * np.sqrt(252))

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        return float(dd.min())

    @staticmethod
    def _max_drawdown_duration(equity: pd.Series) -> int:
        roll_max = equity.cummax()
        in_dd = equity < roll_max
        if not in_dd.any():
            return 0
        groups = (in_dd != in_dd.shift()).cumsum()
        durations = in_dd.groupby(groups).sum()
        return int(durations.max())

    @staticmethod
    def _win_loss_ratio(trades: pd.DataFrame) -> float:
        if trades.empty:
            return 0.0
        wins = trades[trades["net_pnl"] > 0]["net_pnl"]
        losses = trades[trades["net_pnl"] < 0]["net_pnl"]
        avg_win = float(wins.mean()) if len(wins) else 0
        avg_loss = float(abs(losses.mean())) if len(losses) else 1
        return avg_win / max(avg_loss, 1e-8)

    @staticmethod
    def _expectancy(trades: pd.DataFrame) -> float:
        if trades.empty:
            return 0.0
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] < 0]
        win_rate = len(wins) / len(trades)
        avg_win = float(wins["net_pnl"].mean()) if len(wins) else 0
        avg_loss = float(losses["net_pnl"].mean()) if len(losses) else 0
        return win_rate * avg_win + (1 - win_rate) * avg_loss

    @staticmethod
    def _compute_cagr(initial: float, final: float, start: str, end: str) -> float:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
        years = (end_d - start_d).days / 365.25
        if years <= 0:
            return 0.0
        return float((final / initial) ** (1 / years) - 1)

    @staticmethod
    def _calmar(cagr: float, max_dd: float) -> float:
        return cagr / max(abs(max_dd), 1e-8)

    def _empty_result(self, name, start, end, capital) -> BacktestResult:
        return BacktestResult(
            strategy_name=name, start_date=start, end_date=end,
            initial_capital=capital, final_capital=capital,
            total_return=0.0, cagr=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown=0.0, max_drawdown_duration_days=0,
            hit_rate=0.0, win_loss_ratio=0.0, expectancy=0.0,
            calmar=0.0, n_trades=0,
        )
