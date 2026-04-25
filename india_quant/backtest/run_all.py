"""Run all candidate strategies and print a summary table."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from india_quant.backtest.portfolio import (
    PortfolioConfig, run_portfolio, nifty_benchmark,
)
from india_quant.backtest import strategies as strat


REPORT_DIR = Path(__file__).parent.parent.parent / "reports" / "backtest"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _row(r):
    return {
        "strategy": r.strategy_name,
        "trades": r.n_trades,
        "total_ret": f"{r.total_return*100:+.1f}%",
        "cagr": f"{r.cagr*100:+.1f}%",
        "sharpe": f"{r.sharpe:+.2f}",
        "sortino": f"{r.sortino:+.2f}",
        "max_dd": f"{r.max_drawdown*100:.1f}%",
        "calmar": f"{r.calmar:+.2f}",
        "hit": f"{r.hit_rate*100:.1f}%" if r.hit_rate else "—",
        "wl_ratio": f"{r.win_loss_ratio:.2f}" if r.win_loss_ratio else "—",
    }


def run(start: str = "2020-01-01", end: str = "2026-04-23") -> pd.DataFrame:
    cfgs = [
        # Long-short variants
        ("Momentum 12-1 LS · monthly hold",
         strat.momentum_12_1(n=5),
         PortfolioConfig(name="Mom 12-1 LS", n_long=5, n_short=5, hold_days=21)),
        ("Momentum 12-1 LS · quarterly hold",
         strat.momentum_12_1(n=5),
         PortfolioConfig(name="Mom 12-1 LS-q", n_long=5, n_short=5, hold_days=63)),
        ("ST Reversal LS · weekly",
         strat.short_term_reversal(n=5),
         PortfolioConfig(name="ST Rev LS", n_long=5, n_short=5, hold_days=5)),
        ("ML signal LS · weekly",
         strat.ml_signal(n=5, contrarian=False),
         PortfolioConfig(name="ML LS", n_long=5, n_short=5, hold_days=5)),
        ("ML contrarian LS · weekly",
         strat.ml_signal(n=5, contrarian=True),
         PortfolioConfig(name="ML Contra LS", n_long=5, n_short=5, hold_days=5)),

        # Long-only variants (more realistic for retail)
        ("Momentum 12-1 long-only · monthly",
         strat.long_only(strat.momentum_12_1(n=5)),
         PortfolioConfig(name="Mom 12-1 LO", n_long=5, n_short=0, hold_days=21)),
        ("Momentum 12-1 long-only · quarterly",
         strat.long_only(strat.momentum_12_1(n=5)),
         PortfolioConfig(name="Mom 12-1 LO-q", n_long=5, n_short=0, hold_days=63)),
        ("Momentum 12-1 long-only · top 3 quarterly",
         strat.long_only(strat.momentum_12_1(n=3)),
         PortfolioConfig(name="Mom 12-1 LO-3q", n_long=3, n_short=0, hold_days=63)),
        ("Low-vol long-only · monthly",
         strat.long_only(strat.low_vol(n=5)),
         PortfolioConfig(name="Low Vol LO", n_long=5, n_short=0, hold_days=21)),
        ("ST Reversal long-only · weekly",
         strat.long_only(strat.short_term_reversal(n=5)),
         PortfolioConfig(name="ST Rev LO", n_long=5, n_short=0, hold_days=5)),
        ("ML long-only · weekly",
         strat.long_only(strat.ml_signal(n=5, contrarian=False)),
         PortfolioConfig(name="ML LO", n_long=5, n_short=0, hold_days=5)),
    ]

    logger.info(f"=== Backtest panel: {start} → {end} ===")
    rows = []
    for label, selector, cfg in cfgs:
        logger.info(f"--- {label} ---")
        try:
            r = run_portfolio(selector, cfg, start, end)
            rows.append(_row(r))
        except Exception as e:
            logger.error(f"{label}: {e}")
            rows.append({"strategy": cfg.name, "trades": 0, "total_ret": "ERR",
                         "cagr": "—", "sharpe": "—", "sortino": "—",
                         "max_dd": "—", "calmar": "—", "hit": "—", "wl_ratio": "—"})

    # Benchmark
    nifty = nifty_benchmark(start, end)
    rows.append(_row(nifty))

    df = pd.DataFrame(rows)
    out = REPORT_DIR / f"summary_{start}_{end}.json"
    out.write_text(df.to_json(orient="records", indent=2))
    logger.info(f"Summary saved to {out}")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-04-23")
    args = p.parse_args()
    df = run(args.start, args.end)
    print()
    print(df.to_string(index=False))
