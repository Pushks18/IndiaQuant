"""
Run the global cross-market pipeline against today's data and print the
Nifty bias for the next trading session.

No database required — pulls live yfinance data, computes correlations,
classifies regime, and prints the time-ordered signal vector that the
playbook engine consumes.

Usage:
    python scripts/run_global.py
"""
from __future__ import annotations

from datetime import datetime, timedelta

from india_quant.signals.global_context import (
    get_global_context,
    time_ordered_signal_vector,
    instrument_levels,
    REFERENCE_ONLY,
)


def _next_session(now: datetime) -> str:
    nxt = now + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.strftime("%a %d-%b-%Y")


def main() -> None:
    ctx = get_global_context()
    next_open = _next_session(ctx.fetched_at)

    print("=" * 78)
    print(f"GLOBAL PIPELINE  |  fetched {ctx.fetched_at:%Y-%m-%d %H:%M IST}")
    print(f"Bias for next NSE session ({next_open} 09:15 IST open)")
    print("=" * 78)

    print(f"\nRegime         : {ctx.regime}")
    print(f"Drivers        : {', '.join(ctx.regime_drivers) or '—'}")
    print(f"Nifty 1d / 5d  : "
          f"{ctx.nifty_pct_1d:+.2f}% / {ctx.nifty_pct_5d:+.2f}%"
          if ctx.nifty_pct_1d is not None else "Nifty 1d / 5d  : n/a")
    print(f"USD/INR        : {ctx.usdinr:.2f}")
    print(f"\n>> {ctx.nifty_bias_text}\n")

    # Per-group breakdown
    by_group: dict[str, list] = {}
    for s in ctx.signals:
        by_group.setdefault(s.group, []).append(s)

    for group, rows in by_group.items():
        print(f"── {group} " + "─" * (72 - len(group)))
        print(f"  {'Ticker':<12}{'Label':<18}{'1d%':>8}{'5d%':>8}"
              f"{'corr30':>10}{'corr90':>10}  dir")
        for s in rows:
            p1 = f"{s.pct_1d:+.2f}" if s.pct_1d is not None else "  —  "
            p5 = f"{s.pct_5d:+.2f}" if s.pct_5d is not None else "  —  "
            c30 = f"{s.corr_30d:+.2f}" if s.corr_30d is not None else "  —  "
            c90 = f"{s.corr_90d:+.2f}" if s.corr_90d is not None else "  —  "
            print(f"  {s.ticker:<12}{s.label:<18}{p1:>8}{p5:>8}"
                  f"{c30:>10}{c90:>10}  {s.direction}")
        print()

    # Time-ordered weighted vector consumed by the KNN playbook engine
    vec = time_ordered_signal_vector(ctx)
    print("── Time-ordered signal vector (playbook KNN input) " + "─" * 25)
    for k in ("sp_pct_1d", "nikkei_pct_1d", "dxy_pct_1d",
              "usdinr_pct_1d", "vix_price", "crude_pct_1d"):
        v = vec.get(k)
        print(f"  {k:<16}: {v:+.3f}" if isinstance(v, (int, float)) else f"  {k:<16}: n/a")
    print(f"\n  Group-weighted contributions (positive = bullish for Nifty):")
    for g, w in vec["group_weighted"].items():
        print(f"    {g:<12}: {w:+.4f}")
    print(f"  weighted_sum    : {vec['weighted_sum']:+.4f}")
    print()

    # Top tradeable global instruments by |1d move|
    tradeables = [s for s in ctx.signals
                  if s.ticker not in REFERENCE_ONLY and s.pct_1d is not None]
    tradeables.sort(key=lambda s: abs(s.pct_1d), reverse=True)
    print("── Top 5 global movers w/ trade levels (1L capital, 1% risk) " +
          "─" * 14)
    for s in tradeables[:5]:
        lvl = instrument_levels(s, ctx.usdinr, capital=100_000, risk_pct=0.01)
        if not lvl:
            continue
        print(f"  {s.label:<18} {lvl['side']:<5} "
              f"entry {lvl['entry']}  stop {lvl['stop']}  "
              f"t1 {lvl['t1']} (RR {lvl['rr1']})  "
              f"qty {lvl['qty']}  margin ₹{lvl['margin_inr']:.0f}")
    print()


if __name__ == "__main__":
    main()
