"""Signal validation: Harvey-Liu-Zhu gate + McLean-Pontiff IC decay monitor."""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ttest_1samp


@dataclass
class ValidationReport:
    strategy_name: str
    as_of_date: str
    hlz_pass: bool
    ic_tstat: float
    sharpe: float
    max_drawdown: float
    information_ratio: float
    ic_decay_pct: float | None
    ic_decayed: bool
    overall_pass: bool
    notes: list[str]


def harvey_liu_zhu_gate(factor_returns: pd.Series, min_tstat: float = 3.0) -> bool:
    """
    Harvey, Liu & Zhu (2016, RFS): t-stat of mean monthly factor return.
    t = mean / (std / sqrt(n))
    Only approve if t > min_tstat (default 3.0).
    """
    clean = factor_returns.dropna()
    if len(clean) < 12:
        logger.warning("[HLZ] Insufficient data (< 12 months). Gate FAILED.")
        return False
    t_stat, p_value = ttest_1samp(clean, 0)
    t_abs = abs(t_stat)
    passed = t_abs > min_tstat
    logger.info(f"[HLZ] t-stat={t_abs:.3f} (threshold={min_tstat}) → {'PASS' if passed else 'FAIL'}")
    return passed


def mclean_pontiff_monitor(factor_name: str, ic_history: pd.Series) -> dict:
    """
    McLean & Pontiff (2016, JF): check if IC has decayed > 30% vs first-year IC.
    Returns dict with decay_pct and alert string.
    """
    if len(ic_history) < 24:
        return {"decayed": False, "decay_pct": None, "alert": "Insufficient history for decay check"}

    first_year_ic = float(ic_history.iloc[:12].mean())
    recent_ic = float(ic_history.iloc[-12:].mean())

    if abs(first_year_ic) < 1e-6:
        return {"decayed": False, "decay_pct": None, "alert": "First-year IC near zero — cannot compute decay"}

    decay_pct = (first_year_ic - recent_ic) / abs(first_year_ic) * 100
    decayed = decay_pct > 30

    alert = ""
    if decayed:
        alert = (
            f"[McLean-Pontiff] ALERT: {factor_name} IC has decayed {decay_pct:.1f}% "
            f"(first-year: {first_year_ic:.4f}, recent: {recent_ic:.4f})"
        )
        logger.warning(alert)
    else:
        logger.info(f"[McLean-Pontiff] {factor_name}: IC decay {decay_pct:.1f}% — OK")

    return {"decayed": decayed, "decay_pct": decay_pct, "alert": alert}


def run_full_validation(
    strategy_name: str,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    start: str,
    end: str,
    ic_history: pd.Series = None,
    factor_name: str = "model",
) -> ValidationReport:
    """
    Full validation pipeline:
    1. Walk-forward backtest
    2. HLZ gate on Sharpe
    3. Compare to NIFTY 50 benchmark (information ratio)
    4. Stability: rolling 6-month Sharpe, flag if any period < 0
    5. Pass criteria: Sharpe > 1.5, MaxDD < 20%, HLZ passed
    """
    from india_quant.backtest.engine import IndiaBacktestEngine
    engine = IndiaBacktestEngine()

    logger.info(f"[Validation] Running full validation for {strategy_name}")

    result = engine.run_backtest(signals, prices, start, end, strategy_name=strategy_name)

    # 1. HLZ gate on Sharpe
    returns = result.portfolio_returns or pd.Series(dtype=float)
    if not returns.empty:
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) if hasattr(returns, 'resample') else returns
        hlz_pass = harvey_liu_zhu_gate(monthly)
    else:
        hlz_pass = False

    # 2. Benchmark: NIFTY 50 returns for same period
    benchmark_sharpe = _get_nifty_sharpe(start, end)
    ir = (result.sharpe - benchmark_sharpe) / max(returns.std() * np.sqrt(252), 0.01) if not returns.empty else 0.0

    # 3. McLean-Pontiff decay
    decay_result = {"decayed": False, "decay_pct": None}
    if ic_history is not None:
        decay_result = mclean_pontiff_monitor(factor_name, ic_history)

    # 4. Stability check
    notes = []
    if not returns.empty and len(returns) > 126:
        rolling_sharpe = returns.rolling(126).apply(
            lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252)
        ).dropna()
        if (rolling_sharpe < 0).any():
            n_bad = int((rolling_sharpe < 0).sum())
            notes.append(f"Rolling 6-month Sharpe went negative in {n_bad} periods — review stability")

    # 5. Overall pass criteria
    sharpe_pass = result.sharpe > 1.5
    dd_pass = abs(result.max_drawdown) < 0.20
    overall_pass = hlz_pass and sharpe_pass and dd_pass and not decay_result.get("decayed", False)

    if not sharpe_pass:
        notes.append(f"Sharpe {result.sharpe:.2f} < 1.5 threshold")
    if not dd_pass:
        notes.append(f"Max drawdown {result.max_drawdown:.1%} exceeds 20% limit")
    if not hlz_pass:
        notes.append("Harvey-Liu-Zhu t-stat < 3.0 — factor may be false discovery")

    report = ValidationReport(
        strategy_name=strategy_name,
        as_of_date=date.today().isoformat(),
        hlz_pass=hlz_pass,
        ic_tstat=0.0,
        sharpe=result.sharpe,
        max_drawdown=result.max_drawdown,
        information_ratio=ir,
        ic_decay_pct=decay_result.get("decay_pct"),
        ic_decayed=decay_result.get("decayed", False),
        overall_pass=overall_pass,
        notes=notes,
    )

    status = "✅ PASS" if overall_pass else "❌ FAIL"
    logger.info(
        f"[Validation] {strategy_name}: {status} | "
        f"Sharpe={result.sharpe:.2f}, MaxDD={result.max_drawdown:.1%}, "
        f"HLZ={'pass' if hlz_pass else 'fail'}"
    )
    return report


def _get_nifty_sharpe(start: str, end: str) -> float:
    """NIFTY 50 Sharpe ratio for the same period as benchmark."""
    try:
        import yfinance as yf
        data = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            return 0.0
        rets = data["Close"].pct_change().dropna()
        rbi_rf = 0.065 / 252
        excess = rets - rbi_rf
        return float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(252))
    except Exception:
        return 0.0
