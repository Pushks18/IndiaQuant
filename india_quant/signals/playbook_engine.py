"""Global Playbook Engine.

Converts cross-market signals into actionable, plain-English Nifty trade
calls in three time horizons (open / intraday / 5-day swing).

Core flow:
    1. time_ordered_signal_vector(ctx)   live snapshot
    2. knn_lookup(snapshot)              n similar past days from global_signals
    3. match_rules(snapshot)             curated rules from playbook_rules.yaml
    4. reconcile(knn, rules)             HIGH / MEDIUM / LOW (MIXED) conviction
    5. sector_tilt(rules)                sector ranking
    6. pick_stocks(top_sectors, horizon) top 3 F&O picks per favoured sector

Public entrypoint:
    generate_playbook(ctx, capital, risk_pct) -> PlaybookCall
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sqlalchemy import text

from india_quant.signals.global_context import (
    GlobalContext,
    SignalRow,
    time_ordered_signal_vector,
)


# ─── Tunables ────────────────────────────────────────────────────────────────

KNN_K = 8
KNN_MAX_DISTANCE = 3.0
KNN_MIN_VALID = 4

CONVICTION_HIGH_MOVE = 0.4   # |median open %| threshold
CONVICTION_HIGH_IQR  = 0.6   # IQR width threshold
CONVICTION_HIGH_HIT  = 0.6   # hit_rate_long threshold

CONVICTION_MED_MOVE  = 0.2
CONVICTION_MED_IQR   = 1.0

RULES_PATH = Path(__file__).parent / "playbook_rules.yaml"


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class KNNResult:
    n_similar:            int
    median_open_pct:      float
    iqr_low:              float
    iqr_high:             float
    hit_rate_long:        float
    representative_dates: list[date] = field(default_factory=list)


@dataclass
class MatchedRule:
    id:           str
    description:  str
    sector_tilt:  dict[str, float]
    template:    str
    rendered:    str


@dataclass
class StockPick:
    ticker:           str
    sector:           str
    bias:             str       # LONG / SHORT
    score:            float
    entry:            float
    stop:             float
    t1:               float
    t2:               float
    rr1:              float
    qty:              int
    margin_inr:       float
    max_loss_inr:     float
    profit1_inr:      float
    hit_rate_90d:     Optional[float] = None
    avg_return_90d:   Optional[float] = None
    n_trades_90d:     int = 0


@dataclass
class HorizonCall:
    horizon:        str             # "open" | "intraday" | "swing"
    description:    str
    stock_picks:    list[StockPick]


@dataclass
class PlaybookCall:
    fetched_at:     datetime
    snapshot:       dict
    knn:            KNNResult
    matched_rules:  list[MatchedRule]
    conviction:     str             # HIGH / MEDIUM / LOW
    rationale:      str             # plain-English summary
    sector_ranks:   list[tuple[str, float]]
    favoured:       list[str]       # top sectors (positive tilt)
    avoid:          list[str]       # bottom sectors (negative tilt)
    horizons:       dict[str, HorizonCall]
    expected_open_pct: float


# ─── Rule loading ────────────────────────────────────────────────────────────

_RULES_CACHE: list[dict] = []


def load_rules() -> list[dict]:
    global _RULES_CACHE
    if _RULES_CACHE:
        return _RULES_CACHE
    try:
        with open(RULES_PATH) as f:
            _RULES_CACHE = yaml.safe_load(f) or []
    except Exception as e:
        logger.warning(f"playbook_rules.yaml load failed: {e}")
        _RULES_CACHE = []
    return _RULES_CACHE


def _trigger_match(value: Optional[float], spec: dict) -> bool:
    if value is None:
        return False
    if "min" in spec and value < spec["min"]:
        return False
    if "max" in spec and value > spec["max"]:
        return False
    return True


def match_rules(snapshot: dict, ctx: Optional[GlobalContext] = None) -> list[MatchedRule]:
    """Return all rules whose triggers pass against the current snapshot."""
    rules = load_rules()
    matched: list[MatchedRule] = []

    # Extra signals derived from ctx for triggers like hsi_pct_1d, nasdaq_pct_1d
    extras: dict[str, Optional[float]] = {}
    if ctx is not None:
        by_ticker = {s.ticker: s for s in ctx.signals}
        for key, ticker in [
            ("hsi_pct_1d",    "^HSI"),
            ("nasdaq_pct_1d", "^IXIC"),
        ]:
            sig = by_ticker.get(ticker)
            extras[key] = sig.pct_1d if sig else None

    pool = {**snapshot, **extras}

    for rule in rules:
        ok = True
        for key, spec in rule.get("triggers", {}).items():
            if not _trigger_match(pool.get(key), spec):
                ok = False
                break
        if not ok:
            continue
        matched.append(MatchedRule(
            id=rule["id"],
            description=rule.get("description", ""),
            sector_tilt={k: float(v) for k, v in (rule.get("sector_tilt") or {}).items()},
            template=rule.get("template", "").strip(),
            rendered="",  # filled later when KNN is known
        ))
    return matched


def render_rule_templates(
    rules: list[MatchedRule],
    snapshot: dict,
    knn: KNNResult,
    extras: dict,
) -> list[MatchedRule]:
    fmt_args = {
        **{k: (v if v is not None else 0.0)
           for k, v in snapshot.items()
           if not isinstance(v, dict)},
        **extras,
        "median_open_pct": knn.median_open_pct,
        "iqr_low":         knn.iqr_low,
        "iqr_high":        knn.iqr_high,
        "n_similar":       knn.n_similar,
        "k":               KNN_K,
        "hit_rate_long":   knn.hit_rate_long,
    }
    for r in rules:
        try:
            r.rendered = r.template.format(**fmt_args)
        except (KeyError, ValueError):
            r.rendered = r.template
    return rules


# ─── KNN historical analog ────────────────────────────────────────────────────

KNN_FEATURES = [
    "sp_pct_1d", "nikkei_pct_1d", "dxy_pct_1d",
    "usdinr_pct_1d", "vix_price", "crude_pct_1d",
]

# Map snapshot key → (ticker, metric) in global_signals table
KNN_SOURCE = {
    "sp_pct_1d":     ("^GSPC",    "pct_1d"),
    "nikkei_pct_1d": ("^N225",    "pct_1d"),
    "dxy_pct_1d":    ("DX-Y.NYB", "pct_1d"),
    "usdinr_pct_1d": ("USDINR=X", "pct_1d"),
    "vix_price":     ("^VIX",     "pct_1d"),  # treat as level proxy: see _build_history
    "crude_pct_1d":  ("CL=F",     "pct_1d"),
}


def _build_history() -> Optional[pd.DataFrame]:
    """Return a wide DataFrame indexed by date, columns = KNN_FEATURES."""
    from india_quant.data.db import get_session
    try:
        with get_session() as session:
            rows = session.execute(text("""
                SELECT date, ticker, pct_1d
                FROM global_signals
                ORDER BY date, ticker
            """)).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["date", "ticker", "pct_1d"])
        wide = df.pivot(index="date", columns="ticker", values="pct_1d")

        # Rename ticker columns to KNN feature names
        feat = pd.DataFrame(index=wide.index)
        feat["sp_pct_1d"]     = wide.get("^GSPC")
        feat["nikkei_pct_1d"] = wide.get("^N225")
        feat["dxy_pct_1d"]    = wide.get("DX-Y.NYB")
        feat["usdinr_pct_1d"] = wide.get("USDINR=X")
        feat["vix_price"]     = wide.get("^VIX")  # pct_1d, used as volatility proxy
        feat["crude_pct_1d"]  = wide.get("CL=F")
        return feat.dropna(thresh=4)  # require ≥4 of 6 features per row
    except Exception as e:
        logger.warning(f"_build_history failed: {e}")
        return None


_NIFTY_GAP_CACHE: dict = {"data": None, "fetched_at": None}
_NIFTY_GAP_TTL_SECONDS = 6 * 3600  # 6 hours


def _nifty_gap_history() -> dict[date, float]:
    """Map historical date → next-day Nifty open gap %. Fetched via yfinance, cached 6h."""
    import time
    cached = _NIFTY_GAP_CACHE.get("data")
    fetched_at = _NIFTY_GAP_CACHE.get("fetched_at")
    if cached is not None and fetched_at is not None and (time.time() - fetched_at) < _NIFTY_GAP_TTL_SECONDS:
        return cached
    try:
        import yfinance as yf
        df = yf.download("^NSEI", period="2y", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "Close"]].dropna().copy()
        df.index = pd.to_datetime(df.index).date
        df["next_open"] = df["Open"].shift(-1)
        df["next_gap"] = (df["next_open"] / df["Close"] - 1) * 100
        m = {idx: float(row["next_gap"])
             for idx, row in df.iterrows()
             if pd.notna(row["next_gap"])}
        _NIFTY_GAP_CACHE["data"] = m
        _NIFTY_GAP_CACHE["fetched_at"] = time.time()
        return m
    except Exception as e:
        logger.warning(f"_nifty_gap_history failed: {e}")
        return {}


def _next_day_open_returns(history_dates: list[date]) -> dict[date, float]:
    """For each historical date, return next-day Nifty open gap %."""
    if not history_dates:
        return {}
    full = _nifty_gap_history()
    return {d: full[d] for d in history_dates if d in full}


def knn_lookup(snapshot: dict) -> KNNResult:
    """Find k nearest historical days; return Nifty open distribution."""
    hist = _build_history()
    if hist is None or len(hist) < 30:
        return KNNResult(0, 0.0, 0.0, 0.0, 0.0, [])

    # Today's vector
    today_vec = np.array([
        snapshot.get(f) if snapshot.get(f) is not None else hist[f].mean()
        for f in KNN_FEATURES
    ], dtype=float)

    # Z-score history (and today) using history mean/std
    means = hist[KNN_FEATURES].mean()
    stds  = hist[KNN_FEATURES].std().replace(0, 1.0)
    hist_z  = ((hist[KNN_FEATURES] - means) / stds).fillna(0.0).values
    today_z = (today_vec - means.values) / stds.values

    # Euclidean distance + filter
    dists = np.linalg.norm(hist_z - today_z, axis=1)
    order = np.argsort(dists)
    candidates = [
        (hist.index[i], float(dists[i]))
        for i in order
        if dists[i] <= KNN_MAX_DISTANCE
    ][:KNN_K]

    if len(candidates) < KNN_MIN_VALID:
        return KNNResult(len(candidates), 0.0, 0.0, 0.0, 0.0,
                         [d for d, _ in candidates])

    cand_dates = [d for d, _ in candidates]
    next_open = _next_day_open_returns(cand_dates)
    valid_dates  = [d for d in cand_dates if d in next_open]
    valid_values = [next_open[d] for d in valid_dates]

    if len(valid_values) < KNN_MIN_VALID:
        return KNNResult(len(valid_values), 0.0, 0.0, 0.0, 0.0, valid_dates)

    arr = np.asarray(valid_values, dtype=float)
    median = float(np.median(arr))
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    hit_rate = float((arr > 0).sum() / len(arr))
    return KNNResult(
        n_similar=len(valid_values),
        median_open_pct=round(median, 3),
        iqr_low=round(q1, 3),
        iqr_high=round(q3, 3),
        hit_rate_long=round(hit_rate, 3),
        representative_dates=valid_dates,
    )


# ─── Conviction tier + sector tilt ───────────────────────────────────────────

def reconcile(knn: KNNResult, matched: list[MatchedRule]) -> tuple[str, str]:
    """Return (conviction_tier, plain-English rationale)."""
    if knn.n_similar < KNN_MIN_VALID:
        return "LOW", (
            f"Insufficient historical analog ({knn.n_similar} similar days). "
            "No directional play; trade individual setups only."
        )

    iqr_width = knn.iqr_high - knn.iqr_low
    abs_move  = abs(knn.median_open_pct)

    if (
        abs_move    > CONVICTION_HIGH_MOVE and
        iqr_width   < CONVICTION_HIGH_IQR  and
        knn.hit_rate_long > CONVICTION_HIGH_HIT
    ):
        tier = "HIGH"
    elif (
        abs_move  > CONVICTION_MED_MOVE and
        iqr_width < CONVICTION_MED_IQR
    ):
        tier = "MEDIUM"
    else:
        tier = "LOW"

    if tier == "LOW":
        rationale = (
            f"Mixed signals — KNN expected open {knn.median_open_pct:+.2f}%, "
            f"IQR [{knn.iqr_low:+.2f}%, {knn.iqr_high:+.2f}%]. "
            "Conviction insufficient for a directional call; trade individual setups only."
        )
    else:
        # Use first matched rule's rendered template if present
        head = matched[0].rendered if matched else (
            f"KNN over {knn.n_similar} similar past days suggests Nifty opens "
            f"{knn.median_open_pct:+.2f}% (IQR {knn.iqr_low:+.2f}% to {knn.iqr_high:+.2f}%)."
        )
        size = "full" if tier == "HIGH" else "half"
        rationale = (
            f"{head}\n"
            f"{tier} conviction — suggested position size: {size}. "
            f"Hit rate {knn.hit_rate_long:.0%} across {knn.n_similar}/{KNN_K} analog days."
        )
    return tier, rationale


def sector_tilt(matched: list[MatchedRule]) -> list[tuple[str, float]]:
    totals: dict[str, float] = {}
    for r in matched:
        for sector, weight in r.sector_tilt.items():
            totals[sector] = totals.get(sector, 0.0) + weight
    return sorted(totals.items(), key=lambda kv: -kv[1])


# ─── Stock picker ────────────────────────────────────────────────────────────

# Multipliers per time horizon (stop / T1 / T2 expressed as ATR multiples)
HORIZON_PARAMS = {
    "open":     {"stop": 0.5, "t1": 1.0, "t2": 1.8, "label": "Next-day open (exit by 10:00 IST)"},
    "intraday": {"stop": 0.4, "t1": 1.5, "t2": 2.8, "label": "Intraday (5-min ORB style, exit 15:30 IST)"},
    "swing":    {"stop": 1.5, "t1": 3.0, "t2": 5.0, "label": "5-day swing (exit at +5d or T2)"},
}

_DAILY_BARS_CACHE: dict[str, list] = {}


def _bars_for(ticker: str) -> list[dict]:
    """Cache daily bars per ticker for the request lifetime."""
    if ticker in _DAILY_BARS_CACHE:
        return _DAILY_BARS_CACHE[ticker]
    from india_quant.signals.screener import _fetch_daily
    try:
        bars = _fetch_daily(ticker)
    except Exception:
        bars = []
    _DAILY_BARS_CACHE[ticker] = bars
    return bars


def _score_stock(ticker: str, ctx: GlobalContext) -> Optional[tuple[float, dict]]:
    """Compute long-side score + technical context for a single ticker."""
    from india_quant.signals.screener import (
        _wilder_atr, _wilder_adx, _rsi, _macd, _ema_stack_fn,
        _volume_surge, _momentum, _relative_strength, _prev_day_signal,
        _week52, _detect_regime, _score, ADX_PERIOD,
    )
    bars = _bars_for(ticker)
    if len(bars) < ADX_PERIOD * 2 + 10:
        return None

    prev_close = bars[-1]["close"]
    atr  = _wilder_atr(bars) or 0.0
    if atr <= 0:
        return None

    adx, pdi, ndi = _wilder_adx(bars)
    rsi_val = _rsi(bars)
    macd_l, macd_s, macd_h = _macd(bars)
    e9, e21, e50, stk = _ema_stack_fn(bars)
    vol_surge = _volume_surge(bars)
    mom5  = _momentum(bars, 5)
    mom20 = _momentum(bars, 20)
    nifty_5d = ctx.nifty_pct_5d or 0.0
    rs       = _relative_strength(bars, nifty_5d)
    pd_sig   = _prev_day_signal(bars)
    _w52h, _w52l, w52hp, w52lp = _week52(bars)
    regime   = _detect_regime(bars, adx)
    nifty_chg = ctx.nifty_pct_1d or 0.0

    sl, ss = _score(
        nifty_chg, stk, adx, pdi, ndi, rsi_val, macd_h,
        rs, pd_sig, None, mom5, w52hp, w52lp,
        vol_surge, None, prev_close, regime,
        regime_global="NEUTRAL", global_delta=0,
    )
    return sl, {
        "ss": ss, "atr": atr, "prev_close": prev_close,
        "rsi": rsi_val, "regime": regime,
    }


def _pick_for_sector(
    sector: str,
    bias: str,
    horizon: str,
    capital: float,
    risk_pct: float,
    ctx: GlobalContext,
    top_n: int = 3,
) -> list[StockPick]:
    from india_quant.data.fo_universe import tickers_in_sector
    from india_quant.signals.screener import _build_orb_levels, _position_size, INTRADAY_COST_PCT

    params = HORIZON_PARAMS[horizon]
    candidates: list[tuple[float, str, dict]] = []
    for ticker in tickers_in_sector(sector):
        result = _score_stock(ticker, ctx)
        if result is None:
            continue
        sl, info = result
        score = sl if bias == "LONG" else info["ss"]
        if score < 30:
            continue
        candidates.append((score, ticker, info))
    candidates.sort(reverse=True, key=lambda t: t[0])
    candidates = candidates[:top_n]

    picks: list[StockPick] = []
    for score, ticker, info in candidates:
        atr = info["atr"]
        prev_close = info["prev_close"]
        # ORB approximated as prev_high/prev_low; use prev_close ± buffer
        bars = _bars_for(ticker)
        prev_high = bars[-1]["high"]
        prev_low  = bars[-1]["low"]

        # Build levels with horizon-specific multipliers
        if bias == "LONG":
            entry = round(prev_high + prev_high * 0.0005, 2)
            stop  = round(entry - atr * params["stop"], 2)
            t1    = round(entry + atr * params["t1"], 2)
            t2    = round(entry + atr * params["t2"], 2)
        else:
            entry = round(prev_low - prev_low * 0.0005, 2)
            stop  = round(entry + atr * params["stop"], 2)
            t1    = round(entry - atr * params["t1"], 2)
            t2    = round(entry - atr * params["t2"], 2)

        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            continue

        qty, margin_inr, max_loss = _position_size(
            capital, risk_pct, risk_per_unit, entry, vix=None,
        )
        rr1 = round(abs(t1 - entry) / risk_per_unit, 2)
        c = INTRADAY_COST_PCT
        max_loss_inr = round(qty * risk_per_unit + qty * entry * c, 0)
        profit1_inr  = round(qty * abs(t1 - entry) - qty * entry * c, 0)

        picks.append(StockPick(
            ticker=ticker, sector=sector, bias=bias, score=round(score, 1),
            entry=entry, stop=stop, t1=t1, t2=t2, rr1=rr1,
            qty=qty, margin_inr=float(margin_inr), max_loss_inr=max_loss_inr,
            profit1_inr=profit1_inr,
        ))
    return picks


# ─── Public entrypoint ───────────────────────────────────────────────────────

def generate_playbook(
    ctx: GlobalContext,
    capital: float = 200_000,
    risk_pct: float = 0.01,
) -> PlaybookCall:
    """End-to-end: snapshot → KNN → rules → conviction → sector tilt → stock picks."""
    _DAILY_BARS_CACHE.clear()  # fresh per request

    snap = time_ordered_signal_vector(ctx)
    knn  = knn_lookup(snap)

    by_ticker = {s.ticker: s for s in ctx.signals}
    extras = {
        "hsi_pct_1d":    (by_ticker["^HSI"].pct_1d  if "^HSI"  in by_ticker else 0.0) or 0.0,
        "nasdaq_pct_1d": (by_ticker["^IXIC"].pct_1d if "^IXIC" in by_ticker else 0.0) or 0.0,
    }

    matched = match_rules(snap, ctx)
    matched = render_rule_templates(matched, snap, knn, extras)
    tier, rationale = reconcile(knn, matched)
    sector_ranks = sector_tilt(matched)

    favoured = [s for s, w in sector_ranks if w > 0.3]
    avoid    = [s for s, w in sector_ranks if w < -0.3]

    horizons: dict[str, HorizonCall] = {}
    if tier in ("HIGH", "MEDIUM") and favoured:
        for h, params in HORIZON_PARAMS.items():
            picks: list[StockPick] = []
            for sector in favoured[:2]:
                picks.extend(_pick_for_sector(sector, "LONG", h, capital, risk_pct, ctx))
            horizons[h] = HorizonCall(horizon=h, description=params["label"], stock_picks=picks)
    else:
        for h, params in HORIZON_PARAMS.items():
            horizons[h] = HorizonCall(horizon=h, description=params["label"], stock_picks=[])

    return PlaybookCall(
        fetched_at=ctx.fetched_at,
        snapshot=snap,
        knn=knn,
        matched_rules=matched,
        conviction=tier,
        rationale=rationale,
        sector_ranks=sector_ranks,
        favoured=favoured,
        avoid=avoid,
        horizons=horizons,
        expected_open_pct=knn.median_open_pct,
    )
