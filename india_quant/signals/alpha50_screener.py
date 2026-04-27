"""Nifty Alpha 50 — pre-market intraday screener (v3).

Fixes vs v2.1:
  - No pre-market look-ahead: stage 1 uses ONLY completed bars up to T-1.
    Any bar dated today (IST) is dropped before indicator math.
  - Indicators are computed on ASCENDING bars (oldest -> newest).
  - Wilder-smoothed True-Range ATR (replaces simple H-L mean).
  - Proper Wilder ADX with smoothed +DI / -DI and DX.
  - Score-based side selection: independent score_long & score_short.
    Bias = side with higher score; NO TRADE if both below MIN_SIDE_SCORE.
    The fragile "verdict-flip" logic is gone.
  - top_n is a CAP, not a quota. If only 2 names qualify, 2 are returned.
  - Sector cap: max MAX_PER_SECTOR picks per sector to limit concentration.
  - Optional intraday confirmation (ORB + VWAP + Nifty alignment) via
    --confirm flag: validates picks against current 5-min bars before sizing.

Run standalone:
    python -m india_quant.signals.alpha50_screener --capital 200000 --risk 1.0 --top 8

Or import:
    from india_quant.signals.alpha50_screener import run_screener
    plans = run_screener(capital_inr=200_000, risk_per_trade_pct=0.01)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, date

import yfinance as yf
from loguru import logger
from sqlalchemy import text

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    IST = timezone.utc

from india_quant.data.db import get_session
from india_quant.data.fetchers.yfinance_fetcher import YFinanceFetcher

# ── Constants ──────────────────────────────────────────────────────────────────
MIS_LEVERAGE        = 5
INTRADAY_COST_PCT   = 0.0006
MIN_ATR_PCT         = 1.5
MAX_ATR_PCT         = 4.0
MAX_GAP_PCT         = 1.5
VIX_SKIP_THRESHOLD  = 20.0
VIX_HALF_SIZE       = 18.0
MIN_SIDE_SCORE      = 45      # below this, the side is rejected (NO TRADE)
MAX_PER_SECTOR      = 2       # concentration cap
TARGET_1_MULT       = 1.5
TARGET_2_MULT       = 2.5

SECTOR_ETFS = {
    "IT":     "^CNXIT",
    "BANK":   "^NSEBANK",
    "INFRA":  "^CNXINFRA",
    "PHARMA": "^CNXPHARMA",
    "REALTY": "^CNXREALTY",
    "ENERGY": "^CNXENERGY",
}

TICKER_SECTOR = {
    "PERSISTENT.NS": "IT",     "COFORGE.NS": "IT",       "MPHASIS.NS": "IT",
    "HDFCAMC.NS":    "BANK",   "SHRIRAMFIN.NS": "BANK",  "MUTHOOTFIN.NS": "BANK",
    "CHOLAFIN.NS":   "BANK",   "AUBANK.NS": "BANK",      "FEDERALBNK.NS": "BANK",
    "RBLBANK.NS":    "BANK",
    "LUPIN.NS":      "PHARMA", "MANKIND.NS": "PHARMA",   "NAVINFLUOR.NS": "PHARMA",
    "AUROPHARMA.NS": "PHARMA",
    "BEL.NS":        "INFRA",  "BHEL.NS": "INFRA",       "CUMMINSIND.NS": "INFRA",
    "CGPOWER.NS":    "INFRA",  "MAZDOCK.NS": "INFRA",    "BDL.NS": "INFRA",
    "POLYCAB.NS":    "INFRA",  "VOLTAS.NS": "INFRA",     "DIXON.NS": "INFRA",
    "KEI.NS":        "INFRA",  "HITACHIENGY.NS": "INFRA", "INDUSTOWER.NS": "INFRA",
    "PFC.NS":        "ENERGY", "RECLTD.NS": "ENERGY",    "OIL.NS": "ENERGY",
    "NMDC.NS":       "ENERGY", "SUZLON.NS": "ENERGY",    "JSWENERGY.NS": "ENERGY",
    "ADANIPOWER.NS": "ENERGY", "ADANIGREEN.NS": "ENERGY", "ADANIENT.NS": "ENERGY",
    "NHPC.NS":       "ENERGY", "TATAPOWER.NS": "ENERGY",
    "OBEROIRLTY.NS": "REALTY", "PRESTIGE.NS": "REALTY",  "DLF.NS": "REALTY",
    "GODREJPROP.NS": "REALTY",
    "TRENT.NS":      "OTHER",  "MARUTI.NS": "OTHER",     "VBL.NS": "OTHER",
    "FORTIS.NS":     "OTHER",  "MAXHEALTH.NS": "OTHER",  "ZOMATO.NS": "OTHER",
    "POLICYBZR.NS":  "OTHER",  "NYKAA.NS": "OTHER",      "DELHIVERY.NS": "OTHER",
    "INDIGO.NS":     "OTHER",  "CONCOR.NS": "OTHER",
}


# ── Output dataclass ───────────────────────────────────────────────────────────
@dataclass
class AlphaPlan:
    rank:               int
    ticker:             str
    score:              float        # winning side's score
    score_long:         float
    score_short:        float
    bias:               str          # LONG / SHORT / NO_TRADE
    prev_close:         float
    gap_pct:            float
    atr:                float
    atr_pct:            float
    ema9:               float | None
    ema21:              float | None
    ema50:              float | None
    ema_stack:          str
    rs_vs_nifty_5d:     float | None
    prev_day_signal:    str
    week52_high:        float | None
    week52_low:         float | None
    week52_high_pct:    float | None
    week52_low_pct:     float | None
    sector:             str
    sector_momentum:    float | None
    adx:                float | None
    plus_di:            float | None
    minus_di:           float | None
    volume_surge:       float
    momentum_20d:       float
    trigger:            float
    stop:               float
    target1:            float
    target2:            float
    rr1:                float
    rr2:                float
    qty:                int
    margin_inr:         float
    max_loss_inr:       float
    profit1_inr:        float
    profit2_inr:        float
    verdict:            str | None
    conviction:         int | None
    capital_inr:        float
    risk_per_trade_inr: float
    vix_half_size:      bool
    intraday_confirmed: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Market context ─────────────────────────────────────────────────────────────

def _get_market_context() -> dict:
    ctx = {"vix": None, "nifty_bias": "NEUTRAL", "nifty_5d_return": 0.0,
           "sector_returns": {}}
    try:
        vix_df   = yf.Ticker("^INDIAVIX").history(period="3d")
        nifty_df = yf.Ticker("^NSEI").history(period="10d")

        if not vix_df.empty:
            ctx["vix"] = float(vix_df["Close"].iloc[-1])

        if not nifty_df.empty and len(nifty_df) >= 2:
            n_cur  = float(nifty_df["Close"].iloc[-1])
            n_prev = float(nifty_df["Close"].iloc[-2])
            n_5d   = float(nifty_df["Close"].iloc[-6]) if len(nifty_df) >= 6 else n_prev
            chg    = (n_cur / n_prev - 1) * 100
            ctx["nifty_5d_return"] = round((n_cur / n_5d - 1) * 100, 2)
            if chg > 0.3:
                ctx["nifty_bias"] = "LONG"
            elif chg < -0.3:
                ctx["nifty_bias"] = "SHORT"

        for sector, etf in SECTOR_ETFS.items():
            try:
                df = yf.Ticker(etf).history(period="3d")
                if not df.empty and len(df) >= 2:
                    ret = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100
                    ctx["sector_returns"][sector] = round(ret, 2)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Market context fetch failed: {e}")
    return ctx


# ── Price data (ASCENDING, no look-ahead) ──────────────────────────────────────

def _price_rows(ticker: str, n: int = 260) -> list[dict]:
    """Fetch up to n daily bars in ASCENDING time order, dropping any partial
    bar dated today (IST). Stage-1 indicators must NEVER see today's bar."""
    with get_session() as s:
        rows = s.execute(text("""
            SELECT datetime, open, high, low, close, volume
            FROM price_data
            WHERE ticker = :t AND interval = '1d'
            ORDER BY datetime DESC LIMIT :n
        """), {"t": ticker, "n": n}).fetchall()

    today_ist = datetime.now(IST).date()
    bars: list[dict] = []
    for r in rows:
        dt = r[0]
        # Normalize to date in IST
        if isinstance(dt, datetime):
            d = dt.astimezone(IST).date() if dt.tzinfo else dt.date()
        elif isinstance(dt, date):
            d = dt
        else:
            d = today_ist  # unknown; treat as today and drop
        if d >= today_ist:
            continue  # drop today's (possibly partial) bar
        bars.append({
            "datetime": dt,
            "open":   float(r[1] or 0),
            "high":   float(r[2] or 0),
            "low":    float(r[3] or 0),
            "close":  float(r[4] or 0),
            "volume": float(r[5] or 0),
        })
    bars.reverse()  # ascending: bars[-1] is most recent completed bar (T-1)
    return bars


# ── Indicators (ascending input) ───────────────────────────────────────────────

def _ema_series(closes: list[float], period: int) -> list[float] | None:
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    out = [sum(closes[:period]) / period]
    for px in closes[period:]:
        out.append(px * k + out[-1] * (1 - k))
    return out


def _ema_stack(bars: list[dict]) -> tuple[float | None, float | None, float | None, str]:
    closes = [b["close"] for b in bars]
    e9s  = _ema_series(closes, 9)
    e21s = _ema_series(closes, 21)
    e50s = _ema_series(closes, 50)
    e9  = round(e9s[-1], 2)  if e9s  else None
    e21 = round(e21s[-1], 2) if e21s else None
    e50 = round(e50s[-1], 2) if e50s else None
    if e9 and e21 and e50:
        if e9 > e21 > e50:    stack = "bullish"
        elif e9 < e21 < e50:  stack = "bearish"
        else:                  stack = "mixed"
    else:
        stack = "mixed"
    return e9, e21, e50, stack


def _wilder_atr(bars: list[dict], period: int = 14) -> float | None:
    """True-Range ATR with Wilder smoothing. bars in ascending order."""
    if len(bars) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(bars)):
        h, l = bars[i]["high"], bars[i]["low"]
        pc   = bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _wilder_adx(bars: list[dict], period: int = 14) -> tuple[float | None, float | None, float | None]:
    """Returns (adx, +DI, -DI) using Wilder smoothing on ascending bars."""
    if len(bars) < period * 2 + 1:
        return None, None, None
    trs, plus_dms, minus_dms = [], [], []
    for i in range(1, len(bars)):
        h,  l  = bars[i]["high"],     bars[i]["low"]
        ph, pl = bars[i - 1]["high"], bars[i - 1]["low"]
        pc     = bars[i - 1]["close"]
        up   = h - ph
        down = pl - l
        plus_dm  = up   if (up > down and up > 0)   else 0.0
        minus_dm = down if (down > up and down > 0) else 0.0
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr); plus_dms.append(plus_dm); minus_dms.append(minus_dm)

    def wilder(seq: list[float]) -> list[float]:
        out = [sum(seq[:period])]
        for v in seq[period:]:
            out.append(out[-1] - out[-1] / period + v)
        return out

    tr_s    = wilder(trs)
    plus_s  = wilder(plus_dms)
    minus_s = wilder(minus_dms)

    dxs: list[float] = []
    plus_di_last = minus_di_last = None
    for atr_w, p_w, m_w in zip(tr_s, plus_s, minus_s):
        if atr_w == 0:
            continue
        plus_di  = 100 * p_w / atr_w
        minus_di = 100 * m_w / atr_w
        denom = plus_di + minus_di
        dxs.append(100 * abs(plus_di - minus_di) / denom if denom else 0.0)
        plus_di_last, minus_di_last = plus_di, minus_di

    if len(dxs) < period:
        return None, plus_di_last, minus_di_last
    adx = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        adx = (adx * (period - 1) + dx) / period
    return (
        round(adx, 1),
        round(plus_di_last, 1)  if plus_di_last  is not None else None,
        round(minus_di_last, 1) if minus_di_last is not None else None,
    )


def _compute_momentum(bars: list[dict], period: int = 20) -> float:
    if len(bars) < period + 1:
        return 0.0
    cur, prev = bars[-1]["close"], bars[-1 - period]["close"]
    return round((cur / prev - 1) * 100, 2) if prev else 0.0


def _volume_surge(bars: list[dict], avg_period: int = 20) -> float:
    """T-1 volume relative to prior 20-bar average. No today's volume used."""
    if len(bars) < avg_period + 1:
        return 1.0
    last_vol = bars[-1]["volume"]
    avg = sum(b["volume"] for b in bars[-1 - avg_period:-1]) / avg_period
    return round(last_vol / avg, 2) if avg > 0 else 1.0


def _relative_strength(bars: list[dict], nifty_5d_return: float, period: int = 5) -> float | None:
    if len(bars) < period + 1:
        return None
    cur, prev = bars[-1]["close"], bars[-1 - period]["close"]
    if prev == 0:
        return None
    return round((cur / prev - 1) * 100 - nifty_5d_return, 2)


def _prev_day_signal(bars: list[dict]) -> str:
    """T-1 close position relative to T-2 range."""
    if len(bars) < 2:
        return "inside"
    last  = bars[-1]
    prior = bars[-2]
    if last["close"] > prior["high"]: return "above_high"
    if last["close"] < prior["low"]:  return "below_low"
    return "inside"


def _week52(bars: list[dict]) -> tuple[float | None, float | None, float | None, float | None]:
    if len(bars) < 10:
        return None, None, None, None
    period = min(252, len(bars))
    window = bars[-period:]
    w52h   = max(b["high"] for b in window)
    w52l   = min(b["low"]  for b in window)
    cur    = bars[-1]["close"]
    return (round(w52h, 2), round(w52l, 2),
            round((cur / w52h - 1) * 100, 2),
            round((cur / w52l - 1) * 100, 2))


# ── Verdict (DB lookup) ────────────────────────────────────────────────────────

def _is_verdict_stale(ticker: str, max_days: int = 1) -> bool:
    with get_session() as s:
        row = s.execute(text("""
            SELECT created_at FROM debate_result
            WHERE ticker = :t ORDER BY created_at DESC LIMIT 1
        """), {"t": ticker}).fetchone()
    if not row:
        return True
    age = datetime.now(timezone.utc) - row[0].replace(tzinfo=timezone.utc)
    return age.days > max_days


def _latest_verdict(ticker: str) -> tuple[str | None, int | None]:
    if _is_verdict_stale(ticker, max_days=1):
        return None, None
    with get_session() as s:
        row = s.execute(text("""
            SELECT judge_verdict FROM debate_result
            WHERE ticker = :t ORDER BY created_at DESC LIMIT 1
        """), {"t": ticker}).fetchone()
    if row and row[0]:
        try:
            d = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            return d.get("verdict"), d.get("conviction")
        except Exception:
            pass
    return None, None


# ── Side-specific scoring ──────────────────────────────────────────────────────

def _score_side(
    side: str,                   # "LONG" or "SHORT"
    *,
    adx, plus_di, minus_di,
    volume_surge, momentum_20d, atr_pct,
    ema_stack, rs_vs_nifty, prev_day_signal,
    week52_high_pct, week52_low_pct,
    sector_momentum, nifty_bias,
    verdict, conviction,
) -> float:
    """
    Independent score for a side (LONG or SHORT). Each component awards points
    only when it favours `side`; an indicator that opposes the side adds zero
    (not negative) to keep the function additive and bounded.
    """
    s = 0.0
    is_long = side == "LONG"

    # 1. ADX trend strength + DI direction (20 pts)
    if adx is not None:
        di_aligned = (
            (is_long and plus_di and minus_di and plus_di > minus_di) or
            (not is_long and plus_di and minus_di and minus_di > plus_di)
        )
        base = 0
        if   adx >= 30: base = 20
        elif adx >= 25: base = 15
        elif adx >= 20: base = 10
        elif adx >= 15: base = 5
        s += base if di_aligned else base * 0.4

    # 2. Volume surge (15 pts) — neutral on direction
    if   volume_surge >= 2.5: s += 15
    elif volume_surge >= 2.0: s += 12
    elif volume_surge >= 1.5: s += 8
    elif volume_surge >= 1.2: s += 4

    # 3. EMA stack alignment (12 pts)
    if   is_long  and ema_stack == "bullish": s += 12
    elif not is_long and ema_stack == "bearish": s += 12
    elif ema_stack == "mixed": s += 3

    # 4. Relative strength vs Nifty (15 pts)
    if rs_vs_nifty is not None:
        if is_long:
            if   rs_vs_nifty >= 3: s += 15
            elif rs_vs_nifty >= 1: s += 10
            elif rs_vs_nifty >= 0: s += 4
        else:
            if   rs_vs_nifty <= -3: s += 15
            elif rs_vs_nifty <= -1: s += 10
            elif rs_vs_nifty <=  0: s += 4

    # 5. Previous day high/low break (15 pts)
    if   is_long  and prev_day_signal == "above_high": s += 15
    elif not is_long and prev_day_signal == "below_low": s += 15

    # 6. ATR sweet spot (8 pts) — neutral
    if MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT: s += 8
    elif atr_pct < MIN_ATR_PCT:               s += 2

    # 7. 52-week proximity (5 pts)
    if week52_high_pct is not None and week52_low_pct is not None:
        if is_long  and week52_high_pct >= -5: s += 5
        if not is_long and week52_low_pct  <= 10: s += 5

    # 8. Sector momentum (8 pts)
    if sector_momentum is not None:
        if is_long  and sector_momentum >  0.5: s += 8
        if not is_long and sector_momentum < -0.5: s += 8

    # 9. Nifty bias alignment (5 pts)
    if   is_long  and nifty_bias == "LONG":  s += 5
    elif not is_long and nifty_bias == "SHORT": s += 5

    # 10. 20-day momentum (5 pts) — lagging, low weight
    if   is_long and momentum_20d >  3: s += 5
    elif not is_long and momentum_20d < -3: s += 5

    # 11. LLM verdict alignment bonus (+7)
    if conviction and conviction >= 6:
        if (verdict == "Bullish" and is_long) or (verdict == "Bearish" and not is_long):
            s += 7

    return round(min(100.0, s), 1)


# ── Plan builder ───────────────────────────────────────────────────────────────

def _build_plan(
    ticker: str,
    bars: list[dict],
    ctx: dict,
    capital_inr: float,
    risk_per_trade_pct: float,
    target_multiple_1: float,
    target_multiple_2: float,
) -> AlphaPlan | None:
    if len(bars) < 30:
        return None

    prev_close = bars[-1]["close"]   # T-1 close — known pre-market
    if prev_close <= 0:
        return None

    vix        = ctx["vix"]
    nifty_bias = ctx["nifty_bias"]
    n5d_return = ctx.get("nifty_5d_return", 0.0)
    sector     = TICKER_SECTOR.get(ticker, "OTHER")
    sect_mom   = ctx["sector_returns"].get(sector)

    # Indicators
    atr = _wilder_atr(bars)
    if atr is None:
        return None
    atr_pct = round(atr / prev_close * 100, 2)

    adx, plus_di, minus_di = _wilder_adx(bars)
    momentum_20d = _compute_momentum(bars)
    vol_surge    = _volume_surge(bars)
    e9, e21, e50, ema_stk = _ema_stack(bars)
    rs           = _relative_strength(bars, n5d_return)
    pd_signal    = _prev_day_signal(bars)
    w52h, w52l, w52h_pct, w52l_pct = _week52(bars)

    prev2   = bars[-2]["close"] if len(bars) > 1 else prev_close
    gap_pct = round((prev_close / prev2 - 1) * 100, 2) if prev2 > 0 else 0.0

    # Hard filters (volatility / overnight gap on T-1 itself)
    if not (MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT):
        return None
    if abs(gap_pct) > MAX_GAP_PCT:
        return None

    verdict, conviction = _latest_verdict(ticker)

    common = dict(
        adx=adx, plus_di=plus_di, minus_di=minus_di,
        volume_surge=vol_surge, momentum_20d=momentum_20d, atr_pct=atr_pct,
        ema_stack=ema_stk, rs_vs_nifty=rs, prev_day_signal=pd_signal,
        week52_high_pct=w52h_pct, week52_low_pct=w52l_pct,
        sector_momentum=sect_mom, nifty_bias=nifty_bias,
        verdict=verdict, conviction=conviction,
    )
    score_long  = _score_side("LONG",  **common)
    score_short = _score_side("SHORT", **common)

    # Side selection
    if max(score_long, score_short) < MIN_SIDE_SCORE:
        return None
    if score_long >= score_short:
        bias, score = "LONG", score_long
    else:
        bias, score = "SHORT", score_short

    # ORB-style levels around T-1 close
    or_size = atr * 0.4
    or_high = prev_close + 0.5 * or_size
    or_low  = prev_close - 0.5 * or_size
    buf     = prev_close * 0.0005

    if bias == "LONG":
        trigger = round(or_high + buf, 2)
        stop    = round(or_low, 2)
        t1      = round(trigger + target_multiple_1 * or_size, 2)
        t2      = round(trigger + target_multiple_2 * or_size, 2)
        risk    = max(trigger - stop, 0.01)
        rew1    = max(t1 - trigger, 0.01)
        rew2    = max(t2 - trigger, 0.01)
    else:
        trigger = round(or_low - buf, 2)
        stop    = round(or_high, 2)
        t1      = round(trigger - target_multiple_1 * or_size, 2)
        t2      = round(trigger - target_multiple_2 * or_size, 2)
        risk    = max(stop - trigger, 0.01)
        rew1    = max(trigger - t1, 0.01)
        rew2    = max(trigger - t2, 0.01)

    # Position sizing
    half_size   = vix is not None and vix > VIX_HALF_SIZE
    size_factor = 0.5 if half_size else 1.0
    risk_inr    = capital_inr * risk_per_trade_pct * size_factor

    qty = max(1, int(risk_inr // risk))
    if qty * trigger / MIS_LEVERAGE > capital_inr * 0.9:
        qty = max(1, int((capital_inr * 0.9 * MIS_LEVERAGE) // trigger))
    pos_inr = qty * trigger
    cost    = INTRADAY_COST_PCT

    return AlphaPlan(
        rank=0, ticker=ticker, score=score,
        score_long=score_long, score_short=score_short,
        bias=bias,
        prev_close=round(prev_close, 2), gap_pct=gap_pct,
        atr=round(atr, 2), atr_pct=atr_pct,
        ema9=e9, ema21=e21, ema50=e50, ema_stack=ema_stk,
        rs_vs_nifty_5d=rs, prev_day_signal=pd_signal,
        week52_high=w52h, week52_low=w52l,
        week52_high_pct=w52h_pct, week52_low_pct=w52l_pct,
        sector=sector, sector_momentum=sect_mom,
        adx=adx, plus_di=plus_di, minus_di=minus_di,
        volume_surge=vol_surge, momentum_20d=momentum_20d,
        trigger=trigger, stop=stop, target1=t1, target2=t2,
        rr1=round(rew1 / risk, 2), rr2=round(rew2 / risk, 2),
        qty=qty,
        margin_inr=round(pos_inr / MIS_LEVERAGE, 0),
        max_loss_inr=round(qty * risk + pos_inr * cost, 0),
        profit1_inr=round(qty * rew1 - pos_inr * cost, 0),
        profit2_inr=round(qty * rew2 - pos_inr * cost, 0),
        verdict=verdict, conviction=conviction,
        capital_inr=capital_inr,
        risk_per_trade_inr=round(risk_inr, 0),
        vix_half_size=half_size,
    )


# ── Optional intraday confirmation (ORB + VWAP + Nifty alignment) ──────────────

def _intraday_confirm(plan: AlphaPlan, nifty_aligned_required: bool = True) -> bool:
    """Verify a pre-market plan against current 5-min intraday bars.

    Long  passes if: current price > VWAP AND price > opening-range high
                     (and optionally Nifty session change has same sign).
    Short is the symmetric check. Falls back to True (skip) if data unavailable.
    """
    try:
        df = yf.Ticker(plan.ticker).history(period="1d", interval="5m")
        if df.empty or len(df) < 3:
            return True  # no data — don't block
        opens   = df["Open"].astype(float).tolist()
        highs   = df["High"].astype(float).tolist()
        lows    = df["Low"].astype(float).tolist()
        closes  = df["Close"].astype(float).tolist()
        vols    = df["Volume"].astype(float).tolist()

        # Opening range = first 3 bars (~15 min)
        or_n  = min(3, len(df))
        or_hi = max(highs[:or_n])
        or_lo = min(lows[:or_n])

        # VWAP
        cum_pv = cum_v = 0.0
        for h, l, c, v in zip(highs, lows, closes, vols):
            tp = (h + l + c) / 3
            cum_pv += tp * v
            cum_v  += v
        vwap = cum_pv / cum_v if cum_v else closes[-1]
        last = closes[-1]

        if plan.bias == "LONG":
            ok = last > vwap and last > or_hi
        else:
            ok = last < vwap and last < or_lo

        if ok and nifty_aligned_required:
            try:
                ndf = yf.Ticker("^NSEI").history(period="1d", interval="5m")
                if not ndf.empty and len(ndf) >= 2:
                    n_open = float(ndf["Open"].iloc[0])
                    n_last = float(ndf["Close"].iloc[-1])
                    n_chg  = n_last - n_open
                    if plan.bias == "LONG"  and n_chg < 0: ok = False
                    if plan.bias == "SHORT" and n_chg > 0: ok = False
            except Exception:
                pass
        return ok
    except Exception as e:
        logger.warning(f"[Confirm] {plan.ticker} intraday fetch failed: {e}")
        return True


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_screener(
    capital_inr: float = 100_000,
    risk_per_trade_pct: float = 0.01,
    top_n: int = 8,
    target_multiple_1: float = TARGET_1_MULT,
    target_multiple_2: float = TARGET_2_MULT,
    confirm_intraday: bool = False,
    require_nifty_alignment: bool = True,
) -> list[dict]:

    logger.info("[Screener v3] Fetching market context...")
    ctx = _get_market_context()
    vix = ctx["vix"]

    if vix and vix > VIX_SKIP_THRESHOLD:
        logger.warning(f"[Screener] VIX={vix:.1f} > {VIX_SKIP_THRESHOLD} — SKIP ALL TRADES")
        return [{"skip_day": True, "reason": f"VIX={vix:.1f} too high — no trades today", "vix": vix}]

    logger.info(
        f"[Screener] VIX={vix} | Nifty={ctx['nifty_bias']} (5d {ctx['nifty_5d_return']:+.2f}%) | "
        f"Sectors={ctx['sector_returns']}"
    )
    if vix and vix > VIX_HALF_SIZE:
        logger.warning(f"[Screener] VIX={vix:.1f} > {VIX_HALF_SIZE} — halving position sizes")

    tickers = YFinanceFetcher.NIFTY_ALPHA_50
    logger.info(f"[Screener] Screening {len(tickers)} Alpha-50 stocks (pre-market, T-1 data only)...")

    plans:   list[AlphaPlan] = []
    skipped: int = 0

    for ticker in tickers:
        try:
            bars = _price_rows(ticker, n=260)
            if not bars:
                skipped += 1; continue
            plan = _build_plan(
                ticker, bars, ctx,
                capital_inr, risk_per_trade_pct,
                target_multiple_1, target_multiple_2,
            )
            if plan:
                plans.append(plan)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"[Screener] {ticker} failed: {e}")
            skipped += 1

    plans.sort(key=lambda p: -p.score)

    # Sector concentration cap
    sector_count: dict[str, int] = {}
    capped: list[AlphaPlan] = []
    for p in plans:
        if sector_count.get(p.sector, 0) >= MAX_PER_SECTOR:
            continue
        capped.append(p)
        sector_count[p.sector] = sector_count.get(p.sector, 0) + 1
        if len(capped) >= top_n:
            break

    # Optional intraday confirmation
    if confirm_intraday and capped:
        logger.info(f"[Screener] Running intraday confirmation on {len(capped)} picks...")
        confirmed: list[AlphaPlan] = []
        for p in capped:
            ok = _intraday_confirm(p, nifty_aligned_required=require_nifty_alignment)
            p.intraday_confirmed = ok
            if ok:
                confirmed.append(p)
            else:
                logger.info(f"[Confirm] {p.ticker} {p.bias} REJECTED — no intraday confirmation")
        capped = confirmed

    for i, p in enumerate(capped):
        p.rank = i + 1

    logger.info(
        f"[Screener] {len(plans)} qualified → {len(capped)} after sector-cap"
        f"{' + intraday-confirm' if confirm_intraday else ''} ({skipped} skipped)"
    )
    _print_table(capped, ctx)
    return [p.to_dict() for p in capped]


# ── Console output ─────────────────────────────────────────────────────────────

def _print_table(plans: list[AlphaPlan], ctx: dict):
    if not plans:
        logger.info("[Screener] No stocks passed filters today.")
        return
    vix        = ctx.get("vix")
    nifty_bias = ctx.get("nifty_bias")
    logger.info(f"\n{'='*155}")
    vix_str = f"{vix:.1f}" if vix is not None else "?"
    logger.info(f"  NIFTY ALPHA 50 SCREENER v3  |  VIX={vix_str}  |  Market={nifty_bias}")
    logger.info(f"{'='*155}")
    logger.info(
        f"  {'#':<3} {'Ticker':<15} {'Bias':<6} {'Score':<6} {'L/S':<11} "
        f"{'Prev₹':<9} {'ATR%':<5} {'EMA':<8} {'RS5d':<6} {'PDH/L':<11} "
        f"{'52W%':<7} {'Sect%':<7} {'ADX':<5} {'+DI/-DI':<10} {'VolSg':<6} "
        f"{'Trig':<9} {'Stop':<9} {'T1':<9} {'T2':<9} {'R:R':<5} {'Verd'}"
    )
    for p in plans:
        verdict = f"{p.verdict}({p.conviction})" if p.verdict else "—"
        sect    = f"{p.sector_momentum:+.1f}%" if p.sector_momentum is not None else "—"
        w52     = f"{p.week52_high_pct:+.1f}%" if p.week52_high_pct is not None else "—"
        rs      = f"{p.rs_vs_nifty_5d:+.1f}"   if p.rs_vs_nifty_5d  is not None else "—"
        di      = (
            f"{p.plus_di or 0:.0f}/{p.minus_di or 0:.0f}"
            if p.plus_di is not None and p.minus_di is not None else "—"
        )
        ls = f"{p.score_long:.0f}/{p.score_short:.0f}"
        logger.info(
            f"  {p.rank:<3} {p.ticker:<15} {p.bias:<6} {p.score:<6} {ls:<11} "
            f"{p.prev_close:<9} {p.atr_pct:<5} {p.ema_stack:<8} {rs:<6} {p.prev_day_signal:<11} "
            f"{w52:<7} {sect:<7} {str(p.adx or '—'):<5} {di:<10} {p.volume_surge:<6} "
            f"{p.trigger:<9} {p.stop:<9} {p.target1:<9} {p.target2:<9} {p.rr1:<5} {verdict}"
        )
    logger.info(f"{'='*155}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nifty Alpha 50 Intraday Screener v3")
    parser.add_argument("--capital",  type=float, default=100_000, help="Capital in INR")
    parser.add_argument("--risk",     type=float, default=1.0,     help="Risk per trade %")
    parser.add_argument("--top",      type=int,   default=8,       help="Max picks (cap, not quota)")
    parser.add_argument("--target1",  type=float, default=1.5)
    parser.add_argument("--target2",  type=float, default=2.5)
    parser.add_argument("--confirm",  action="store_true",
                        help="Run intraday ORB+VWAP confirmation (use after market open)")
    parser.add_argument("--no-nifty-align", action="store_true",
                        help="With --confirm, skip the Nifty intraday alignment check")
    args = parser.parse_args()
    run_screener(
        capital_inr=args.capital,
        risk_per_trade_pct=args.risk / 100,
        top_n=args.top,
        target_multiple_1=args.target1,
        target_multiple_2=args.target2,
        confirm_intraday=args.confirm,
        require_nifty_alignment=not args.no_nifty_align,
    )
