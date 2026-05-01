"""
NSE Intraday Screener v4
========================
A fully self-contained pre-market screener for NSE/BSE intraday trading.
No database required — runs entirely on yfinance.

Key improvements over v3:
  - Real ORB: fetches actual 9:15-9:30 opening range from 5-min bars
  - Real VWAP: computed from intraday 5-min bars (not a placeholder flag)
  - RSI(14) added to scoring
  - MACD crossover added to scoring
  - Regime detection: trending vs choppy day → different strategy weight
  - Intraday momentum score (15-min trend)
  - Fully standalone: pure yfinance, no SQLAlchemy, no custom fetchers
  - Rich colour terminal output

Usage:
    python -m india_quant.signals.screener
    python -m india_quant.signals.screener --capital 200000 --risk 1.0 --top 8

    # After 9:30 AM (real ORB mode):
    python -m india_quant.signals.screener --live

Requirements:
    pip install yfinance pandas tabulate colorama
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from datetime import datetime, time, timezone, timedelta
from typing import Optional

import yfinance as yf

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE  (Nifty Alpha 50 + extras)
# ─────────────────────────────────────────────────────────────────────────────
TICKER_SECTOR: dict[str, str] = {
    # IT
    "PERSISTENT.NS": "IT",    "COFORGE.NS": "IT",       "MPHASIS.NS": "IT",
    "LTIM.NS":       "IT",    "WIPRO.NS":   "IT",
    # BANK / FINANCE
    "HDFCAMC.NS":    "BANK",  "SHRIRAMFIN.NS": "BANK",  "MUTHOOTFIN.NS": "BANK",
    "CHOLAFIN.NS":   "BANK",  "AUBANK.NS":  "BANK",     "FEDERALBNK.NS": "BANK",
    "RBLBANK.NS":    "BANK",  "M&MFIN.NS":  "BANK",
    # PHARMA
    "LUPIN.NS":      "PHARMA","MANKIND.NS": "PHARMA",   "NAVINFLUOR.NS": "PHARMA",
    "AUROPHARMA.NS": "PHARMA","SUNPHARMA.NS":"PHARMA",
    # INFRA / CAPITAL GOODS
    "BEL.NS":        "INFRA", "BHEL.NS":    "INFRA",    "CUMMINSIND.NS": "INFRA",
    "CGPOWER.NS":    "INFRA", "POLYCAB.NS": "INFRA",    "VOLTAS.NS":     "INFRA",
    "DIXON.NS":      "INFRA", "KEI.NS":     "INFRA",    "INDUSTOWER.NS": "INFRA",
    "ABB.NS":        "INFRA", "SIEMENS.NS": "INFRA",
    # ENERGY
    "PFC.NS":        "ENERGY","RECLTD.NS":  "ENERGY",   "NMDC.NS":       "ENERGY",
    "SUZLON.NS":     "ENERGY","JSWENERGY.NS":"ENERGY",  "TATAPOWER.NS":  "ENERGY",
    "NHPC.NS":       "ENERGY","ADANIGREEN.NS":"ENERGY",
    # REALTY
    "OBEROIRLTY.NS": "REALTY","PRESTIGE.NS":"REALTY",   "DLF.NS":        "REALTY",
    "GODREJPROP.NS": "REALTY",
    # CONSUMER / OTHER
    "TRENT.NS":      "CONS",  "MARUTI.NS":  "CONS",     "VBL.NS":        "CONS",
    "ZOMATO.NS":     "CONS",  "NYKAA.NS":   "CONS",     "DELHIVERY.NS":  "CONS",
    "INDIGO.NS":     "CONS",  "TITAN.NS":   "CONS",
    # AUTO
    "M&M.NS":        "AUTO",  "TATAMOTORS.NS":"AUTO",   "BAJAJ-AUTO.NS": "AUTO",
    "EICHERMOT.NS":  "AUTO",
}

SECTOR_ETFS: dict[str, str] = {
    "IT":     "^CNXIT",
    "BANK":   "^NSEBANK",
    "INFRA":  "^CNXINFRA",
    "PHARMA": "^CNXPHARMA",
    "REALTY": "^CNXREALTY",
    "ENERGY": "^CNXENERGY",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MIS_LEVERAGE        = 5
INTRADAY_COST_PCT   = 0.0006      # brokerage + STT + charges (round trip)
MIN_ATR_PCT         = 1.2
MAX_ATR_PCT         = 5.0
MAX_GAP_PCT         = 2.0         # skip if overnight gap > this %
VIX_SKIP            = 22.0        # skip all trades above this
VIX_HALF_SIZE       = 19.0        # halve sizes above this
MIN_SCORE           = 42          # minimum composite score to trade
MAX_PER_SECTOR      = 2           # sector concentration cap
TARGET_1_MULT       = 1.5
TARGET_2_MULT       = 2.8
ADX_PERIOD          = 14
ATR_PERIOD          = 14
RSI_PERIOD          = 14
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
ORB_MINUTES         = 15          # opening range = first 15 mins (9:15–9:30)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TradePlan:
    rank:               int
    ticker:             str
    sector:             str
    side:               str               # LONG | SHORT

    score:              float
    score_long:         float
    score_short:        float

    # Yesterday's price context
    prev_close:         float
    prev_high:          float
    prev_low:           float
    gap_pct:            float
    atr:                float
    atr_pct:            float

    # Technical indicators
    ema9:               Optional[float]
    ema21:              Optional[float]
    ema50:              Optional[float]
    ema_stack:          str               # bullish | bearish | mixed
    rsi:                Optional[float]
    macd:               Optional[float]
    macd_signal:        Optional[float]
    macd_hist:          Optional[float]
    adx:                Optional[float]
    plus_di:            Optional[float]
    minus_di:           Optional[float]
    rs_vs_nifty_5d:     Optional[float]
    prev_day_signal:    str               # above_high | below_low | inside
    week52_high_pct:    Optional[float]
    week52_low_pct:     Optional[float]
    volume_surge:       float
    momentum_5d:        Optional[float]
    momentum_20d:       Optional[float]
    sector_momentum:    Optional[float]
    regime:             str               # TRENDING | CHOPPY | UNKNOWN

    # Trade levels
    orb_high:           float
    orb_low:            float
    orb_source:         str               # "real" | "estimated"
    vwap:               Optional[float]
    vwap_aligned:       bool

    long_trigger:       float
    long_stop:          float
    long_target1:       float
    long_target2:       float
    long_rr1:           float
    long_rr2:           float
    long_qty:           int
    long_margin_inr:    float
    long_max_loss_inr:  float
    long_profit1_inr:   float
    long_profit2_inr:   float

    short_trigger:      float
    short_stop:         float
    short_target1:      float
    short_target2:      float
    short_rr1:          float
    short_rr2:          float
    short_qty:          int
    short_margin_inr:   float
    short_max_loss_inr: float
    short_profit1_inr:  float
    short_profit2_inr:  float

    # Meta
    capital_inr:        float
    risk_per_trade_inr: float
    vix_half_size:      bool

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flat aliases expected by the dashboard template and live_tracker
        is_long = self.side == "LONG"
        d["bias"]         = self.side
        d["trigger"]      = self.long_trigger      if is_long else self.short_trigger
        d["stop"]         = self.long_stop         if is_long else self.short_stop
        d["target1"]      = self.long_target1      if is_long else self.short_target1
        d["target2"]      = self.long_target2      if is_long else self.short_target2
        d["rr1"]          = self.long_rr1          if is_long else self.short_rr1
        d["rr2"]          = self.long_rr2          if is_long else self.short_rr2
        d["qty"]          = self.long_qty          if is_long else self.short_qty
        d["margin_inr"]   = self.long_margin_inr   if is_long else self.short_margin_inr
        d["max_loss_inr"] = self.long_max_loss_inr if is_long else self.short_max_loss_inr
        d["profit1_inr"]  = self.long_profit1_inr  if is_long else self.short_profit1_inr
        d["profit2_inr"]  = self.long_profit2_inr  if is_long else self.short_profit2_inr
        d["verdict"]      = None
        d["conviction"]   = None
        return d


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR HELPERS  (all no-look-ahead: bars[-1] = yesterday)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    k   = 2.0 / (period + 1)
    val = sum(closes[:period]) / period
    for p in closes[period:]:
        val = p * k + val * (1 - k)
    return round(val, 2)


def _wilder_smooth(values: list[float], period: int) -> list[float]:
    """Wilder's smoothing (used for ATR, ADX, RSI)."""
    if len(values) < period:
        return []
    result = [sum(values[:period])]
    for v in values[period:]:
        result.append(result[-1] - result[-1] / period + v)
    return result


def _wilder_atr(bars: list[dict], period: int = ATR_PERIOD) -> Optional[float]:
    if len(bars) < period + 2:
        return None
    trs = [
        max(bars[i]["high"] - bars[i]["low"],
            abs(bars[i]["high"] - bars[i-1]["close"]),
            abs(bars[i]["low"]  - bars[i-1]["close"]))
        for i in range(1, len(bars))
    ]
    smoothed = _wilder_smooth(trs, period)
    return round(smoothed[-1] / period, 4) if smoothed else None


def _wilder_adx(bars: list[dict], period: int = ADX_PERIOD
                ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if len(bars) < period * 2 + 2:
        return None, None, None
    pdms, ndms, trs = [], [], []
    for i in range(1, len(bars)):
        h,  l  = bars[i]["high"],     bars[i]["low"]
        ph, pl = bars[i-1]["high"],   bars[i-1]["low"]
        pc     = bars[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        up, dn = h - ph, pl - l
        pdms.append(up if (up > dn and up > 0) else 0.0)
        ndms.append(dn if (dn > up and dn > 0) else 0.0)

    def ws(lst): return _wilder_smooth(lst, period)
    str_ = ws(trs); spdm = ws(pdms); sndm = ws(ndms)
    if not str_:
        return None, None, None

    dx_list = []
    for i in range(len(str_)):
        if str_[i] == 0:
            continue
        pdi = 100 * spdm[i] / str_[i]
        ndi = 100 * sndm[i] / str_[i]
        s   = pdi + ndi
        dx_list.append((100 * abs(pdi - ndi) / s if s else 0, pdi, ndi))

    if not dx_list:
        return None, None, None

    adx = sum(d[0] for d in dx_list[:period]) / period
    for d in dx_list[period:]:
        adx = (adx * (period - 1) + d[0]) / period
    return round(adx, 1), round(dx_list[-1][1], 1), round(dx_list[-1][2], 1)


def _rsi(bars: list[dict], period: int = RSI_PERIOD) -> Optional[float]:
    """Wilder RSI. bars[-1] = yesterday."""
    if len(bars) < period + 2:
        return None
    closes = [b["close"] for b in bars]
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


def _macd(bars: list[dict]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (macd_line, signal_line, histogram)."""
    closes = [b["close"] for b in bars]
    if len(closes) < MACD_SLOW + MACD_SIGNAL:
        return None, None, None
    fast_ema = _ema(closes, MACD_FAST)
    slow_ema = _ema(closes, MACD_SLOW)
    if fast_ema is None or slow_ema is None:
        return None, None, None
    macd_series = []
    k_fast = 2 / (MACD_FAST + 1)
    k_slow = 2 / (MACD_SLOW + 1)
    ema_f  = sum(closes[:MACD_FAST]) / MACD_FAST
    ema_s  = sum(closes[:MACD_SLOW]) / MACD_SLOW
    for i in range(MACD_SLOW, len(closes)):
        ema_f = closes[i] * k_fast + ema_f * (1 - k_fast)
        ema_s = closes[i] * k_slow + ema_s * (1 - k_slow)
        macd_series.append(ema_f - ema_s)
    if len(macd_series) < MACD_SIGNAL:
        return None, None, None
    k_sig  = 2 / (MACD_SIGNAL + 1)
    sig    = sum(macd_series[:MACD_SIGNAL]) / MACD_SIGNAL
    for v in macd_series[MACD_SIGNAL:]:
        sig = v * k_sig + sig * (1 - k_sig)
    macd_val  = macd_series[-1]
    hist      = macd_val - sig
    return round(macd_val, 4), round(sig, 4), round(hist, 4)


def _ema_stack_fn(bars: list[dict]
                  ) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    closes = [b["close"] for b in bars]
    e9  = _ema(closes, 9)
    e21 = _ema(closes, 21)
    e50 = _ema(closes, 50)
    if e9 and e21 and e50:
        if e9 > e21 > e50:   stack = "bullish"
        elif e9 < e21 < e50: stack = "bearish"
        else:                 stack = "mixed"
    else:
        stack = "mixed"
    return e9, e21, e50, stack


def _volume_surge(bars: list[dict], avg_period: int = 20) -> float:
    if len(bars) < avg_period + 2:
        return 1.0
    yvol   = bars[-1]["volume"]
    avg    = sum(b["volume"] for b in bars[-avg_period-1:-1]) / avg_period
    return round(yvol / avg, 2) if avg > 0 else 1.0


def _momentum(bars: list[dict], period: int) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    cur, prev = bars[-1]["close"], bars[-period-1]["close"]
    return round((cur / prev - 1) * 100, 2) if prev else None


def _relative_strength(bars: list[dict], nifty_5d: float) -> Optional[float]:
    if len(bars) < 6:
        return None
    cur, prev = bars[-1]["close"], bars[-6]["close"]
    return round((cur / prev - 1) * 100 - nifty_5d, 2) if prev else None


def _prev_day_signal(bars: list[dict]) -> str:
    if len(bars) < 2:
        return "inside"
    yc  = bars[-1]["close"]
    dbh = bars[-2]["high"]
    dbl = bars[-2]["low"]
    if yc > dbh: return "above_high"
    if yc < dbl: return "below_low"
    return "inside"


def _week52(bars: list[dict]) -> tuple[Optional[float], Optional[float],
                                        Optional[float], Optional[float]]:
    if len(bars) < 10:
        return None, None, None, None
    subset = bars[-min(252, len(bars)):]
    w52h   = max(b["high"] for b in subset)
    w52l   = min(b["low"]  for b in subset)
    cur    = bars[-1]["close"]
    return (round(w52h, 2), round(w52l, 2),
            round((cur / w52h - 1) * 100, 2),
            round((cur / w52l - 1) * 100, 2))


def _detect_regime(bars: list[dict], adx: Optional[float]) -> str:
    """
    TRENDING : ADX > 25 OR strong recent momentum
    CHOPPY   : ADX < 20 AND low momentum
    UNKNOWN  : insufficient data
    """
    if adx is None:
        return "UNKNOWN"
    mom5  = _momentum(bars, 5)
    if adx >= 25:
        return "TRENDING"
    if adx <= 20 and (mom5 is None or abs(mom5) < 1.5):
        return "CHOPPY"
    return "TRENDING" if adx >= 22 else "CHOPPY"


# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY DATA — REAL ORB + VWAP
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_intraday_5min(ticker: str) -> list[dict]:
    """Fetch today's 5-min bars (or last session's if pre-market)."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="5m")
        if df.empty:
            return []
        df = df.reset_index()
        bars = []
        for _, row in df.iterrows():
            bars.append({
                "datetime": row.get("Datetime", row.get("Date")),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row["Volume"]),
            })
        return bars
    except Exception:
        return []


def _compute_real_orb(intraday_bars: list[dict]
                      ) -> tuple[Optional[float], Optional[float], str]:
    """
    Compute real ORB from 5-min bars.
    Opening range = 9:15 to 9:30 (first 3 bars of NSE session).
    """
    if not intraday_bars:
        return None, None, "no_data"
    orb_bars = []
    for b in intraday_bars:
        dt = b["datetime"]
        if hasattr(dt, "time"):
            t = dt.time()
            session_start = time(9, 15)
            orb_end       = time(9, 15 + ORB_MINUTES)
            if session_start <= t <= orb_end:
                orb_bars.append(b)
    if not orb_bars:
        return None, None, "pre_market"
    orb_high = max(b["high"] for b in orb_bars)
    orb_low  = min(b["low"]  for b in orb_bars)
    return round(orb_high, 2), round(orb_low, 2), "real"


def _compute_vwap(intraday_bars: list[dict]) -> Optional[float]:
    """VWAP = Σ(typical_price × volume) / Σ(volume)"""
    if not intraday_bars:
        return None
    total_tpv = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"]
                    for b in intraday_bars)
    total_vol = sum(b["volume"] for b in intraday_bars)
    if total_vol == 0:
        return None
    return round(total_tpv / total_vol, 2)


# ─────────────────────────────────────────────────────────────────────────────
# DAILY DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_daily(ticker: str, n: int = 300) -> list[dict]:
    """
    Fetch last n daily bars, ascending order (oldest → newest).
    bars[-1] = most recent closed session (yesterday).
    """
    try:
        df = yf.Ticker(ticker).history(period=f"{n}d")
        if df.empty:
            return []
        df = df.reset_index().sort_values("Date")
        bars = []
        for _, row in df.iterrows():
            bars.append({
                "datetime": row["Date"],
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row["Volume"]),
            })
        return bars
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MARKET CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def _get_market_context() -> dict:
    ctx = {
        "vix":               None,
        "nifty_prev_chg_pct": 0.0,
        "nifty_5d_return":    0.0,
        "sector_returns":     {},
    }
    try:
        vix_df = yf.Ticker("^INDIAVIX").history(period="5d")
        if not vix_df.empty:
            ctx["vix"] = float(vix_df["Close"].iloc[-1])

        nifty_df = yf.Ticker("^NSEI").history(period="15d")
        if not nifty_df.empty and len(nifty_df) >= 2:
            nc  = float(nifty_df["Close"].iloc[-1])
            np_ = float(nifty_df["Close"].iloc[-2])
            n5  = float(nifty_df["Close"].iloc[-6]) if len(nifty_df) >= 6 else np_
            ctx["nifty_prev_chg_pct"] = round((nc / np_ - 1) * 100, 2)
            ctx["nifty_5d_return"]    = round((nc / n5  - 1) * 100, 2)
    except Exception:
        pass

    for sector, etf in SECTOR_ETFS.items():
        try:
            df = yf.Ticker(etf).history(period="5d")
            if not df.empty and len(df) >= 2:
                ret = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100
                ctx["sector_returns"][sector] = round(ret, 2)
        except Exception:
            pass
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE  (0–100 points, independent LONG and SHORT scores)
# ─────────────────────────────────────────────────────────────────────────────

def _score(
    nifty_chg:      float,
    ema_stack:      str,
    adx:            Optional[float],
    plus_di:        Optional[float],
    minus_di:       Optional[float],
    rsi:            Optional[float],
    macd_hist:      Optional[float],
    rs_vs_nifty:    Optional[float],
    prev_day_sig:   str,
    sect_mom:       Optional[float],
    mom_5d:         Optional[float],
    w52h_pct:       Optional[float],
    w52l_pct:       Optional[float],
    vol_surge:      float,
    vwap:           Optional[float],
    prev_close:     float,
    regime:         str,
) -> tuple[float, float]:
    """
    Compute LONG and SHORT scores independently.
    Returns (score_long, score_short).

    Weight table (total possible = 100):
      Nifty direction         : 18
      ADX + DI direction      : 16
      Prev-day high/low break : 14
      RSI                     : 12
      RS vs Nifty             : 10
      MACD histogram          : 10
      EMA stack               : 8
      Sector momentum         : 6
      5-day momentum          : 4
      52-week proximity       : 4
      Volume surge bonus      : +5 (additive)
      VWAP alignment bonus    : +3 (additive, intraday only)
    """
    sl = ss = 0.0

    # ── 1. Nifty direction (18 pts) ──────────────────────────────────────
    if nifty_chg > 0.6:       sl += 18
    elif nifty_chg > 0.3:     sl += 12
    elif nifty_chg > 0.1:     sl += 6
    elif nifty_chg < -0.6:    ss += 18
    elif nifty_chg < -0.3:    ss += 12
    elif nifty_chg < -0.1:    ss += 6

    # ── 2. ADX + DI (16 pts) ─────────────────────────────────────────────
    if adx and plus_di and minus_di:
        regime_factor = 0.6 if regime == "CHOPPY" else 1.0
        adx_pts = 0
        if adx >= 30:    adx_pts = 16
        elif adx >= 25:  adx_pts = 12
        elif adx >= 20:  adx_pts = 7
        elif adx >= 15:  adx_pts = 3
        adx_pts = int(adx_pts * regime_factor)
        if plus_di > minus_di: sl += adx_pts
        else:                  ss += adx_pts

    # ── 3. Prev-day high/low break (14 pts) ──────────────────────────────
    if prev_day_sig == "above_high": sl += 14
    elif prev_day_sig == "below_low": ss += 14

    # ── 4. RSI (12 pts) ──────────────────────────────────────────────────
    if rsi is not None:
        if rsi >= 60:    sl += 12
        elif rsi >= 55:  sl += 8
        elif rsi >= 50:  sl += 4
        elif rsi <= 40:  ss += 12
        elif rsi <= 45:  ss += 8
        elif rsi <= 50:  ss += 4
        if regime == "CHOPPY":
            if rsi >= 75:   sl -= 4
            elif rsi <= 25: ss -= 4

    # ── 5. RS vs Nifty (10 pts) ──────────────────────────────────────────
    if rs_vs_nifty is not None:
        if rs_vs_nifty >= 3:    sl += 10
        elif rs_vs_nifty >= 1:  sl += 6
        elif rs_vs_nifty >= 0:  sl += 3
        elif rs_vs_nifty <= -3: ss += 10
        elif rs_vs_nifty <= -1: ss += 6
        else:                   ss += 3

    # ── 6. MACD histogram (10 pts) ───────────────────────────────────────
    if macd_hist is not None:
        norm = macd_hist / prev_close * 100
        if norm > 0.05:    sl += 10
        elif norm > 0.02:  sl += 6
        elif norm > 0:     sl += 3
        elif norm < -0.05: ss += 10
        elif norm < -0.02: ss += 6
        elif norm < 0:     ss += 3

    # ── 7. EMA stack (8 pts) ─────────────────────────────────────────────
    if ema_stack == "bullish":   sl += 8
    elif ema_stack == "bearish": ss += 8

    # ── 8. Sector momentum (6 pts) ───────────────────────────────────────
    if sect_mom is not None:
        if sect_mom > 1.0:    sl += 6
        elif sect_mom > 0.3:  sl += 3
        elif sect_mom < -1.0: ss += 6
        elif sect_mom < -0.3: ss += 3

    # ── 9. 5-day momentum (4 pts) ────────────────────────────────────────
    if mom_5d is not None:
        if mom_5d > 3:    sl += 4
        elif mom_5d > 1:  sl += 2
        elif mom_5d < -3: ss += 4
        elif mom_5d < -1: ss += 2

    # ── 10. 52-week proximity (4 pts) ────────────────────────────────────
    if w52h_pct is not None:
        if w52h_pct >= -3:    sl += 4
        elif w52h_pct <= -40: ss += 4
    if w52l_pct is not None:
        if w52l_pct <= 5:     ss += 4

    # ── Bonus: Volume surge (+5) ─────────────────────────────────────────
    if vol_surge >= 1.5:
        if sl >= ss: sl = min(100, sl + 5)
        else:        ss = min(100, ss + 5)

    # ── Bonus: VWAP alignment (+3) ───────────────────────────────────────
    if vwap is not None:
        if prev_close > vwap:  sl = min(100, sl + 3)
        else:                  ss = min(100, ss + 3)

    return round(max(sl, 0), 1), round(max(ss, 0), 1)


# ─────────────────────────────────────────────────────────────────────────────
# ORB LEVELS
# ─────────────────────────────────────────────────────────────────────────────

def _build_orb_levels(
    orb_high:   float,
    orb_low:    float,
    atr:        float,
    t1_mult:    float,
    t2_mult:    float,
) -> dict:
    or_size = orb_high - orb_low
    if or_size <= 0:
        or_size = atr * 0.4
    buf = orb_high * 0.0005

    lt  = round(orb_high + buf, 2)
    ls  = round(orb_low  - buf, 2)
    lt1 = round(lt + t1_mult * or_size, 2)
    lt2 = round(lt + t2_mult * or_size, 2)
    lr  = max(lt - ls, 0.01)

    st  = round(orb_low  - buf, 2)
    ss  = round(orb_high + buf, 2)
    st1 = round(st - t1_mult * or_size, 2)
    st2 = round(st - t2_mult * or_size, 2)
    sr  = max(ss - st, 0.01)

    return {
        "lt": lt,   "ls": ls,   "lt1": lt1, "lt2": lt2,
        "lr": lr,   "lrew1": max(lt1-lt, 0.01), "lrew2": max(lt2-lt, 0.01),
        "st": st,   "ss": ss,   "st1": st1, "st2": st2,
        "sr": sr,   "srew1": max(st-st1, 0.01), "srew2": max(st-st2, 0.01),
    }


def _position_size(
    capital: float,
    risk_pct: float,
    risk_per_share: float,
    trigger: float,
    vix: Optional[float],
) -> tuple[int, float, float]:
    half   = vix is not None and vix > VIX_HALF_SIZE
    factor = 0.5 if half else 1.0
    risk   = capital * risk_pct * factor
    qty    = max(1, int(risk // risk_per_share)) if risk_per_share > 0 else 1
    pos    = qty * trigger
    if pos / MIS_LEVERAGE > capital * 0.9:
        qty = max(1, int(capital * 0.9 * MIS_LEVERAGE / trigger))
        pos = qty * trigger
    return qty, round(pos / MIS_LEVERAGE, 0), round(risk, 0)


# ─────────────────────────────────────────────────────────────────────────────
# PLAN BUILDER  (per ticker)
# ─────────────────────────────────────────────────────────────────────────────

def _build_plan(
    ticker:       str,
    bars:         list[dict],
    ctx:          dict,
    capital:      float,
    risk_pct:     float,
    t1_mult:      float,
    t2_mult:      float,
    live_mode:    bool,
) -> Optional[TradePlan]:

    if len(bars) < ADX_PERIOD * 2 + 10:
        return None

    prev_close = bars[-1]["close"]
    prev_high  = bars[-1]["high"]
    prev_low   = bars[-1]["low"]
    if prev_close <= 0:
        return None

    vix        = ctx["vix"]
    nifty_chg  = ctx.get("nifty_prev_chg_pct", 0.0)
    n5d        = ctx.get("nifty_5d_return", 0.0)
    sector     = TICKER_SECTOR.get(ticker, "OTHER")
    sect_mom   = ctx["sector_returns"].get(sector)

    atr = _wilder_atr(bars)
    if atr is None:
        return None
    atr_pct = round(atr / prev_close * 100, 2)
    if not (MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT):
        return None

    if len(bars) < 2:
        return None
    gap_pct = round((prev_close / bars[-2]["close"] - 1) * 100, 2) if bars[-2]["close"] else 0.0
    if abs(gap_pct) > MAX_GAP_PCT:
        return None

    adx, pdi, ndi     = _wilder_adx(bars)
    e9, e21, e50, stk = _ema_stack_fn(bars)
    rsi_val           = _rsi(bars)
    macd_l, macd_s, macd_h = _macd(bars)
    rs                = _relative_strength(bars, n5d)
    pd_sig            = _prev_day_signal(bars)
    w52h, w52l, w52hp, w52lp = _week52(bars)
    vol_surge         = _volume_surge(bars)
    mom5              = _momentum(bars, 5)
    mom20             = _momentum(bars, 20)
    regime            = _detect_regime(bars, adx)

    vwap        = None
    orb_source  = "estimated"
    orb_high    = round(prev_close + 0.2 * atr, 2)
    orb_low     = round(prev_close - 0.2 * atr, 2)

    if live_mode:
        intra = _fetch_intraday_5min(ticker)
        if intra:
            roh, rol, src = _compute_real_orb(intra)
            if roh and rol:
                orb_high   = roh
                orb_low    = rol
                orb_source = src
            vwap = _compute_vwap(intra)

    orb = _build_orb_levels(orb_high, orb_low, atr, t1_mult, t2_mult)

    vwap_aligned = False
    if vwap and prev_close:
        vwap_aligned = prev_close > vwap

    sl, ss = _score(
        nifty_chg, stk, adx, pdi, ndi, rsi_val, macd_h,
        rs, pd_sig, sect_mom, mom5, w52hp, w52lp,
        vol_surge, vwap, prev_close, regime,
    )

    if sl >= ss and sl >= MIN_SCORE:
        side, score = "LONG", sl
    elif ss > sl and ss >= MIN_SCORE:
        side, score = "SHORT", ss
    else:
        return None

    half = vix is not None and vix > VIX_HALF_SIZE
    lq, lm, lr_inr = _position_size(capital, risk_pct, orb["lr"], orb["lt"], vix)
    sq, sm, sr_inr = _position_size(capital, risk_pct, orb["sr"], orb["st"], vix)
    c = INTRADAY_COST_PCT

    return TradePlan(
        rank=0, ticker=ticker, sector=sector, side=side,
        score=score, score_long=sl, score_short=ss,
        prev_close=round(prev_close, 2),
        prev_high=round(prev_high, 2),
        prev_low=round(prev_low, 2),
        gap_pct=gap_pct, atr=round(atr, 2), atr_pct=atr_pct,
        ema9=e9, ema21=e21, ema50=e50, ema_stack=stk,
        rsi=rsi_val,
        macd=macd_l, macd_signal=macd_s, macd_hist=macd_h,
        adx=adx, plus_di=pdi, minus_di=ndi,
        rs_vs_nifty_5d=rs, prev_day_signal=pd_sig,
        week52_high_pct=w52hp, week52_low_pct=w52lp,
        volume_surge=vol_surge, momentum_5d=mom5, momentum_20d=mom20,
        sector_momentum=sect_mom, regime=regime,
        orb_high=orb_high, orb_low=orb_low, orb_source=orb_source,
        vwap=vwap, vwap_aligned=vwap_aligned,
        long_trigger=orb["lt"],   long_stop=orb["ls"],
        long_target1=orb["lt1"],  long_target2=orb["lt2"],
        long_rr1=round(orb["lrew1"] / orb["lr"], 2),
        long_rr2=round(orb["lrew2"] / orb["lr"], 2),
        long_qty=lq, long_margin_inr=lm,
        long_max_loss_inr=round(lq * orb["lr"] + lq * orb["lt"] * c, 0),
        long_profit1_inr=round(lq * orb["lrew1"] - lq * orb["lt"] * c, 0),
        long_profit2_inr=round(lq * orb["lrew2"] - lq * orb["lt"] * c, 0),
        short_trigger=orb["st"],  short_stop=orb["ss"],
        short_target1=orb["st1"], short_target2=orb["st2"],
        short_rr1=round(orb["srew1"] / orb["sr"], 2),
        short_rr2=round(orb["srew2"] / orb["sr"], 2),
        short_qty=sq, short_margin_inr=sm,
        short_max_loss_inr=round(sq * orb["sr"] + sq * orb["st"] * c, 0),
        short_profit1_inr=round(sq * orb["srew1"] - sq * orb["st"] * c, 0),
        short_profit2_inr=round(sq * orb["srew2"] - sq * orb["st"] * c, 0),
        capital_inr=capital,
        risk_per_trade_inr=lr_inr if side == "LONG" else sr_inr,
        vix_half_size=half,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCREENER
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(
    capital_inr:        float = 100_000,
    risk_per_trade_pct: float = 0.01,
    top_n:              int   = 8,
    t1_mult:            float = TARGET_1_MULT,
    t2_mult:            float = TARGET_2_MULT,
    live_mode:          bool  = False,
) -> list[dict]:
    """
    Run the screener against the built-in universe.
    live_mode=True fetches real ORB + VWAP from intraday 5-min bars
    (run after 9:30 AM IST for best results).
    Returns a list of dicts; each dict includes flat aliases (trigger, stop,
    target1, target2, bias, qty, etc.) for dashboard/tracker compatibility.
    """
    print("\n" + _col("=" * 72, "CYAN"))
    print(_col("  NSE INTRADAY SCREENER v4  |  Fetching market context...", "CYAN"))
    print(_col("=" * 72, "CYAN"))

    ctx = _get_market_context()
    vix = ctx["vix"]

    if vix and vix > VIX_SKIP:
        msg = f"  VIX = {vix:.1f} > {VIX_SKIP} — SKIP ALL TRADES TODAY"
        print(_col(msg, "RED"))
        return [{"skip_day": True, "reason": msg, "vix": vix}]

    print(f"  VIX      : {_fmt_vix(vix)}")
    print(f"  Nifty Δ  : {_col_chg(ctx['nifty_prev_chg_pct'])}")
    print(f"  Nifty 5d : {_col_chg(ctx['nifty_5d_return'])}")
    print(f"  Sectors  : { {k: f'{v:+.1f}%' for k, v in ctx['sector_returns'].items()} }")
    if live_mode:
        print(_col("  MODE     : LIVE (real ORB + VWAP)", "GREEN"))
    else:
        print(_col("  MODE     : PRE-MARKET (estimated ORB)", "YELLOW"))
    if vix and vix > VIX_HALF_SIZE:
        print(_col(f"  VIX > {VIX_HALF_SIZE} — position sizes halved", "YELLOW"))
    print()

    tickers = list(TICKER_SECTOR.keys())
    print(f"  Scanning {len(tickers)} stocks...")

    plans: list[TradePlan] = []
    skipped = 0
    for ticker in tickers:
        try:
            bars = _fetch_daily(ticker)
            if len(bars) < ADX_PERIOD * 2 + 10:
                skipped += 1
                continue
            plan = _build_plan(
                ticker, bars, ctx, capital_inr,
                risk_per_trade_pct, t1_mult, t2_mult, live_mode,
            )
            if plan:
                plans.append(plan)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    plans.sort(key=lambda p: -p.score)

    sector_count: dict[str, int] = {}
    capped: list[TradePlan] = []
    for p in plans:
        cnt = sector_count.get(p.sector, 0)
        if cnt < MAX_PER_SECTOR:
            capped.append(p)
            sector_count[p.sector] = cnt + 1
        if len(capped) >= top_n:
            break

    for i, p in enumerate(capped):
        p.rank = i + 1

    print(f"  {_col(str(len(plans)), 'GREEN')} passed filters → "
          f"{_col(str(len(capped)), 'CYAN')} after sector cap | "
          f"{skipped} skipped\n")

    _print_table(capped, ctx)
    return [p.to_dict() for p in capped]


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _col(text: str, colour: str) -> str:
    if not HAS_COLOR:
        return text
    colors = {
        "RED":    Fore.RED,   "GREEN": Fore.GREEN,
        "YELLOW": Fore.YELLOW,"CYAN":  Fore.CYAN,
        "WHITE":  Fore.WHITE, "BOLD":  Style.BRIGHT,
    }
    return f"{colors.get(colour, '')}{text}{Style.RESET_ALL}"


def _col_chg(val: float) -> str:
    s = f"{val:+.2f}%"
    return _col(s, "GREEN" if val >= 0 else "RED")


def _fmt_vix(vix: Optional[float]) -> str:
    if vix is None:
        return "N/A"
    s = f"{vix:.1f}"
    if vix > VIX_SKIP:      return _col(s, "RED")
    if vix > VIX_HALF_SIZE: return _col(s, "YELLOW")
    return _col(s, "GREEN")


def _print_table(plans: list[TradePlan], ctx: dict):
    if not plans:
        print(_col("  No stocks passed all filters — no trades today.", "YELLOW"))
        return

    rows = []
    for p in plans:
        is_long = p.side == "LONG"
        trigger = p.long_trigger  if is_long else p.short_trigger
        stop    = p.long_stop     if is_long else p.short_stop
        t1      = p.long_target1  if is_long else p.short_target1
        rr1     = p.long_rr1      if is_long else p.short_rr1
        qty     = p.long_qty      if is_long else p.short_qty
        margin  = p.long_margin_inr if is_long else p.short_margin_inr
        loss    = p.long_max_loss_inr if is_long else p.short_max_loss_inr
        profit1 = p.long_profit1_inr  if is_long else p.short_profit1_inr

        side_str = _col("▲ LONG",  "GREEN") if is_long else _col("▼ SHORT", "RED")
        reg_str  = _col(p.regime[:5], "GREEN") if p.regime == "TRENDING" else _col("CHOPY", "YELLOW")
        orb_str  = "✓" if p.orb_source == "real" else "~"
        vwap_str = "✓" if p.vwap_aligned else "—"
        rsi_str  = f"{p.rsi:.0f}" if p.rsi else "—"
        macd_str = ("▲" if p.macd_hist and p.macd_hist > 0 else "▼") if p.macd_hist else "—"

        rows.append([
            p.rank,
            p.ticker.replace(".NS", ""),
            p.sector[:5],
            side_str,
            f"{p.score:.0f}",
            f"₹{p.prev_close:.0f}",
            f"{p.atr_pct:.1f}%",
            reg_str,
            rsi_str,
            macd_str,
            orb_str,
            vwap_str,
            f"₹{trigger:.1f}",
            f"₹{stop:.1f}",
            f"₹{t1:.1f}",
            f"{rr1:.1f}R",
            f"{qty}",
            f"₹{margin:.0f}",
            f"-₹{loss:.0f}",
            f"+₹{profit1:.0f}",
        ])

    headers = [
        "#", "TICKER", "SECT", "SIDE", "SCR",
        "CLOSE", "ATR%", "REGM", "RSI", "MAC",
        "ORB", "VWP",
        "TRIGGER", "STOP", "T1", "R:R",
        "QTY", "MARGIN", "MAX LOSS", "PROFIT@T1"
    ]

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline",
                       colalign=("right",) + ("left",) * (len(headers) - 1)))
    else:
        print("  " + "  ".join(f"{h:<10}" for h in headers))
        for r in rows:
            print("  " + "  ".join(f"{str(c):<10}" for c in r))

    print()
    orb_note = "ORB=✓ real  |  ORB=~ estimated (run --live after 9:30 AM)"
    print(_col(f"  {orb_note}", "YELLOW"))
    print(_col("  Enter ONLY if price hits Trigger. Never chase reversals.", "YELLOW"))
    print(_col("  Use Trigger as entry, Stop as stop-loss, T1 as first target.", "YELLOW"))
    print()

    print(_col("  ─── TRADE DETAILS ───", "CYAN"))
    for p in plans:
        is_long = p.side == "LONG"
        trigger = p.long_trigger  if is_long else p.short_trigger
        stop    = p.long_stop     if is_long else p.short_stop
        t1      = p.long_target1  if is_long else p.short_target1
        t2      = p.long_target2  if is_long else p.short_target2
        qty     = p.long_qty      if is_long else p.short_qty
        margin  = p.long_margin_inr if is_long else p.short_margin_inr
        loss    = p.long_max_loss_inr if is_long else p.short_max_loss_inr
        p1      = p.long_profit1_inr  if is_long else p.short_profit1_inr
        p2      = p.long_profit2_inr  if is_long else p.short_profit2_inr
        rr1     = p.long_rr1 if is_long else p.short_rr1
        rr2     = p.long_rr2 if is_long else p.short_rr2
        side_c  = "GREEN" if is_long else "RED"
        ticker_short = p.ticker.replace(".NS", "")

        print(f"\n  {_col(f'#{p.rank}  {ticker_short}  {p.side}', side_c)}"
              f"  |  Score: {p.score:.0f}/100"
              f"  |  Sector: {p.sector}"
              f"  |  Regime: {p.regime}")
        print(f"      Entry : ₹{trigger:.2f}  (ORB {p.orb_source})")
        print(f"      Stop  : ₹{stop:.2f}  "
              f"({'below ORB low' if is_long else 'above ORB high'})")
        print(f"      T1    : ₹{t1:.2f}  (R:R {rr1:.1f}R)  "
              f"→ Profit: +₹{p1:.0f}")
        print(f"      T2    : ₹{t2:.2f}  (R:R {rr2:.1f}R)  "
              f"→ Profit: +₹{p2:.0f}")
        print(f"      Qty   : {qty} shares  |  Margin: ₹{margin:.0f}  "
              f"|  Max loss: ₹{loss:.0f}")
        if p.vix_half_size:
            print(_col("      VIX high — position halved", "YELLOW"))
        rsi_note = f"RSI={p.rsi:.0f}" if p.rsi else "RSI=—"
        atr_note = f"ATR%={p.atr_pct:.1f}"
        ema_note = f"EMA={p.ema_stack}"
        adx_note = f"ADX={p.adx:.0f}" if p.adx else "ADX=—"
        rs_note  = f"RS5d={p.rs_vs_nifty_5d:+.1f}%" if p.rs_vs_nifty_5d else ""
        macd_dir = "↑" if (p.macd_hist and p.macd_hist > 0) else "↓"
        print(f"      Signals: {rsi_note}  {adx_note}  {atr_note}  "
              f"{ema_note}  MACD{macd_dir}  {rs_note}")
        if p.vwap:
            vwap_side = "above" if p.vwap_aligned else "below"
            print(f"      VWAP  : ₹{p.vwap:.2f}  (price is {vwap_side} VWAP)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSE Intraday Screener v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m india_quant.signals.screener                    # pre-market, ₹1L capital
  python -m india_quant.signals.screener --capital 200000   # ₹2L capital
  python -m india_quant.signals.screener --risk 0.8 --top 5
  python -m india_quant.signals.screener --live             # after 9:30 AM, real ORB
        """,
    )
    parser.add_argument("--capital",   type=float, default=100_000)
    parser.add_argument("--risk",      type=float, default=1.0,
                        help="Risk per trade %% (default: 1.0)")
    parser.add_argument("--top",       type=int,   default=8)
    parser.add_argument("--target1",   type=float, default=TARGET_1_MULT)
    parser.add_argument("--target2",   type=float, default=TARGET_2_MULT)
    parser.add_argument("--live",      action="store_true",
                        help="Fetch real ORB + VWAP (run after 9:30 AM IST)")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    args = parser.parse_args()

    import india_quant.signals.screener as _self
    _self.MIN_SCORE = args.min_score

    run_screener(
        capital_inr=args.capital,
        risk_per_trade_pct=args.risk / 100,
        top_n=args.top,
        t1_mult=args.target1,
        t2_mult=args.target2,
        live_mode=args.live,
    )
