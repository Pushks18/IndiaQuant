"""Bull and Bear researchers — LLM-narrated via OpenRouter, rule-based fallback."""
import json

from loguru import logger

from india_quant.llm import get_client as get_llm


def _safe(d: dict | None, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _slim_summary(analyst_summary: dict, ml_signal: dict) -> dict:
    """Extract only the data the LLM needs — keeps token count tight."""
    tech = analyst_summary.get("technical_analyst") or analyst_summary.get("technical") or {}
    fund = analyst_summary.get("fundamental_analyst") or analyst_summary.get("fundamental") or {}
    sent = analyst_summary.get("sentiment_analyst") or analyst_summary.get("sentiment") or {}
    macro = analyst_summary.get("macro_analyst") or analyst_summary.get("macro") or {}
    return {
        "technical": {
            "trend": tech.get("trend"),
            "rsi": tech.get("rsi"),
            "macd_signal": tech.get("macd_signal"),
            "cmp": _safe(tech, "indicators", "cmp"),
            "atr": _safe(tech, "indicators", "atr"),
            "support": _safe(tech, "key_levels", "support"),
            "resistance": _safe(tech, "key_levels", "resistance"),
            "outlook": tech.get("outlook"),
        },
        "fundamental": {
            "verdict": fund.get("verdict"),
            "pe": fund.get("pe"),
            "sector_median_pe": fund.get("sector_median_pe"),
            "roe": fund.get("roe"),
            "eps_growth": fund.get("eps_growth"),
            "red_flags": fund.get("red_flags") or [],
            "sector": fund.get("sector"),
        },
        "sentiment": {
            "verdict": sent.get("verdict"),
            "score": sent.get("sentiment_score"),
            "article_count": sent.get("article_count"),
            "key_themes": sent.get("key_themes") or [],
            "high_impact_event": sent.get("high_impact_event"),
            "rbi_relevance": sent.get("rbi_relevance"),
            "sebi_news": sent.get("sebi_news"),
        },
        "macro": {
            "regime_label": macro.get("regime_label"),
            "india_vix": macro.get("india_vix"),
            "above_200ema": macro.get("above_200ema"),
        },
        "ml": {
            "predicted_return_1d": ml_signal.get("predicted_return_1d"),
            "signal_rank": ml_signal.get("signal_rank"),
        },
    }


_BULL_SYSTEM = """You are a bull-case researcher for an Indian equity (NSE).
Build the strongest possible argument FOR going long, grounded in the supplied data.

Cite specific numbers (RSI, EMAs, P/E vs sector median, sentiment score).
India-specific catalysts to surface when present: PLI scheme, FII inflow, PEAD drift,
index inclusion, valuation below sector median, promoter buying.

Return ONLY this JSON object — no prose, no markdown:
{
  "bull_case": str,                   // 2-3 sentences, specific numbers
  "key_catalysts": [str, ...],        // 3-5 bullets, each <= 12 words
  "target_price": float | null,       // realistic upside, in INR
  "confidence": int,                  // 1-10
  "key_risk": str                     // single sentence
}"""

_BEAR_SYSTEM = """You are a bear-case researcher for an Indian equity (NSE).
Build the strongest possible argument AGAINST going long, grounded in the supplied data.

India-specific risks to surface when present: F&O ban risk, SEBI insider blackout,
RBI tightening, FII outflow, P/E > 1.5x sector median without growth, promoter
pledging, circuit-filter proximity.

Return ONLY this JSON object — no prose, no markdown:
{
  "bear_case": str,                   // 2-3 sentences, specific numbers
  "key_risks": [str, ...],            // 3-5 bullets, each <= 12 words
  "downside_target": float | null,    // realistic downside, in INR
  "confidence": int,                  // 1-10
  "thesis_breaker": str               // what would invalidate the bear thesis
}"""


def _llm_run(system_prompt: str, slim: dict) -> dict | None:
    client = get_llm()
    if client is None:
        return None
    user = "Data:\n" + json.dumps(slim, indent=2, default=str)
    return client.chat_json(system_prompt, user, max_tokens=600, temperature=0.2)


class BullAgent:
    """LLM-first bullish narrative; rule-based fallback."""

    def run(self, analyst_summary: dict, ml_signal: dict) -> dict:
        slim = _slim_summary(analyst_summary, ml_signal)
        out = _llm_run(_BULL_SYSTEM, slim)
        if out and isinstance(out.get("bull_case"), str):
            out.setdefault("key_catalysts", [])
            out.setdefault("target_price", None)
            try:
                out["confidence"] = max(1, min(10, int(out.get("confidence") or 5)))
            except (TypeError, ValueError):
                out["confidence"] = 5
            out.setdefault("key_risk", "Standard market risk.")
            out["source"] = "llm"
            return out
        return self._rule_based(analyst_summary, ml_signal)

    def _rule_based(self, analyst_summary: dict, ml_signal: dict) -> dict:
        tech = analyst_summary.get("technical_analyst") or analyst_summary.get("technical") or {}
        fund = analyst_summary.get("fundamental_analyst") or analyst_summary.get("fundamental") or {}
        sent = analyst_summary.get("sentiment_analyst") or analyst_summary.get("sentiment") or {}
        macro = analyst_summary.get("macro_analyst") or analyst_summary.get("macro") or {}

        catalysts = []
        argument = []

        trend = tech.get("trend")
        rsi = tech.get("rsi")
        macd_signal = tech.get("macd_signal")
        cmp_ = _safe(tech, "indicators", "cmp")
        resistance = _safe(tech, "key_levels", "resistance")
        support = _safe(tech, "key_levels", "support")

        if trend == "uptrend":
            catalysts.append("Price above EMA50/EMA200 — uptrend intact")
            argument.append(f"Technical: uptrend with RSI {rsi}, MACD {macd_signal}.")
        if macd_signal == "bullish_crossover":
            catalysts.append("MACD bullish crossover")
        if rsi is not None and 50 <= rsi <= 65:
            catalysts.append(f"RSI {rsi} in healthy bullish zone")

        if fund.get("verdict") == "bullish":
            argument.append(
                f"Fundamentals: P/E {fund.get('pe')} vs sector median {fund.get('sector_median_pe')}, "
                f"ROE {fund.get('roe')}%, EPS growth {fund.get('eps_growth')}%."
            )
            catalysts.append("Fundamentals: undervalued vs sector with healthy ROE")
        elif fund.get("verdict") == "neutral" and (fund.get("eps_growth") or 0) > 0:
            argument.append(f"Fundamentals: EPS growth {fund.get('eps_growth')}% supports thesis.")

        if sent.get("verdict") == "positive":
            argument.append(f"Sentiment: 7-day average {sent.get('sentiment_score')} on {sent.get('article_count')} articles.")
            catalysts.append("Positive news flow")

        if macro.get("regime_label") == "Bull":
            argument.append(f"Macro: NIFTY in Bull regime, India VIX {macro.get('india_vix')}.")
            catalysts.append("Bull macro regime")
        elif macro.get("regime_label") == "Sideways" and macro.get("above_200ema"):
            catalysts.append("NIFTY holding above 200-EMA")

        pred = ml_signal.get("predicted_return_1d")
        rank = ml_signal.get("signal_rank")
        if pred is not None and pred > 0:
            argument.append(f"ML: predicted 1d return +{pred*100:.2f}% (rank {rank}).")
            if pred > 0.005:
                catalysts.append(f"ML signal high-conviction (+{pred*100:.2f}%)")

        target = None
        if cmp_ and resistance and resistance > cmp_:
            target = round(resistance, 2)
        elif cmp_:
            target = round(cmp_ * 1.04, 2)

        confidence = min(10, max(1, len(catalysts) * 2))
        risk = "Setup invalidates if price closes below support."
        if support:
            risk = f"Setup invalidates on close below {support}."

        return {
            "bull_case": " ".join(argument) or "Insufficient bullish data points.",
            "key_catalysts": catalysts,
            "target_price": target,
            "confidence": confidence,
            "key_risk": risk,
        }


class BearAgent:
    """LLM-first bearish narrative; rule-based fallback."""

    def run(self, analyst_summary: dict, ml_signal: dict) -> dict:
        slim = _slim_summary(analyst_summary, ml_signal)
        out = _llm_run(_BEAR_SYSTEM, slim)
        if out and isinstance(out.get("bear_case"), str):
            out.setdefault("key_risks", [])
            out.setdefault("downside_target", None)
            try:
                out["confidence"] = max(1, min(10, int(out.get("confidence") or 5)))
            except (TypeError, ValueError):
                out["confidence"] = 5
            out.setdefault("thesis_breaker", "Bear thesis breaks on close above resistance with volume.")
            out["source"] = "llm"
            return out
        return self._rule_based(analyst_summary, ml_signal)

    def _rule_based(self, analyst_summary: dict, ml_signal: dict) -> dict:
        tech = analyst_summary.get("technical_analyst") or analyst_summary.get("technical") or {}
        fund = analyst_summary.get("fundamental_analyst") or analyst_summary.get("fundamental") or {}
        sent = analyst_summary.get("sentiment_analyst") or analyst_summary.get("sentiment") or {}
        macro = analyst_summary.get("macro_analyst") or analyst_summary.get("macro") or {}

        risks = []
        argument = []

        trend = tech.get("trend")
        rsi = tech.get("rsi")
        macd_signal = tech.get("macd_signal")
        cmp_ = _safe(tech, "indicators", "cmp")
        support = _safe(tech, "key_levels", "support")

        if trend == "downtrend":
            risks.append("Price below EMA50/EMA200 — downtrend")
            argument.append(f"Technical: downtrend with RSI {rsi}, MACD {macd_signal}.")
        if macd_signal == "bearish_crossover":
            risks.append("MACD bearish crossover")
        if rsi is not None and rsi >= 70:
            risks.append(f"RSI {rsi} overbought — pullback risk")

        red_flags = fund.get("red_flags") or []
        for rf in red_flags:
            risks.append(rf)
        if fund.get("verdict") == "bearish":
            argument.append(
                f"Fundamentals: P/E {fund.get('pe')} vs sector {fund.get('sector_median_pe')}, "
                f"ROE {fund.get('roe')}%, red flags: {len(red_flags)}."
            )

        if sent.get("verdict") == "negative":
            argument.append(f"Sentiment: 7-day average {sent.get('sentiment_score')} on {sent.get('article_count')} articles.")
            risks.append("Negative news flow")
        if sent.get("sebi_news"):
            risks.append("SEBI/insider keyword in recent headlines")
        if sent.get("rbi_relevance"):
            risks.append(f"RBI policy in {sent.get('rbi_days_away')} days — event risk")

        if macro.get("regime_label") == "Bear":
            argument.append(f"Macro: Bear regime, India VIX {macro.get('india_vix')}.")
            risks.append("Bear macro regime")
        elif macro.get("regime_label") == "High-Vol":
            risks.append(f"High-Vol regime (VIX {macro.get('india_vix')})")

        pred = ml_signal.get("predicted_return_1d")
        rank = ml_signal.get("signal_rank")
        if pred is not None and pred < 0:
            argument.append(f"ML: predicted 1d return {pred*100:.2f}% (rank {rank}).")
            if pred < -0.005:
                risks.append(f"ML signal high-conviction ({pred*100:.2f}%)")

        downside_target = None
        if cmp_ and support and support < cmp_:
            downside_target = round(support, 2)
        elif cmp_:
            downside_target = round(cmp_ * 0.96, 2)

        confidence = min(10, max(1, len(risks) * 2))
        breaker = "Bear thesis breaks if price closes above resistance with volume."

        return {
            "bear_case": " ".join(argument) or "Insufficient bearish data points.",
            "key_risks": risks,
            "downside_target": downside_target,
            "confidence": confidence,
            "thesis_breaker": breaker,
            "source": "rule",
        }
