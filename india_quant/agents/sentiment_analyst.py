"""Sentiment analyst — LLM-first via OpenRouter, deterministic FinBERT/VADER fallback."""
from datetime import date

from loguru import logger

from india_quant.agents.base import BaseAnalystAgent
from india_quant.config import cfg
from india_quant.llm import get_client as get_llm


def _get_sentiment_aggregate(ticker: str, days: int = 7) -> dict:
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(
                text(f"""
                    SELECT date, avg_score, article_count FROM sentiment_aggregate
                    WHERE ticker = :t AND date >= NOW() - INTERVAL '{int(days)} days'
                    ORDER BY date DESC
                """),
                {"t": ticker},
            ).fetchall()
        if not rows:
            return {"avg_score_7d": 0.0, "total_articles": 0, "daily": []}
        avg = sum(float(r[1] or 0) for r in rows) / len(rows)
        total = sum(int(r[2] or 0) for r in rows)
        return {
            "avg_score_7d": round(avg, 3),
            "total_articles": total,
            "daily": [{"date": str(r[0]), "score": float(r[1] or 0), "count": int(r[2] or 0)} for r in rows],
        }
    except Exception as e:
        return {"avg_score_7d": 0.0, "total_articles": 0, "error": str(e)}


def _get_recent_headlines(ticker: str, n: int = 10) -> list[dict]:
    try:
        from india_quant.data.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT headline, sentiment_score, timestamp FROM news_article
                    WHERE :t = ANY(tickers)
                    ORDER BY timestamp DESC LIMIT :n
                """),
                {"t": ticker, "n": n},
            ).fetchall()
        return [
            {"headline": r[0], "score": float(r[1] or 0), "timestamp": str(r[2])}
            for r in rows
        ]
    except Exception:
        return []


def _rbi_calendar_status() -> dict:
    rbi_dates = [date.fromisoformat(d) for d in cfg.rbi_policy_dates]
    today = date.today()
    upcoming = [d for d in rbi_dates if d >= today]
    if not upcoming:
        return {"days_to_next": None, "next_date": None}
    nxt = min(upcoming)
    return {"days_to_next": (nxt - today).days, "next_date": nxt.isoformat()}


def _has_sebi_keyword(headlines: list[dict]) -> bool:
    keywords = ("sebi", "insider", "fraud", "investigation", "probe")
    return any(any(k in (h.get("headline") or "").lower() for k in keywords) for h in headlines)


_LLM_SYSTEM = """You are a sentiment analyst for Indian equities (NSE/BSE).
Given a list of recent news headlines about a single stock, return a single JSON object:

{
  "score": float in [-1, 1],          // overall directional sentiment
  "verdict": "positive" | "negative" | "neutral",
  "confidence": int in [1, 10],        // higher = stronger signal
  "key_themes": [str, ...],            // 1-3 short bullets
  "high_impact_event": str | null,     // e.g. "earnings beat", "M&A rumour", "downgrade", "fraud probe"
  "rbi_relevance": bool,               // headlines reference RBI policy / rates
  "sebi_news": bool                    // headlines reference SEBI / insider trading / fraud
}

Rules:
- Weigh recent + concrete events (earnings, guidance, regulatory, M&A) over generic commentary.
- "Stock target raised" or "buy rating" without numbers is weak evidence — score moderate.
- Article count < 3 → cap confidence at 4.
- Output JSON only — no prose, no markdown fences."""


class SentimentAnalystAgent(BaseAnalystAgent):
    def __init__(self):
        super().__init__("sentiment_analyst")

    def _tool_registry(self) -> dict:
        return {
            "get_sentiment_aggregate": _get_sentiment_aggregate,
            "get_recent_headlines": _get_recent_headlines,
        }

    # ── Rule-based fallback ──────────────────────────────────────────────

    def _rule_based(self, ticker: str, date_str: str, agg: dict, headlines: list[dict],
                    rbi: dict) -> dict:
        avg = agg.get("avg_score_7d", 0.0)
        n = agg.get("total_articles", 0)
        if n < 3:
            verdict = "neutral"
        elif avg >= 0.2:
            verdict = "positive"
        elif avg <= -0.2:
            verdict = "negative"
        else:
            verdict = "neutral"
        return {
            "ticker": ticker,
            "date": date_str,
            "sentiment_score": round(float(avg), 3),
            "article_count": int(n),
            "top_headlines": [h["headline"] for h in headlines[:5]],
            "rbi_relevance": rbi.get("days_to_next") is not None and rbi["days_to_next"] <= 5,
            "sebi_news": _has_sebi_keyword(headlines),
            "verdict": verdict,
            "confidence": min(8, n) if n else 1,
            "key_themes": [],
            "high_impact_event": None,
            "source": "rule",
            "rbi_next_date": rbi.get("next_date"),
            "rbi_days_away": rbi.get("days_to_next"),
        }

    # ── LLM-augmented path ───────────────────────────────────────────────

    def _llm_path(self, ticker: str, date_str: str, headlines: list[dict],
                  agg: dict, rbi: dict) -> dict | None:
        client = get_llm()
        if client is None or not headlines:
            return None
        ticker_short = ticker.replace(".NS", "").replace(".BO", "")
        joined = "\n".join(
            f"- ({h.get('timestamp','')}) {h.get('headline','')}"
            for h in headlines[:15]
        )
        user = (
            f"Stock: {ticker_short} (NSE)\n"
            f"As of: {date_str}\n"
            f"Recent headlines (most recent first):\n{joined}\n\n"
            "Return the JSON object."
        )
        out = client.chat_json(_LLM_SYSTEM, user, max_tokens=400, temperature=0.1)
        if not out:
            return None
        # Sanitize / clamp
        try:
            score = float(out.get("score", 0.0))
            score = max(-1.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.0
        verdict = (out.get("verdict") or "neutral").lower()
        if verdict not in ("positive", "negative", "neutral"):
            verdict = "neutral"
        return {
            "ticker": ticker,
            "date": date_str,
            "sentiment_score": round(score, 3),
            "article_count": agg.get("total_articles", len(headlines)),
            "top_headlines": [h["headline"] for h in headlines[:5]],
            "rbi_relevance": bool(out.get("rbi_relevance")) or (
                rbi.get("days_to_next") is not None and rbi["days_to_next"] <= 5
            ),
            "sebi_news": bool(out.get("sebi_news")),
            "verdict": verdict,
            "confidence": int(out.get("confidence") or 5),
            "key_themes": list(out.get("key_themes") or [])[:5],
            "high_impact_event": out.get("high_impact_event") or None,
            "source": "llm",
            "rbi_next_date": rbi.get("next_date"),
            "rbi_days_away": rbi.get("days_to_next"),
        }

    def analyze(self, context: dict, date_str: str) -> dict:
        ticker = context.get("ticker", "")
        agg = _get_sentiment_aggregate(ticker, 7)
        headlines = _get_recent_headlines(ticker, 15)
        rbi = _rbi_calendar_status()

        report = self._llm_path(ticker, date_str, headlines, agg, rbi)
        if report is None:
            report = self._rule_based(ticker, date_str, agg, headlines, rbi)
        else:
            logger.info(f"[sentiment_analyst] {ticker}: LLM verdict={report['verdict']} "
                        f"conf={report['confidence']} event={report['high_impact_event']}")
        return report
