"""News fetcher + FinBERT sentiment scorer for Indian equities."""
import hashlib
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from urllib.parse import quote_plus

import feedparser
import requests
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from india_quant.config import cfg
from india_quant.data.db import get_session
from india_quant.data.models import NewsArticle, SentimentAggregate

# Lazy imports for heavy models
_finbert = None
_tokenizer = None
_finbert_failed = False
_vader = None


def _load_finbert():
    """Load FinBERT directly via BertTokenizer (AutoTokenizer's converter is unreliable)."""
    global _finbert, _tokenizer, _finbert_failed
    if _finbert_failed:
        return None, None
    if _finbert is None:
        try:
            from transformers.models.bert.tokenization_bert import BertTokenizer
            from transformers import BertForSequenceClassification
            logger.info("Loading FinBERT model (first run — ~400MB download)...")
            _tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            _finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            _finbert.eval()
            logger.info("FinBERT loaded.")
        except Exception as e:
            logger.warning(f"FinBERT load failed ({e}); falling back to VADER lexicon.")
            _finbert_failed = True
            return None, None
    return _tokenizer, _finbert


def _load_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


class NewsFetcher:
    GOOGLE_RSS_URL = (
        "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    FINNHUB_URL = "https://finnhub.io/api/v1/news?category=general&token={key}"
    INDIA_KEYWORDS = {"NSE", "BSE", "SEBI", "RBI", "India", "NIFTY", "SENSEX", "Nifty", "BSE"}

    def fetch_google_rss(self, ticker_name: str, max_articles: int = 20) -> list[dict]:
        """Fetch articles from Google News RSS for a ticker."""
        query = quote_plus(f"{ticker_name} NSE stock India")
        url = self.GOOGLE_RSS_URL.format(query=query)
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:max_articles]:
                articles.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", ""),
                    "source": "google_rss",
                })
            return articles
        except Exception as e:
            logger.error(f"Google RSS fetch failed for {ticker_name}: {e}")
            return []

    def fetch_finnhub(self, category: str = "general", count: int = 50) -> list[dict]:
        """Fetch news from Finnhub, filtering for India-relevant articles."""
        url = self.FINNHUB_URL.format(key=cfg.finnhub_key)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            news = resp.json()
            india_news = []
            for article in news[:count]:
                headline = article.get("headline", "") + " " + article.get("summary", "")
                if any(kw in headline for kw in self.INDIA_KEYWORDS):
                    india_news.append({
                        "title": article.get("headline", ""),
                        "link": article.get("url", ""),
                        "published": str(article.get("datetime", "")),
                        "summary": article.get("summary", ""),
                        "source": "finnhub",
                    })
            return india_news
        except Exception as e:
            logger.error(f"Finnhub fetch failed: {e}")
            return []

    def fetch_newsapi(self, query: str, days_back: int = 3) -> list[dict]:
        """Fetch from NewsAPI, focused on Indian financial domains."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "domains": "economictimes.indiatimes.com,livemint.com,moneycontrol.com,business-standard.com",
            "from": from_date,
            "language": "en",
            "apiKey": cfg.newsapi_key,
            "pageSize": 20,
        }
        try:
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "link": a.get("url", ""),
                    "published": a.get("publishedAt", ""),
                    "summary": a.get("description", ""),
                    "source": "newsapi",
                }
                for a in articles
            ]
        except Exception as e:
            logger.error(f"NewsAPI fetch failed for '{query}': {e}")
            return []

    def score_sentiment(self, articles: list[dict]) -> list[dict]:
        """Score each article's headline with FinBERT (or VADER fallback). Returns score in [-1, +1]."""
        if not articles:
            return []
        tokenizer, model = _load_finbert()
        if tokenizer is None or model is None:
            return self._score_with_vader(articles)

        import torch
        headlines = [a["title"] for a in articles]
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        scored = []
        batch_size = 16

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i: i + batch_size]
            try:
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).numpy()
                labels = model.config.id2label
                for j, prob in enumerate(probs):
                    top_idx = prob.argmax()
                    label = labels[top_idx].lower()
                    confidence = float(prob[top_idx])
                    score = label_map.get(label, 0.0) * confidence
                    article = articles[i + j].copy()
                    article["sentiment_score"] = score
                    scored.append(article)
            except Exception as e:
                logger.error(f"FinBERT scoring batch {i}: {e}")
                vader_scored = self._score_with_vader(articles[i: i + batch_size])
                scored.extend(vader_scored)

        return scored

    def _score_with_vader(self, articles: list[dict]) -> list[dict]:
        """Lexicon-based fallback when FinBERT is unavailable."""
        analyzer = _load_vader()
        scored = []
        for a in articles:
            try:
                vs = analyzer.polarity_scores(a.get("title", ""))
                score = float(vs.get("compound", 0.0))
            except Exception:
                score = 0.0
            article = a.copy()
            article["sentiment_score"] = score
            scored.append(article)
        return scored

    def fetch_and_store(self, tickers: list[str]) -> int:
        """Fetch news from all sources, score, upsert to DB, aggregate by ticker-day."""
        total = 0
        for ticker in tickers:
            ticker_name = ticker.replace(".NS", "").replace(".BO", "")
            articles = (
                self.fetch_google_rss(ticker_name)
                + self.fetch_newsapi(ticker_name)
            )
            articles = self.score_sentiment(articles)

            daily_scores: dict[str, list[float]] = {}

            with get_session() as session:
                for a in articles:
                    url = a.get("link", "")
                    if not url:
                        continue
                    pub_raw = a.get("published", "")
                    try:
                        pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                    except Exception:
                        pub_dt = datetime.now(tz=timezone.utc)

                    score = a.get("sentiment_score", 0.0)
                    day_key = pub_dt.date().isoformat()
                    daily_scores.setdefault(day_key, []).append(score)

                    stmt = insert(NewsArticle).values(
                        timestamp=pub_dt,
                        source=a.get("source", ""),
                        tickers=[ticker],
                        headline=a.get("title", ""),
                        content=a.get("summary", ""),
                        url=url,
                        sentiment_score=score,
                    ).on_conflict_do_nothing(index_elements=["url"])
                    session.execute(stmt)
                    total += 1

            # Write daily sentiment aggregates
            with get_session() as session:
                for day_str, scores in daily_scores.items():
                    import datetime as dt_mod
                    day = dt_mod.date.fromisoformat(day_str)
                    stmt = insert(SentimentAggregate).values(
                        ticker=ticker,
                        date=day,
                        avg_score=sum(scores) / len(scores),
                        article_count=len(scores),
                    ).on_conflict_do_update(
                        index_elements=["ticker", "date"],
                        set_={"avg_score": sum(scores) / len(scores), "article_count": len(scores)},
                    )
                    session.execute(stmt)

        return total


if __name__ == "__main__":
    fetcher = NewsFetcher()
    articles = fetcher.fetch_google_rss("RELIANCE NSE stock")
    print(f"Google RSS: {len(articles)} articles")
    if articles:
        scored = fetcher.score_sentiment(articles[:5])
        for a in scored[:3]:
            print(f"  {a['sentiment_score']:.3f}  {a['title'][:80]}")
