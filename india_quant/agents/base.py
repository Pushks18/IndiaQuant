"""Base analyst agent — rule-based, no LLM. Stores reports + manages memory.

Subclasses override `analyze(context, date_str) -> dict` to return a report
matching the appropriate Pydantic schema below. The base class handles tool
dispatch (kept so subclasses can call their data getters by name) and DB
persistence.
"""
import json
from datetime import datetime, date as date_type
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert

from india_quant.data.db import get_session
from india_quant.data.models import AnalystReport

MEMORY_DIR = Path(__file__).parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)


# ── Report schemas ────────────────────────────────────────────────────────────

class TechnicalReport(BaseModel):
    ticker: str
    date: str
    trend: str
    rsi: float | None = None
    macd_signal: str | None = None
    key_levels: dict = {}
    pattern: str | None = None
    outlook: str
    confidence: int


class FundamentalReport(BaseModel):
    ticker: str
    date: str
    pe: float | None = None
    pb: float | None = None
    roe: float | None = None
    eps_growth: float | None = None
    promoter_pct: float | None = None
    fii_pct: float | None = None
    verdict: str
    red_flags: list[str] = []


class SentimentReport(BaseModel):
    ticker: str
    date: str
    sentiment_score: float
    article_count: int
    top_headlines: list[str] = []
    rbi_relevance: bool = False
    sebi_news: bool = False
    verdict: str


class MacroReport(BaseModel):
    date: str
    nifty_regime: str
    rbi_stance: str
    fii_flow_7d: float | None = None
    usd_inr: float | None = None
    india_vix: float | None = None
    market_breadth: dict = {}
    regime_label: str


class AnalystSummary(BaseModel):
    tech: TechnicalReport | None = None
    fundamental: FundamentalReport | None = None
    sentiment: SentimentReport | None = None
    macro: MacroReport | None = None
    combined_verdict: str


# ── Base agent ────────────────────────────────────────────────────────────────

class BaseAnalystAgent:
    def __init__(self, name: str, tools: list[dict] | None = None):
        self.name = name
        self.tools = tools or []
        self._memory = self._load_memory()

    def _load_memory(self) -> dict:
        path = MEMORY_DIR / f"{self.name}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {"lessons": [], "principles": []}

    def _save_memory(self):
        path = MEMORY_DIR / f"{self.name}.json"
        path.write_text(json.dumps(self._memory, indent=2))

    def _call_tool(self, tool_name: str, tool_input: dict):
        """Helper: invoke a registered tool by name. Returns the raw result (not JSON-stringified)."""
        fn = self._tool_registry().get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**tool_input)
        except Exception as e:
            logger.error(f"[{self.name}] Tool {tool_name} error: {e}")
            return {"error": str(e)}

    def _tool_registry(self) -> dict:
        return {}

    # ── Public API ────────────────────────────────────────────────────────

    def analyze(self, context: dict, date_str: str) -> dict:
        """Deterministic analysis. Subclasses MUST override."""
        raise NotImplementedError

    def run(self, context: dict, date: str) -> dict:
        """Run rule-based analysis and persist the report."""
        try:
            report = self.analyze(context, date)
        except Exception as e:
            logger.error(f"[{self.name}] analyze() failed: {e}")
            report = {"error": str(e), "ticker": context.get("ticker", ""), "date": date}
        self._store_report(report, date, context.get("ticker", ""))
        return report

    def _store_report(self, report: dict, date_str: str, ticker: str):
        try:
            d = date_type.fromisoformat(date_str) if isinstance(date_str, str) else date_str
            with get_session() as session:
                stmt = insert(AnalystReport).values(
                    ticker=ticker,
                    date=d,
                    agent_name=self.name,
                    report_json=json.dumps(report, default=str),
                )
                session.execute(stmt)
        except Exception as e:
            logger.error(f"[{self.name}] Failed to store report: {e}")

    def update_memory(self, lesson: str):
        """Append a lesson. Periodically distil into principles via simple aggregation."""
        self._memory["lessons"].append({
            "date": datetime.now().isoformat(),
            "lesson": lesson,
        })
        if len(self._memory["lessons"]) % 30 == 0:
            recent = [l["lesson"] for l in self._memory["lessons"][-30:]]
            self._memory["principles"] = list(dict.fromkeys(recent))[:10]
        self._save_memory()
