"""FinCon-style memory manager: record outcomes, extract lessons, update principles."""
import json
from datetime import date, datetime
from pathlib import Path

import anthropic
from loguru import logger

from india_quant.config import cfg

MEMORY_DIR = Path(__file__).parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)
OUTCOMES_FILE = MEMORY_DIR / "trade_outcomes.json"
PRINCIPLES_FILE = MEMORY_DIR / "agent_principles.json"

MODEL = "claude-sonnet-4-20250514"


class MemoryManager:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._outcomes = self._load_json(OUTCOMES_FILE, default=[])
        self._principles = self._load_json(PRINCIPLES_FILE, default={})

    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return default

    def record_trade_outcome(self, trade_proposal: dict, actual_outcome: dict):
        """
        Record what happened after the trade's time horizon passed.
        actual_outcome: {
            hit_target_1: bool,
            hit_stop: bool,
            max_adverse_excursion: float,
            max_favorable_excursion: float,
            realized_pnl_pct: float,
        }
        """
        entry = {
            "date": date.today().isoformat(),
            "ticker": trade_proposal.get("ticker"),
            "instrument": trade_proposal.get("instrument"),
            "direction": trade_proposal.get("direction"),
            "time_horizon": trade_proposal.get("time_horizon"),
            "rationale": trade_proposal.get("rationale", "")[:200],
            "nse_risks": trade_proposal.get("nse_risks", []),
            "outcome": actual_outcome,
            "lesson": None,
        }

        # Extract lesson via Claude
        lesson = self.extract_lesson(trade_proposal, actual_outcome)
        entry["lesson"] = lesson

        self._outcomes.append(entry)
        OUTCOMES_FILE.write_text(json.dumps(self._outcomes, indent=2))
        logger.info(f"[MemoryManager] Recorded outcome for {entry['ticker']}: {lesson}")

        # Update principles every 30 lessons
        if len(self._outcomes) % 30 == 0:
            self.update_agent_principles()

    def extract_lesson(self, trade_proposal: dict, outcome: dict) -> str:
        """Call Claude to extract a one-sentence lesson from this trade outcome."""
        prompt = (
            f"Trade: {json.dumps(trade_proposal, default=str)}\n\n"
            f"Outcome: {json.dumps(outcome, default=str)}\n\n"
            f"Extract a single, concise lesson for future Indian equity trading. "
            f"Be specific: cite the ticker, setup, and what happened. "
            f"Example: 'BANKNIFTY long trades before RBI week had 70% stop rate — avoid.'"
        )
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"[MemoryManager] Lesson extraction failed: {e}")
            return f"Trade {'won' if outcome.get('hit_target_1') else 'lost'}: {trade_proposal.get('ticker', 'unknown')}"

    def update_agent_principles(self):
        """Every 30 lessons, distill into 5-10 principles for all agents."""
        lessons = [o["lesson"] for o in self._outcomes if o.get("lesson")]
        if not lessons:
            return

        lessons_text = "\n".join(f"- {l}" for l in lessons[-60:])
        prompt = (
            f"Distill these trading lessons for an Indian equity analyst into "
            f"5-10 actionable, concise principles. Be specific to Indian markets.\n\n"
            f"{lessons_text}\n\nReturn a JSON list of strings."
        )
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            principles = json.loads(response.content[0].text)
            self._principles["general"] = principles
            self._principles["updated_at"] = datetime.now().isoformat()
            PRINCIPLES_FILE.write_text(json.dumps(self._principles, indent=2))
            logger.info(f"[MemoryManager] Principles updated: {len(principles)} principles")
        except Exception as e:
            logger.error(f"[MemoryManager] Principle update failed: {e}")

    def get_relevant_lessons(self, context: dict) -> list[str]:
        """
        Simple keyword retrieval: match ticker, sector, event_type against stored lessons.
        Returns top 3 most relevant lessons.
        """
        ticker = context.get("ticker", "").replace(".NS", "").upper()
        sector = context.get("sector", "").lower()
        event = context.get("event_type", "").lower()

        scored = []
        for outcome in self._outcomes:
            lesson = outcome.get("lesson", "") or ""
            score = 0
            if ticker and ticker in lesson.upper():
                score += 3
            if sector and sector in lesson.lower():
                score += 2
            if event and event in lesson.lower():
                score += 2
            if "expiry" in lesson.lower() and "expiry" in event:
                score += 1
            if score > 0:
                scored.append((score, lesson))

        scored.sort(reverse=True)
        return [lesson for _, lesson in scored[:3]]

    def get_principles(self, agent: str = "general") -> list[str]:
        """Return current principles for an agent."""
        return self._principles.get(agent, self._principles.get("general", []))
