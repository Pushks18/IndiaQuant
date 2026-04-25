"""Thin OpenRouter client (OpenAI-compatible chat completions).

Designed to fail safely: every call returns either a parsed dict or None — never
raises into the agent. Callers fall back to rule-based logic when None is returned.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests
from loguru import logger

from india_quant.config import cfg

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 2

# Match the first balanced {...} block in a string. Free models occasionally
# wrap JSON in prose or markdown fences — this trims that down.
_JSON_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def is_available() -> bool:
    return bool(cfg.openrouter_api_key)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        self.api_key = api_key or cfg.openrouter_api_key
        self.model = model or cfg.openrouter_model
        self.fallback_model = fallback_model or cfg.openrouter_model_fallback
        self.timeout = timeout
        self.retries = retries

    # ── Core HTTP ────────────────────────────────────────────────────────

    def _post(self, payload: dict, model: str) -> dict | None:
        if not self.api_key:
            return None
        body = {**payload, "model": model}
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost/india-quant",
                    "X-Title": "India Quant",
                },
                json=body,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            logger.warning(f"[LLM] {model}: network error {e}")
            return None

        if resp.status_code == 429:
            logger.warning(f"[LLM] {model}: rate-limited")
            return None
        if resp.status_code >= 500:
            logger.warning(f"[LLM] {model}: HTTP {resp.status_code}")
            return None
        try:
            data = resp.json()
        except Exception:
            logger.warning(f"[LLM] {model}: non-JSON response: {resp.text[:120]}")
            return None
        if "error" in data:
            logger.warning(f"[LLM] {model}: provider error: {data['error'].get('message','')[:120]}")
            return None
        return data

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str | None:
        """Return raw text from the assistant or None on failure."""
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        models = [self.model, self.fallback_model]
        for attempt in range(self.retries):
            for m in models:
                data = self._post(payload, m)
                if not data:
                    continue
                try:
                    text = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    continue
                if text and text.strip():
                    return text
            time.sleep(0.5 * (attempt + 1))
        return None

    def chat_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> dict | None:
        """Like chat() but parses the response as JSON, robust to stray prose."""
        text = self.chat(system, user, max_tokens=max_tokens,
                         temperature=temperature, json_mode=True)
        if not text:
            return None
        return parse_json_loose(text)


# ── Helpers ─────────────────────────────────────────────────────────────────


def parse_json_loose(text: str) -> dict | None:
    """Try strict JSON, then extract the first balanced object."""
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip ```json fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Last resort: first balanced object
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


_singleton: OpenRouterClient | None = None


def get_client() -> OpenRouterClient | None:
    global _singleton
    if not is_available():
        return None
    if _singleton is None:
        _singleton = OpenRouterClient()
    return _singleton
