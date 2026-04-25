"""LLM integration layer. All callers should import from openrouter."""
from india_quant.llm.openrouter import (
    OpenRouterClient,
    get_client,
    is_available,
)

__all__ = ["OpenRouterClient", "get_client", "is_available"]
