"""Multi-tier LLM extraction orchestrator (Phase III).

Ties together the fallback chain, chunking, and rate-limit handling:

  * For each record, walk the provider chain (Gemini -> Groq -> DeepSeek).
  * Before each call, chunk the payload to THAT provider's token budget so a
    413 is structurally impossible; if a 413 still occurs, halve the budget and
    retry once on the same provider before falling through.
  * On 429, back off (exponential + jitter, respecting Retry-After when given)
    up to N times, then fall through to the next provider.
  * On provider-unavailable, fall through immediately.
  * If every provider fails, raise — the caller quarantines the record (it is
    NOT emitted with guessed data).
"""
from __future__ import annotations

import asyncio
import random

from src.core.logging import get_logger
from src.llm.chunking import count_tokens, salient_truncate
from src.llm.prompts import SYSTEM, build_user_prompt
from src.llm.providers import (
    LLMProvider,
    ProviderPayloadTooLarge,
    ProviderRateLimited,
    ProviderUnavailable,
    build_provider_chain,
)

log = get_logger(__name__)

_PROMPT_OVERHEAD = 800  # tokens reserved for system + schema + wrapper


class AllProvidersFailed(Exception):
    pass


class LLMOrchestrator:
    def __init__(self, chain: list[LLMProvider] | None = None, max_429_retries: int = 4) -> None:
        self._chain = chain or build_provider_chain()
        self._max_429 = max_429_retries

    async def extract(self, record_type: str, text: str) -> dict:
        last_err: Exception | None = None
        for provider in self._chain:
            try:
                return await self._call_with_backoff(provider, record_type, text)
            except (ProviderUnavailable, AllProvidersFailed) as e:
                last_err = e
                log.warning("provider_fell_through", provider=provider.name, error=str(e))
                continue
        raise AllProvidersFailed(str(last_err) if last_err else "no providers")

    async def _call_with_backoff(
        self, provider: LLMProvider, record_type: str, text: str
    ) -> dict:
        budget = provider.max_input_tokens - _PROMPT_OVERHEAD
        payload = self._fit(text, budget)

        for attempt in range(self._max_429 + 1):
            try:
                user = build_user_prompt(record_type, payload)
                return await provider.complete_json(SYSTEM, user)
            except ProviderRateLimited as e:
                if attempt == self._max_429:
                    raise ProviderUnavailable("429 budget exhausted") from e
                delay = e.retry_after or _backoff(attempt)
                log.info("llm_429_backoff", provider=provider.name, attempt=attempt, delay=delay)
                await asyncio.sleep(delay)
            except ProviderPayloadTooLarge:
                budget //= 2
                payload = self._fit(text, budget)
                log.info("llm_413_reduce", provider=provider.name, new_budget=budget)
                if budget < 500:
                    raise ProviderUnavailable("413 even after chunking") from None
        raise ProviderUnavailable("unreachable")

    @staticmethod
    def _fit(text: str, budget: int) -> str:
        if count_tokens(text) <= budget:
            return text
        return salient_truncate(text, max_tokens=budget)


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter: base 1s, cap 60s."""
    return random.uniform(0, min(60.0, 2**attempt))
