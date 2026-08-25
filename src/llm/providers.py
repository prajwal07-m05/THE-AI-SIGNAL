"""LLM provider adapters with a uniform async interface.

Each provider exposes `async complete_json(system, user) -> dict`. They raise
`ProviderRateLimited` on 429 and `ProviderPayloadTooLarge` on 413 so the
orchestrator can decide whether to back off (same provider) or fall through
(next provider). Providers requiring context beyond their budget are handled by
the chunker before they are ever called.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from src.core.logging import get_logger
from src.settings import get_settings

log = get_logger(__name__)


class ProviderRateLimited(Exception):
    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("429 rate limited")
        self.retry_after = retry_after


class ProviderPayloadTooLarge(Exception):
    """413 — caller must re-chunk with a smaller budget and retry."""


class ProviderUnavailable(Exception):
    """Auth/quota/network failure — orchestrator should fall through."""


class LLMProvider(ABC):
    name: str
    #: max input tokens we allow before chunking kicks in
    max_input_tokens: int

    @abstractmethod
    async def complete_json(self, system: str, user: str) -> dict:
        ...

    @staticmethod
    def _loads(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].removeprefix("json").strip()
        return json.loads(raw)


class GeminiProvider(LLMProvider):
    name = "gemini-1.5-flash"
    max_input_tokens = 900_000  # 1M context; leave headroom

    def __init__(self, api_key: str) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")

    async def complete_json(self, system: str, user: str) -> dict:
        from google.api_core import exceptions as gexc

        try:
            resp = await self._model.generate_content_async(
                [system, user],
                generation_config={"response_mime_type": "application/json"},
            )
            return self._loads(resp.text)
        except gexc.ResourceExhausted as e:
            raise ProviderRateLimited() from e
        except gexc.InvalidArgument as e:
            if "too large" in str(e).lower() or "413" in str(e):
                raise ProviderPayloadTooLarge() from e
            raise ProviderUnavailable(str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise ProviderUnavailable(str(e)) from e


class GroqProvider(LLMProvider):
    name = "groq-llama-3.3-70b"
    max_input_tokens = 30_000  # 32k context

    def __init__(self, api_key: str) -> None:
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=api_key)

    async def complete_json(self, system: str, user: str) -> dict:
        from groq import APIStatusError, RateLimitError

        try:
            resp = await self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return self._loads(resp.choices[0].message.content or "{}")
        except RateLimitError as e:
            raise ProviderRateLimited() from e
        except APIStatusError as e:
            if e.status_code == 413:
                raise ProviderPayloadTooLarge() from e
            raise ProviderUnavailable(str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise ProviderUnavailable(str(e)) from e


class DeepSeekProvider(LLMProvider):
    name = "deepseek-chat"
    max_input_tokens = 60_000  # 64k context

    def __init__(self, api_key: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key, base_url="https://api.deepseek.com"
        )

    async def complete_json(self, system: str, user: str) -> dict:
        from openai import APIStatusError, RateLimitError

        try:
            resp = await self._client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return self._loads(resp.choices[0].message.content or "{}")
        except RateLimitError as e:
            raise ProviderRateLimited() from e
        except APIStatusError as e:
            if e.status_code == 413:
                raise ProviderPayloadTooLarge() from e
            raise ProviderUnavailable(str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise ProviderUnavailable(str(e)) from e


def build_provider_chain() -> list[LLMProvider]:
    """Instantiate providers in fallback order, skipping any without a key."""
    s = get_settings()
    chain: list[LLMProvider] = []
    if s.gemini_api_key:
        chain.append(GeminiProvider(s.gemini_api_key))
    if s.groq_api_key:
        chain.append(GroqProvider(s.groq_api_key))
    if s.deepseek_api_key:
        chain.append(DeepSeekProvider(s.deepseek_api_key))
    if not chain:
        raise RuntimeError("No LLM API keys configured — set at least one in .env")
    log.info("llm_chain", providers=[p.name for p in chain])
    return chain
