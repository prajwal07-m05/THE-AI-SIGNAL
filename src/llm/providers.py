"""LLM provider adapters with a uniform async interface.

Each provider exposes `async complete_json(system, user) -> dict`. They raise
`ProviderRateLimited` on 429 and `ProviderPayloadTooLarge` on 413 so the
orchestrator can decide whether to back off (same provider) or fall through
(next provider). Providers requiring context beyond their budget are handled by
the chunker before they are ever called.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

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
    #: Maximum input tokens we allow before chunking kicks in.
    max_input_tokens: int

    @abstractmethod
    async def complete_json(self, system: str, user: str) -> dict:
        ...

    @staticmethod
    def _loads(raw: str) -> dict:
        raw = raw.strip()

        if raw.startswith("```"):
            parts = raw.split("```", 2)

            if len(parts) >= 2:
                raw = parts[1].strip()

                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

        result = json.loads(raw)

        if not isinstance(result, dict):
            raise ValueError("LLM response must be a JSON object")

        return result


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the current google-genai SDK."""

    name = "gemini-2.5-flash"
    max_input_tokens = 900_000

    def __init__(self, api_key: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)

    async def complete_json(
        self,
        system: str,
        user: str,
    ) -> dict:
        from google.genai import types

        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.name,
                contents=[
                    system,
                    user,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )

            text = response.text or ""

            return self._loads(text)

        except Exception as e:  # noqa: BLE001
            status_code = getattr(e, "status_code", None)

            message = str(e).lower()

            if status_code == 429 or "429" in message:
                retry_after = _extract_retry_after(e)
                raise ProviderRateLimited(
                    retry_after=retry_after,
                ) from e

            if (
                status_code == 413
                or "413" in message
                or "payload too large" in message
                or "request too large" in message
                or "too large" in message
            ):
                raise ProviderPayloadTooLarge() from e

            raise ProviderUnavailable(str(e)) from e


class GroqProvider(LLMProvider):
    name = "groq-llama-3.3-70b"
    max_input_tokens = 30_000

    def __init__(self, api_key: str) -> None:
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=api_key)

    async def complete_json(
        self,
        system: str,
        user: str,
    ) -> dict:
        from groq import APIStatusError, RateLimitError

        try:
            resp = await self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
                response_format={
                    "type": "json_object",
                },
                temperature=0,
            )

            return self._loads(
                resp.choices[0].message.content or "{}"
            )

        except RateLimitError as e:
            retry_after = _extract_retry_after(e)

            raise ProviderRateLimited(
                retry_after=retry_after,
            ) from e

        except APIStatusError as e:
            if e.status_code == 413:
                raise ProviderPayloadTooLarge() from e

            raise ProviderUnavailable(str(e)) from e

        except Exception as e:  # noqa: BLE001
            raise ProviderUnavailable(str(e)) from e


class DeepSeekProvider(LLMProvider):
    name = "deepseek-chat"
    max_input_tokens = 60_000

    def __init__(self, api_key: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    async def complete_json(
        self,
        system: str,
        user: str,
    ) -> dict:
        from openai import APIStatusError, RateLimitError

        try:
            resp = await self._client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
                response_format={
                    "type": "json_object",
                },
                temperature=0,
            )

            return self._loads(
                resp.choices[0].message.content or "{}"
            )

        except RateLimitError as e:
            retry_after = _extract_retry_after(e)

            raise ProviderRateLimited(
                retry_after=retry_after,
            ) from e

        except APIStatusError as e:
            if e.status_code == 413:
                raise ProviderPayloadTooLarge() from e

            raise ProviderUnavailable(str(e)) from e

        except Exception as e:  # noqa: BLE001
            raise ProviderUnavailable(str(e)) from e


def _extract_retry_after(error: Any) -> float | None:
    """Best-effort extraction of Retry-After from provider exceptions."""

    response = getattr(error, "response", None)

    if response is None:
        return None

    headers = getattr(response, "headers", None)

    if not headers:
        return None

    value = headers.get("retry-after")

    if value is None:
        value = headers.get("Retry-After")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_provider_chain() -> list[LLMProvider]:
    """Instantiate providers in fallback order, skipping any without a key."""

    settings = get_settings()

    chain: list[LLMProvider] = []

    if settings.gemini_api_key:
        chain.append(
            GeminiProvider(
                settings.gemini_api_key,
            )
        )

    if settings.groq_api_key:
        chain.append(
            GroqProvider(
                settings.groq_api_key,
            )
        )

    if settings.deepseek_api_key:
        chain.append(
            DeepSeekProvider(
                settings.deepseek_api_key,
            )
        )

    if not chain:
        raise RuntimeError(
            "No LLM API keys configured — "
            "set at least one in .env"
        )

    log.info(
        "llm_chain",
        providers=[provider.name for provider in chain],
    )

    return chain