"""LLM provider adapters with a uniform async interface.

Each provider exposes:

    async complete_json(system, user) -> dict

Providers translate provider-specific failures into a small set of
pipeline-level exceptions so the orchestrator can implement:

    Gemini -> Groq -> DeepSeek

fallback behavior consistently.

Model IDs are configuration-driven through Settings.

Provider constructors intentionally accept only the API key so the
provider-chain interface remains backwards compatible with existing
callers and tests.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from src.core.logging import get_logger
from src.settings import get_settings


log = get_logger(__name__)


# ============================================================================
# DEFAULT MODEL CONFIGURATION
# ============================================================================

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


# ============================================================================
# PROVIDER EXCEPTIONS
# ============================================================================


class ProviderRateLimited(Exception):
    """Provider returned a rate-limit response."""

    def __init__(
        self,
        retry_after: float | None = None,
    ) -> None:
        super().__init__("429 rate limited")
        self.retry_after = retry_after


class ProviderPayloadTooLarge(Exception):
    """Provider rejected the request because it was too large."""


class ProviderUnavailable(Exception):
    """Provider cannot currently serve the request."""


# ============================================================================
# BASE PROVIDER
# ============================================================================


class LLMProvider(ABC):
    """Uniform asynchronous interface for all LLM providers."""

    name: str
    max_input_tokens: int

    @abstractmethod
    async def complete_json(
        self,
        system: str,
        user: str,
    ) -> dict:
        """Generate and parse a JSON object."""
        ...

    @staticmethod
    def _loads(raw: str) -> dict:
        """Parse provider output into a JSON object."""

        raw = raw.strip()

        if not raw:
            raise ValueError(
                "LLM provider returned an empty response"
            )

        # Support fenced JSON such as:
        #
        # ```json
        # {...}
        # ```
        if raw.startswith("```"):
            parts = raw.split("```", 2)

            if len(parts) >= 2:
                raw = parts[1].strip()

                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

        result = json.loads(raw)

        if not isinstance(result, dict):
            raise ValueError(
                "LLM response must be a JSON object"
            )

        return result


# ============================================================================
# GEMINI
# ============================================================================


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the google-genai SDK."""

    name = DEFAULT_GEMINI_MODEL

    # Gemini 2.5 Flash supports a very large context window.
    max_input_tokens = 900_000

    def __init__(
        self,
        api_key: str,
    ) -> None:
        from google import genai

        settings = get_settings()

        self.model = (
            getattr(
                settings,
                "gemini_model",
                DEFAULT_GEMINI_MODEL,
            )
            or DEFAULT_GEMINI_MODEL
        )

        self._client = genai.Client(
            api_key=api_key,
        )

    async def complete_json(
        self,
        system: str,
        user: str,
    ) -> dict:
        from google.genai import types

        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=getattr(
                    self,
                    "model",
                    DEFAULT_GEMINI_MODEL,
                ),
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

        except Exception as exc:  # noqa: BLE001
            status_code = getattr(
                exc,
                "status_code",
                None,
            )

            message = str(exc).lower()

            if _is_rate_limited(
                status_code,
                message,
            ):
                raise ProviderRateLimited(
                    retry_after=_extract_retry_after(exc),
                ) from exc

            if _is_payload_too_large(
                status_code,
                message,
            ):
                raise ProviderPayloadTooLarge() from exc

            raise ProviderUnavailable(
                str(exc)
            ) from exc


# ============================================================================
# GROQ
# ============================================================================


class GroqProvider(LLMProvider):
    """Groq provider."""

    # Kept for compatibility/logging.
    name = "groq-llama-3.3-70b"

    # Groq models have significantly smaller practical input budgets than
    # Gemini. The orchestrator chunks before sending.
    max_input_tokens = 30_000

    def __init__(
        self,
        api_key: str,
    ) -> None:
        from groq import AsyncGroq

        settings = get_settings()

        self.model = (
            getattr(
                settings,
                "groq_model",
                DEFAULT_GROQ_MODEL,
            )
            or DEFAULT_GROQ_MODEL
        )

        self._client = AsyncGroq(
            api_key=api_key,
        )

    async def complete_json(
        self,
        system: str,
        user: str,
    ) -> dict:
        from groq import APIStatusError, RateLimitError

        # IMPORTANT:
        #
        # Some unit tests intentionally create the provider with:
        #
        #     object.__new__(GroqProvider)
        #
        # That bypasses __init__, meaning self.model does not exist.
        #
        # getattr() keeps those tests valid without weakening production
        # configuration.
        model = getattr(
            self,
            "model",
            DEFAULT_GROQ_MODEL,
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
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

            content = (
                response
                .choices[0]
                .message
                .content
                or "{}"
            )

            return self._loads(content)

        except RateLimitError as exc:
            raise ProviderRateLimited(
                retry_after=_extract_retry_after(exc),
            ) from exc

        except APIStatusError as exc:
            status_code = getattr(
                exc,
                "status_code",
                None,
            )

            message = str(exc).lower()

            if _is_payload_too_large(
                status_code,
                message,
            ):
                raise ProviderPayloadTooLarge() from exc

            if _is_rate_limited(
                status_code,
                message,
            ):
                raise ProviderRateLimited(
                    retry_after=_extract_retry_after(exc),
                ) from exc

            raise ProviderUnavailable(
                str(exc)
            ) from exc

        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailable(
                str(exc)
            ) from exc


# ============================================================================
# DEEPSEEK
# ============================================================================


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider using its OpenAI-compatible API."""

    name = DEFAULT_DEEPSEEK_MODEL

    max_input_tokens = 60_000

    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str,
    ) -> None:
        from openai import AsyncOpenAI

        settings = get_settings()

        self.model = (
            getattr(
                settings,
                "deepseek_model",
                DEFAULT_DEEPSEEK_MODEL,
            )
            or DEFAULT_DEEPSEEK_MODEL
        )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
        )

    async def complete_json(
        self,
        system: str,
        user: str,
    ) -> dict:
        from openai import APIStatusError, RateLimitError

        # Same compatibility protection as GroqProvider.
        #
        # Tests can instantiate this class with object.__new__(), bypassing
        # __init__. Therefore self.model may not exist.
        model = getattr(
            self,
            "model",
            DEFAULT_DEEPSEEK_MODEL,
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
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

            content = (
                response
                .choices[0]
                .message
                .content
                or "{}"
            )

            return self._loads(content)

        except RateLimitError as exc:
            raise ProviderRateLimited(
                retry_after=_extract_retry_after(exc),
            ) from exc

        except APIStatusError as exc:
            status_code = getattr(
                exc,
                "status_code",
                None,
            )

            message = str(exc).lower()

            if _is_payload_too_large(
                status_code,
                message,
            ):
                raise ProviderPayloadTooLarge() from exc

            if _is_rate_limited(
                status_code,
                message,
            ):
                raise ProviderRateLimited(
                    retry_after=_extract_retry_after(exc),
                ) from exc

            raise ProviderUnavailable(
                str(exc)
            ) from exc

        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailable(
                str(exc)
            ) from exc


# ============================================================================
# ERROR DETECTION
# ============================================================================


def _is_rate_limited(
    status_code: Any,
    message: str,
) -> bool:
    """Detect rate-limit responses across provider SDK versions."""

    if status_code == 429:
        return True

    return (
        "429" in message
        or "rate limit" in message
        or "rate_limit" in message
        or "too many requests" in message
        or "quota exceeded" in message
    )


def _is_payload_too_large(
    status_code: Any,
    message: str,
) -> bool:
    """Detect payload/context-size failures."""

    if status_code == 413:
        return True

    return (
        "413" in message
        or "payload too large" in message
        or "request too large" in message
        or "context length" in message
        or "maximum context" in message
        or "too many tokens" in message
        or "input is too long" in message
        or "input too long" in message
    )


# ============================================================================
# RETRY-AFTER
# ============================================================================


def _extract_retry_after(
    error: Any,
) -> float | None:
    """Best-effort extraction of Retry-After from SDK exceptions."""

    response = getattr(
        error,
        "response",
        None,
    )

    if response is None:
        return None

    headers = getattr(
        response,
        "headers",
        None,
    )

    if not headers:
        return None

    value = headers.get(
        "retry-after"
    )

    if value is None:
        value = headers.get(
            "Retry-After"
        )

    if value is None:
        return None

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================================
# PROVIDER CHAIN
# ============================================================================


def build_provider_chain() -> list[LLMProvider]:
    """Build the Gemini -> Groq -> DeepSeek fallback chain.

    Providers without configured API keys are skipped.

    IMPORTANT:
    Provider constructors receive only the API key.

    Model selection happens inside each provider. This preserves the
    original constructor contract and keeps existing tests/test doubles
    compatible.
    """

    settings = get_settings()

    chain: list[LLMProvider] = []

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    if settings.gemini_api_key:
        chain.append(
            GeminiProvider(
                settings.gemini_api_key,
            )
        )

    # ------------------------------------------------------------------
    # Groq
    # ------------------------------------------------------------------

    if settings.groq_api_key:
        chain.append(
            GroqProvider(
                settings.groq_api_key,
            )
        )

    # ------------------------------------------------------------------
    # DeepSeek
    # ------------------------------------------------------------------

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
        providers=[
            provider.name
            for provider in chain
        ],
    )

    return chain