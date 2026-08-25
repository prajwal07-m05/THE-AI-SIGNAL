"""Unit tests for LLM provider adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.llm.providers import (
    DeepSeekProvider,
    GeminiProvider,
    GroqProvider,
    ProviderPayloadTooLarge,
    ProviderRateLimited,
    ProviderUnavailable,
    build_provider_chain,
)


def test_provider_loads_plain_json():
    raw = '{"role_family": "AI Engineer", "score": 0.95}'

    result = GeminiProvider._loads(raw)

    assert result == {
        "role_family": "AI Engineer",
        "score": 0.95,
    }


def test_provider_loads_markdown_json():
    raw = '```json\n{"role_family": "AI Engineer"}\n```'

    result = GeminiProvider._loads(raw)

    assert result == {
        "role_family": "AI Engineer",
    }


def test_provider_loads_json_without_language_marker():
    raw = '```\n{"role_family": "AI Engineer"}\n```'

    result = GeminiProvider._loads(raw)

    assert result == {
        "role_family": "AI Engineer",
    }


def test_provider_rejects_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        GeminiProvider._loads("not json")


def test_rate_limited_preserves_retry_after():
    error = ProviderRateLimited(retry_after=2.5)

    assert error.retry_after == 2.5
    assert str(error) == "429 rate limited"


def test_payload_too_large_is_distinct_exception():
    error = ProviderPayloadTooLarge()

    assert isinstance(error, ProviderPayloadTooLarge)


def test_provider_unavailable_is_distinct_exception():
    error = ProviderUnavailable("temporary failure")

    assert str(error) == "temporary failure"


def test_build_provider_chain_skips_missing_keys(monkeypatch):
    class FakeSettings:
        gemini_api_key = "gemini-key"
        groq_api_key = None
        deepseek_api_key = "deepseek-key"

    class FakeGemini:
        name = "gemini"

        def __init__(self, api_key):
            self.api_key = api_key

    class FakeDeepSeek:
        name = "deepseek"

        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr(
        "src.llm.providers.get_settings",
        lambda: FakeSettings(),
    )
    monkeypatch.setattr(
        "src.llm.providers.GeminiProvider",
        FakeGemini,
    )
    monkeypatch.setattr(
        "src.llm.providers.DeepSeekProvider",
        FakeDeepSeek,
    )

    chain = build_provider_chain()

    assert [provider.name for provider in chain] == [
        "gemini",
        "deepseek",
    ]

    assert chain[0].api_key == "gemini-key"
    assert chain[1].api_key == "deepseek-key"


def test_build_provider_chain_preserves_fallback_order(monkeypatch):
    class FakeSettings:
        gemini_api_key = "gemini-key"
        groq_api_key = "groq-key"
        deepseek_api_key = "deepseek-key"

    class FakeProvider:
        def __init__(self, api_key):
            self.api_key = api_key

    class FakeGemini(FakeProvider):
        name = "gemini"

    class FakeGroq(FakeProvider):
        name = "groq"

    class FakeDeepSeek(FakeProvider):
        name = "deepseek"

    monkeypatch.setattr(
        "src.llm.providers.get_settings",
        lambda: FakeSettings(),
    )
    monkeypatch.setattr(
        "src.llm.providers.GeminiProvider",
        FakeGemini,
    )
    monkeypatch.setattr(
        "src.llm.providers.GroqProvider",
        FakeGroq,
    )
    monkeypatch.setattr(
        "src.llm.providers.DeepSeekProvider",
        FakeDeepSeek,
    )

    chain = build_provider_chain()

    assert [provider.name for provider in chain] == [
        "gemini",
        "groq",
        "deepseek",
    ]

    assert [provider.api_key for provider in chain] == [
        "gemini-key",
        "groq-key",
        "deepseek-key",
    ]


def test_build_provider_chain_requires_at_least_one_key(monkeypatch):
    class FakeSettings:
        gemini_api_key = None
        groq_api_key = None
        deepseek_api_key = None

    monkeypatch.setattr(
        "src.llm.providers.get_settings",
        lambda: FakeSettings(),
    )

    with pytest.raises(
        RuntimeError,
        match="No LLM API keys configured",
    ):
        build_provider_chain()


@pytest.mark.asyncio
async def test_groq_rate_limit_is_translated(monkeypatch):
    provider = object.__new__(GroqProvider)

    class FakeRateLimitError(Exception):
        pass

    class FakeAPIStatusError(Exception):
        pass

    class FakeCompletions:
        async def create(self, **kwargs):
            raise FakeRateLimitError("rate limited")

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    provider._client = FakeClient()

    import groq

    monkeypatch.setattr(
        groq,
        "RateLimitError",
        FakeRateLimitError,
    )
    monkeypatch.setattr(
        groq,
        "APIStatusError",
        FakeAPIStatusError,
    )

    with pytest.raises(ProviderRateLimited):
        await provider.complete_json(
            "system",
            "user",
        )


@pytest.mark.asyncio
async def test_groq_413_is_translated(monkeypatch):
    provider = object.__new__(GroqProvider)

    class FakeRateLimitError(Exception):
        pass

    class FakeAPIStatusError(Exception):
        status_code = 413

    class FakeCompletions:
        async def create(self, **kwargs):
            raise FakeAPIStatusError("payload too large")

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    provider._client = FakeClient()

    import groq

    monkeypatch.setattr(
        groq,
        "RateLimitError",
        FakeRateLimitError,
    )
    monkeypatch.setattr(
        groq,
        "APIStatusError",
        FakeAPIStatusError,
    )

    with pytest.raises(ProviderPayloadTooLarge):
        await provider.complete_json(
            "system",
            "user",
        )


@pytest.mark.asyncio
async def test_deepseek_rate_limit_is_translated(monkeypatch):
    provider = object.__new__(DeepSeekProvider)

    class FakeRateLimitError(Exception):
        pass

    class FakeAPIStatusError(Exception):
        status_code = 500

    class FakeCompletions:
        async def create(self, **kwargs):
            raise FakeRateLimitError("rate limited")

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    provider._client = FakeClient()

    import openai

    monkeypatch.setattr(
        openai,
        "RateLimitError",
        FakeRateLimitError,
    )
    monkeypatch.setattr(
        openai,
        "APIStatusError",
        FakeAPIStatusError,
    )

    with pytest.raises(ProviderRateLimited):
        await provider.complete_json(
            "system",
            "user",
        )


@pytest.mark.asyncio
async def test_deepseek_413_is_translated(monkeypatch):
    provider = object.__new__(DeepSeekProvider)

    class FakeRateLimitError(Exception):
        pass

    class FakeAPIStatusError(Exception):
        status_code = 413

    class FakeCompletions:
        async def create(self, **kwargs):
            raise FakeAPIStatusError("payload too large")

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    provider._client = FakeClient()

    import openai

    monkeypatch.setattr(
        openai,
        "RateLimitError",
        FakeRateLimitError,
    )
    monkeypatch.setattr(
        openai,
        "APIStatusError",
        FakeAPIStatusError,
    )

    with pytest.raises(ProviderPayloadTooLarge):
        await provider.complete_json(
            "system",
            "user",
        )


@pytest.mark.asyncio
async def test_groq_success_parses_json(monkeypatch):
    provider = object.__new__(GroqProvider)

    class FakeMessage:
        content = '{"role_family": "AI Engineer"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    provider._client = FakeClient()

    import groq

    class FakeRateLimitError(Exception):
        pass

    class FakeAPIStatusError(Exception):
        pass

    monkeypatch.setattr(
        groq,
        "RateLimitError",
        FakeRateLimitError,
    )
    monkeypatch.setattr(
        groq,
        "APIStatusError",
        FakeAPIStatusError,
    )

    result = await provider.complete_json(
        "system",
        "user",
    )

    assert result == {
        "role_family": "AI Engineer",
    }