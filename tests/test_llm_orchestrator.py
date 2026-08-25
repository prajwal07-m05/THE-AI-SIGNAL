"""Unit tests for the multi-provider LLM orchestrator."""

from __future__ import annotations

import pytest

from src.llm.orchestrator import (
    AllProvidersFailed,
    LLMOrchestrator,
)
from src.llm.providers import (
    ProviderPayloadTooLarge,
    ProviderRateLimited,
    ProviderUnavailable,
)


_RECORD_TYPE = "RESEARCH_PAPER"


class FakeProvider:
    """Deterministic fake LLM provider for orchestrator tests."""

    def __init__(
        self,
        name: str,
        *,
        max_input_tokens: int = 1000,
        responses: list | None = None,
        errors: list[Exception] | None = None,
    ):
        self.name = name
        self.max_input_tokens = max_input_tokens
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls: list[tuple[str, str]] = []

    async def complete_json(
        self,
        system: str,
        user: str,
    ) -> dict:
        self.calls.append((system, user))

        if self.errors:
            error = self.errors.pop(0)
            raise error

        if self.responses:
            return self.responses.pop(0)

        return {"ok": True}


@pytest.mark.asyncio
async def test_first_provider_success():
    first = FakeProvider(
        "gemini",
        responses=[
            {"provider": "gemini"},
        ],
    )

    second = FakeProvider(
        "groq",
        responses=[
            {"provider": "groq"},
        ],
    )

    orchestrator = LLMOrchestrator(
        chain=[
            first,
            second,
        ],
    )

    result = await orchestrator.extract(
        _RECORD_TYPE,
        "short paper text",
    )

    assert result == {
        "provider": "gemini",
    }

    assert len(first.calls) == 1
    assert len(second.calls) == 0


@pytest.mark.asyncio
async def test_provider_unavailable_falls_through():
    first = FakeProvider(
        "gemini",
        errors=[
            ProviderUnavailable(
                "gemini unavailable"
            ),
        ],
    )

    second = FakeProvider(
        "groq",
        responses=[
            {"provider": "groq"},
        ],
    )

    orchestrator = LLMOrchestrator(
        chain=[
            first,
            second,
        ],
    )

    result = await orchestrator.extract(
        _RECORD_TYPE,
        "paper text",
    )

    assert result == {
        "provider": "groq",
    }

    assert len(first.calls) == 1
    assert len(second.calls) == 1


@pytest.mark.asyncio
async def test_all_providers_failed():
    first = FakeProvider(
        "gemini",
        errors=[
            ProviderUnavailable(
                "gemini unavailable"
            ),
        ],
    )

    second = FakeProvider(
        "groq",
        errors=[
            ProviderUnavailable(
                "groq unavailable"
            ),
        ],
    )

    orchestrator = LLMOrchestrator(
        chain=[
            first,
            second,
        ],
    )

    with pytest.raises(AllProvidersFailed):
        await orchestrator.extract(
            _RECORD_TYPE,
            "paper text",
        )


@pytest.mark.asyncio
async def test_429_retries_same_provider(
    monkeypatch,
):
    provider = FakeProvider(
        "gemini",
        errors=[
            ProviderRateLimited(),
        ],
        responses=[
            {"ok": True},
        ],
    )

    sleeps: list[float] = []

    async def fake_sleep(
        delay: float,
    ):
        sleeps.append(delay)

    monkeypatch.setattr(
        "src.llm.orchestrator.asyncio.sleep",
        fake_sleep,
    )

    monkeypatch.setattr(
        "src.llm.orchestrator._backoff",
        lambda attempt: 0.25,
    )

    orchestrator = LLMOrchestrator(
        chain=[
            provider,
        ],
        max_429_retries=2,
    )

    result = await orchestrator.extract(
        _RECORD_TYPE,
        "paper text",
    )

    assert result == {
        "ok": True,
    }

    assert len(provider.calls) == 2
    assert sleeps == [
        0.25,
    ]


@pytest.mark.asyncio
async def test_429_retry_after_is_respected(
    monkeypatch,
):
    provider = FakeProvider(
        "gemini",
        errors=[
            ProviderRateLimited(
                retry_after=3.5,
            ),
        ],
        responses=[
            {"ok": True},
        ],
    )

    sleeps: list[float] = []

    async def fake_sleep(
        delay: float,
    ):
        sleeps.append(delay)

    monkeypatch.setattr(
        "src.llm.orchestrator.asyncio.sleep",
        fake_sleep,
    )

    orchestrator = LLMOrchestrator(
        chain=[
            provider,
        ],
        max_429_retries=1,
    )

    result = await orchestrator.extract(
        _RECORD_TYPE,
        "paper text",
    )

    assert result == {
        "ok": True,
    }

    assert sleeps == [
        3.5,
    ]


@pytest.mark.asyncio
async def test_429_budget_exhaustion_falls_through():
    first = FakeProvider(
        "gemini",
        errors=[
            ProviderRateLimited(),
            ProviderRateLimited(),
        ],
    )

    second = FakeProvider(
        "groq",
        responses=[
            {"provider": "groq"},
        ],
    )

    orchestrator = LLMOrchestrator(
        chain=[
            first,
            second,
        ],
        max_429_retries=1,
    )

    result = await orchestrator.extract(
        _RECORD_TYPE,
        "paper text",
    )

    assert result == {
        "provider": "groq",
    }

    assert len(first.calls) == 2
    assert len(second.calls) == 1


@pytest.mark.asyncio
async def test_413_reduces_payload_budget(
    monkeypatch,
):
    provider = FakeProvider(
        "groq",
        max_input_tokens=2200,
        errors=[
            ProviderPayloadTooLarge(),
        ],
        responses=[
            {"ok": True},
        ],
    )

    fitted_budgets: list[int] = []

    original_fit = LLMOrchestrator._fit

    def fake_fit(
        text: str,
        budget: int,
    ) -> str:
        fitted_budgets.append(budget)

        return original_fit(
            text,
            budget,
        )

    monkeypatch.setattr(
        LLMOrchestrator,
        "_fit",
        staticmethod(fake_fit),
    )

    orchestrator = LLMOrchestrator(
        chain=[
            provider,
        ],
    )

    result = await orchestrator.extract(
        _RECORD_TYPE,
        "paper text",
    )

    assert result == {
        "ok": True,
    }

    assert len(fitted_budgets) == 2

    assert fitted_budgets[1] < fitted_budgets[0]

    assert fitted_budgets[1] >= 500


@pytest.mark.asyncio
async def test_413_repeatedly_reduces_until_provider_succeeds():
    provider = FakeProvider(
        "groq",
        max_input_tokens=4000,
        errors=[
            ProviderPayloadTooLarge(),
            ProviderPayloadTooLarge(),
        ],
        responses=[
            {"ok": True},
        ],
    )

    orchestrator = LLMOrchestrator(
        chain=[
            provider,
        ],
    )

    result = await orchestrator.extract(
        _RECORD_TYPE,
        "paper text",
    )

    assert result == {
        "ok": True,
    }

    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_413_below_minimum_falls_through():
    first = FakeProvider(
        "groq",
        max_input_tokens=900,
        errors=[
            ProviderPayloadTooLarge(),
        ],
    )

    second = FakeProvider(
        "deepseek",
        responses=[
            {"provider": "deepseek"},
        ],
    )

    orchestrator = LLMOrchestrator(
        chain=[
            first,
            second,
        ],
    )

    result = await orchestrator.extract(
        _RECORD_TYPE,
        "paper text",
    )

    assert result == {
        "provider": "deepseek",
    }


def test_fit_keeps_short_text_unchanged():
    text = "short text"

    assert LLMOrchestrator._fit(
        text,
        1000,
    ) == text