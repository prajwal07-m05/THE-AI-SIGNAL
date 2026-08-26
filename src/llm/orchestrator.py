"""Multi-tier LLM extraction orchestrator (Phase III).

Ties together the fallback chain, chunking, rate-limit handling, and
provider cooldowns.

Behavior:

  * Walk providers in configured fallback order.
  * Skip providers whose temporary circuit is open.
  * Fit source text to each provider's input-token budget.
  * On 429, retry the SAME provider with exponential full-jitter backoff.
  * Respect Retry-After when supplied by the provider.
  * After exhausting 429 retries, temporarily cool down that provider.
  * On 413, reduce the payload budget and retry with a smaller payload.
  * If a provider remains unable to process the request, fall through.
  * Never fabricate data when all providers fail; raise AllProvidersFailed.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

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


# Tokens reserved for:
#   * system prompt
#   * extraction schema
#   * JSON wrapper
#   * provider-specific overhead
_PROMPT_OVERHEAD = 800

# Once repeated 413 responses reduce the payload below this threshold,
# continuing to retry is not useful.
_MIN_PAYLOAD_BUDGET = 500

# Temporary provider cooldown after exhausting the 429 retry budget.
_DEFAULT_PROVIDER_COOLDOWN = 60.0


class AllProvidersFailed(Exception):
    """Raised when no configured LLM provider can process the request."""


@dataclass
class _ProviderState:
    """Runtime state for a single provider."""

    cooldown_until: float = 0.0

    @property
    def is_cooling_down(self) -> bool:
        """Return True while the provider circuit is open."""

        return time.monotonic() < self.cooldown_until


class LLMOrchestrator:
    """Fallback-aware asynchronous LLM extraction orchestrator."""

    def __init__(
        self,
        chain: list[LLMProvider] | None = None,
        max_429_retries: int = 4,
        provider_cooldown: float = _DEFAULT_PROVIDER_COOLDOWN,
    ) -> None:
        if max_429_retries < 0:
            raise ValueError(
                "max_429_retries must be >= 0"
            )

        if provider_cooldown < 0:
            raise ValueError(
                "provider_cooldown must be >= 0"
            )

        self._chain = chain or build_provider_chain()

        # Keep the public constructor parameter name while using the
        # internal attribute consistently throughout the orchestrator.
        self._max_429 = max_429_retries

        self._provider_cooldown = float(
            provider_cooldown
        )

        self._state: dict[str, _ProviderState] = {
            provider.name: _ProviderState()
            for provider in self._chain
        }

    async def extract(
        self,
        record_type: str,
        text: str,
    ) -> dict:
        """Extract structured data using the provider fallback chain."""

        last_err: Exception | None = None

        for provider in self._chain:
            if self._is_provider_cooling_down(provider):
                log.info(
                    "provider_circuit_open",
                    provider=provider.name,
                )
                continue

            try:
                result = await self._call_with_backoff(
                    provider,
                    record_type,
                    text,
                )

                # A successful request proves the provider is healthy.
                self._close_provider(provider)

                return result

            except ProviderUnavailable as exc:
                last_err = exc

                log.warning(
                    "provider_fell_through",
                    provider=provider.name,
                    error=str(exc),
                )

                continue

            except AllProvidersFailed as exc:
                last_err = exc

                log.warning(
                    "provider_fell_through",
                    provider=provider.name,
                    error=str(exc),
                )

                continue

        if last_err is not None:
            raise AllProvidersFailed(
                str(last_err)
            ) from last_err

        raise AllProvidersFailed(
            "no providers available; "
            "all providers are cooling down"
        )

    async def _call_with_backoff(
        self,
        provider: LLMProvider,
        record_type: str,
        text: str,
    ) -> dict:
        """Call one provider with token fitting and retry handling."""

        # Do not reject small providers before their first request.
        #
        # Some lightweight/test providers legitimately advertise a small
        # context window. The minimum payload threshold is enforced only
        # after a genuine 413 response.
        budget = self._initial_budget(provider)

        payload = self._fit(
            text,
            budget,
        )

        # Number of 429 retries performed for the current payload.
        #
        # Important:
        #   * 429 increments this counter.
        #   * 413 does NOT consume this budget.
        #   * After a 413, the payload is reduced and the 429 retry budget
        #     is reset.
        attempt = 0

        while True:
            try:
                user = build_user_prompt(
                    record_type,
                    payload,
                )

                return await provider.complete_json(
                    SYSTEM,
                    user,
                )

            except ProviderRateLimited as exc:
                # We have exhausted this provider's retry budget.
                #
                # The correct internal attribute is _max_429.
                if attempt >= self._max_429:
                    self._open_provider(
                        provider,
                        reason="429 budget exhausted",
                    )

                    raise ProviderUnavailable(
                        "429 budget exhausted"
                    ) from exc

                if exc.retry_after is not None:
                    delay = max(
                        0.0,
                        float(exc.retry_after),
                    )
                else:
                    delay = _backoff(attempt)

                log.info(
                    "llm_429_backoff",
                    provider=provider.name,
                    attempt=attempt,
                    delay=delay,
                )

                await asyncio.sleep(delay)

                attempt += 1

            except ProviderPayloadTooLarge:
                previous_budget = budget

                # Reduce the source-text budget by half.
                budget //= 2

                # Guarantee that every 413 changes the budget.
                if budget >= previous_budget:
                    budget = previous_budget - 1

                log.info(
                    "llm_413_reduce",
                    provider=provider.name,
                    previous_budget=previous_budget,
                    new_budget=budget,
                )

                if budget < _MIN_PAYLOAD_BUDGET:
                    raise ProviderUnavailable(
                        "413 even after chunking"
                    ) from None

                payload = self._fit(
                    text,
                    budget,
                )

                # 413 and 429 are independent failure modes.
                #
                # A payload reduction should not consume the 429 budget.
                attempt = 0

    @staticmethod
    def _initial_budget(
        provider: LLMProvider,
    ) -> int:
        """Calculate the initial usable source-text token budget."""

        budget = (
            provider.max_input_tokens
            - _PROMPT_OVERHEAD
        )

        # If the advertised context is smaller than the reserved prompt
        # overhead, still allow a minimal first request.
        return max(
            1,
            budget,
        )

    @staticmethod
    def _fit(
        text: str,
        budget: int,
    ) -> str:
        """Fit source text within the requested token budget."""

        if budget <= 0:
            raise ValueError(
                "token budget must be greater than zero"
            )

        if count_tokens(text) <= budget:
            return text

        return salient_truncate(
            text,
            max_tokens=budget,
        )

    def _is_provider_cooling_down(
        self,
        provider: LLMProvider,
    ) -> bool:
        """Return whether the provider's temporary circuit is open."""

        state = self._state.setdefault(
            provider.name,
            _ProviderState(),
        )

        if not state.is_cooling_down:
            # Cooldown expired. Reset the state.
            if state.cooldown_until:
                state.cooldown_until = 0.0

            return False

        return True

    def _open_provider(
        self,
        provider: LLMProvider,
        *,
        reason: str,
    ) -> None:
        """Temporarily stop using a provider after repeated failures."""

        state = self._state.setdefault(
            provider.name,
            _ProviderState(),
        )

        state.cooldown_until = (
            time.monotonic()
            + self._provider_cooldown
        )

        log.warning(
            "provider_circuit_opened",
            provider=provider.name,
            cooldown_s=self._provider_cooldown,
            reason=reason,
        )

    def _close_provider(
        self,
        provider: LLMProvider,
    ) -> None:
        """Clear a provider's cooldown after successful recovery."""

        state = self._state.setdefault(
            provider.name,
            _ProviderState(),
        )

        if state.cooldown_until:
            state.cooldown_until = 0.0

            log.info(
                "provider_circuit_closed",
                provider=provider.name,
            )


def _backoff(
    attempt: int,
) -> float:
    """Exponential backoff with full jitter.

    Base delay: 1 second.
    Maximum delay: 60 seconds.
    """

    return random.uniform(
        0.0,
        min(
            60.0,
            2**attempt,
        ),
    )