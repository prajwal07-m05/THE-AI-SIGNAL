"""Production async HTTP client.

Provides a single HTTP boundary for all pipeline network traffic.

Guarantees:
    * Global concurrency limiting.
    * Per-host concurrency limiting.
    * Global request-rate limiting.
    * Retry with exponential backoff and jitter.
    * Explicit 429 and 5xx handling.
    * Retry-After support for both delta-seconds and HTTP-date values.
    * Transport and timeout retries.
    * Rotating User-Agent headers.
    * HTTP/2 support.
    * Configurable request timeout.
"""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.core.logging import get_logger
from src.settings import get_settings

log = get_logger(__name__)


class RetryableHTTPError(Exception):
    """Raised for HTTP responses that should be retried."""

    def __init__(
        self,
        status: int,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            f"retryable HTTP {status}"
        )

        self.status = status
        self.retry_after = retry_after


def _is_retryable(
    exc: BaseException,
) -> bool:
    """Return whether an exception is safe to retry."""
    return isinstance(
        exc,
        (
            RetryableHTTPError,
            httpx.TransportError,
            httpx.TimeoutException,
        ),
    )


class AsyncFetcher:
    """Reusable async HTTP fetcher.

    The fetcher should normally be used as an async context manager:

        async with AsyncFetcher() as fetcher:
            response = await fetcher.get(url)
    """

    def __init__(self) -> None:
        settings = get_settings()

        self._settings = settings

        self._global_sem = asyncio.Semaphore(
            settings.max_concurrency
        )

        self._host_sems: dict[
            str,
            asyncio.Semaphore,
        ] = defaultdict(
            lambda: asyncio.Semaphore(
                settings.per_host_concurrency
            )
        )

        self._limiter = AsyncLimiter(
            max_rate=settings.global_rps,
            time_period=1.0,
        )

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(
        self,
    ) -> "AsyncFetcher":
        """Create the underlying HTTP client."""
        self._client = httpx.AsyncClient(
            http2=True,
            follow_redirects=True,
            timeout=self._settings.request_timeout_s,
            limits=httpx.Limits(
                max_connections=self._settings.max_concurrency,
                max_keepalive_connections=(
                    self._settings.max_concurrency
                ),
            ),
        )

        return self

    async def __aexit__(
        self,
        *exc: object,
    ) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()

        self._client = None

    def _headers(
        self,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build default request headers."""
        headers = {
            "User-Agent": random.choice(
                self._settings.user_agents
            ),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        if extra:
            headers.update(extra)

        return headers

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        """GET with concurrency, rate limiting, and retries."""
        host = (
            httpx.URL(url).host
            or "unknown"
        )

        @retry(
            retry=retry_if_exception(
                _is_retryable
            ),
            wait=wait_exponential_jitter(
                initial=1,
                max=60,
                jitter=2,
            ),
            stop=stop_after_attempt(6),
            reraise=True,
        )
        async def _do() -> httpx.Response:
            assert self._client is not None

            async with (
                self._global_sem,
                self._host_sems[host],
            ):
                async with self._limiter:
                    response = await self._client.get(
                        url,
                        headers=self._headers(
                            headers
                        ),
                        params=params,
                    )

            if response.status_code == 429:
                retry_after = _parse_retry_after(
                    response.headers.get(
                        "Retry-After"
                    )
                )

                log.warning(
                    "rate_limited",
                    url=url,
                    retry_after=retry_after,
                )

                if retry_after is not None:
                    await asyncio.sleep(
                        retry_after
                    )

                raise RetryableHTTPError(
                    429,
                    retry_after,
                )

            if response.status_code >= 500:
                raise RetryableHTTPError(
                    response.status_code
                )

            response.raise_for_status()

            return response

        return await _do()

    async def post_json(
        self,
        url: str,
        body: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST JSON and return the decoded JSON response.

        The request receives the same concurrency, rate-limit, and retry
        guarantees as GET requests.
        """
        host = (
            httpx.URL(url).host
            or "unknown"
        )

        request_headers = self._headers(
            headers
        )

        request_headers.setdefault(
            "Content-Type",
            "application/json",
        )

        @retry(
            retry=retry_if_exception(
                _is_retryable
            ),
            wait=wait_exponential_jitter(
                initial=1,
                max=60,
                jitter=2,
            ),
            stop=stop_after_attempt(6),
            reraise=True,
        )
        async def _do() -> dict[str, Any]:
            assert self._client is not None

            async with (
                self._global_sem,
                self._host_sems[host],
            ):
                async with self._limiter:
                    response = await self._client.post(
                        url,
                        headers=request_headers,
                        content=body,
                    )

            if response.status_code == 429:
                retry_after = _parse_retry_after(
                    response.headers.get(
                        "Retry-After"
                    )
                )

                if retry_after is not None:
                    await asyncio.sleep(
                        retry_after
                    )

                raise RetryableHTTPError(
                    429,
                    retry_after,
                )

            if response.status_code >= 500:
                raise RetryableHTTPError(
                    response.status_code
                )

            response.raise_for_status()

            return response.json()

        return await _do()


def _parse_retry_after(
    value: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse an HTTP Retry-After header.

    Supports both forms defined by HTTP:

        Retry-After: 120

    and:

        Retry-After: Wed, 26 Aug 2026 12:00:00 GMT

    HTTP-date values are converted to a non-negative delay in seconds.

    ``now`` is injectable to make the function deterministic in tests.
    """
    if not value:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        delay = float(value)

        return max(
            0.0,
            delay,
        )

    except ValueError:
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if retry_at is None:
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(
            tzinfo=timezone.utc
        )

    retry_at = retry_at.astimezone(
        timezone.utc
    )

    if now is None:
        current_time = datetime.now(
            timezone.utc
        )
    else:
        current_time = now.astimezone(
            timezone.utc
        )

    return max(
        0.0,
        (
            retry_at - current_time
        ).total_seconds(),
    )