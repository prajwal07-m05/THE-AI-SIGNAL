"""HTTP client resilience and request-policy tests."""

from __future__ import annotations

import httpx
import pytest

from src.core.http_client import (
    AsyncFetcher,
    RetryableHTTPError,
    _parse_retry_after,
)


def test_parse_retry_after_seconds():
    assert _parse_retry_after("5") == 5.0


def test_parse_retry_after_empty_value():
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None


def test_parse_retry_after_invalid_value():
    assert _parse_retry_after("not-a-number") is None


@pytest.mark.asyncio
async def test_get_success(monkeypatch):
    async with AsyncFetcher() as fetcher:
        assert fetcher._client is not None

        async def mock_get(*args, **kwargs):
            return httpx.Response(
                200,
                request=httpx.Request(
                    "GET",
                    "https://example.com",
                ),
                text="ok",
            )

        monkeypatch.setattr(
            fetcher._client,
            "get",
            mock_get,
        )

        response = await fetcher.get(
            "https://example.com"
        )

        assert response.status_code == 200
        assert response.text == "ok"


@pytest.mark.asyncio
async def test_get_retries_on_500(monkeypatch):
    async with AsyncFetcher() as fetcher:
        assert fetcher._client is not None

        attempts = 0

        async def mock_get(*args, **kwargs):
            nonlocal attempts
            attempts += 1

            if attempts < 3:
                return httpx.Response(
                    500,
                    request=httpx.Request(
                        "GET",
                        "https://example.com",
                    ),
                )

            return httpx.Response(
                200,
                request=httpx.Request(
                    "GET",
                    "https://example.com",
                ),
                text="recovered",
            )

        monkeypatch.setattr(
            fetcher._client,
            "get",
            mock_get,
        )

        # Avoid waiting for real exponential backoff in unit tests.
        monkeypatch.setattr(
            "src.core.http_client.asyncio.sleep",
            _no_sleep,
        )

        response = await fetcher.get(
            "https://example.com"
        )

        assert response.status_code == 200
        assert response.text == "recovered"
        assert attempts == 3


@pytest.mark.asyncio
async def test_get_retries_on_429(monkeypatch):
    async with AsyncFetcher() as fetcher:
        assert fetcher._client is not None

        attempts = 0

        async def mock_get(*args, **kwargs):
            nonlocal attempts
            attempts += 1

            if attempts == 1:
                return httpx.Response(
                    429,
                    headers={
                        "Retry-After": "0",
                    },
                    request=httpx.Request(
                        "GET",
                        "https://example.com",
                    ),
                )

            return httpx.Response(
                200,
                request=httpx.Request(
                    "GET",
                    "https://example.com",
                ),
                text="recovered",
            )

        monkeypatch.setattr(
            fetcher._client,
            "get",
            mock_get,
        )

        monkeypatch.setattr(
            "src.core.http_client.asyncio.sleep",
            _no_sleep,
        )

        response = await fetcher.get(
            "https://example.com"
        )

        assert response.status_code == 200
        assert attempts == 2


@pytest.mark.asyncio
async def test_get_retries_on_transport_error(
    monkeypatch,
):
    async with AsyncFetcher() as fetcher:
        assert fetcher._client is not None

        attempts = 0

        async def mock_get(*args, **kwargs):
            nonlocal attempts
            attempts += 1

            if attempts == 1:
                raise httpx.ConnectError(
                    "temporary failure"
                )

            return httpx.Response(
                200,
                request=httpx.Request(
                    "GET",
                    "https://example.com",
                ),
                text="recovered",
            )

        monkeypatch.setattr(
            fetcher._client,
            "get",
            mock_get,
        )

        monkeypatch.setattr(
            "src.core.http_client.asyncio.sleep",
            _no_sleep,
        )

        response = await fetcher.get(
            "https://example.com"
        )

        assert response.status_code == 200
        assert attempts == 2


@pytest.mark.asyncio
async def test_get_raises_after_retry_exhaustion(
    monkeypatch,
):
    async with AsyncFetcher() as fetcher:
        assert fetcher._client is not None

        attempts = 0

        async def mock_get(*args, **kwargs):
            nonlocal attempts
            attempts += 1

            return httpx.Response(
                503,
                request=httpx.Request(
                    "GET",
                    "https://example.com",
                ),
            )

        monkeypatch.setattr(
            fetcher._client,
            "get",
            mock_get,
        )

        monkeypatch.setattr(
            "src.core.http_client.asyncio.sleep",
            _no_sleep,
        )

        with pytest.raises(RetryableHTTPError):
            await fetcher.get(
                "https://example.com"
            )

        assert attempts == 6


@pytest.mark.asyncio
async def test_post_json_success(monkeypatch):
    async with AsyncFetcher() as fetcher:
        assert fetcher._client is not None

        async def mock_post(*args, **kwargs):
            assert kwargs["content"] == '{"query":"ai"}'
            assert (
                kwargs["headers"]["Content-Type"]
                == "application/json"
            )

            return httpx.Response(
                200,
                request=httpx.Request(
                    "POST",
                    "https://example.com/search",
                ),
                json={
                    "results": [
                        {"id": 1}
                    ]
                },
            )

        monkeypatch.setattr(
            fetcher._client,
            "post",
            mock_post,
        )

        result = await fetcher.post_json(
            "https://example.com/search",
            '{"query":"ai"}',
        )

        assert result == {
            "results": [
                {"id": 1}
            ]
        }


async def _no_sleep(*args, **kwargs):
    """Replace retry sleeps during unit tests."""
    return None