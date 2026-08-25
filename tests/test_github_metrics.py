"""Tests for deterministic GitHub repository correlation and live metrics."""

from __future__ import annotations

import httpx
import pytest

from src.scrapers.github_metrics import fetch_stars, find_repo


def test_find_repo_from_candidate_link():
    result = find_repo(
        [
            "https://github.com/openai/example-repo",
            "https://arxiv.org/abs/1234.5678",
        ],
        "",
    )

    assert result == ("openai", "example-repo")


def test_find_repo_from_git_suffix():
    result = find_repo(
        ["https://github.com/openai/example-repo.git"],
        "",
    )

    assert result == ("openai", "example-repo")


def test_find_repo_from_abstract():
    result = find_repo(
        [],
        "Implementation available at https://github.com/openai/example-repo",
    )

    assert result == ("openai", "example-repo")


def test_find_repo_prefers_candidate_links():
    result = find_repo(
        ["https://github.com/owner/candidate-repo"],
        "https://github.com/owner/abstract-repo",
    )

    assert result == ("owner", "candidate-repo")


def test_find_repo_returns_none_when_missing():
    assert find_repo([], "No repository is mentioned here.") is None


@pytest.mark.asyncio
async def test_fetch_stars_returns_live_github_metrics(monkeypatch):
    class FakeResponse:
        def json(self):
            return {
                "html_url": "https://github.com/openai/example-repo",
                "stargazers_count": 1234,
            }

    class FakeFetcher:
        async def get(self, url, headers=None):
            assert (
                url
                == "https://api.github.com/repos/openai/example-repo"
            )
            assert headers["Accept"] == "application/vnd.github+json"
            return FakeResponse()

    monkeypatch.setattr(
        "src.scrapers.github_metrics.get_settings",
        lambda: type("Settings", (), {"github_token": None})(),
    )

    result = await fetch_stars(
        FakeFetcher(),
        "openai",
        "example-repo",
    )

    assert result == {
        "github_url": "https://github.com/openai/example-repo",
        "github_stars": 1234,
    }


@pytest.mark.asyncio
async def test_fetch_stars_uses_github_token(monkeypatch):
    captured = {}

    class FakeResponse:
        def json(self):
            return {
                "html_url": "https://github.com/acme/repo",
                "stargazers_count": 42,
            }

    class FakeFetcher:
        async def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(
        "src.scrapers.github_metrics.get_settings",
        lambda: type(
            "Settings",
            (),
            {"github_token": "test-token"},
        )(),
    )

    result = await fetch_stars(
        FakeFetcher(),
        "acme",
        "repo",
    )

    assert result == {
        "github_url": "https://github.com/acme/repo",
        "github_stars": 42,
    }
    assert (
        captured["headers"]["Authorization"]
        == "Bearer test-token"
    )


@pytest.mark.asyncio
async def test_fetch_stars_returns_none_on_http_failure(monkeypatch):
    class FakeFetcher:
        async def get(self, *args, **kwargs):
            request = httpx.Request(
                "GET",
                "https://api.github.com/repos/acme/missing",
            )
            response = httpx.Response(
                404,
                request=request,
            )
            response.raise_for_status()

    monkeypatch.setattr(
        "src.scrapers.github_metrics.get_settings",
        lambda: type("Settings", (), {"github_token": None})(),
    )

    result = await fetch_stars(
        FakeFetcher(),
        "acme",
        "missing",
    )

    assert result is None
