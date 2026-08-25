"""GitHub correlation + live star metrics (Phase I — dynamic metrics).

Given a paper's candidate links + abstract, find an associated GitHub repo and
fetch its CURRENT star count from the official GitHub REST API. Stars are a
dynamic metric, so we always read them live at collection time (never cached).

Correlation strategy (deterministic, no hallucination):
  1. If any candidate link is a github.com/<owner>/<repo> URL, use it directly.
  2. Else, search the arXiv abstract text for a github.com URL.
  3. Else, no repo -> github_url/stars stay null (allowed by schema).

Authenticated requests (GITHUB_TOKEN) get 5,000 req/hr vs 60 unauthenticated.
"""
from __future__ import annotations

import re

from src.core.http_client import AsyncFetcher
from src.settings import get_settings

_REPO_RE = re.compile(
    r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git|/|#|\)|\s|$)"
)


def find_repo(candidate_links: list[str], abstract: str) -> tuple[str, str] | None:
    """Return (owner, repo) if a GitHub repo can be located, else None."""
    for url in candidate_links:
        if m := _REPO_RE.search(url or ""):
            return m.group(1), m.group(2)
    if m := _REPO_RE.search(abstract or ""):
        return m.group(1), m.group(2)
    return None


async def fetch_stars(fetcher: AsyncFetcher, owner: str, repo: str) -> dict | None:
    """Fetch live repo metadata; returns {'github_url', 'github_stars'} or None."""
    token = get_settings().github_token
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = await fetcher.get(
            f"https://api.github.com/repos/{owner}/{repo}", headers=headers
        )
        data = resp.json()
        return {
            "github_url": data["html_url"],
            "github_stars": int(data.get("stargazers_count", 0)),
        }
    except Exception:  # noqa: BLE001 — repo may be gone/renamed; not fatal
        return None
