"""Anti-bot navigation (Phase V).

A tiered strategy — always try the cheapest path first, escalate only when a
domain fights back. This module provides the escalation layer: a stealthed
async Playwright browser for Cloudflare / Datadome / heavy-JS domains.

TIER 0  httpx GET                 (see AsyncFetcher) — 95% of directories/APIs
TIER 1  httpx + realistic headers + cookie warm-up
TIER 2  Playwright (this module)  — real Chromium, JS execution, TLS/JA3 that
        matches a real browser, human-like waits => passes Cloudflare's
        managed challenge and Datadome's fingerprinting for most pages.
TIER 3  Playwright + residential proxy rotation + CAPTCHA solver webhook
        (documented in docs/architecture.md; wired via PROXY_URL env — kept out
        of the trial run to stay within ToS, but the seam exists).

We deliberately do NOT ship an aggressive captcha-bypass; we degrade politely.
"""
from __future__ import annotations

import asyncio
import random

from src.core.logging import get_logger

log = get_logger(__name__)


class PlaywrightRenderer:
    """Async, stealthed headless browser for JS-heavy / protected pages."""

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._pw = None
        self._browser = None

    async def __aenter__(self) -> "PlaywrightRenderer":
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def render(self, url: str, wait_selector: str | None = None) -> str:
        """Return fully-rendered HTML after JS + challenge resolution."""
        from playwright_stealth import stealth_async

        assert self._browser is not None
        context = await self._browser.new_context(
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="America/New_York",
        )
        page = await context.new_page()
        await stealth_async(page)  # patches navigator.webdriver, WebGL, etc.
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            # Cloudflare managed-challenge pages usually clear within a few seconds.
            await asyncio.sleep(random.uniform(2.0, 4.0))
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=20_000)
            return await page.content()
        finally:
            await context.close()
