"""Centralized, env-driven configuration (12-factor).

All tunables live here so the pipeline scales by changing infrastructure /
environment variables — never code. This satisfies the Phase I requirement that
the architecture scale to 500k+ records "without requiring code changes".
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM fallback chain (in priority order)
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    deepseek_api_key: str | None = None

    github_token: str | None = None
    redis_url: str | None = None

    google_service_account_json: str | None = None
    google_sheet_id: str | None = None

    # Concurrency / politeness
    max_concurrency: int = 64
    per_host_concurrency: int = 8
    global_rps: float = 50.0
    request_timeout_s: float = 30.0

    freshness_window_hours: int = 24

    log_level: str = "INFO"
    log_json: bool = True

    # A realistic desktop UA rotation pool for polite scraping.
    user_agents: list[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.6 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        ]
    )

    @property
    def has_any_llm(self) -> bool:
        return any([self.gemini_api_key, self.groq_api_key, self.deepseek_api_key])


@lru_cache
def get_settings() -> Settings:
    return Settings()
