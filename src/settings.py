"""Centralized, environment-driven configuration.

All runtime tunables live here so the pipeline can scale and adapt through
environment variables rather than source-code changes.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM API credentials
    # ------------------------------------------------------------------

    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    deepseek_api_key: str | None = None

    # ------------------------------------------------------------------
    # LLM model configuration
    #
    # These are configurable through:
    #
    # GEMINI_MODEL
    # GROQ_MODEL
    # DEEPSEEK_MODEL
    #
    # Keeping model IDs outside provider code prevents provider model
    # changes from requiring source-code modifications.
    # ------------------------------------------------------------------

    gemini_model: str = "gemini-2.5-flash"

    groq_model: str = "openai/gpt-oss-120b"

    deepseek_model: str = "deepseek-chat"

    # ------------------------------------------------------------------
    # Other integrations
    # ------------------------------------------------------------------

    github_token: str | None = None
    redis_url: str | None = None

    google_service_account_json: str | None = None
    google_sheet_id: str | None = None

    # ------------------------------------------------------------------
    # Concurrency / politeness
    # ------------------------------------------------------------------

    max_concurrency: int = 64
    per_host_concurrency: int = 8
    global_rps: float = 50.0
    request_timeout_s: float = 30.0

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    freshness_window_hours: int = 24

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: str = "INFO"
    log_json: bool = True

    # ------------------------------------------------------------------
    # HTTP user-agent rotation
    # ------------------------------------------------------------------

    user_agents: list[str] = Field(
        default_factory=lambda: [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.6 Safari/605.1.15"
            ),
            (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Derived configuration
    # ------------------------------------------------------------------

    @property
    def has_any_llm(self) -> bool:
        """Return True when at least one LLM API key is configured."""

        return any(
            [
                self.gemini_api_key,
                self.groq_api_key,
                self.deepseek_api_key,
            ]
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""

    return Settings()