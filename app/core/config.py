"""Centralised, validated application configuration.

Every runtime knob enters the application through this module. Business logic
never calls ``os.getenv`` directly, which means configuration is validated once,
at startup, and is discoverable in a single place.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TimeAnchor(StrEnum):
    """What "now" means when a lookback window is resolved.

    The shipped dataset is a fixed synthetic snapshot, so anchoring windows to
    wall-clock time would make every run and every test depend on the date it is
    executed. ``DATASET`` anchors to the newest transaction in the data instead.
    A deployment reading a live source system sets ``NOW``.
    """

    DATASET = "dataset"
    NOW = "now"


class LLMProvider(StrEnum):
    """Which chat-model backend RegGuard talks to."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    STUB = "stub"
    """Deterministic in-process test double. No network, no API key.

    Used by the test suite and by ``make demo`` so the project is runnable by a
    reviewer who has not been given credentials.
    """


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and ``.env``."""

    # --- application -------------------------------------------------------
    app_name: str = "RegGuard"
    environment: str = "development"
    log_level: str = "INFO"
    log_format: str = Field(
        default="json",
        description="'json' for machine-readable logs, 'text' for local dev.",
    )

    # --- model -------------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.STUB
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: int = Field(default=30, gt=0, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- data ---------------------------------------------------------------
    time_anchor: TimeAnchor = Field(
        default=TimeAnchor.DATASET,
        description="Anchor lookback windows to the dataset snapshot (default, "
        "deterministic) or to wall-clock time (for live source systems).",
    )

    # --- agent guardrails --------------------------------------------------
    max_supervisor_steps: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Hard ceiling on LLM routing decisions per investigation.",
    )
    max_tool_iterations: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Hard ceiling on tool-calling rounds inside one specialist.",
    )
    max_visits_per_agent: int = Field(
        default=2,
        ge=1,
        le=10,
        description="How often one specialist may be re-entered before the "
        "supervisor's choice is overridden as a loop.",
    )
    graph_recursion_limit: int = Field(default=40, ge=5, le=200)

    # --- retention ----------------------------------------------------------
    max_retained_investigations: int = Field(
        default=200,
        ge=1,
        le=100_000,
        description="How many investigations the in-memory checkpointer keeps "
        "before evicting the least recently written. Prevents unbounded growth "
        "in a long-lived process; irrelevant once a durable checkpointer is "
        "configured.",
    )

    # --- authentication -----------------------------------------------------
    auth_enabled: bool = Field(
        default=False,
        description="Require a bearer token on the investigation endpoints. "
        "Off by default so the repository is runnable out of the box; "
        "mandatory in production, which is enforced below.",
    )
    api_tokens: str = Field(
        default="",
        description="Comma-separated token definitions, each "
        "'token:principal:role' where role is 'analyst' or 'approver'. In a "
        "real deployment these come from an identity provider, not an "
        "environment variable — this is the smallest thing that makes the "
        "approver identity authenticated rather than self-declared.",
    )

    # --- human in the loop -------------------------------------------------
    require_approval_for_high_risk: bool = Field(
        default=True,
        description="HIGH-risk investigations pause for a human before the "
        "report is released. Never disable in production.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _validate_provider_credentials(self) -> Settings:
        """Fail fast at startup rather than mid-investigation."""
        if self.llm_provider is LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set in .env"
            )
        if self.llm_provider is LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set in .env"
            )
        if self.auth_enabled and not self.api_tokens.strip():
            raise ValueError("AUTH_ENABLED=true requires API_TOKENS to be set")
        if self.is_production:
            if not self.auth_enabled:
                raise ValueError(
                    "ENVIRONMENT=production requires AUTH_ENABLED=true: an "
                    "approval must be attributable to an authenticated "
                    "reviewer, not to a name supplied in the request body."
                )
            if self.llm_provider is LLMProvider.STUB:
                raise ValueError(
                    "ENVIRONMENT=production cannot run with LLM_PROVIDER=stub: "
                    "the stub is a test double and makes no routing decisions."
                )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
