"""The configuration surface itself: the time anchor, and .env.example.

``.env.example`` is the file a new operator copies. If a value in it does not
parse the way it looks, that is a production incident waiting to happen, so it is
loaded and asserted here rather than trusted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from app.core.config import Settings, TimeAnchor, get_settings
from app.tools import repository

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


class TestTimeAnchor:
    """ "Now" is a configuration decision, and both answers must work."""

    def test_dataset_is_the_default_and_is_deterministic(self):
        assert get_settings().time_anchor is TimeAnchor.DATASET
        assert repository.reference_date() == repository.dataset_reference_date()
        assert repository.reference_date() == repository.reference_date()

    def test_dataset_anchor_sees_the_shipped_activity(self):
        assert len(repository.get_transactions("C001", 30)) == 12

    def test_wall_clock_anchor_tracks_real_time(self, monkeypatch):
        monkeypatch.setenv("TIME_ANCHOR", "now")
        get_settings.cache_clear()

        anchored = repository.reference_date()
        assert anchored.tzinfo is not None
        assert abs(anchored - datetime.now(UTC)) < timedelta(seconds=5)
        assert anchored > repository.dataset_reference_date()

    def test_wall_clock_anchor_excludes_the_stale_fixtures(self, monkeypatch):
        """The fixtures predate today, so a live clock sees less of them.

        This is the behaviour a reviewer needs to understand before reading any
        window-based number out of this system.
        """
        monkeypatch.setenv("TIME_ANCHOR", "now")
        get_settings.cache_clear()

        with_live_clock = len(repository.get_transactions("C001", 30))
        assert with_live_clock < 12

    def test_an_unknown_anchor_is_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings.model_validate({"time_anchor": "yesterday"})


class TestEnvExample:
    def test_the_file_exists_and_documents_every_setting(self):
        assert ENV_EXAMPLE.exists()
        text = ENV_EXAMPLE.read_text()
        for key in (
            "LLM_PROVIDER",
            "TIME_ANCHOR",
            "AUTH_ENABLED",
            "API_TOKENS",
            "MAX_RETAINED_INVESTIGATIONS",
            "MAX_SUPERVISOR_STEPS",
            "REQUIRE_APPROVAL_FOR_HIGH_RISK",
        ):
            assert f"{key}=" in text, f"{key} is not documented in .env.example"

    def test_it_parses_to_the_values_it_appears_to_declare(self, monkeypatch):
        """Inline '# comment' suffixes must not leak into values."""
        for variable in (
            "LOG_FORMAT",
            "LLM_PROVIDER",
            "MAX_SUPERVISOR_STEPS",
            "TIME_ANCHOR",
            "REQUIRE_APPROVAL_FOR_HIGH_RISK",
        ):
            monkeypatch.delenv(variable, raising=False)

        class ExampleSettings(Settings):
            model_config = SettingsConfigDict(
                env_file=str(ENV_EXAMPLE),
                env_file_encoding="utf-8",
                extra="ignore",
            )

        settings = ExampleSettings()

        assert settings.log_format == "json"  # not 'json            # json | text'
        assert settings.max_supervisor_steps == 8
        assert settings.max_tool_iterations == 4
        assert settings.max_visits_per_agent == 2
        assert settings.graph_recursion_limit == 40
        assert settings.time_anchor is TimeAnchor.DATASET
        assert settings.require_approval_for_high_risk is True
        assert settings.auth_enabled is False
        assert settings.llm_temperature == 0.0
