"""Configuration validation and the provider swap.

The claim "changing provider is a configuration change, not a code change" is
worth testing rather than asserting in a README, so these tests construct the
real OpenAI and Anthropic clients (construction only — no network call).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import LLMProvider, Settings
from app.core.exceptions import ConfigurationError
from app.core.llm import build_chat_model
from app.core.stub_llm import ScriptedChatModel


def _settings(**overrides) -> Settings:
    base: dict[str, object] = {
        "llm_provider": "stub",
        "llm_model": "gpt-4.1-mini",
        "openai_api_key": "",
        "anthropic_api_key": "",
    }
    base.update(overrides)
    return Settings.model_validate(base)


class TestSettingsValidation:
    def test_defaults_are_safe_for_a_fresh_clone(self):
        settings = _settings()
        assert settings.llm_provider is LLMProvider.STUB
        assert settings.require_approval_for_high_risk is True
        assert settings.llm_temperature == 0.0

    def test_openai_without_a_key_fails_at_startup_not_mid_run(self):
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            _settings(llm_provider="openai")

    def test_anthropic_without_a_key_fails_at_startup(self):
        with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
            _settings(llm_provider="anthropic")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_supervisor_steps", 0),
            ("max_supervisor_steps", 999),
            ("max_tool_iterations", 0),
            ("llm_timeout_seconds", 0),
            ("llm_temperature", 5.0),
            ("graph_recursion_limit", 1),
        ],
    )
    def test_guardrail_limits_are_bounded(self, field, value):
        with pytest.raises(ValidationError):
            _settings(**{field: value})

    def test_production_flag_is_derived_from_environment(self):
        assert _settings(environment="production").is_production is True
        assert _settings(environment="development").is_production is False


class TestModelFactory:
    def test_stub_provider_returns_the_deterministic_model(self):
        assert isinstance(build_chat_model(_settings()), ScriptedChatModel)

    def test_openai_client_is_constructed_from_configuration(self):
        model = build_chat_model(
            _settings(
                llm_provider="openai",
                llm_model="gpt-4.1-mini",
                openai_api_key="sk-test",
            )
        )
        assert model.model_name == "gpt-4.1-mini"
        assert model.temperature == 0.0

    def test_anthropic_client_is_constructed_from_configuration(self):
        model = build_chat_model(
            _settings(
                llm_provider="anthropic",
                llm_model="claude-sonnet-4-5",
                anthropic_api_key="sk-ant-test",
            )
        )
        assert model.model == "claude-sonnet-4-5"

    def test_mismatched_provider_and_model_is_rejected_with_a_clear_message(self):
        with pytest.raises(ConfigurationError, match="looks like an OpenAI model"):
            build_chat_model(
                _settings(
                    llm_provider="anthropic",
                    llm_model="gpt-4.1-mini",
                    anthropic_api_key="sk-ant-test",
                )
            )

    def test_every_provider_exposes_the_interface_the_agents_rely_on(self):
        from app.tools import CUSTOMER_TOOLS

        for settings in (
            _settings(),
            _settings(llm_provider="openai", openai_api_key="sk-test"),
            _settings(
                llm_provider="anthropic",
                llm_model="claude-sonnet-4-5",
                anthropic_api_key="sk-ant-test",
            ),
        ):
            model = build_chat_model(settings)
            assert hasattr(model, "bind_tools")
            assert hasattr(model, "with_structured_output")
            assert model.bind_tools(CUSTOMER_TOOLS) is not None
