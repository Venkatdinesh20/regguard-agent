"""The single place a chat model is constructed.

Nothing else in RegGuard instantiates a provider client. That gives one place to
change the model, one place to set timeouts and retries, and one place to attach
tracing later — and it makes the provider a configuration decision rather than a
code decision.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import LLMProvider, Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.core.stub_llm import ScriptedChatModel

logger = get_logger(__name__)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Construct the chat model described by ``settings``."""
    provider = settings.llm_provider

    if provider is LLMProvider.STUB:
        logger.warning(
            "llm.stub_selected",
            extra={
                "detail": "Deterministic stub model in use — no LLM is being "
                "called. Set LLM_PROVIDER=openai or anthropic for real routing."
            },
        )
        return ScriptedChatModel()

    if provider is LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        logger.info(
            "llm.initialised",
            extra={"provider": "openai", "model": settings.llm_model},
        )
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider is LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        if settings.llm_model.startswith("gpt"):
            raise ConfigurationError(
                "LLM_PROVIDER=anthropic but LLM_MODEL looks like an OpenAI "
                f"model ('{settings.llm_model}'). Set LLM_MODEL to a Claude "
                "model identifier."
            )
        logger.info(
            "llm.initialised",
            extra={"provider": "anthropic", "model": settings.llm_model},
        )
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    raise ConfigurationError(  # pragma: no cover - Enum is exhaustive
        f"Unsupported LLM provider: {provider}"
    )


@lru_cache
def get_chat_model() -> BaseChatModel:
    """Return the process-wide chat model, constructed once."""
    return build_chat_model(get_settings())


def reset_chat_model_cache() -> None:
    """Drop the cached model. Used by tests that switch provider."""
    get_chat_model.cache_clear()
