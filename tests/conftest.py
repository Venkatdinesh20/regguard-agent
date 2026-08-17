"""Shared fixtures.

The whole suite runs against the deterministic stub provider, so tests are fast,
free and network-free while still exercising the real graph, tools, guardrails
and API.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "text")


@pytest.fixture(autouse=True)
def _isolated_caches() -> Iterator[None]:
    """Reset the memoised settings, model and graph around every test."""
    from app.core.config import get_settings
    from app.core.llm import reset_chat_model_cache
    from app.graph.build import reset_graph_cache

    get_settings.cache_clear()
    reset_chat_model_cache()
    reset_graph_cache()
    yield
    get_settings.cache_clear()
    reset_chat_model_cache()
    reset_graph_cache()


@pytest.fixture
def client() -> Iterator[object]:
    from fastapi.testclient import TestClient

    from app.api import app

    with TestClient(app) as test_client:
        yield test_client
