"""Retention bounds and checkpoint serialisation compatibility."""

from __future__ import annotations

import warnings

import pytest

from app.graph.build import build_graph
from app.graph.checkpointer import (
    REGGUARD_MSGPACK_TYPES,
    BoundedMemorySaver,
    regguard_serializer,
)
from app.graph.state import initial_state
from app.schemas.investigation import InvestigationReport, RiskLevel

CLEAN_CASE = "Review customer C002 following a monitoring alert"


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}


def _run(graph, case_id: str, thread_id: str) -> None:
    graph.invoke(
        initial_state(case_id, CLEAN_CASE, "C002", 60),
        config=_config(thread_id),
    )


class TestRetentionBound:
    """A long-lived process must not accumulate checkpoints for ever."""

    def test_oldest_investigations_are_evicted(self):
        saver = BoundedMemorySaver(max_threads=2)
        graph = build_graph(checkpointer=saver)

        for index in range(4):
            _run(graph, f"CASE-{index}", f"t{index}")

        assert saver.retained_threads == ["t2", "t3"]
        assert graph.get_state(_config("t0")).values == {}
        assert graph.get_state(_config("t1")).values == {}
        assert graph.get_state(_config("t3")).values["case_id"] == "CASE-3"

    def test_rewriting_a_thread_refreshes_its_position(self):
        saver = BoundedMemorySaver(max_threads=2)
        graph = build_graph(checkpointer=saver)

        _run(graph, "CASE-A", "ta")
        _run(graph, "CASE-B", "tb")
        graph.get_state(_config("ta"))  # a read must not count as a write
        _run(graph, "CASE-C", "tc")

        assert saver.retained_threads == ["tb", "tc"]

    def test_a_zero_bound_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            BoundedMemorySaver(max_threads=0)

    def test_the_default_graph_is_bounded_by_configuration(self, monkeypatch):
        from app.core.config import get_settings
        from app.graph.build import get_graph, reset_graph_cache

        monkeypatch.setenv("MAX_RETAINED_INVESTIGATIONS", "7")
        get_settings.cache_clear()
        reset_graph_cache()

        checkpointer = get_graph().checkpointer
        assert isinstance(checkpointer, BoundedMemorySaver)
        assert checkpointer.max_threads == 7


class TestCheckpointSerialisation:
    """Our own state types must survive a round trip through the checkpointer."""

    def test_every_registered_type_is_importable(self):
        import importlib

        for module_name, type_name in REGGUARD_MSGPACK_TYPES:
            module = importlib.import_module(module_name)
            assert hasattr(module, type_name), f"{module_name}.{type_name} is gone"

    def test_state_round_trips_under_strict_msgpack(self, monkeypatch):
        """Simulates the LangGraph release that blocks unregistered types.

        Without the allowlist in ``regguard_serializer`` this is exactly how a
        minor dependency upgrade would break reading and resuming a paused case.
        """
        monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")

        saver = BoundedMemorySaver(max_threads=4, serde=regguard_serializer())
        graph = build_graph(checkpointer=saver)
        config = _config("strict")

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a serde warning fails the test
            graph.invoke(
                initial_state(
                    "CASE-STRICT",
                    "Investigate unusual cash deposits for customer C001",
                    "C001",
                    60,
                ),
                config=config,
            )
            snapshot = graph.get_state(config)

        report = snapshot.values["report"]
        assert isinstance(report, InvestigationReport)
        assert report.risk_level is RiskLevel.HIGH
        assert [f.agent.value for f in snapshot.values["findings"]] == [
            "CUSTOMER",
            "TRANSACTION",
            "FRAUD",
            "POLICY",
        ]
        assert snapshot.tasks[0].interrupts, "the approval gate should still be open"
