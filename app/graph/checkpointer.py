"""A memory-bounded in-process checkpointer.

LangGraph's ``MemorySaver`` keeps every checkpoint of every thread for the
lifetime of the process. That is fine for a script and wrong for a long-lived
API: each investigation writes roughly one checkpoint per node, so a service
answering requests all day grows without limit.

:class:`BoundedMemorySaver` keeps the ``max_threads`` most recently written
investigations and evicts the oldest, logging each eviction so a resumed-too-late
approval has a traceable explanation rather than looking like data loss.

This is a *bound*, not durability. A deployment that must survive a restart —
or resume a paused case on a different worker — should use
``langgraph-checkpoint-postgres`` instead; the graph definition does not change.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.core.logging import get_logger

logger = get_logger(__name__)

REGGUARD_MSGPACK_TYPES: tuple[tuple[str, str], ...] = (
    ("app.schemas.routing", "AgentRoute"),
    ("app.schemas.routing", "RouteDecision"),
    ("app.schemas.findings", "Finding"),
    ("app.schemas.investigation", "RiskLevel"),
    ("app.schemas.investigation", "InvestigationReport"),
    ("app.schemas.investigation", "HumanApproval"),
)
"""Domain types that appear in checkpointed state.

LangGraph deserialises unregistered types with a warning today and will refuse to
in a future release. Since our state channels carry Pydantic models and enums,
leaving them unregistered means a minor dependency upgrade would silently break
reading and resuming persisted investigations. Registering them explicitly is
also the safer posture: the allowlist says exactly which types may be
reconstructed from a checkpoint.

Verified under ``LANGGRAPH_STRICT_MSGPACK=true`` by
``tests/test_checkpointer.py``, which is the behaviour of that future release.
"""


def regguard_serializer() -> JsonPlusSerializer:
    """Serializer that permits exactly RegGuard's own checkpointed types."""
    return JsonPlusSerializer(allowed_msgpack_modules=REGGUARD_MSGPACK_TYPES)


class BoundedMemorySaver(MemorySaver):
    """In-memory checkpointer with least-recently-written thread eviction."""

    def __init__(self, max_threads: int, **kwargs: Any) -> None:
        if max_threads < 1:
            raise ValueError("max_threads must be at least 1")
        kwargs.setdefault("serde", regguard_serializer())
        super().__init__(**kwargs)
        self._max_threads = max_threads
        self._threads: OrderedDict[str, None] = OrderedDict()

    @property
    def max_threads(self) -> int:
        return self._max_threads

    @property
    def retained_threads(self) -> list[str]:
        """Thread ids currently retained, oldest write first."""
        return list(self._threads)

    def put(self, config: Any, *args: Any, **kwargs: Any) -> Any:
        checkpoint_config = super().put(config, *args, **kwargs)
        self._record(config)
        return checkpoint_config

    async def aput(self, config: Any, *args: Any, **kwargs: Any) -> Any:
        checkpoint_config = await super().aput(config, *args, **kwargs)
        self._record(config)
        return checkpoint_config

    def _record(self, config: Any) -> None:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not thread_id:  # pragma: no cover - LangGraph always supplies one
            return

        self._threads.pop(thread_id, None)
        self._threads[thread_id] = None

        while len(self._threads) > self._max_threads:
            evicted, _ = self._threads.popitem(last=False)
            self.delete_thread(evicted)
            logger.info(
                "checkpointer.thread_evicted",
                extra={
                    "thread_id": evicted,
                    "retained": len(self._threads),
                    "max_threads": self._max_threads,
                    "detail": "Least recently written investigation dropped from "
                    "the in-memory checkpointer; it can no longer be read or "
                    "resumed.",
                },
            )
