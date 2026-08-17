"""Graph assembly: the topology inside which the LLM decides the path.

    START
      │
      ▼
   intake ───────────────► supervisor ◄──────────────────┐
   (deterministic)              │                        │
                                │ LLM-chosen route       │ findings appended
              ┌─────────────────┼─────────────────┬──────┴──────┐
              ▼                 ▼                 ▼             ▼
          customer         transaction          fraud         policy
                                │
                                ▼ (FINISH)
                             report
                                │ risk == HIGH?
                    ┌───────────┴───────────┐
                    ▼                       ▼
             human_approval                END
                    │
                    ▼
                   END

The edges out of ``supervisor`` are the important part: the *set* of legal
transitions is fixed and reviewable, while the *choice* among them on every turn
belongs to the LLM. That is the difference between a state machine with an LLM
inside it and an unbounded agent loop.
"""

from __future__ import annotations

from collections.abc import Hashable
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.reporter import human_approval_node, intake_node, report_node
from app.agents.specialists import (
    customer_node,
    fraud_node,
    policy_node,
    transaction_node,
)
from app.agents.supervisor import route_from_state, supervisor_node
from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.checkpointer import BoundedMemorySaver
from app.graph.state import InvestigationState
from app.schemas.investigation import RiskLevel

logger = get_logger(__name__)

SPECIALIST_NODES: dict[Hashable, str] = {
    "CUSTOMER": "customer",
    "TRANSACTION": "transaction",
    "FRAUD": "fraud",
    "POLICY": "policy",
}


def needs_human_approval(state: InvestigationState) -> str:
    """Decide whether the outcome requires human authorisation.

    Deliberately *not* an LLM decision. Who must sign off on an adverse outcome
    is a governance rule, and a model may not opt out of it.
    """
    if not get_settings().require_approval_for_high_risk:
        return "end"
    report = state.get("report")
    if report is not None and report.risk_level is RiskLevel.HIGH:
        return "approve"
    return "end"


def build_graph(checkpointer: Any | None = None) -> Any:
    """Construct and compile the investigation graph."""
    builder: StateGraph[
        InvestigationState, Any, InvestigationState, InvestigationState
    ] = StateGraph(InvestigationState)

    builder.add_node("intake", intake_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("customer", customer_node)
    builder.add_node("transaction", transaction_node)
    builder.add_node("fraud", fraud_node)
    builder.add_node("policy", policy_node)
    builder.add_node("report", report_node)
    builder.add_node("human_approval", human_approval_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "supervisor")

    # The LLM's routing decision, dispatched through a fixed, legal edge set.
    builder.add_conditional_edges(
        "supervisor",
        route_from_state,
        {**SPECIALIST_NODES, "FINISH": "report"},
    )

    # Every specialist reports back to the supervisor — never to another
    # specialist. Coordination stays centralised and auditable.
    for node_name in SPECIALIST_NODES.values():
        builder.add_edge(node_name, "supervisor")

    builder.add_conditional_edges(
        "report",
        needs_human_approval,
        {"approve": "human_approval", "end": END},
    )
    builder.add_edge("human_approval", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("graph.compiled", extra={"checkpointer": type(checkpointer).__name__})
    return compiled


@lru_cache
def get_graph() -> Any:
    """Process-wide compiled graph with a memory-bounded checkpointer.

    :class:`~app.graph.checkpointer.BoundedMemorySaver` keeps this repository
    runnable with no infrastructure while capping retention at
    ``MAX_RETAINED_INVESTIGATIONS`` threads, so a long-lived API process does not
    grow without limit. For a real deployment swap in
    ``langgraph-checkpoint-postgres`` so paused investigations survive a restart
    and can be resumed by a different worker — the graph definition above does
    not change.
    """
    return build_graph(
        checkpointer=BoundedMemorySaver(
            max_threads=get_settings().max_retained_investigations
        )
    )


def reset_graph_cache() -> None:
    """Drop the cached graph. Used by tests that change configuration."""
    get_graph.cache_clear()
