"""The shared state that flows through the graph.

Two properties matter for a compliance system:

1. **Append-only evidence.** ``findings``, ``route_history``, ``decisions`` and
   ``guardrail_events`` are reducer channels: a node returns only what it adds,
   and LangGraph merges. No node can quietly overwrite another's evidence, and
   the full sequence of decisions survives into the audit trail.
2. **Everything needed to explain the outcome lives in state.** Given a final
   state you can reconstruct which specialist ran, why the supervisor chose it,
   what it found, and who approved the result.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.schemas.findings import Finding
from app.schemas.investigation import HumanApproval, InvestigationReport, RiskLevel
from app.schemas.routing import RouteDecision


class InvestigationState(TypedDict, total=False):
    """State channels for one investigation."""

    # --- immutable inputs --------------------------------------------------
    case_id: str
    query: str
    customer_id: str | None
    lookback_days: int

    # --- accumulated evidence (append-only) -------------------------------
    findings: Annotated[list[Finding], operator.add]
    route_history: Annotated[list[str], operator.add]
    decisions: Annotated[list[RouteDecision], operator.add]
    guardrail_events: Annotated[list[str], operator.add]

    # --- control ----------------------------------------------------------
    next_agent: str
    step_count: int

    # --- outputs ----------------------------------------------------------
    overall_risk: RiskLevel
    report: InvestigationReport | None
    approval: HumanApproval | None
    status: str


def initial_state(
    case_id: str,
    query: str,
    customer_id: str | None,
    lookback_days: int,
) -> InvestigationState:
    """Build the starting state for an investigation."""
    return InvestigationState(
        case_id=case_id,
        query=query,
        customer_id=customer_id,
        lookback_days=lookback_days,
        findings=[],
        route_history=[],
        decisions=[],
        guardrail_events=[],
        step_count=0,
        status="running",
        report=None,
        approval=None,
    )
