"""Application service layer: the boundary the API and CLI both call.

Keeping this between the transport layer and the graph means the graph is never
coupled to HTTP, and the API never has to know about checkpoint configs, thread
identifiers or interrupt payloads.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import case_context, get_logger
from app.graph.build import get_graph
from app.graph.state import initial_state
from app.schemas.findings import Finding
from app.schemas.investigation import (
    HumanApproval,
    InvestigationReport,
    InvestigationRequest,
    RiskLevel,
)
from app.schemas.routing import RouteDecision

logger = get_logger(__name__)


class InvestigationOutcome(BaseModel):
    """Everything a caller needs, including the full audit trail."""

    case_id: str
    thread_id: str = Field(
        description="Identifier used to resume a paused investigation."
    )
    status: str = Field(
        description="running | reported | awaiting_approval | approved | rejected"
    )
    risk_level: RiskLevel | None = None
    report: InvestigationReport | None = None
    findings: list[Finding] = Field(default_factory=list)
    decisions: list[RouteDecision] = Field(
        default_factory=list,
        description="Every routing decision the LLM made, in order.",
    )
    route_history: list[str] = Field(default_factory=list)
    guardrail_events: list[str] = Field(
        default_factory=list,
        description="Deterministic overrides applied to the model's choices.",
    )
    steps_used: int = 0
    approval: HumanApproval | None = None
    pending_approval: dict[str, Any] | None = Field(
        default=None,
        description="Present when the graph is paused for human authorisation.",
    )


def _config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": get_settings().graph_recursion_limit,
    }


def _pending_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"payload": value}


def _to_outcome(
    case_id: str,
    thread_id: str,
    result: dict[str, Any],
) -> InvestigationOutcome:
    pending = _pending_interrupt(result)
    status = result.get("status", "running")
    if pending:
        status = "awaiting_approval"

    return InvestigationOutcome(
        case_id=result.get("case_id", case_id),
        thread_id=thread_id,
        status=status,
        risk_level=result.get("overall_risk"),
        report=result.get("report"),
        findings=result.get("findings") or [],
        decisions=result.get("decisions") or [],
        route_history=result.get("route_history") or [],
        guardrail_events=result.get("guardrail_events") or [],
        steps_used=result.get("step_count", 0),
        approval=result.get("approval"),
        pending_approval=pending,
    )


def run_investigation(
    request: InvestigationRequest,
    thread_id: str | None = None,
) -> InvestigationOutcome:
    """Run an investigation to completion, or until it pauses for a human."""
    thread = thread_id or f"{request.case_id}-{uuid.uuid4().hex[:8]}"
    graph = get_graph()

    with case_context(request.case_id):
        logger.info(
            "investigation.started",
            extra={"thread_id": thread, "query": request.query},
        )
        result = graph.invoke(
            initial_state(
                case_id=request.case_id,
                query=request.query,
                customer_id=request.customer_id,
                lookback_days=request.lookback_days,
            ),
            config=_config(thread),
        )
        outcome = _to_outcome(request.case_id, thread, result)
        logger.info(
            "investigation.finished",
            extra={
                "thread_id": thread,
                "status": outcome.status,
                "risk_level": outcome.risk_level.value if outcome.risk_level else None,
                "route_history": outcome.route_history,
                "steps_used": outcome.steps_used,
            },
        )
        return outcome


def resume_investigation(
    thread_id: str,
    approval: HumanApproval,
) -> InvestigationOutcome:
    """Resume a paused investigation with a human reviewer's decision."""
    from langgraph.types import Command

    graph = get_graph()
    snapshot = graph.get_state(_config(thread_id))
    if not snapshot.values:
        raise KeyError(f"No investigation found for thread_id '{thread_id}'")

    case_id = snapshot.values.get("case_id", "UNKNOWN")
    with case_context(case_id):
        logger.info(
            "investigation.resumed",
            extra={
                "thread_id": thread_id,
                "approver": approval.approver,
                "approved": approval.approved,
            },
        )
        result = graph.invoke(
            Command(resume=approval.model_dump()),
            config=_config(thread_id),
        )
        return _to_outcome(case_id, thread_id, result)


def get_investigation(thread_id: str) -> InvestigationOutcome:
    """Read the current state of an investigation without advancing it."""
    graph = get_graph()
    snapshot = graph.get_state(_config(thread_id))
    if not snapshot.values:
        raise KeyError(f"No investigation found for thread_id '{thread_id}'")

    values = dict(snapshot.values)
    pending: dict[str, Any] | None = None
    for task in snapshot.tasks or ():
        for task_interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(task_interrupt, "value", None)
            if isinstance(value, dict):
                pending = value
                break
    if pending:
        values["__interrupt__"] = [type("I", (), {"value": pending})()]

    return _to_outcome(values.get("case_id", "UNKNOWN"), thread_id, values)
