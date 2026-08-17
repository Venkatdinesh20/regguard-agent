"""The supervisor: the node that makes RegGuard's control flow LLM-decided.

The supervisor owns no tools and gathers no evidence. On every turn it reads the
accumulated findings and emits a validated :class:`RouteDecision`, and the
graph's conditional edge dispatches on it. Nothing in the code says "customer,
then transactions, then fraud" — that sequence is the model's decision, and it
changes with the case.

What the code *does* own is a set of deterministic guardrails applied to the
model's choice. LLM-decided control flow is not the same as unbounded control
flow: an agent that can loop forever, re-run an expensive specialist, or score
risk before it has any data is not production ready. The supervisor trusts the
model to decide, and verifies the decision before acting on it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import ValidationError

from app.agents.context import supervisor_context
from app.agents.prompts import SUPERVISOR_PROMPT
from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.core.logging import get_logger
from app.graph.state import InvestigationState
from app.schemas.routing import AgentRoute, RouteDecision

logger = get_logger(__name__)

FALLBACK_CONFIDENCE = 0.0


def _ask_llm(state: InvestigationState) -> RouteDecision:
    """Get a validated routing decision, degrading safely on failure."""
    from langchain_core.messages import HumanMessage, SystemMessage

    router = get_chat_model().with_structured_output(RouteDecision)
    try:
        decision = router.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=supervisor_context(state)),
            ]
        )
    except (ValidationError, ValueError) as exc:
        logger.error("supervisor.invalid_decision", extra={"error": str(exc)})
        return RouteDecision(
            next_agent=AgentRoute.FINISH,
            reasoning=(
                "The routing model did not return a schema-valid decision, so "
                "the investigation stops rather than dispatching blindly."
            ),
            confidence=FALLBACK_CONFIDENCE,
        )
    if not isinstance(decision, RouteDecision):  # pragma: no cover - provider guard
        decision = RouteDecision.model_validate(decision)
    return decision


def apply_guardrails(
    decision: RouteDecision,
    state: InvestigationState,
) -> tuple[RouteDecision, list[str]]:
    """Check the model's choice against deterministic limits.

    Returns the decision to act on plus any guardrail events to record. Every
    override is written to the audit trail: we never silently disagree with the
    model.
    """
    settings = get_settings()
    events: list[str] = []
    history = state.get("route_history") or []
    findings = state.get("findings") or []
    step_count = state.get("step_count", 0)

    # 1. Step budget — an investigation may not run forever.
    if step_count > settings.max_supervisor_steps:
        events.append(
            f"STEP_BUDGET_EXHAUSTED: {step_count} routing steps exceeded the "
            f"limit of {settings.max_supervisor_steps}; forcing FINISH "
            f"(model wanted {decision.next_agent.value})."
        )
        return _override(decision, AgentRoute.FINISH, events[-1]), events

    if decision.next_agent is AgentRoute.FINISH:
        return decision, events

    # 2. Evidence ordering — never score risk before there is activity data.
    collected = {finding.agent for finding in findings}
    if (
        decision.next_agent is AgentRoute.FRAUD
        and AgentRoute.TRANSACTION not in collected
    ):
        events.append(
            "ORDERING_VIOLATION: FRAUD requested before TRANSACTION evidence "
            "existed; redirected to TRANSACTION."
        )
        return _override(decision, AgentRoute.TRANSACTION, events[-1]), events

    # 3. Loop protection — a specialist may not be re-entered indefinitely.
    visits = Counter(history)
    if visits[decision.next_agent.value] >= settings.max_visits_per_agent:
        events.append(
            f"LOOP_DETECTED: {decision.next_agent.value} already ran "
            f"{visits[decision.next_agent.value]} time(s), at the limit of "
            f"{settings.max_visits_per_agent}; forcing FINISH."
        )
        return _override(decision, AgentRoute.FINISH, events[-1]), events

    return decision, events


def _override(
    decision: RouteDecision,
    route: AgentRoute,
    reason: str,
) -> RouteDecision:
    return decision.model_copy(
        update={
            "next_agent": route,
            "reasoning": f"[guardrail override] {reason} "
            f"Model reasoning was: {decision.reasoning}",
        }
    )


def supervisor_node(state: InvestigationState) -> dict[str, Any]:
    """Graph node: decide the next step and record why."""
    step_count = state.get("step_count", 0) + 1
    decision = _ask_llm(state)
    final_decision, events = apply_guardrails(
        decision, {**state, "step_count": step_count}
    )

    logger.info(
        "supervisor.decision",
        extra={
            "step": step_count,
            "model_choice": decision.next_agent.value,
            "dispatched": final_decision.next_agent.value,
            "confidence": final_decision.confidence,
            "reasoning": final_decision.reasoning,
            "guardrail_events": events,
        },
    )

    return {
        "next_agent": final_decision.next_agent.value,
        "decisions": [final_decision],
        "step_count": step_count,
        "guardrail_events": events,
    }


def route_from_state(state: InvestigationState) -> str:
    """Conditional-edge function: map the recorded decision to a node name."""
    choice = state.get("next_agent", AgentRoute.FINISH.value)
    return choice if choice in {route.value for route in AgentRoute} else "FINISH"
