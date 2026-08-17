"""The four specialist agents, built from one specification table.

Each specialist is a node that receives state, gathers evidence with *only* its
own tools, and appends exactly one :class:`Finding`. Specialists never call each
other and never write to another channel of state — all coordination goes
through the supervisor. That is what makes the system's behaviour reconstructable
from the audit trail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from app.agents.base import run_specialist
from app.agents.context import specialist_context
from app.agents.prompts import (
    CUSTOMER_PROMPT,
    FRAUD_PROMPT,
    POLICY_PROMPT,
    TRANSACTION_PROMPT,
)
from app.core.logging import get_logger
from app.graph.state import InvestigationState
from app.schemas.routing import AgentRoute
from app.tools import CUSTOMER_TOOLS, FRAUD_TOOLS, POLICY_TOOLS, TRANSACTION_TOOLS

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpecialistSpec:
    """Everything that distinguishes one specialist from another."""

    route: AgentRoute
    system_prompt: str
    tools: Sequence[BaseTool]


SPECIALISTS: dict[AgentRoute, SpecialistSpec] = {
    AgentRoute.CUSTOMER: SpecialistSpec(
        route=AgentRoute.CUSTOMER,
        system_prompt=CUSTOMER_PROMPT,
        tools=CUSTOMER_TOOLS,
    ),
    AgentRoute.TRANSACTION: SpecialistSpec(
        route=AgentRoute.TRANSACTION,
        system_prompt=TRANSACTION_PROMPT,
        tools=TRANSACTION_TOOLS,
    ),
    AgentRoute.FRAUD: SpecialistSpec(
        route=AgentRoute.FRAUD,
        system_prompt=FRAUD_PROMPT,
        tools=FRAUD_TOOLS,
    ),
    AgentRoute.POLICY: SpecialistSpec(
        route=AgentRoute.POLICY,
        system_prompt=POLICY_PROMPT,
        tools=POLICY_TOOLS,
    ),
}


def run_specialist_step(
    route: AgentRoute,
    state: InvestigationState,
) -> dict[str, Any]:
    """Run one specialist and return only the channels it is allowed to write."""
    spec = SPECIALISTS[route]
    logger.info("specialist.start", extra={"agent": route.value})
    finding = run_specialist(
        route=spec.route,
        system_prompt=spec.system_prompt,
        tools=spec.tools,
        context=specialist_context(state, route),
    )
    return {"findings": [finding], "route_history": [route.value]}


def customer_node(state: InvestigationState) -> dict[str, Any]:
    """Customer due-diligence specialist."""
    return run_specialist_step(AgentRoute.CUSTOMER, state)


def transaction_node(state: InvestigationState) -> dict[str, Any]:
    """Transaction analysis specialist."""
    return run_specialist_step(AgentRoute.TRANSACTION, state)


def fraud_node(state: InvestigationState) -> dict[str, Any]:
    """Risk scoring specialist."""
    return run_specialist_step(AgentRoute.FRAUD, state)


def policy_node(state: InvestigationState) -> dict[str, Any]:
    """Regulatory policy specialist."""
    return run_specialist_step(AgentRoute.POLICY, state)
