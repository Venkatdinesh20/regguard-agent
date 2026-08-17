"""The deterministic provider used by the suite, and its contract with prompts."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.context import FINDINGS_HEADER, supervisor_context
from app.core import stub_llm
from app.core.stub_llm import ScriptedChatModel
from app.graph.state import InvestigationState
from app.schemas.routing import AgentRoute, RouteDecision
from app.tools import CUSTOMER_TOOLS

MODEL = ScriptedChatModel()


def _route(state: dict) -> RouteDecision:
    return MODEL.with_structured_output(RouteDecision).invoke(
        [
            SystemMessage(content="You may route to CUSTOMER: TRANSACTION: FRAUD:"),
            HumanMessage(content=supervisor_context(cast(InvestigationState, state))),
        ]
    )


def _state(**overrides) -> dict:
    base = {
        "case_id": "CASE-T",
        "query": "Investigate unusual cash deposits for customer C001",
        "customer_id": "C001",
        "lookback_days": 30,
        "findings": [],
        "route_history": [],
        "step_count": 0,
    }
    base.update(overrides)
    return base


def test_findings_header_stays_in_sync_with_prompt_builder():
    """app.core may not import app.agents, so the marker is duplicated."""
    assert stub_llm.FINDINGS_HEADER == FINDINGS_HEADER


def test_system_prompt_mentioning_agents_is_not_mistaken_for_evidence():
    decision = _route(_state())
    assert decision.next_agent is AgentRoute.CUSTOMER


def test_routing_follows_the_evidence_sequence():
    from app.schemas.findings import Finding

    findings = [
        Finding(
            agent=AgentRoute.CUSTOMER,
            summary="Profile established for the customer under review.",
        )
    ]
    decision = _route(_state(findings=findings, route_history=["CUSTOMER"]))
    assert decision.next_agent is AgentRoute.TRANSACTION


def test_out_of_scope_query_finishes_immediately():
    decision = _route(_state(query="What will the weather be in Toronto tomorrow?"))
    assert decision.next_agent is AgentRoute.FINISH
    assert "compliance" in decision.reasoning.lower()


def test_bound_tools_produce_schema_valid_arguments():
    bound = MODEL.bind_tools(CUSTOMER_TOOLS)
    response = bound.invoke(
        [HumanMessage(content="case_id=CASE-T\ncustomer_id=C001\nlookback_days=30")]
    )
    assert response.tool_calls
    call = response.tool_calls[0]
    assert call["name"] == "get_customer_profile"
    assert call["args"] == {"customer_id": "C001"}


def test_model_stops_calling_tools_once_results_exist():
    from langchain_core.messages import AIMessage, ToolMessage

    bound = MODEL.bind_tools(CUSTOMER_TOOLS)
    response = bound.invoke(
        [
            HumanMessage(content="customer_id=C001"),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(content='{"kyc_status": "VERIFIED"}', tool_call_id="x"),
        ]
    )
    assert not response.tool_calls
