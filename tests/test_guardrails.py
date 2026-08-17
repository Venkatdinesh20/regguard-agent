"""The supervisor's guardrails: trust the model's decision, verify it anyway."""

from __future__ import annotations

from app.agents.base import execute_tool_call
from app.agents.supervisor import apply_guardrails, route_from_state
from app.core.config import get_settings
from app.schemas.findings import Finding
from app.schemas.investigation import RiskLevel
from app.schemas.routing import AgentRoute, RouteDecision
from app.tools import CUSTOMER_TOOLS


def _decision(route: AgentRoute) -> RouteDecision:
    return RouteDecision(
        next_agent=route,
        reasoning="A model-generated decision used as test input.",
        confidence=0.8,
    )


def _finding(route: AgentRoute) -> Finding:
    return Finding(
        agent=route,
        summary="Evidence gathered by this specialist during the test.",
        risk_signals=[],
        assessed_risk=RiskLevel.LOW,
        confidence=0.8,
    )


class TestApplyGuardrails:
    def test_reasonable_decision_passes_through_untouched(self):
        state = {"step_count": 1, "route_history": [], "findings": []}
        decision, events = apply_guardrails(_decision(AgentRoute.CUSTOMER), state)
        assert decision.next_agent is AgentRoute.CUSTOMER
        assert events == []

    def test_step_budget_forces_finish(self):
        limit = get_settings().max_supervisor_steps
        state = {
            "step_count": limit + 1,
            "route_history": ["CUSTOMER"],
            "findings": [_finding(AgentRoute.CUSTOMER)],
        }
        decision, events = apply_guardrails(_decision(AgentRoute.TRANSACTION), state)
        assert decision.next_agent is AgentRoute.FINISH
        assert any("STEP_BUDGET_EXHAUSTED" in event for event in events)
        assert "guardrail override" in decision.reasoning

    def test_fraud_before_transaction_evidence_is_redirected(self):
        """A risk score without activity data is not defensible."""
        state = {
            "step_count": 2,
            "route_history": ["CUSTOMER"],
            "findings": [_finding(AgentRoute.CUSTOMER)],
        }
        decision, events = apply_guardrails(_decision(AgentRoute.FRAUD), state)
        assert decision.next_agent is AgentRoute.TRANSACTION
        assert any("ORDERING_VIOLATION" in event for event in events)

    def test_fraud_allowed_once_transaction_evidence_exists(self):
        state = {
            "step_count": 3,
            "route_history": ["CUSTOMER", "TRANSACTION"],
            "findings": [
                _finding(AgentRoute.CUSTOMER),
                _finding(AgentRoute.TRANSACTION),
            ],
        }
        decision, events = apply_guardrails(_decision(AgentRoute.FRAUD), state)
        assert decision.next_agent is AgentRoute.FRAUD
        assert events == []

    def test_repeated_specialist_is_treated_as_a_loop(self):
        limit = get_settings().max_visits_per_agent
        state = {
            "step_count": 3,
            "route_history": ["CUSTOMER"] * limit,
            "findings": [_finding(AgentRoute.CUSTOMER)],
        }
        decision, events = apply_guardrails(_decision(AgentRoute.CUSTOMER), state)
        assert decision.next_agent is AgentRoute.FINISH
        assert any("LOOP_DETECTED" in event for event in events)

    def test_finish_is_always_permitted(self):
        state = {"step_count": 99, "route_history": [], "findings": []}
        decision, _ = apply_guardrails(_decision(AgentRoute.FINISH), state)
        assert decision.next_agent is AgentRoute.FINISH


class TestRouteFromState:
    def test_known_route_is_dispatched(self):
        assert route_from_state({"next_agent": "FRAUD"}) == "FRAUD"

    def test_unknown_route_falls_back_to_finish(self):
        """Defence in depth: even a corrupted channel cannot mis-dispatch."""
        assert route_from_state({"next_agent": "SEND_MONEY"}) == "FINISH"
        assert route_from_state({}) == "FINISH"


class TestToolFailureHandling:
    """A failing tool must produce a message the agent can react to."""

    def setup_method(self):
        self.tools_by_name = {tool.name: tool for tool in CUSTOMER_TOOLS}

    def test_unknown_tool_is_reported_not_raised(self):
        message = execute_tool_call(
            self.tools_by_name,
            {"name": "wire_money", "args": {}, "id": "call_1"},
        )
        assert message.status == "error"
        assert "does not exist" in message.content

    def test_invalid_arguments_are_reported_not_raised(self):
        message = execute_tool_call(
            self.tools_by_name,
            {"name": "get_customer_profile", "args": {}, "id": "call_2"},
        )
        assert message.status == "error"
        assert "invalid arguments" in message.content

    def test_domain_error_is_reported_not_raised(self):
        message = execute_tool_call(
            self.tools_by_name,
            {
                "name": "get_customer_profile",
                "args": {"customer_id": "C999"},
                "id": "call_3",
            },
        )
        assert message.status == "error"
        assert "C999" in message.content

    def test_successful_call_returns_serialised_payload(self):
        message = execute_tool_call(
            self.tools_by_name,
            {
                "name": "get_customer_profile",
                "args": {"customer_id": "C001"},
                "id": "call_4",
            },
        )
        assert message.status != "error"
        assert "VERIFIED" in message.content
