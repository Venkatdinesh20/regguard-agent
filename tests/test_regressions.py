"""Regression tests for defects found in review.

Each test here corresponds to a specific bug that existed and was fixed. They are
grouped separately from the feature suites so the invariant each one protects
stays legible.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from app.agents.base import invalid_tool_call_message, run_specialist
from app.agents.supervisor import _ask_llm
from app.core.exceptions import InvestigationNotPausedError, ThreadAlreadyUsedError
from app.core.stub_llm import ScriptedChatModel
from app.schemas.investigation import (
    CustomerProfile,
    HumanApproval,
    InvestigationRequest,
    RiskLevel,
    Transaction,
)
from app.schemas.routing import AgentRoute
from app.service import resume_investigation, run_investigation
from app.tools import CUSTOMER_TOOLS
from app.tools.fraud import score_customer_risk

# --------------------------------------------------------------------- data ---


def _profile(**overrides: Any) -> CustomerProfile:
    base: dict[str, Any] = {
        "customer_id": "CX",
        "full_name": "Test Person",
        "country": "Canada",
        "account_age_days": 900,
        "average_monthly_spend": 3000.0,
        "kyc_status": "VERIFIED",
        "pep_flag": False,
        "sanctions_hit": False,
    }
    base.update(overrides)
    return CustomerProfile.model_validate(base)


def _cash(tid: str, when: str, amount: float) -> Transaction:
    return Transaction(
        transaction_id=tid,
        customer_id="CX",
        timestamp=when,
        amount=amount,
        currency="CAD",
        direction="CREDIT",
        channel="CASH",
        counterparty="Branch deposit",
        country="Canada",
    )


@pytest.fixture
def synthetic(monkeypatch):
    """Point the risk engine at an in-test customer and transaction set."""

    def _install(profile: CustomerProfile, transactions: list[Transaction]):
        monkeypatch.setattr("app.tools.fraud.get_customer", lambda _cid: profile)
        monkeypatch.setattr(
            "app.tools.fraud.get_transactions", lambda _cid, _days=30: transactions
        )

    return _install


class TestRiskEngineMonotonicity:
    """Adding suspicious activity must never lower the score."""

    def test_aggregation_rule_survives_a_reportable_deposit_in_the_window(
        self, synthetic
    ):
        sub_threshold = [
            _cash("A1", "2026-07-14T09:00:00Z", 9_000),
            _cash("A2", "2026-07-14T12:00:00Z", 9_000),
        ]
        synthetic(_profile(), sub_threshold)
        without = score_customer_risk("CX", 30)

        synthetic(
            _profile(),
            [*sub_threshold, _cash("A3", "2026-07-14T15:00:00Z", 15_000)],
        )
        with_large = score_customer_risk("CX", 30)

        rules_without = {r["rule_id"] for r in without["triggered_rules"]}
        rules_with = {r["rule_id"] for r in with_large["triggered_rules"]}

        assert "R02_THRESHOLD_AGGREGATION" in rules_without
        assert "R02_THRESHOLD_AGGREGATION" in rules_with
        assert with_large["risk_score"] >= without["risk_score"]


class TestMandatoryEscalation:
    """Some findings are an obligation, not a contribution to a score."""

    def test_sanctions_match_alone_reaches_high(self, synthetic):
        synthetic(_profile(sanctions_hit=True), [])
        result = score_customer_risk("CX", 30)

        assert result["risk_score"] == 50  # below the HIGH threshold of 60
        assert result["risk_level"] == "HIGH"
        assert result["escalated_by"] == ["R08_SANCTIONS_HIT"]

    def test_structuring_alone_reaches_high(self, synthetic):
        synthetic(
            _profile(),
            [
                _cash("S1", "2026-07-14T09:00:00Z", 8_000),
                _cash("S2", "2026-07-15T09:00:00Z", 8_100),
                _cash("S3", "2026-07-16T08:00:00Z", 8_200),
            ],
        )
        result = score_customer_risk("CX", 30)
        assert result["risk_level"] == "HIGH"
        assert "R01_STRUCTURING" in result["escalated_by"]

    def test_escalation_never_lowers_a_level(self, synthetic):
        """A floor of HIGH must not pull a 100-point case down, or a clean one up."""
        synthetic(_profile(), [])
        assert score_customer_risk("CX", 30)["risk_level"] == "LOW"


# ------------------------------------------------------- provider robustness --


class _NoStructuredOutputModel(ScriptedChatModel):
    """Imitates a real provider whose parser returns ``None``.

    OpenAI's and Anthropic's ``with_structured_output`` return ``None`` when the
    model answers without calling the structured-output tool — a refusal, a
    safety response, or plain text. That must degrade, not raise.
    """

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        return RunnableLambda(lambda _prompt: None)


class _InvalidToolCallModel(ScriptedChatModel):
    """Imitates a provider returning a tool call whose arguments are broken JSON."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="done"))]
            )
        message = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "get_customer_profile",
                    "args": '{"customer_id": ',
                    "id": "call_broken",
                    "error": "Unterminated JSON",
                    "type": "invalid_tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class TestStructuredOutputFailures:
    def test_supervisor_degrades_to_finish_when_no_object_is_returned(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "app.agents.supervisor.get_chat_model", _NoStructuredOutputModel
        )
        decision = _ask_llm(
            {"case_id": "CASE-X", "query": "Investigate customer C001", "findings": []}
        )
        assert decision.next_agent is AgentRoute.FINISH
        assert decision.confidence == 0.0
        assert "schema-valid" in decision.reasoning

    def test_specialist_returns_a_degraded_finding_instead_of_raising(self):
        finding = run_specialist(
            route=AgentRoute.CUSTOMER,
            system_prompt="You are a test specialist.",
            tools=list(CUSTOMER_TOOLS),
            context="case_id=CASE-X\ncustomer_id=C001\nlookback_days=30",
            model=_NoStructuredOutputModel(),
        )
        assert finding.agent is AgentRoute.CUSTOMER
        assert finding.confidence == 0.0
        assert finding.assessed_risk is RiskLevel.LOW


class TestInvalidToolCalls:
    def test_helper_produces_an_error_message_bound_to_the_call_id(self):
        message = invalid_tool_call_message(
            {
                "name": "get_customer_profile",
                "args": '{"customer_id": ',
                "id": "call_broken",
                "error": "Unterminated JSON",
            }
        )
        assert message.status == "error"
        assert message.tool_call_id == "call_broken"
        assert "not valid" in message.content

    def test_unparsable_arguments_are_answered_so_the_history_stays_valid(self):
        """An unanswered tool call is a 400 from OpenAI and Anthropic alike."""
        model = _InvalidToolCallModel()
        seen: list[list[BaseMessage]] = []

        original = ScriptedChatModel.with_structured_output

        def _capture(self, schema, **kwargs):
            runnable = original(self, schema, **kwargs)
            return RunnableLambda(
                lambda prompt: (seen.append(list(prompt)), runnable.invoke(prompt))[1]
            )

        model.__class__.with_structured_output = _capture  # type: ignore[method-assign]
        try:
            finding = run_specialist(
                route=AgentRoute.CUSTOMER,
                system_prompt="You are a test specialist.",
                tools=list(CUSTOMER_TOOLS),
                context="case_id=CASE-X\ncustomer_id=C001",
                model=model,
            )
        finally:
            model.__class__.with_structured_output = original  # type: ignore[method-assign]

        assert finding.agent is AgentRoute.CUSTOMER
        history = seen[-1]
        tool_call_ids = {
            call["id"]
            for message in history
            if isinstance(message, AIMessage)
            for call in (message.tool_calls or []) + (message.invalid_tool_calls or [])
        }
        answered = {
            message.tool_call_id
            for message in history
            if getattr(message, "type", "") == "tool"
        }
        assert tool_call_ids, "the model under test must have requested a tool"
        assert tool_call_ids <= answered, "every tool call must be answered"


# ------------------------------------------------------------ thread safety ---


class TestThreadHandling:
    def test_approving_a_case_that_is_not_paused_is_refused(self):
        outcome = run_investigation(
            InvestigationRequest(
                case_id="CASE-NOTPAUSED",
                query="Review customer C002 after a monitoring alert",
                customer_id="C002",
                lookback_days=60,
            )
        )
        assert outcome.status == "reported"

        with pytest.raises(InvestigationNotPausedError):
            resume_investigation(
                outcome.thread_id,
                HumanApproval(approved=True, approver="someone@bank.example"),
            )

    def test_a_case_cannot_be_approved_twice(self):
        outcome = run_investigation(
            InvestigationRequest(
                case_id="CASE-TWICE",
                query="Investigate cash deposits for customer C001",
                customer_id="C001",
                lookback_days=60,
            )
        )
        resume_investigation(
            outcome.thread_id,
            HumanApproval(approved=True, approver="first.reviewer"),
        )
        with pytest.raises(InvestigationNotPausedError):
            resume_investigation(
                outcome.thread_id,
                HumanApproval(approved=False, approver="second.reviewer"),
            )

    def test_reusing_a_thread_for_a_new_case_is_refused(self):
        """Append-only channels mean one thread holds exactly one investigation."""
        run_investigation(
            InvestigationRequest(
                case_id="CASE-T1",
                query="Review customer C002 after a monitoring alert",
                customer_id="C002",
                lookback_days=60,
            ),
            thread_id="shared-thread",
        )
        with pytest.raises(ThreadAlreadyUsedError):
            run_investigation(
                InvestigationRequest(
                    case_id="CASE-T2",
                    query="Investigate customer C004 large inbound transfer",
                    customer_id="C004",
                    lookback_days=60,
                ),
                thread_id="shared-thread",
            )


class TestStepBudget:
    def test_the_budget_is_not_exceeded_and_costs_no_extra_model_call(
        self, monkeypatch
    ):
        from app.agents import supervisor
        from app.core.config import get_settings
        from app.core.llm import reset_chat_model_cache
        from app.graph.build import reset_graph_cache

        monkeypatch.setenv("MAX_SUPERVISOR_STEPS", "2")
        get_settings.cache_clear()
        reset_chat_model_cache()
        reset_graph_cache()

        calls: list[int] = []
        real_ask = supervisor._ask_llm
        monkeypatch.setattr(
            supervisor,
            "_ask_llm",
            lambda state: (calls.append(1), real_ask(state))[1],
        )

        outcome = run_investigation(
            InvestigationRequest(
                case_id="CASE-BUDGET-2",
                query="Investigate cash deposits for customer C001",
                customer_id="C001",
            )
        )

        assert len(calls) == 2, "a routing call must not be paid for and discarded"
        assert len(outcome.route_history) == 2
        assert any("STEP_BUDGET_EXHAUSTED" in e for e in outcome.guardrail_events)
        assert "without another model call" in outcome.decisions[-1].reasoning
        assert outcome.report is not None


class TestApiConflicts:
    def test_approving_an_unpaused_investigation_returns_409(self, client):
        opened = client.post(
            "/investigations",
            json={
                "case_id": "CASE-API-409",
                "query": "Review customer C002 following a monitoring alert",
                "customer_id": "C002",
                "lookback_days": 60,
            },
        )
        assert opened.status_code == 200
        thread_id = opened.json()["thread_id"]

        response = client.post(
            f"/investigations/{thread_id}/approval",
            json={"approved": True, "approver": "officer@bank.example"},
        )
        assert response.status_code == 409
        assert "not awaiting authorisation" in response.json()["detail"]
