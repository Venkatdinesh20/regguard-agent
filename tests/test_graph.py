"""End-to-end graph behaviour, including the human-in-the-loop pause."""

from __future__ import annotations

from app.schemas.investigation import HumanApproval, InvestigationRequest, RiskLevel
from app.schemas.routing import AgentRoute
from app.service import get_investigation, resume_investigation, run_investigation


def _request(case_id: str, query: str, customer_id: str | None = None, **kwargs):
    return InvestigationRequest(
        case_id=case_id, query=query, customer_id=customer_id, **kwargs
    )


class TestHighRiskInvestigation:
    def setup_method(self):
        self.outcome = run_investigation(
            _request(
                "CASE-HIGH",
                "Investigate unusual cash deposit activity for customer C001",
                "C001",
                lookback_days=60,
            )
        )

    def test_supervisor_visited_every_specialist_in_a_defensible_order(self):
        assert self.outcome.route_history == [
            "CUSTOMER",
            "TRANSACTION",
            "FRAUD",
            "POLICY",
        ]

    def test_every_step_was_an_llm_routing_decision(self):
        assert len(self.outcome.decisions) == 5  # four specialists, then FINISH
        assert self.outcome.decisions[-1].next_agent is AgentRoute.FINISH
        assert all(d.reasoning for d in self.outcome.decisions)

    def test_no_guardrail_had_to_intervene_on_a_well_behaved_run(self):
        assert self.outcome.guardrail_events == []

    def test_findings_are_attributed_to_the_specialist_that_ran(self):
        agents = [finding.agent.value for finding in self.outcome.findings]
        assert agents == self.outcome.route_history

    def test_high_risk_pauses_for_human_authorisation(self):
        assert self.outcome.status == "awaiting_approval"
        assert self.outcome.pending_approval is not None
        assert self.outcome.pending_approval["risk_level"] == "HIGH"
        assert self.outcome.pending_approval["requires_sar_filing"] is True

    def test_report_is_produced_before_the_pause(self):
        assert self.outcome.report is not None
        assert self.outcome.report.risk_level is RiskLevel.HIGH
        assert self.outcome.report.case_id == "CASE-HIGH"
        assert self.outcome.report.customer_id == "C001"

    def test_state_can_be_read_back_while_paused(self):
        snapshot = get_investigation(self.outcome.thread_id)
        assert snapshot.status == "awaiting_approval"
        assert snapshot.case_id == "CASE-HIGH"

    def test_approval_resumes_and_is_recorded(self):
        resumed = resume_investigation(
            self.outcome.thread_id,
            HumanApproval(
                approved=True, approver="j.reviewer", notes="Evidence reviewed."
            ),
        )
        assert resumed.status == "approved"
        assert resumed.approval is not None
        assert resumed.approval.approver == "j.reviewer"
        assert resumed.pending_approval is None

    def test_rejection_is_recorded_as_rejected(self):
        outcome = run_investigation(
            _request("CASE-REJECT", "Investigate customer C001 cash deposits", "C001")
        )
        resumed = resume_investigation(
            outcome.thread_id,
            HumanApproval(approved=False, approver="k.officer", notes="Insufficient."),
        )
        assert resumed.status == "rejected"
        assert resumed.approval.approved is False


class TestLowRiskInvestigation:
    def test_clean_customer_completes_without_human_approval(self):
        outcome = run_investigation(
            _request(
                "CASE-LOW",
                "Review customer C002 after an alert",
                "C002",
                lookback_days=60,
            )
        )
        assert outcome.status == "reported"
        assert outcome.pending_approval is None
        assert outcome.report is not None
        assert outcome.report.risk_level is RiskLevel.LOW
        assert outcome.report.requires_sar_filing is False


class TestOutOfScopeQuery:
    def test_supervisor_finishes_immediately_without_spending_tool_calls(self):
        outcome = run_investigation(
            _request("CASE-SCOPE", "What will the weather be in Toronto tomorrow?")
        )
        assert outcome.route_history == []
        assert outcome.findings == []
        assert len(outcome.decisions) == 1
        assert outcome.decisions[0].next_agent is AgentRoute.FINISH
        assert outcome.status == "reported"


class TestAuditTrail:
    def test_outcome_serialises_for_transport_and_storage(self):
        outcome = run_investigation(
            _request(
                "CASE-AUDIT",
                "Investigate customer C004 large inbound transfer",
                "C004",
                lookback_days=60,
            )
        )
        payload = outcome.model_dump(mode="json")
        assert payload["case_id"] == "CASE-AUDIT"
        assert payload["risk_level"] == "MEDIUM"
        assert len(payload["decisions"]) == len(outcome.decisions)
        assert all("reasoning" in decision for decision in payload["decisions"])
