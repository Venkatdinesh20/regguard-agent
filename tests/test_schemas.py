"""Boundary validation: bad data must never reach business logic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.findings import Finding
from app.schemas.investigation import (
    CustomerProfile,
    InvestigationReport,
    InvestigationRequest,
    RiskLevel,
    Transaction,
)
from app.schemas.routing import AgentRoute, RouteDecision


class TestRouteDecision:
    def test_valid_decision(self):
        decision = RouteDecision(
            next_agent="CUSTOMER",
            reasoning="No profile evidence has been collected yet.",
            confidence=0.9,
        )
        assert decision.next_agent is AgentRoute.CUSTOMER

    def test_hallucinated_route_is_rejected(self):
        """The core safety property of LLM-decided control flow."""
        with pytest.raises(ValidationError):
            RouteDecision(
                next_agent="FREEZE_THE_ACCOUNT",
                reasoning="An invented destination that does not exist.",
                confidence=0.9,
            )

    @pytest.mark.parametrize("confidence", [-0.1, 1.5, 42.0])
    def test_confidence_must_be_a_probability(self, confidence):
        with pytest.raises(ValidationError):
            RouteDecision(
                next_agent="FRAUD",
                reasoning="Confidence outside the permitted range.",
                confidence=confidence,
            )

    def test_reasoning_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            RouteDecision(next_agent="FINISH", reasoning="ok", confidence=0.5)


class TestInvestigationRequest:
    def test_valid_request(self):
        request = InvestigationRequest(
            case_id="CASE-1001",
            query="Investigate unusual cash deposits for customer C001",
        )
        assert request.lookback_days == 30

    @pytest.mark.parametrize(
        "case_id",
        ["", "CASE 1001", "CASE/1001", "x" * 65, "DROP TABLE cases;"],
    )
    def test_malformed_case_ids_are_rejected(self, case_id):
        with pytest.raises(ValidationError):
            InvestigationRequest(case_id=case_id, query="A valid enough query")

    def test_query_must_be_substantive(self):
        with pytest.raises(ValidationError):
            InvestigationRequest(case_id="CASE-1", query="hi")

    @pytest.mark.parametrize("days", [0, -5, 400])
    def test_lookback_window_is_bounded(self, days):
        with pytest.raises(ValidationError):
            InvestigationRequest(
                case_id="CASE-1",
                query="Investigate the account activity",
                lookback_days=days,
            )


class TestCustomerProfile:
    def test_negative_account_age_is_rejected(self):
        with pytest.raises(ValidationError):
            CustomerProfile(
                customer_id="C001",
                full_name="Test Person",
                country="Canada",
                account_age_days=-100,
                average_monthly_spend=2500,
                kyc_status="VERIFIED",
            )

    def test_unknown_kyc_status_is_rejected(self):
        with pytest.raises(ValidationError):
            CustomerProfile(
                customer_id="C001",
                full_name="Test Person",
                country="Canada",
                account_age_days=100,
                average_monthly_spend=2500,
                kyc_status="PROBABLY_FINE",
            )

    def test_kyc_status_is_normalised(self):
        profile = CustomerProfile(
            customer_id="C001",
            full_name="Test Person",
            country="Canada",
            account_age_days=100,
            average_monthly_spend=2500,
            kyc_status="verified",
        )
        assert profile.kyc_status == "VERIFIED"


class TestTransaction:
    def test_zero_amount_is_rejected(self):
        with pytest.raises(ValidationError):
            Transaction(
                transaction_id="T1",
                customer_id="C001",
                timestamp="2026-07-01T00:00:00Z",
                amount=0,
                direction="DEBIT",
                channel="CARD",
                counterparty="Shop",
                country="Canada",
            )

    def test_unknown_direction_is_rejected(self):
        with pytest.raises(ValidationError):
            Transaction(
                transaction_id="T1",
                customer_id="C001",
                timestamp="2026-07-01T00:00:00Z",
                amount=10,
                direction="SIDEWAYS",
                channel="CARD",
                counterparty="Shop",
                country="Canada",
            )


class TestReportAndFinding:
    def test_report_requires_at_least_one_finding_and_action(self):
        with pytest.raises(ValidationError):
            InvestigationReport(
                case_id="CASE-1",
                risk_level=RiskLevel.LOW,
                summary="A summary long enough to satisfy the minimum length.",
                key_findings=[],
                recommended_actions=[],
                confidence=0.5,
            )

    def test_risk_levels_are_ordered(self):
        assert RiskLevel.HIGH.rank > RiskLevel.MEDIUM.rank > RiskLevel.LOW.rank

    def test_finding_renders_context_line(self):
        finding = Finding(
            agent=AgentRoute.FRAUD,
            summary="Rule engine scored the customer at 65/100.",
            risk_signals=["R01_STRUCTURING"],
            assessed_risk=RiskLevel.HIGH,
            confidence=0.9,
        )
        line = finding.as_context_line()
        assert line.startswith("FRAUD:")
        assert "risk=HIGH" in line
        assert "R01_STRUCTURING" in line
