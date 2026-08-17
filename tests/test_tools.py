"""Tools and the deterministic risk engine.

The scoring engine is the part of the system an auditor would challenge, so it
is tested by expected outcome per fixture customer, not just for "no crash".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.exceptions import RecordNotFoundError
from app.tools.customer import get_customer_profile
from app.tools.fraud import describe_risk_rules, score_customer_risk, score_fraud_risk
from app.tools.policy import search_policies, search_policy
from app.tools.repository import get_customer, get_transactions, known_customer_ids
from app.tools.transactions import summarise_transactions

WINDOW = 60


class TestRepository:
    def test_fixtures_load_and_validate(self):
        assert known_customer_ids() == ["C001", "C002", "C003", "C004"]

    def test_unknown_customer_raises_domain_error(self):
        with pytest.raises(RecordNotFoundError) as exc:
            get_customer("C999")
        assert "C999" in str(exc.value)
        assert "C001" in str(exc.value)  # error message helps the agent recover

    def test_lookback_window_filters_transactions(self):
        wide = get_transactions("C001", 60)
        narrow = get_transactions("C001", 1)
        assert len(wide) > len(narrow)


class TestRiskEngine:
    def test_structuring_case_scores_high(self):
        result = score_customer_risk("C001", WINDOW)
        rules = {rule["rule_id"] for rule in result["triggered_rules"]}
        assert result["risk_level"] == "HIGH"
        assert "R01_STRUCTURING" in rules
        assert "R02_THRESHOLD_AGGREGATION" in rules
        assert result["risk_score"] == 65

    def test_clean_customer_scores_low_with_no_rules(self):
        result = score_customer_risk("C002", WINDOW)
        assert result["risk_level"] == "LOW"
        assert result["risk_score"] == 0
        assert result["clean"] is True

    def test_pep_pass_through_case_scores_high(self):
        result = score_customer_risk("C003", WINDOW)
        rules = {rule["rule_id"] for rule in result["triggered_rules"]}
        assert result["risk_level"] == "HIGH"
        assert {"R03_PASS_THROUGH", "R09_PEP", "R10_KYC_NOT_VERIFIED"} <= rules

    def test_mule_pattern_scores_medium(self):
        result = score_customer_risk("C004", WINDOW)
        rules = {rule["rule_id"] for rule in result["triggered_rules"]}
        assert result["risk_level"] == "MEDIUM"
        assert "R06_RAPID_DISPERSAL" in rules

    def test_score_is_capped_at_100(self):
        assert score_customer_risk("C003", WINDOW)["risk_score"] <= 100

    def test_scoring_is_deterministic(self):
        first = score_customer_risk("C001", WINDOW)
        second = score_customer_risk("C001", WINDOW)
        assert first == second

    def test_every_rule_carries_evidence(self):
        for rule in score_customer_risk("C001", WINDOW)["triggered_rules"]:
            assert rule["evidence"], f"{rule['rule_id']} has no evidence"

    def test_rule_catalogue_is_exposed_for_explanation(self):
        catalogue = describe_risk_rules.invoke({})
        assert catalogue["rules_version"].startswith("regguard-rules-")
        assert len(catalogue["rules"]) == 10


class TestToolArgumentValidation:
    """LLM-supplied arguments are untrusted input."""

    def test_out_of_range_lookback_is_rejected(self):
        with pytest.raises(ValidationError):
            score_fraud_risk.invoke({"customer_id": "C001", "lookback_days": 9999})

    def test_missing_required_argument_is_rejected(self):
        with pytest.raises(ValidationError):
            get_customer_profile.invoke({})

    def test_empty_customer_id_is_rejected(self):
        with pytest.raises(ValidationError):
            get_customer_profile.invoke({"customer_id": ""})

    def test_valid_arguments_succeed(self):
        payload = get_customer_profile.invoke({"customer_id": "C001"})
        assert payload["kyc_status"] == "VERIFIED"
        assert payload["is_new_account"] is False


class TestTransactionSummary:
    def test_summary_exposes_baseline_comparison(self):
        summary = summarise_transactions.invoke(
            {"customer_id": "C001", "lookback_days": WINDOW}
        )
        assert summary["transaction_count"] == 12
        assert summary["cash_credit_total"] == pytest.approx(28_300.0)
        assert summary["credit_to_baseline_ratio"] > 5

    def test_summary_handles_empty_window(self):
        """C001's newest activity predates the dataset's reference date."""
        summary = summarise_transactions.invoke(
            {"customer_id": "C001", "lookback_days": 1}
        )
        assert summary["transaction_count"] == 0
        assert "No transactions" in summary["note"]


class TestPolicyRetrieval:
    def test_structuring_query_retrieves_structuring_policy(self):
        results = search_policies("structuring cash deposits below threshold")[
            "results"
        ]
        assert results[0]["policy_id"] == "POL-STR-001"

    def test_reporting_query_retrieves_reporting_policy(self):
        results = search_policies("SAR filing deadline suspicious transaction")[
            "results"
        ]
        assert results[0]["policy_id"] == "POL-STR-002"

    def test_no_match_returns_guidance_rather_than_noise(self):
        payload = search_policies("zzzz unrelated aardvark topic")
        assert payload["results"] == []
        assert "try different terms" in payload["note"]

    def test_top_k_is_bounded(self):
        with pytest.raises(ValidationError):
            search_policy.invoke({"topic": "structuring", "top_k": 99})
