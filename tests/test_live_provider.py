"""Opt-in verification against a real LLM.

Everything else in this suite runs against the deterministic stub, which proves
the graph, guardrails and API but cannot prove that a real provider's tool
calling and structured output behave as the code expects. This module closes that
gap, and it is the check to run before claiming the system works end to end.

    LLM_PROVIDER=openai LLM_MODEL=gpt-4.1-mini OPENAI_API_KEY=sk-... \
        REGGUARD_LIVE_TEST=1 pytest -m live -v

or:

    make test-live

Skipped by default: it costs money, needs network, and is not deterministic.
``pyproject.toml`` excludes the ``live`` marker from the default run.
"""

from __future__ import annotations

import os

import pytest

from app.core.config import LLMProvider, get_settings
from app.schemas.investigation import InvestigationRequest, RiskLevel
from app.schemas.routing import AgentRoute
from app.service import run_investigation

pytestmark = pytest.mark.live


def _reason_to_skip() -> str | None:
    if os.getenv("REGGUARD_LIVE_TEST", "").lower() not in {"1", "true", "yes"}:
        return "set REGGUARD_LIVE_TEST=1 to run the live-provider check"

    settings = get_settings()
    if settings.llm_provider is LLMProvider.STUB:
        return "set LLM_PROVIDER=openai or anthropic for the live-provider check"
    if settings.llm_provider is LLMProvider.OPENAI and not settings.openai_api_key:
        return "OPENAI_API_KEY is not set"
    if (
        settings.llm_provider is LLMProvider.ANTHROPIC
        and not settings.anthropic_api_key
    ):
        return "ANTHROPIC_API_KEY is not set"
    return None


@pytest.fixture(autouse=True)
def _requires_live_provider():
    reason = _reason_to_skip()
    if reason:
        pytest.skip(reason)


@pytest.fixture(scope="module")
def live_outcome():
    """One real investigation, shared by the assertions below to limit spend."""
    if _reason_to_skip():
        pytest.skip("live provider not configured")

    return run_investigation(
        InvestigationRequest(
            case_id="CASE-LIVE-1",
            query="Investigate unusual cash deposit activity for customer C001",
            customer_id="C001",
            lookback_days=60,
        )
    )


class TestLiveProvider:
    def test_the_model_returned_parseable_routing_decisions(self, live_outcome):
        """The whole design rests on structured output actually validating."""
        assert live_outcome.decisions, "the supervisor produced no decisions"
        for decision in live_outcome.decisions:
            assert isinstance(decision.next_agent, AgentRoute)
            assert 0.0 <= decision.confidence <= 1.0
            assert len(decision.reasoning) >= 10

    def test_no_decision_was_a_degraded_fallback(self, live_outcome):
        """Confidence 0.0 is what the code emits when parsing failed."""
        fallbacks = [d for d in live_outcome.decisions if d.confidence == 0.0]
        assert not fallbacks, f"schema failures or guardrail overrides: {fallbacks}"

    def test_tool_calling_worked_and_produced_real_evidence(self, live_outcome):
        assert live_outcome.route_history, "no specialist was ever dispatched"
        assert live_outcome.findings, "no evidence was gathered"
        for finding in live_outcome.findings:
            assert finding.confidence > 0.0, f"{finding.agent} degraded: {finding}"
            assert len(finding.summary) >= 10

    def test_the_deterministic_score_survived_the_model(self, live_outcome):
        """The rule engine's verdict must reach the report unaltered."""
        fraud = [f for f in live_outcome.findings if f.agent is AgentRoute.FRAUD]
        if not fraud:
            pytest.skip("the model chose not to route to FRAUD on this run")
        assert fraud[0].assessed_risk is RiskLevel.HIGH
        assert any("R01_STRUCTURING" in signal for signal in fraud[0].risk_signals)

    def test_a_high_risk_case_still_stops_for_a_human(self, live_outcome):
        if live_outcome.risk_level is not RiskLevel.HIGH:
            pytest.skip(f"model-driven run assessed {live_outcome.risk_level}")
        assert live_outcome.status == "awaiting_approval"
        assert live_outcome.pending_approval is not None
        assert live_outcome.report is not None
        assert live_outcome.report.requires_sar_filing is True

    def test_an_out_of_scope_query_costs_almost_nothing(self):
        """A real model should also decline work that is not compliance work."""
        outcome = run_investigation(
            InvestigationRequest(
                case_id="CASE-LIVE-2",
                query="What will the weather be in Toronto tomorrow?",
            )
        )
        assert outcome.decisions[0].next_agent is AgentRoute.FINISH
        assert outcome.route_history == []
