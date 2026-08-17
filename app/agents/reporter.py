"""Intake, reporting and the human authorisation gate."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import ValidationError

from app.agents.context import report_context
from app.agents.prompts import REPORT_PROMPT
from app.core.llm import get_chat_model
from app.core.logging import get_logger
from app.graph.state import InvestigationState
from app.schemas.findings import Finding
from app.schemas.investigation import (
    HumanApproval,
    InvestigationReport,
    RiskLevel,
)

logger = get_logger(__name__)

_CUSTOMER_IN_QUERY = re.compile(r"\b(C\d{3,})\b")


def intake_node(state: InvestigationState) -> dict[str, Any]:
    """Deterministic pre-processing before any model is called.

    Resolving the subject of the investigation is a parsing problem, not a
    reasoning problem, so it is done in code: cheaper, and it cannot hallucinate
    a customer identifier.
    """
    customer_id = state.get("customer_id")
    if not customer_id:
        match = _CUSTOMER_IN_QUERY.search(state.get("query", ""))
        customer_id = match.group(1) if match else None

    logger.info(
        "intake.resolved",
        extra={
            "customer_id": customer_id,
            "lookback_days": state.get("lookback_days"),
        },
    )
    return {"customer_id": customer_id, "status": "running"}


def derive_overall_risk(findings: list[Finding]) -> RiskLevel:
    """The case risk is the highest risk any specialist evidenced.

    Deliberately computed, not asked of the model: the headline risk level of a
    compliance case must be reproducible from the evidence.
    """
    if not findings:
        return RiskLevel.LOW
    return max((finding.assessed_risk for finding in findings), key=lambda r: r.rank)


def report_node(state: InvestigationState) -> dict[str, Any]:
    """Write the final report as validated structured output."""
    findings = state.get("findings") or []
    overall_risk = derive_overall_risk(findings)
    enriched: InvestigationState = {**state, "overall_risk": overall_risk}

    writer = get_chat_model().with_structured_output(InvestigationReport)
    try:
        report = writer.invoke(
            [
                SystemMessage(content=REPORT_PROMPT),
                HumanMessage(content=report_context(enriched)),
            ]
        )
        if not isinstance(report, InvestigationReport):  # pragma: no cover
            report = InvestigationReport.model_validate(report)
    except (ValidationError, ValueError) as exc:
        logger.error("report.invalid", extra={"error": str(exc)})
        report = _fallback_report(state, overall_risk, str(exc))

    # The model may not restate provenance or overrule the computed risk level.
    report = report.model_copy(
        update={
            "case_id": state.get("case_id", report.case_id),
            "customer_id": state.get("customer_id"),
            "risk_level": overall_risk,
            "requires_sar_filing": (
                report.requires_sar_filing or overall_risk is RiskLevel.HIGH
            ),
        }
    )

    logger.info(
        "report.written",
        extra={
            "risk_level": report.risk_level.value,
            "requires_sar_filing": report.requires_sar_filing,
            "findings_used": len(findings),
        },
    )
    return {
        "report": report,
        "overall_risk": overall_risk,
        "status": "reported",
    }


def _fallback_report(
    state: InvestigationState,
    risk: RiskLevel,
    error: str,
) -> InvestigationReport:
    findings = state.get("findings") or []
    return InvestigationReport(
        case_id=state.get("case_id", "UNKNOWN-CASE"),
        customer_id=state.get("customer_id"),
        risk_level=risk,
        summary=(
            "The report writer failed to produce schema-valid output, so this "
            "report was assembled deterministically from the specialist "
            f"findings. Underlying error: {error}"
        ),
        key_findings=[finding.as_context_line() for finding in findings]
        or ["No specialist findings were recorded."],
        recommended_actions=["Route this case to a human analyst for manual review."],
        regulatory_considerations=[],
        requires_sar_filing=risk is RiskLevel.HIGH,
        confidence=0.0,
    )


def human_approval_node(state: InvestigationState) -> dict[str, Any]:
    """Pause the graph until a qualified human authorises the outcome.

    ``interrupt()`` suspends execution and persists the checkpoint. The API
    returns the pending payload to the caller; when a reviewer answers, the graph
    resumes from exactly this point with their decision. Nothing adverse — no
    report filed, no account restricted — happens before that.
    """
    report = state.get("report")
    logger.info("approval.requested", extra={"case_id": state.get("case_id")})

    response = interrupt(
        {
            "type": "approval_required",
            "case_id": state.get("case_id"),
            "customer_id": state.get("customer_id"),
            "risk_level": report.risk_level.value if report else RiskLevel.LOW.value,
            "requires_sar_filing": bool(report and report.requires_sar_filing),
            "summary": report.summary if report else "",
            "recommended_actions": list(report.recommended_actions) if report else [],
            "question": (
                "Authorise this recommendation? Respond with approved, approver "
                "and optional notes."
            ),
        }
    )

    try:
        approval = (
            response
            if isinstance(response, HumanApproval)
            else HumanApproval.model_validate(response)
        )
    except ValidationError as exc:
        logger.error("approval.invalid_payload", extra={"error": str(exc)})
        approval = HumanApproval(
            approved=False,
            approver="system",
            notes=f"Malformed approval payload rejected: {exc}",
        )

    logger.info(
        "approval.recorded",
        extra={"approved": approval.approved, "approver": approval.approver},
    )
    return {
        "approval": approval,
        "status": "approved" if approval.approved else "rejected",
    }
