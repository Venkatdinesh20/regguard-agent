"""Command-line entrypoint: run investigations and print the audit trail.

python -m app.main                    # run the built-in demo cases
python -m app.main --customer C003    # investigate one customer
python -m app.main --query "..."      # investigate a free-text query
"""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.schemas.investigation import HumanApproval, InvestigationRequest
from app.service import InvestigationOutcome, resume_investigation, run_investigation

DEMO_CASES: list[tuple[str, str, str | None]] = [
    (
        "CASE-1001",
        "Investigate unusual cash deposit activity for customer C001",
        "C001",
    ),
    (
        "CASE-1002",
        "Review customer C002 following an automated monitoring alert",
        "C002",
    ),
    (
        "CASE-1003",
        "Assess wire activity and source of funds for customer C003",
        "C003",
    ),
    (
        "CASE-1004",
        "Investigate a large inbound transfer and rapid dispersal for customer C004",
        "C004",
    ),
    (
        "CASE-1005",
        "What will the weather be in Toronto tomorrow?",
        None,
    ),
]

RULE = "=" * 78


def _print_outcome(outcome: InvestigationOutcome) -> None:
    print(RULE)
    print(f"{outcome.case_id}  status={outcome.status}")
    print(RULE)

    print("\nControl flow chosen by the LLM supervisor:")
    for index, decision in enumerate(outcome.decisions, start=1):
        print(
            f"  {index}. -> {decision.next_agent.value:<12}"
            f"(confidence {decision.confidence:.2f})"
        )
        print(f"     {decision.reasoning}")

    if outcome.guardrail_events:
        print("\nGuardrail interventions:")
        for event in outcome.guardrail_events:
            print(f"  ! {event}")

    if outcome.findings:
        print("\nEvidence collected:")
        for finding in outcome.findings:
            print(f"  [{finding.agent.value}] {finding.summary}")
            for signal in finding.risk_signals:
                print(f"      - {signal}")

    report = outcome.report
    if report:
        print(f"\nRisk level: {report.risk_level.value}")
        print(f"SAR/STR recommended: {report.requires_sar_filing}")
        print(f"\nSummary:\n  {report.summary}")
        print("\nRecommended actions:")
        for action in report.recommended_actions:
            print(f"  - {action}")
        if report.regulatory_considerations:
            print("\nRegulatory considerations:")
            for item in report.regulatory_considerations:
                print(f"  - {item}")

    if outcome.pending_approval:
        print("\n*** PAUSED FOR HUMAN AUTHORISATION ***")
        print(f"  {outcome.pending_approval.get('question')}")
        print(f"  thread_id={outcome.thread_id}")

    if outcome.approval:
        verdict = "APPROVED" if outcome.approval.approved else "REJECTED"
        print(
            f"\nHuman decision: {verdict} by {outcome.approval.approver}"
            + (f" — {outcome.approval.notes}" if outcome.approval.notes else "")
        )
    print()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="regguard",
        description="Run a RegGuard financial-crime investigation.",
    )
    parser.add_argument("--case-id", default="CASE-CLI", help="Case reference.")
    parser.add_argument("--query", help="What to investigate, in plain language.")
    parser.add_argument("--customer", help="Customer identifier, e.g. C001.")
    parser.add_argument(
        "--lookback-days", type=int, default=30, help="Review window in days."
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Simulate a human reviewer approving any paused investigation. "
        "Demo convenience only — never a production behaviour. Without it, a "
        "HIGH-risk case stops at the authorisation gate, as it should.",
    )
    return parser.parse_args(argv)


def _run_one(
    case_id: str,
    query: str,
    customer_id: str | None,
    lookback_days: int,
    auto_approve: bool,
) -> InvestigationOutcome:
    outcome = run_investigation(
        InvestigationRequest(
            case_id=case_id,
            query=query,
            customer_id=customer_id,
            lookback_days=lookback_days,
        )
    )
    _print_outcome(outcome)

    if outcome.status == "awaiting_approval" and auto_approve:
        print(">>> simulating human authorisation\n")
        outcome = resume_investigation(
            outcome.thread_id,
            HumanApproval(
                approved=True,
                approver="demo.reviewer",
                notes="Approved in demo mode; evidence reviewed.",
            ),
        )
        _print_outcome(outcome)
    return outcome


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    print(
        f"\n{settings.app_name} — provider={settings.llm_provider.value} "
        f"model={settings.llm_model}\n"
    )

    if args.query or args.customer:
        query = args.query or (
            f"Investigate potentially suspicious activity for customer {args.customer}"
        )
        _run_one(
            args.case_id,
            query,
            args.customer,
            args.lookback_days,
            args.auto_approve,
        )
        return 0

    for case_id, query, customer_id in DEMO_CASES:
        _run_one(case_id, query, customer_id, args.lookback_days, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
