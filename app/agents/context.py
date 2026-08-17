"""Rendering graph state into prompts.

Every prompt the agents receive is built here, in one deterministic format:
machine-readable ``key=value`` lines first, then a human-readable evidence
block. That makes prompts diffable in logs and makes it obvious which piece of
state an agent could possibly have acted on.
"""

from __future__ import annotations

from app.graph.state import InvestigationState
from app.schemas.findings import Finding
from app.schemas.routing import AgentRoute

FINDINGS_HEADER = "Findings collected so far:"


def _header(state: InvestigationState) -> list[str]:
    return [
        f"case_id={state.get('case_id', 'UNKNOWN')}",
        f"customer_id={state.get('customer_id') or 'unknown'}",
        f"lookback_days={state.get('lookback_days', 30)}",
        f"query: {state.get('query', '')}",
    ]


def evidence_block(findings: list[Finding]) -> str:
    if not findings:
        return f"{FINDINGS_HEADER}\n- (none yet)"
    lines = "\n".join(f"- {finding.as_context_line()}" for finding in findings)
    return f"{FINDINGS_HEADER}\n{lines}"


def supervisor_context(state: InvestigationState) -> str:
    """What the supervisor sees before choosing the next specialist."""
    findings = state.get("findings") or []
    already = list(state.get("route_history") or [])
    lines = _header(state)
    lines.append(f"steps_taken={state.get('step_count', 0)}")
    lines.append(f"specialists_already_run={', '.join(already) or 'none'}")
    return (
        "\n".join(lines)
        + "\n\n"
        + evidence_block(findings)
        + ("\n\nDecide which specialist should act next.")
    )


def specialist_context(state: InvestigationState, route: AgentRoute) -> str:
    """What a specialist sees before choosing its tools."""
    findings = state.get("findings") or []
    lines = _header(state)
    lines.append(f"your_role={route.value}")
    return (
        "\n".join(lines)
        + "\n\n"
        + evidence_block(findings)
        + ("\n\nGather the evidence that is yours to gather, using your tools.")
    )


def report_context(state: InvestigationState) -> str:
    """What the report writer sees. Overall risk is supplied, not inferred."""
    findings = state.get("findings") or []
    risk = state.get("overall_risk")
    signals = [signal for finding in findings for signal in finding.risk_signals]
    lines = _header(state)
    lines.append(f"overall_risk={risk.value if risk else 'LOW'}")
    lines.append(f"specialists_consulted={', '.join(state.get('route_history') or [])}")
    block = evidence_block(findings)
    signal_block = (
        "\n\nRed flags reported by specialists:\n"
        + "\n".join(f"- {signal}" for signal in signals)
        if signals
        else ""
    )
    return (
        "\n".join(lines)
        + "\n\n"
        + block
        + signal_block
        + "\n\nWrite the final case report."
    )
