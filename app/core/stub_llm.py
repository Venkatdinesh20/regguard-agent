"""A deterministic in-process chat model used for tests and the keyless demo.

Why this exists
---------------
An agent system whose only execution path requires a paid API key is hard to
test and hard to review. ``ScriptedChatModel`` implements the same
:class:`~langchain_core.language_models.chat_models.BaseChatModel` interface the
real providers do — including ``bind_tools`` and ``with_structured_output`` — so
the graph, the guardrails, the tool layer, the API and the human-in-the-loop
pause are all exercised end to end by ``pytest`` with no network access.

It is a **test double, not business logic**. Its "decisions" are simple rules
over the prompt it is handed, chosen to imitate a competent supervisor: gather
identity, then activity, then score, then policy, then stop. Swapping
``LLM_PROVIDER`` to ``openai`` or ``anthropic`` replaces it with a real model and
changes nothing else in the codebase.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.schemas.findings import Finding
from app.schemas.investigation import InvestigationReport, RiskLevel
from app.schemas.routing import AgentRoute, RouteDecision

_CUSTOMER_ID_RE = re.compile(r"customer_id\s*[=:]\s*([A-Za-z0-9_-]+)")
_LOOSE_CUSTOMER_RE = re.compile(r"\b(C\d{3,})\b")
_LOOKBACK_RE = re.compile(r"lookback_days\s*[=:]\s*(\d+)")
_CASE_ID_RE = re.compile(r"case_id\s*[=:]\s*([A-Za-z0-9._-]+)")
_QUERY_RE = re.compile(r"(?:investigation )?query\s*[=:]\s*(.+)", re.IGNORECASE)
_FINDING_LINE_RE = re.compile(r"^-\s+(CUSTOMER|TRANSACTION|FRAUD|POLICY):")

COMPLIANCE_VOCABULARY = (
    "structuring",
    "cash",
    "deposit",
    "threshold",
    "wire",
    "transfer",
    "transaction",
    "suspicious",
    "laundering",
    "fraud",
    "sanctions",
    "pep",
    "kyc",
    "dormant",
    "dispersal",
    "mule",
    "sar",
    "str",
    "aml",
    "risk",
    "customer",
    "account",
    "policy",
    "reporting",
    "layering",
    "jurisdiction",
    "unusual",
    "investigate",
    "compliance",
    "escalation",
)

FINDINGS_HEADER = "Findings collected so far:"
"""Mirrors ``app.agents.context.FINDINGS_HEADER``.

Duplicated deliberately: ``app.core`` is the bottom layer and must not import
from ``app.agents``. ``tests/test_stub_llm.py`` asserts the two stay identical.
"""

_ROUTE_SEQUENCE = (
    AgentRoute.CUSTOMER,
    AgentRoute.TRANSACTION,
    AgentRoute.FRAUD,
    AgentRoute.POLICY,
)


def _text_of(messages: Sequence[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.content
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(parts)


def _extract_customer_id(text: str) -> str:
    match = _CUSTOMER_ID_RE.search(text)
    if match and match.group(1).lower() not in {"none", "null", "unknown"}:
        return match.group(1)
    loose = _LOOSE_CUSTOMER_RE.search(text)
    return loose.group(1) if loose else "C001"


def _extract_lookback(text: str) -> int:
    match = _LOOKBACK_RE.search(text)
    return int(match.group(1)) if match else 30


def _tool_payloads(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


class ScriptedChatModel(BaseChatModel):
    """Deterministic stand-in for a tool-calling, structured-output chat model."""

    name: str = "regguard-stub"

    @property
    def _llm_type(self) -> str:
        return "regguard-scripted"

    # -- tool calling -------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Record the available tools the same way a real provider would."""
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools: list[dict[str, Any]] = kwargs.get("tools") or []
        text = _text_of(messages)
        already_called = any(isinstance(m, ToolMessage) for m in messages)

        if tools and not already_called:
            primary = tools[0]["function"]
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": primary["name"],
                        "args": self._arguments_for(primary, text),
                        "id": f"call_{primary['name']}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content="Evidence gathered; ready to summarise.")

        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _arguments_for(function_schema: dict[str, Any], text: str) -> dict[str, Any]:
        """Fill a tool's arguments from the prompt, honouring its schema."""
        properties = (function_schema.get("parameters") or {}).get("properties", {})
        args: dict[str, Any] = {}
        if "customer_id" in properties:
            args["customer_id"] = _extract_customer_id(text)
        if "lookback_days" in properties:
            args["lookback_days"] = _extract_lookback(text)
        if "topic" in properties:
            args["topic"] = ScriptedChatModel._policy_topic(text)
        if "top_k" in properties:
            args["top_k"] = 3
        return args

    @staticmethod
    def _policy_topic(text: str) -> str:
        lowered = text.lower()
        hits = [word for word in COMPLIANCE_VOCABULARY if word in lowered]
        return " ".join(hits[:8]) or "suspicious transaction reporting"

    # -- structured output --------------------------------------------------

    def with_structured_output(
        self,
        schema: Any,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Return a runnable that fabricates a valid instance of ``schema``."""

        def _build(prompt: Any) -> Any:
            messages = _coerce_messages(prompt)
            # Routing and reporting read only the state rendered into the human
            # turn; the system prompt names every specialist and would otherwise
            # look like evidence that they had already run.
            human_text = _text_of([m for m in messages if m.type == "human"])
            if schema is RouteDecision:
                return _decide_route(human_text)
            if schema is Finding:
                return _build_finding(messages, _text_of(messages))
            if schema is InvestigationReport:
                return _build_report(human_text)
            raise NotImplementedError(  # pragma: no cover - defensive
                f"ScriptedChatModel has no script for {schema!r}"
            )

        return RunnableLambda(_build)


def _coerce_messages(prompt: Any) -> list[BaseMessage]:
    if isinstance(prompt, BaseMessage):
        return [prompt]
    if isinstance(prompt, str):
        from langchain_core.messages import HumanMessage

        return [HumanMessage(content=prompt)]
    if isinstance(prompt, Sequence):
        return [m for m in prompt if isinstance(m, BaseMessage)]
    if hasattr(prompt, "to_messages"):  # PromptValue
        messages: list[BaseMessage] = prompt.to_messages()
        return messages
    return []  # pragma: no cover - defensive


# --------------------------------------------------------------- scripts ----


def _decide_route(text: str) -> RouteDecision:
    """Imitate a competent supervisor: identity, activity, score, policy, stop."""
    query_match = _QUERY_RE.search(text)
    query = query_match.group(1).strip().lower() if query_match else text.lower()

    # Only the evidence block counts as "already collected".
    scope = text.split(FINDINGS_HEADER, 1)[1] if FINDINGS_HEADER in text else ""
    completed = {route for route in _ROUTE_SEQUENCE if f"{route.value}:" in scope}

    if not completed and not any(word in query for word in COMPLIANCE_VOCABULARY):
        return RouteDecision(
            next_agent=AgentRoute.FINISH,
            reasoning=(
                "The query does not describe a financial-crime compliance "
                "matter, so no specialist applies."
            ),
            confidence=0.9,
        )

    for route in _ROUTE_SEQUENCE:
        if route not in completed:
            return RouteDecision(
                next_agent=route,
                reasoning=(
                    f"{route.value} evidence has not been collected yet; it is "
                    "the next prerequisite for a defensible assessment."
                ),
                confidence=0.85,
            )

    return RouteDecision(
        next_agent=AgentRoute.FINISH,
        reasoning=(
            "Customer, transaction, risk-scoring and policy evidence are all "
            "present; the report can be written."
        ),
        confidence=0.9,
    )


def _build_finding(messages: Sequence[BaseMessage], text: str) -> Finding:
    payloads = _tool_payloads(messages)
    merged = {key: value for payload in payloads for key, value in payload.items()}

    if "risk_score" in merged:
        rules = merged.get("triggered_rules") or []
        signals = [f"{rule['rule_id']}: {rule['description']}" for rule in rules][:10]
        return Finding(
            agent=AgentRoute.FRAUD,
            summary=(
                f"Deterministic rule engine {merged.get('rules_version')} scored "
                f"customer {merged.get('customer_id')} at "
                f"{merged.get('risk_score')}/100 = {merged.get('risk_level')} "
                f"across {merged.get('transactions_examined')} transactions, "
                f"triggering {len(rules)} rule(s)."
            ),
            risk_signals=signals,
            assessed_risk=RiskLevel(merged.get("risk_level", "LOW")),
            confidence=0.9,
        )

    if "kyc_status" in merged:
        signals = []
        if merged.get("pep_flag"):
            signals.append("Customer is a politically exposed person.")
        if merged.get("sanctions_hit"):
            signals.append("Sanctions screening returned a match.")
        if merged.get("kyc_status") != "VERIFIED":
            signals.append(f"KYC status is {merged.get('kyc_status')}.")
        if merged.get("is_new_account"):
            signals.append("Account is less than 180 days old.")
        return Finding(
            agent=AgentRoute.CUSTOMER,
            summary=(
                f"{merged.get('full_name')} ({merged.get('customer_id')}), "
                f"resident of {merged.get('country')}, account age "
                f"{merged.get('account_age_days')} days, KYC "
                f"{merged.get('kyc_status')}, monthly spend baseline "
                f"{merged.get('average_monthly_spend')}."
            ),
            risk_signals=signals,
            assessed_risk=RiskLevel.MEDIUM if signals else RiskLevel.LOW,
            confidence=0.85,
        )

    if "transaction_count" in merged:
        signals = []
        cash_total = merged.get("cash_credit_total") or 0
        if cash_total >= 10_000:
            signals.append(f"Cash credits total {cash_total:,.2f} in the window.")
        ratio = merged.get("credit_to_baseline_ratio")
        if isinstance(ratio, int | float) and ratio >= 5:
            signals.append(f"Credits are {ratio}x the monthly baseline.")
        if merged.get("foreign_transaction_count"):
            signals.append(
                f"{merged['foreign_transaction_count']} cross-border transactions."
            )
        return Finding(
            agent=AgentRoute.TRANSACTION,
            summary=(
                f"{merged.get('transaction_count')} transactions in the last "
                f"{merged.get('window_days')} days: credits "
                f"{merged.get('total_credit')}, debits {merged.get('total_debit')}, "
                f"largest credit {merged.get('largest_credit')}, "
                f"{merged.get('distinct_counterparties')} distinct counterparties."
            ),
            risk_signals=signals,
            assessed_risk=RiskLevel.MEDIUM if signals else RiskLevel.LOW,
            confidence=0.85,
        )

    if "results" in merged or "policies" in merged:
        results = merged.get("results") or merged.get("policies") or []
        citations = [
            f"{item.get('policy_id')}: {item.get('title')}" for item in results
        ][:10]
        return Finding(
            agent=AgentRoute.POLICY,
            summary=(
                f"Retrieved {len(results)} applicable policy document(s) for "
                f"topic '{merged.get('topic', 'compliance obligations')}'. "
                "Obligations include reporting deadlines, enhanced due "
                "diligence triggers and the prohibition on tipping off."
            ),
            risk_signals=citations,
            assessed_risk=RiskLevel.LOW,
            confidence=0.8,
        )

    return Finding(
        agent=AgentRoute.CUSTOMER,
        summary=(
            "No tool output was available to this specialist, so no evidence "
            f"could be established. Context length {len(text)} characters."
        ),
        risk_signals=[],
        assessed_risk=RiskLevel.LOW,
        confidence=0.2,
    )


def _build_report(text: str) -> InvestigationReport:
    case_match = _CASE_ID_RE.search(text)
    case_id = case_match.group(1) if case_match else "UNKNOWN-CASE"
    customer_id = _extract_customer_id(text) if "customer_id" in text else None

    risk = RiskLevel.LOW
    for level in (RiskLevel.HIGH, RiskLevel.MEDIUM):
        if f"risk={level.value}" in text or f"overall_risk={level.value}" in text:
            risk = level
            break

    findings = [
        line.strip().lstrip("- ").strip()
        for line in text.splitlines()
        if _FINDING_LINE_RE.match(line.strip())
    ][:10] or ["No specialist findings were recorded for this investigation."]

    if risk is RiskLevel.HIGH:
        actions = [
            "Escalate to a compliance officer for suspicious transaction "
            "report review.",
            "Apply enhanced due diligence and request source-of-funds evidence.",
            "Do not contact the customer about the report (no tipping off).",
        ]
    elif risk is RiskLevel.MEDIUM:
        actions = [
            "Place the account under enhanced monitoring for 90 days.",
            "Request supporting documentation for the anomalous activity.",
        ]
    else:
        actions = ["Close the case with no further action and retain the record."]

    return InvestigationReport(
        case_id=case_id,
        customer_id=customer_id,
        risk_level=risk,
        summary=(
            f"Investigation {case_id} assessed overall risk as {risk.value} on "
            f"the basis of {len(findings)} specialist finding(s). "
            "See key findings for the evidence relied upon and the policy "
            "references that apply."
        ),
        key_findings=findings,
        recommended_actions=actions,
        regulatory_considerations=[
            "POL-STR-002: report within 30 days of establishing suspicion.",
            "POL-HITL-006: automated systems may not file reports unilaterally.",
        ],
        requires_sar_filing=risk is RiskLevel.HIGH,
        confidence=0.8,
    )
