"""The shared tool-calling loop used by every specialist agent.

One implementation, four specialists. Each specialist differs only in its system
prompt and the tools it is allowed to touch, which keeps the failure handling,
the iteration cap and the provenance guarantees identical across all of them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.agents.prompts import SUMMARISE_INSTRUCTION
from app.core.config import get_settings
from app.core.exceptions import RegGuardError
from app.core.llm import get_chat_model
from app.core.logging import get_logger
from app.schemas.findings import Finding
from app.schemas.investigation import RiskLevel
from app.schemas.routing import AgentRoute

logger = get_logger(__name__)

MAX_TOOL_RESULT_CHARS = 8_000
"""Tool output is truncated before it reaches the model: a runaway result must
not blow the context window or the token bill."""


def _serialise(result: Any) -> str:
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        return text[:MAX_TOOL_RESULT_CHARS] + '... [truncated]"'
    return text


def execute_tool_call(
    tools_by_name: dict[str, BaseTool],
    tool_call: dict[str, Any],
) -> ToolMessage:
    """Run one LLM-requested tool call, converting any failure into a message.

    A tool failure must never crash the graph. The error is handed back to the
    model as an error ``ToolMessage`` so it can correct its arguments or choose
    another tool — the same way a human analyst reacts to "no such customer".
    """
    name = tool_call.get("name", "")
    call_id = tool_call.get("id") or f"call_{name}"
    arguments = tool_call.get("args") or {}

    tool = tools_by_name.get(name)
    if tool is None:
        logger.warning("tool.unknown", extra={"tool": name})
        return ToolMessage(
            content=(
                f"Error: tool '{name}' does not exist. Available tools: "
                f"{sorted(tools_by_name)}."
            ),
            tool_call_id=call_id,
            status="error",
        )

    try:
        result = tool.invoke(arguments)
    except ValidationError as exc:
        logger.warning(
            "tool.invalid_arguments",
            extra={"tool": name, "arguments": arguments, "error": str(exc)},
        )
        return ToolMessage(
            content=f"Error: invalid arguments for '{name}': {exc}",
            tool_call_id=call_id,
            status="error",
        )
    except RegGuardError as exc:
        logger.warning("tool.domain_error", extra={"tool": name, "error": str(exc)})
        return ToolMessage(
            content=f"Error from '{name}': {exc}",
            tool_call_id=call_id,
            status="error",
        )
    except Exception as exc:  # deliberate backstop: no tool may crash the graph
        logger.exception("tool.unexpected_error", extra={"tool": name})
        return ToolMessage(
            content=f"Unexpected error from '{name}': {type(exc).__name__}: {exc}",
            tool_call_id=call_id,
            status="error",
        )

    logger.info("tool.executed", extra={"tool": name, "arguments": arguments})
    return ToolMessage(content=_serialise(result), tool_call_id=call_id)


def run_specialist(
    route: AgentRoute,
    system_prompt: str,
    tools: Sequence[BaseTool],
    context: str,
    model: BaseChatModel | None = None,
) -> Finding:
    """Let a specialist gather evidence with its tools, then return a Finding."""
    settings = get_settings()
    chat_model = model or get_chat_model()
    tools_by_name = {tool.name: tool for tool in tools}
    bound = chat_model.bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    used_tools: list[str] = []
    completed = False

    for iteration in range(settings.max_tool_iterations):
        response = bound.invoke(messages)
        messages.append(response)
        tool_calls = list(getattr(response, "tool_calls", None) or [])

        if not tool_calls:
            completed = True
            break

        for tool_call in tool_calls:
            used_tools.append(tool_call.get("name", "?"))
            messages.append(execute_tool_call(tools_by_name, tool_call))

        logger.info(
            "specialist.tool_round",
            extra={
                "agent": route.value,
                "iteration": iteration + 1,
                "tool_calls": [call.get("name") for call in tool_calls],
            },
        )

    if not completed:
        logger.warning(
            "guardrail.tool_iteration_cap",
            extra={"agent": route.value, "cap": settings.max_tool_iterations},
        )

    finding = _summarise(chat_model, route, messages)

    logger.info(
        "specialist.finding",
        extra={
            "agent": route.value,
            "tools_used": used_tools,
            "assessed_risk": finding.assessed_risk.value,
            "signal_count": len(finding.risk_signals),
        },
    )
    return finding


def _summarise(
    chat_model: BaseChatModel,
    route: AgentRoute,
    messages: list[BaseMessage],
) -> Finding:
    """Turn the conversation into a validated Finding, with one retry."""
    summariser = chat_model.with_structured_output(Finding)
    prompt = [*messages, HumanMessage(content=SUMMARISE_INSTRUCTION)]

    for attempt in (1, 2):
        try:
            finding = summariser.invoke(prompt)
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "specialist.structured_output_failed",
                extra={"agent": route.value, "attempt": attempt, "error": str(exc)},
            )
            prompt = [
                *prompt,
                HumanMessage(
                    content=(
                        "Your previous response did not satisfy the required "
                        f"schema ({exc}). Respond again, obeying every field "
                        "constraint exactly."
                    )
                ),
            ]
            continue

        if not isinstance(finding, Finding):  # pragma: no cover - provider guard
            finding = Finding.model_validate(finding)

        # Provenance is asserted by the graph, never by the model: a specialist
        # cannot claim to be a different specialist.
        if finding.agent is not route:
            logger.warning(
                "guardrail.finding_provenance_corrected",
                extra={"agent": route.value, "claimed": finding.agent.value},
            )
            finding = finding.model_copy(update={"agent": route})
        return finding

    logger.error("specialist.degraded_finding", extra={"agent": route.value})
    return Finding(
        agent=route,
        summary=(
            f"The {route.value} specialist could not produce a schema-valid "
            "summary of its evidence. This finding is recorded as incomplete so "
            "the case is not silently under-evidenced."
        ),
        risk_signals=["Specialist output failed schema validation twice."],
        assessed_risk=RiskLevel.LOW,
        confidence=0.0,
    )
