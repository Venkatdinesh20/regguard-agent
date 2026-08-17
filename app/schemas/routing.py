"""The schema that makes LLM-decided control flow safe.

The supervisor LLM does not emit free text that we then pattern-match. It emits
a :class:`RouteDecision`, validated by Pydantic before the graph acts on it. An
invented route name is a ``ValidationError``, not a mis-dispatch.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AgentRoute(StrEnum):
    """Every destination the supervisor is allowed to choose."""

    CUSTOMER = "CUSTOMER"
    TRANSACTION = "TRANSACTION"
    FRAUD = "FRAUD"
    POLICY = "POLICY"
    FINISH = "FINISH"


class RouteDecision(BaseModel):
    """The supervisor's decision about which specialist acts next.

    Field descriptions are not comments — they are serialised into the JSON
    schema sent to the model, so they are part of the prompt.
    """

    next_agent: AgentRoute = Field(
        description=(
            "The single specialist that should act next. Use FINISH when the "
            "findings already collected are sufficient to write the report, or "
            "when the query falls outside compliance investigation scope."
        )
    )
    reasoning: str = Field(
        min_length=10,
        max_length=500,
        description=(
            "Why this specialist, given the findings collected so far. This is "
            "written to the audit trail."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this routing decision, from 0.0 to 1.0.",
    )
