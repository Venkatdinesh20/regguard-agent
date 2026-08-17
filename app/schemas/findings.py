"""What a specialist agent hands back to the supervisor."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.investigation import RiskLevel
from app.schemas.routing import AgentRoute


class Finding(BaseModel):
    """One specialist's contribution to the evidence file.

    Specialists never mutate shared state directly and never talk to each
    other; they append a ``Finding``. The supervisor routes on the accumulated
    list. That keeps the graph's state append-only and auditable.
    """

    agent: AgentRoute = Field(description="Which specialist produced this finding.")
    summary: str = Field(
        min_length=10,
        max_length=1200,
        description="What this specialist established, in plain language.",
    )
    risk_signals: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Concrete red flags observed, empty if none.",
    )
    assessed_risk: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="This specialist's own risk read, scoped to its evidence.",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def as_context_line(self) -> str:
        """Render for inclusion in the supervisor prompt."""
        signals = "; ".join(self.risk_signals) if self.risk_signals else "none"
        return (
            f"{self.agent.value}: {self.summary} "
            f"[risk={self.assessed_risk.value}, signals={signals}]"
        )
