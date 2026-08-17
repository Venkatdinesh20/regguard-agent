"""Domain schemas for the investigation boundary.

These models are the contract between the outside world (API callers, source
systems, LLM output) and RegGuard's business logic. Anything that fails
validation here never reaches an agent.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def rank(self) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[self.value]


class InvestigationRequest(BaseModel):
    """An analyst's request to open an investigation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    case_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Caller-supplied case reference, used as the audit key.",
    )
    query: str = Field(
        min_length=5,
        max_length=2000,
        description="Plain-language description of what must be investigated.",
    )
    customer_id: str | None = Field(
        default=None,
        max_length=32,
        description="Subject of the investigation, if already known.",
    )
    lookback_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Transaction window the specialists should consider.",
    )


class CustomerProfile(BaseModel):
    """KYC view of a customer, as returned by the customer source system."""

    customer_id: str = Field(min_length=1, max_length=32)
    full_name: str
    country: str = Field(min_length=2, max_length=56)
    account_age_days: int = Field(ge=0)
    average_monthly_spend: float = Field(ge=0)
    kyc_status: str
    pep_flag: bool = False
    sanctions_hit: bool = False
    occupation: str | None = None

    @field_validator("kyc_status")
    @classmethod
    def _known_kyc_status(cls, value: str) -> str:
        allowed = {"VERIFIED", "PENDING", "EXPIRED", "REJECTED"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"kyc_status must be one of {sorted(allowed)}")
        return upper


class Transaction(BaseModel):
    """A single monetary movement."""

    transaction_id: str = Field(min_length=1, max_length=32)
    customer_id: str = Field(min_length=1, max_length=32)
    timestamp: str
    amount: float = Field(gt=0, description="Positive magnitude in account currency.")
    currency: str = Field(default="CAD", min_length=3, max_length=3)
    direction: str
    channel: str
    counterparty: str
    country: str

    @field_validator("direction")
    @classmethod
    def _known_direction(cls, value: str) -> str:
        upper = value.upper()
        if upper not in {"DEBIT", "CREDIT"}:
            raise ValueError("direction must be DEBIT or CREDIT")
        return upper


class InvestigationReport(BaseModel):
    """The deliverable. Produced by the report agent as structured output."""

    case_id: str
    customer_id: str | None = None
    risk_level: RiskLevel
    summary: str = Field(
        min_length=20,
        max_length=3000,
        description="Analyst-facing narrative of what was found.",
    )
    key_findings: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Evidence bullets, each traceable to a specialist's output.",
    )
    recommended_actions: list[str] = Field(min_length=1, max_length=20)
    regulatory_considerations: list[str] = Field(default_factory=list, max_length=20)
    requires_sar_filing: bool = Field(
        default=False,
        description="Whether a Suspicious Activity/Transaction Report is advised.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


class HumanApproval(BaseModel):
    """A human decision recorded against a paused investigation."""

    approved: bool
    approver: str = Field(min_length=1, max_length=128)
    notes: str = Field(default="", max_length=1000)
