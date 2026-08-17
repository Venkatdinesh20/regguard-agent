"""Argument schemas for every tool an agent may call.

An LLM chooses tool arguments, which means tool arguments are untrusted input.
Every tool that takes arguments therefore declares an explicit Pydantic
``args_schema``: a malformed or out-of-range argument is rejected before the tool
body runs, and the model receives a validation error it can correct on the next
turn. (The three no-argument tools need no schema.)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerLookupArgs(BaseModel):
    customer_id: str = Field(
        min_length=1,
        max_length=32,
        description="Identifier of the customer, for example 'C001'.",
    )


class TransactionQueryArgs(BaseModel):
    customer_id: str = Field(
        min_length=1,
        max_length=32,
        description="Identifier of the customer, for example 'C001'.",
    )
    lookback_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Size of the review window in days, counted back from the "
        "most recent activity on the account.",
    )


class PolicySearchArgs(BaseModel):
    topic: str = Field(
        min_length=3,
        max_length=200,
        description="What to look up, for example 'structuring cash deposits' "
        "or 'SAR filing deadline'.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=6,
        description="How many policy documents to return.",
    )
