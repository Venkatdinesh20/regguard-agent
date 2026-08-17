"""HTTP surface for RegGuard.

Three ideas are load-bearing here:

* **Validation at the edge.** Request bodies are Pydantic models, so malformed
  input is rejected with a 422 before any model is called or any money is spent.
* **Pause and resume as first-class operations.** A HIGH-risk investigation ends
  in ``202 Accepted`` with a ``thread_id``, not in a filed report. Authorisation
  is a separate, authenticated action.
* **The audit trail is part of the response.** Callers receive every routing
  decision, every guardrail override and every finding — not just the verdict.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import RegGuardError
from app.core.logging import configure_logging, get_logger
from app.schemas.investigation import HumanApproval, InvestigationRequest
from app.service import (
    InvestigationOutcome,
    get_investigation,
    resume_investigation,
    run_investigation,
)

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    llm_provider: str
    llm_model: str


class ApprovalRequest(BaseModel):
    """A human reviewer's authorisation decision."""

    approved: bool = Field(description="True to authorise the recommendation.")
    approver: str = Field(
        min_length=1,
        max_length=128,
        description="Identity of the authorising reviewer, recorded in the audit "
        "trail. In a real deployment this comes from the authenticated "
        "principal, not the request body.",
    )
    notes: str = Field(default="", max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "api.startup",
        extra={
            "environment": settings.environment,
            "llm_provider": settings.llm_provider.value,
        },
    )
    yield
    logger.info("api.shutdown")


app = FastAPI(
    title="RegGuard",
    version="1.0.0",
    description=(
        "Multi-agent financial-crime investigation service. An LLM supervisor "
        "decides which specialist agent runs at each step; deterministic "
        "guardrails bound its choices, and HIGH-risk outcomes require human "
        "authorisation before release."
    ),
    lifespan=lifespan,
)


@app.exception_handler(RegGuardError)
async def regguard_error_handler(_: object, exc: RegGuardError) -> JSONResponse:
    logger.warning("api.domain_error", extra={"error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness and configuration probe."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
        llm_provider=settings.llm_provider.value,
        llm_model=settings.llm_model,
    )


@app.post(
    "/investigations",
    response_model=InvestigationOutcome,
    status_code=status.HTTP_200_OK,
    tags=["investigations"],
)
def create_investigation(
    request: InvestigationRequest,
    response: Response,
) -> InvestigationOutcome:
    """Open an investigation and run it until it concludes or needs a human.

    Returns ``202 Accepted`` when the graph paused for authorisation, ``200 OK``
    when the case completed without needing it.
    """
    outcome = run_investigation(request)
    if outcome.status == "awaiting_approval":
        response.status_code = status.HTTP_202_ACCEPTED
    return outcome


@app.get(
    "/investigations/{thread_id}",
    response_model=InvestigationOutcome,
    tags=["investigations"],
)
def read_investigation(thread_id: str) -> InvestigationOutcome:
    """Fetch the current state and audit trail of an investigation."""
    try:
        return get_investigation(thread_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@app.post(
    "/investigations/{thread_id}/approval",
    response_model=InvestigationOutcome,
    tags=["investigations"],
)
def approve_investigation(
    thread_id: str,
    decision: ApprovalRequest = Body(...),
) -> InvestigationOutcome:
    """Record a human authorisation decision and resume the paused graph."""
    try:
        return resume_investigation(
            thread_id,
            HumanApproval(
                approved=decision.approved,
                approver=decision.approver,
                notes=decision.notes,
            ),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
