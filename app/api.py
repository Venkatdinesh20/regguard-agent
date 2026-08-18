"""HTTP surface for RegGuard.

Three ideas are load-bearing here:

* **Validation at the edge.** Request bodies are Pydantic models, so malformed
  input is rejected with a 422 before any model is called or any money is spent.
* **Pause and resume as first-class operations.** A HIGH-risk investigation ends
  in ``202 Accepted`` with a ``thread_id``, not in a filed report. Authorisation
  is a separate, authenticated action.
* **The audit trail is part of the response.** Callers receive every routing
  decision, every guardrail override and every finding — not just the verdict.
* **An approval is attributed to an authenticated principal.** When
  ``AUTH_ENABLED`` is set, the approver's identity comes from their bearer token
  and any name in the request body is ignored. Production refuses to start
  without it (``app/core/config.py``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    InvestigationNotPausedError,
    RegGuardError,
    ThreadAlreadyUsedError,
)
from app.core.logging import configure_logging, get_logger
from app.core.security import Principal, principal_for_token
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
    auth_enabled: bool = Field(
        description="Whether the investigation endpoints require a bearer token."
    )
    time_anchor: str = Field(
        description="Whether lookback windows are anchored to the dataset "
        "snapshot or to wall-clock time."
    )


class ApprovalRequest(BaseModel):
    """A human reviewer's authorisation decision."""

    approved: bool = Field(description="True to authorise the recommendation.")
    approver: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Only used when AUTH_ENABLED is false, for local runs. When "
        "authentication is on, the approver is taken from the bearer token and "
        "this field is ignored.",
    )
    notes: str = Field(default="", max_length=1000)


_bearer = HTTPBearer(auto_error=False, description="Bearer token from API_TOKENS.")


def authenticate(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> Principal | None:
    """Resolve the caller, or ``None`` when authentication is disabled.

    Disabled is the documented default so the repository runs out of the box;
    ``ENVIRONMENT=production`` refuses to start in that state.
    """
    if not settings.auth_enabled:
        return None

    token = credentials.credentials if credentials else ""
    principal = principal_for_token(token, settings)
    if principal is None:
        logger.warning("api.auth_failed", extra={"token_supplied": bool(token)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_approver(
    principal: Principal | None = Depends(authenticate),
) -> Principal | None:
    """Additionally require the 'approver' role to authorise an outcome."""
    if principal is not None and not principal.can_approve:
        logger.warning(
            "api.authorisation_denied",
            extra={"principal": principal.name, "role": principal.role.value},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This principal may not authorise investigations.",
        )
    return principal


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "api.startup",
        extra={
            "environment": settings.environment,
            "llm_provider": settings.llm_provider.value,
            "auth_enabled": settings.auth_enabled,
            "time_anchor": settings.time_anchor.value,
        },
    )
    if not settings.auth_enabled:
        logger.warning(
            "api.auth_disabled",
            extra={
                "detail": "Investigation endpoints are unauthenticated and an "
                "approver's identity is self-declared. Acceptable for local "
                "runs only."
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


# Resolve the static directory relative to this file so the app works
# regardless of the working directory it is launched from.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/index.html")


@app.exception_handler(RegGuardError)
async def regguard_error_handler(_: Request, exc: RegGuardError) -> JSONResponse:
    """Map RegGuard's own errors to status codes; anything else stays a 500."""
    conflict = isinstance(exc, ThreadAlreadyUsedError | InvestigationNotPausedError)
    logger.warning("api.domain_error", extra={"error": str(exc)})
    return JSONResponse(
        status_code=(
            status.HTTP_409_CONFLICT if conflict else status.HTTP_400_BAD_REQUEST
        ),
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
        auth_enabled=settings.auth_enabled,
        time_anchor=settings.time_anchor.value,
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
    principal: Principal | None = Depends(authenticate),
) -> InvestigationOutcome:
    """Open an investigation and run it until it concludes or needs a human.

    Returns ``202 Accepted`` when the graph paused for authorisation, ``200 OK``
    when the case completed without needing it.
    """
    logger.info(
        "api.investigation_requested",
        extra={"principal": principal.name if principal else "unauthenticated"},
    )
    outcome = run_investigation(request)
    if outcome.status == "awaiting_approval":
        response.status_code = status.HTTP_202_ACCEPTED
    return outcome


@app.get(
    "/investigations/{thread_id}",
    response_model=InvestigationOutcome,
    tags=["investigations"],
)
def read_investigation(
    thread_id: str,
    principal: Principal | None = Depends(authenticate),
) -> InvestigationOutcome:
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
    principal: Principal | None = Depends(require_approver),
) -> InvestigationOutcome:
    """Record a human authorisation decision and resume the paused graph.

    Returns ``409 Conflict`` if the investigation is not awaiting authorisation:
    a reviewer must never be told their decision was recorded when it was not.
    """
    approver = _resolve_approver(principal, decision)
    try:
        return resume_investigation(
            thread_id,
            HumanApproval(
                approved=decision.approved,
                approver=approver,
                notes=decision.notes,
            ),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except InvestigationNotPausedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


def _resolve_approver(principal: Principal | None, decision: ApprovalRequest) -> str:
    """Decide whose name goes into the audit trail.

    An authenticated principal always wins: the body cannot be used to record an
    authorisation under someone else's name, and an attempt to do so is logged.
    """
    if principal is not None:
        if decision.approver and decision.approver != principal.name:
            logger.warning(
                "api.approver_overridden",
                extra={
                    "authenticated": principal.name,
                    "claimed": decision.approver,
                    "detail": "Body approver ignored in favour of the "
                    "authenticated principal.",
                },
            )
        return principal.name

    if not decision.approver:
        raise HTTPException(
            # Literal 422: the Starlette constant for it was renamed, and this
            # code should not depend on which spelling the pinned version uses.
            status_code=422,
            detail="approver is required while AUTH_ENABLED is false.",
        )
    return decision.approver
