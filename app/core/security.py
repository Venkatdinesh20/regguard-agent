"""Authentication, and the identity an approval is attributed to.

The point of this module is one property: **a reviewer's identity is
authenticated, not self-declared.** Before it existed, the approval endpoint
took ``approver`` from the request body, so any caller could record an
authorisation in someone else's name — worthless as an audit trail, and the one
thing a financial-crime workflow cannot get wrong.

Scope is deliberately small. Tokens live in configuration, which is *not* how a
bank does this: real deployments authenticate against an identity provider (OIDC)
and read the principal and its roles from a verified token. The seam is
:func:`principal_for_token` — replace it and the endpoints below do not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class Role(StrEnum):
    """What a principal is allowed to do."""

    ANALYST = "analyst"
    """May open investigations and read their audit trail."""

    APPROVER = "approver"
    """May additionally authorise a HIGH-risk outcome. A superset of analyst."""


@dataclass(frozen=True)
class Principal:
    """An authenticated caller."""

    name: str
    role: Role

    @property
    def can_approve(self) -> bool:
        return self.role is Role.APPROVER


def parse_api_tokens(raw: str) -> dict[str, Principal]:
    """Parse ``API_TOKENS`` into a token → principal map.

    Format: comma-separated ``token:principal:role`` entries, where ``role``
    defaults to ``analyst`` if omitted. Malformed configuration raises rather
    than silently granting or denying access.
    """
    tokens: dict[str, Principal] = {}

    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue

        fields = [field.strip() for field in entry.split(":")]
        if len(fields) == 2:
            token, name = fields
            role_name = Role.ANALYST.value
        elif len(fields) == 3:
            token, name, role_name = fields
        else:
            raise ConfigurationError(
                f"Malformed API_TOKENS entry '{entry}': expected "
                "'token:principal' or 'token:principal:role'."
            )

        if not token or not name:
            raise ConfigurationError(
                f"Malformed API_TOKENS entry '{entry}': token and principal "
                "must both be non-empty."
            )
        try:
            role = Role(role_name.lower())
        except ValueError as exc:
            roles = ", ".join(sorted(role.value for role in Role))
            raise ConfigurationError(
                f"Unknown role '{role_name}' in API_TOKENS. Valid roles: {roles}."
            ) from exc
        if token in tokens:
            raise ConfigurationError(
                f"Duplicate token in API_TOKENS for principal '{name}'."
            )

        tokens[token] = Principal(name=name, role=role)

    return tokens


def principal_for_token(token: str, settings: Settings) -> Principal | None:
    """Resolve a bearer token to a principal, or ``None`` if it is not valid."""
    if not token:
        return None
    return parse_api_tokens(settings.api_tokens).get(token)


def configured_principals(settings: Settings) -> list[Principal]:
    """Every principal the current configuration recognises. For diagnostics."""
    return list(parse_api_tokens(settings.api_tokens).values())
