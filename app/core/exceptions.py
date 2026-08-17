"""RegGuard exception hierarchy.

A single root exception lets the API layer distinguish *our* failures from
unexpected ones, and lets callers catch a meaningful category rather than
``Exception``.
"""

from __future__ import annotations


class RegGuardError(Exception):
    """Base class for every error RegGuard raises deliberately."""


class ConfigurationError(RegGuardError):
    """Required configuration is missing or invalid."""


class ToolExecutionError(RegGuardError):
    """A domain tool failed while executing.

    Raised by the tool layer and surfaced back to the calling agent as an error
    ``ToolMessage`` so the LLM can react instead of the graph crashing.
    """


class GuardrailViolation(RegGuardError):
    """The agent attempted something the deterministic guardrails forbid.

    Examples: exceeding the step budget, re-entering a specialist too often,
    or requesting a route that does not exist.
    """


class RecordNotFoundError(RegGuardError):
    """A requested customer or record does not exist in the source system."""


class InvestigationNotPausedError(RegGuardError):
    """An approval was submitted for an investigation that is not awaiting one.

    Silently accepting it would tell a reviewer their authorisation was recorded
    when it was discarded — unacceptable in an approval workflow.
    """


class ThreadAlreadyUsedError(RegGuardError):
    """A caller tried to start a new investigation on an existing thread.

    State channels are append-only, so reusing a thread would mix two cases'
    evidence and let one case be reported on another's findings.
    """
