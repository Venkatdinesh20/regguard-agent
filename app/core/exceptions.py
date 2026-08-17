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
