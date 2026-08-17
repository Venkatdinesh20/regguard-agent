"""Structured logging with per-investigation correlation.

Compliance systems must be auditable: for any decision we need to reconstruct
which case it belonged to, which agent made it, and in what order. Every log
record therefore carries the current ``case_id``, propagated through a
:class:`contextvars.ContextVar` so nothing has to thread it manually.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_case_id: ContextVar[str] = ContextVar("case_id", default="-")

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


@contextmanager
def case_context(case_id: str) -> Iterator[None]:
    """Bind ``case_id`` to every log record emitted inside the block."""
    token = _case_id.set(case_id)
    try:
        yield
    finally:
        _case_id.reset(token)


def current_case_id() -> str:
    return _case_id.get()


class CaseIdFilter(logging.Filter):
    """Attach the ambient ``case_id`` to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.case_id = _case_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, suitable for shipping to a log store."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "case_id": getattr(record, "case_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install RegGuard's root logging configuration. Idempotent."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(CaseIdFilter())

    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(case_id)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party chatter stays out of the audit trail.
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
