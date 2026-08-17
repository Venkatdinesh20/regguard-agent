"""Data access layer for the investigation tools.

The tools never read files or shape raw dictionaries themselves — they go
through this repository, which validates every record into a Pydantic model on
the way in. In production the JSON fixtures behind it are replaced by the core
banking / case-management adapters; the tool layer above does not change.

Time handling
-------------
``reference_date`` decides what "now" means when a lookback window is resolved,
and it is a configuration choice rather than a hard-coded one:

``TIME_ANCHOR=dataset`` (default)
    Anchor to the newest transaction in the shipped snapshot. Every run and every
    test is then deterministic regardless of the date it executes.
``TIME_ANCHOR=now``
    Anchor to wall-clock UTC, which is what a deployment reading a live source
    system needs.

The distinction matters when reading results: under ``dataset``, "the last 30
days" means the 30 days before the data ends, not before today.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import TimeAnchor, get_settings
from app.core.exceptions import RecordNotFoundError
from app.schemas.investigation import CustomerProfile, Transaction

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_json(filename: str) -> Any:
    path = DATA_DIR / filename
    if not path.exists():  # pragma: no cover - packaging guard
        raise FileNotFoundError(f"Missing data fixture: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z``."""
    normalised = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalised)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@lru_cache
def all_customers() -> dict[str, CustomerProfile]:
    """Every customer, validated, keyed by ``customer_id``."""
    records = _load_json("customers.json")
    return {
        record["customer_id"]: CustomerProfile.model_validate(record)
        for record in records
    }


@lru_cache
def all_transactions() -> tuple[Transaction, ...]:
    """Every transaction, validated, ordered by timestamp."""
    records = _load_json("transactions.json")
    parsed = [Transaction.model_validate(record) for record in records]
    parsed.sort(key=lambda item: parse_timestamp(item.timestamp))
    return tuple(parsed)


@lru_cache
def all_policies() -> tuple[dict[str, Any], ...]:
    """The policy corpus searched by the policy specialist."""
    return tuple(_load_json("policies.json"))


@lru_cache
def high_risk_jurisdictions() -> frozenset[str]:
    """Internal high-risk jurisdiction list."""
    reference = _load_json("reference.json")
    return frozenset(reference["high_risk_jurisdictions"])


@lru_cache
def dataset_reference_date() -> datetime:
    """The newest transaction timestamp in the shipped dataset."""
    return max(parse_timestamp(item.timestamp) for item in all_transactions())


def reference_date() -> datetime:
    """What "now" means for lookback windows, per ``TIME_ANCHOR``.

    Not cached: under ``TIME_ANCHOR=now`` a cached value would freeze the clock
    for the lifetime of the process.
    """
    if get_settings().time_anchor is TimeAnchor.NOW:
        return datetime.now(UTC)
    return dataset_reference_date()


def get_customer(customer_id: str) -> CustomerProfile:
    """Look up one customer or raise :class:`RecordNotFoundError`."""
    try:
        return all_customers()[customer_id]
    except KeyError as exc:
        known = ", ".join(sorted(all_customers()))
        raise RecordNotFoundError(
            f"No customer with id '{customer_id}'. Known customers: {known}."
        ) from exc


def get_transactions(customer_id: str, lookback_days: int = 30) -> list[Transaction]:
    """Transactions for a customer inside the lookback window.

    Raises :class:`RecordNotFoundError` if the customer itself is unknown, so a
    typo is reported as a bad identifier rather than as "no activity".
    """
    get_customer(customer_id)  # existence check
    cutoff = reference_date() - timedelta(days=lookback_days)
    return [
        item
        for item in all_transactions()
        if item.customer_id == customer_id and parse_timestamp(item.timestamp) >= cutoff
    ]


def known_customer_ids() -> list[str]:
    return sorted(all_customers())
