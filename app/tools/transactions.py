"""Tools owned by the transaction specialist."""

from __future__ import annotations

from collections import Counter
from typing import Any

from langchain_core.tools import StructuredTool

from app.core.logging import get_logger
from app.schemas.tool_args import TransactionQueryArgs
from app.tools.repository import get_customer, get_transactions, reference_date

logger = get_logger(__name__)

MAX_RETURNED_TRANSACTIONS = 50
"""Cap on rows handed to the model: context is a budget, not a bucket."""


def _list_transactions(customer_id: str, lookback_days: int = 30) -> dict[str, Any]:
    """Return the raw transactions for a customer inside the review window."""
    transactions = get_transactions(customer_id, lookback_days)
    logger.info(
        "transactions.listed",
        extra={
            "customer_id": customer_id,
            "lookback_days": lookback_days,
            "count": len(transactions),
        },
    )
    rows = [
        {
            "transaction_id": item.transaction_id,
            "timestamp": item.timestamp,
            "amount": item.amount,
            "currency": item.currency,
            "direction": item.direction,
            "channel": item.channel,
            "counterparty": item.counterparty,
            "country": item.country,
        }
        for item in transactions[:MAX_RETURNED_TRANSACTIONS]
    ]
    return {
        "customer_id": customer_id,
        "window_days": lookback_days,
        "window_end": reference_date().isoformat(),
        "returned": len(rows),
        "total_matching": len(transactions),
        "truncated": len(transactions) > MAX_RETURNED_TRANSACTIONS,
        "transactions": rows,
    }


def _summarise_transactions(
    customer_id: str, lookback_days: int = 30
) -> dict[str, Any]:
    """Aggregate a customer's activity and compare it against their baseline."""
    profile = get_customer(customer_id)
    transactions = get_transactions(customer_id, lookback_days)

    if not transactions:
        return {
            "customer_id": customer_id,
            "window_days": lookback_days,
            "transaction_count": 0,
            "note": "No transactions in the review window.",
        }

    credits = [item for item in transactions if item.direction == "CREDIT"]
    debits = [item for item in transactions if item.direction == "DEBIT"]
    cash_credits = [item for item in credits if item.channel in {"CASH", "ATM"}]

    total_credit = round(sum(item.amount for item in credits), 2)
    total_debit = round(sum(item.amount for item in debits), 2)
    baseline = profile.average_monthly_spend or 0.0

    summary: dict[str, Any] = {
        "customer_id": customer_id,
        "window_days": lookback_days,
        "window_end": reference_date().isoformat(),
        "transaction_count": len(transactions),
        "credit_count": len(credits),
        "debit_count": len(debits),
        "total_credit": total_credit,
        "total_debit": total_debit,
        "largest_credit": round(max((i.amount for i in credits), default=0.0), 2),
        "largest_debit": round(max((i.amount for i in debits), default=0.0), 2),
        "cash_credit_count": len(cash_credits),
        "cash_credit_total": round(sum(item.amount for item in cash_credits), 2),
        "baseline_monthly_spend": baseline,
        "debit_to_baseline_ratio": (
            round(total_debit / baseline, 2) if baseline else None
        ),
        "credit_to_baseline_ratio": (
            round(total_credit / baseline, 2) if baseline else None
        ),
        "by_channel": dict(Counter(item.channel for item in transactions)),
        "by_country": dict(Counter(item.country for item in transactions)),
        "distinct_counterparties": len({item.counterparty for item in transactions}),
        "foreign_transaction_count": sum(
            1 for item in transactions if item.country != profile.country
        ),
    }
    logger.info(
        "transactions.summarised",
        extra={"customer_id": customer_id, "count": len(transactions)},
    )
    return summary


list_transactions = StructuredTool.from_function(
    func=_list_transactions,
    name="list_transactions",
    description=(
        "List individual transactions for a customer within a lookback window: "
        "timestamp, amount, direction, channel, counterparty and country. Use "
        "when specific transactions or their sequencing matter."
    ),
    args_schema=TransactionQueryArgs,
)

summarise_transactions = StructuredTool.from_function(
    func=_summarise_transactions,
    name="summarise_transactions",
    description=(
        "Aggregate a customer's transaction activity over a lookback window: "
        "totals, largest amounts, cash deposit totals, channel and country "
        "breakdowns, and ratios against the customer's spending baseline. Use "
        "this first; it is cheaper than listing every transaction."
    ),
    args_schema=TransactionQueryArgs,
)

TRANSACTION_TOOLS = [summarise_transactions, list_transactions]
