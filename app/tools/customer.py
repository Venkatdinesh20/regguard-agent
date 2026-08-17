"""Tools owned by the customer specialist."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from app.core.logging import get_logger
from app.schemas.tool_args import CustomerLookupArgs
from app.tools.repository import get_customer, known_customer_ids

logger = get_logger(__name__)


def _get_customer_profile(customer_id: str) -> dict[str, Any]:
    """Retrieve the KYC profile for a customer.

    Returns identity, residence, account tenure, spending baseline, KYC
    verification status, politically-exposed-person flag and sanctions status.
    """
    profile = get_customer(customer_id)
    logger.info("customer.profile.retrieved", extra={"customer_id": customer_id})
    payload = profile.model_dump()
    payload["account_age_years"] = round(profile.account_age_days / 365, 2)
    payload["is_new_account"] = profile.account_age_days < 180
    return payload


def _list_known_customers() -> dict[str, Any]:
    """List the customer identifiers available in the source system.

    Use this when a query names a customer you cannot find, to check whether
    the identifier is simply wrong before concluding there is no data.
    """
    ids = known_customer_ids()
    logger.info("customer.directory.listed", extra={"count": len(ids)})
    return {"customer_ids": ids, "count": len(ids)}


get_customer_profile = StructuredTool.from_function(
    func=_get_customer_profile,
    name="get_customer_profile",
    description=(
        "Retrieve the KYC profile for one customer: residence country, account "
        "age in days, average monthly spend baseline, KYC verification status, "
        "PEP flag and sanctions screening result."
    ),
    args_schema=CustomerLookupArgs,
)

list_known_customers = StructuredTool.from_function(
    func=_list_known_customers,
    name="list_known_customers",
    description=(
        "List every customer identifier that exists in the source system. Use "
        "when a customer identifier appears to be missing or mistyped."
    ),
)

CUSTOMER_TOOLS = [get_customer_profile, list_known_customers]
