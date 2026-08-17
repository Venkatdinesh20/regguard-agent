"""Domain tools, grouped by the specialist that owns them.

Tool ownership is deliberate: the customer agent cannot call the fraud scorer,
and the fraud agent cannot rewrite policy. Least privilege applies to agents the
same way it applies to services.
"""

from __future__ import annotations

from app.tools.customer import CUSTOMER_TOOLS
from app.tools.fraud import FRAUD_TOOLS
from app.tools.policy import POLICY_TOOLS
from app.tools.transactions import TRANSACTION_TOOLS

__all__ = [
    "CUSTOMER_TOOLS",
    "FRAUD_TOOLS",
    "POLICY_TOOLS",
    "TRANSACTION_TOOLS",
]
