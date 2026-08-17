"""Deterministic risk scoring, exposed to the fraud specialist as a tool.

Design decision worth stating explicitly: **the LLM does not invent the risk
score.** Scoring is a transparent, versioned, deterministic rule engine. The
agent decides *when* to score, reads the triggered rules, and explains them.

That split matters in a regulated setting. A model-generated number cannot be
reproduced, back-tested, or defended to an auditor; a rule engine can, and the
same evidence always yields the same score. The rule table below is the natural
place a bank would later swap in a governed ML model (for example a gradient
boosted classifier) — behind the same interface, with the same explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from langchain_core.tools import StructuredTool

from app.core.logging import get_logger
from app.schemas.investigation import RiskLevel, Transaction
from app.schemas.tool_args import TransactionQueryArgs
from app.tools.repository import (
    get_customer,
    get_transactions,
    high_risk_jurisdictions,
    parse_timestamp,
)

logger = get_logger(__name__)

RULES_VERSION = "regguard-rules-1.0.0"

HIGH_RISK_SCORE = 60
MEDIUM_RISK_SCORE = 30

STRUCTURING_FLOOR = 7_500.0
REPORTING_THRESHOLD = 10_000.0
STRUCTURING_MIN_COUNT = 3
STRUCTURING_WINDOW = timedelta(hours=72)
AGGREGATION_WINDOW = timedelta(hours=24)
PASS_THROUGH_WINDOW = timedelta(hours=72)
PASS_THROUGH_RATIO = 0.7
BASELINE_ANOMALY_MULTIPLE = 10
DISPERSAL_MULTIPLE = 5
NEW_ACCOUNT_DAYS = 180


@dataclass
class RuleHit:
    """One triggered rule, with the evidence that triggered it."""

    rule_id: str
    description: str
    weight: int
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "weight": self.weight,
            "evidence": self.evidence,
        }


def _windowed_groups(
    items: list[Transaction], window: timedelta, minimum: int
) -> list[list[Transaction]]:
    """Every maximal run of ``items`` that fits inside ``window``."""
    groups: list[list[Transaction]] = []
    for index, anchor in enumerate(items):
        anchor_time = parse_timestamp(anchor.timestamp)
        group = [
            candidate
            for candidate in items[index:]
            if parse_timestamp(candidate.timestamp) - anchor_time <= window
        ]
        if len(group) >= minimum:
            groups.append(group)
    return groups


def score_customer_risk(customer_id: str, lookback_days: int = 30) -> dict[str, Any]:
    """Score a customer's financial-crime risk from evidence, deterministically."""
    profile = get_customer(customer_id)
    transactions = get_transactions(customer_id, lookback_days)
    risky_countries = high_risk_jurisdictions()

    hits: list[RuleHit] = []

    credits = [item for item in transactions if item.direction == "CREDIT"]
    debits = [item for item in transactions if item.direction == "DEBIT"]
    cash_credits = [item for item in credits if item.channel in {"CASH", "ATM"}]
    baseline = profile.average_monthly_spend or 0.0

    # R01 — structuring: repeated deposits parked just under the threshold.
    just_under = [
        item
        for item in cash_credits
        if STRUCTURING_FLOOR <= item.amount < REPORTING_THRESHOLD
    ]
    groups = _windowed_groups(just_under, STRUCTURING_WINDOW, STRUCTURING_MIN_COUNT)
    if groups:
        biggest = max(groups, key=len)
        hits.append(
            RuleHit(
                "R01_STRUCTURING",
                f"{len(biggest)} cash deposits between "
                f"{STRUCTURING_FLOOR:,.0f} and {REPORTING_THRESHOLD:,.0f} within "
                f"{int(STRUCTURING_WINDOW.total_seconds() // 3600)}h — consistent "
                "with deliberate avoidance of the reporting threshold.",
                40,
                [
                    f"{item.transaction_id} {item.timestamp} "
                    f"{item.amount:,.2f} {item.currency} via {item.channel} "
                    f"({item.counterparty})"
                    for item in biggest
                ],
            )
        )

    # R02 — aggregation: sub-threshold deposits that sum above it in one day.
    daily_groups = _windowed_groups(cash_credits, AGGREGATION_WINDOW, 2)
    for group in daily_groups:
        total = sum(item.amount for item in group)
        if total >= REPORTING_THRESHOLD and all(
            item.amount < REPORTING_THRESHOLD for item in group
        ):
            hits.append(
                RuleHit(
                    "R02_THRESHOLD_AGGREGATION",
                    f"Cash deposits totalling {total:,.2f} within 24h, each "
                    "individually below the reporting threshold.",
                    15,
                    [f"{item.transaction_id} {item.amount:,.2f}" for item in group],
                )
            )
            break

    # R03 — pass-through / layering out of a high-risk jurisdiction.
    for credit in credits:
        if credit.country not in risky_countries or credit.amount < REPORTING_THRESHOLD:
            continue
        credit_time = parse_timestamp(credit.timestamp)
        onward = [
            debit
            for debit in debits
            if 0
            <= (parse_timestamp(debit.timestamp) - credit_time).total_seconds()
            <= PASS_THROUGH_WINDOW.total_seconds()
            and debit.amount >= credit.amount * PASS_THROUGH_RATIO
        ]
        if onward:
            hits.append(
                RuleHit(
                    "R03_PASS_THROUGH",
                    f"Funds of {credit.amount:,.2f} received from "
                    f"{credit.country} left the account within 72h at a similar "
                    "amount — layering indicator.",
                    30,
                    [
                        f"IN {credit.transaction_id} {credit.amount:,.2f} "
                        f"from {credit.counterparty} ({credit.country})",
                        *[
                            f"OUT {item.transaction_id} {item.amount:,.2f} "
                            f"to {item.counterparty} ({item.country})"
                            for item in onward
                        ],
                    ],
                )
            )
            break

    # R04 — exposure to high-risk jurisdictions at all.
    touched = sorted(
        {item.country for item in transactions if item.country in risky_countries}
    )
    if touched:
        hits.append(
            RuleHit(
                "R04_HIGH_RISK_GEOGRAPHY",
                "Activity involves jurisdictions on the internal high-risk list: "
                + ", ".join(touched),
                min(10 * len(touched), 20),
                touched,
            )
        )

    # R05 — a credit far outside the account's established baseline.
    if baseline and credits:
        largest = max(credits, key=lambda item: item.amount)
        if largest.amount >= baseline * BASELINE_ANOMALY_MULTIPLE:
            hits.append(
                RuleHit(
                    "R05_BASELINE_ANOMALY",
                    f"Largest credit {largest.amount:,.2f} is "
                    f"{largest.amount / baseline:.1f}x the account's monthly "
                    f"baseline of {baseline:,.2f}.",
                    25,
                    [
                        f"{largest.transaction_id} {largest.amount:,.2f} "
                        f"from {largest.counterparty} ({largest.country})"
                    ],
                )
            )

    # R06 — rapid dispersal to multiple third parties (money-mule shape).
    if baseline:
        for credit in credits:
            if credit.amount < baseline * DISPERSAL_MULTIPLE:
                continue
            credit_time = parse_timestamp(credit.timestamp)
            onward = [
                debit
                for debit in debits
                if 0
                <= (parse_timestamp(debit.timestamp) - credit_time).total_seconds()
                <= PASS_THROUGH_WINDOW.total_seconds()
            ]
            payees = {item.counterparty for item in onward}
            dispersed = sum(item.amount for item in onward)
            if len(payees) >= 2 and dispersed >= credit.amount * PASS_THROUGH_RATIO:
                hits.append(
                    RuleHit(
                        "R06_RAPID_DISPERSAL",
                        f"A credit of {credit.amount:,.2f} was dispersed to "
                        f"{len(payees)} distinct third parties within 72h "
                        f"({dispersed:,.2f} total).",
                        20,
                        [
                            f"{item.transaction_id} {item.amount:,.2f} "
                            f"to {item.counterparty}"
                            for item in onward
                        ],
                    )
                )
                break

    # R07 — young account moving large wires.
    if profile.account_age_days < NEW_ACCOUNT_DAYS:
        large_wires = [
            item
            for item in transactions
            if item.channel == "WIRE" and item.amount > REPORTING_THRESHOLD
        ]
        if large_wires:
            hits.append(
                RuleHit(
                    "R07_NEW_ACCOUNT_LARGE_WIRES",
                    f"Account is {profile.account_age_days} days old and has "
                    f"{len(large_wires)} wires above "
                    f"{REPORTING_THRESHOLD:,.0f}.",
                    15,
                    [
                        f"{item.transaction_id} {item.amount:,.2f} {item.counterparty}"
                        for item in large_wires
                    ],
                )
            )

    # R08/R09/R10 — static customer risk attributes.
    if profile.sanctions_hit:
        hits.append(
            RuleHit(
                "R08_SANCTIONS_HIT",
                "Customer matches a sanctions list entry.",
                50,
                [profile.customer_id],
            )
        )
    if profile.pep_flag:
        hits.append(
            RuleHit(
                "R09_PEP",
                "Customer is a politically exposed person; enhanced due "
                "diligence applies.",
                10,
                [profile.occupation or "PEP flag set"],
            )
        )
    if profile.kyc_status != "VERIFIED" and transactions:
        hits.append(
            RuleHit(
                "R10_KYC_NOT_VERIFIED",
                f"Account is transacting while KYC status is {profile.kyc_status}.",
                10,
                [f"kyc_status={profile.kyc_status}"],
            )
        )

    raw_score = sum(hit.weight for hit in hits)
    score = min(raw_score, 100)
    if score >= HIGH_RISK_SCORE:
        level = RiskLevel.HIGH
    elif score >= MEDIUM_RISK_SCORE:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    logger.info(
        "fraud.scored",
        extra={
            "customer_id": customer_id,
            "risk_score": score,
            "risk_level": level.value,
            "rules_triggered": [hit.rule_id for hit in hits],
        },
    )

    return {
        "customer_id": customer_id,
        "rules_version": RULES_VERSION,
        "window_days": lookback_days,
        "transactions_examined": len(transactions),
        "risk_score": score,
        "risk_level": level.value,
        "thresholds": {"HIGH": HIGH_RISK_SCORE, "MEDIUM": MEDIUM_RISK_SCORE},
        "triggered_rules": [hit.to_dict() for hit in hits],
        "clean": not hits,
    }


def _rule_catalogue() -> dict[str, Any]:
    """Describe every rule the scoring engine can trigger."""
    return {
        "rules_version": RULES_VERSION,
        "scoring": (
            "Weights of triggered rules are summed and capped at 100. "
            f"score >= {HIGH_RISK_SCORE} is HIGH, "
            f">= {MEDIUM_RISK_SCORE} is MEDIUM, otherwise LOW."
        ),
        "rules": [
            {"rule_id": "R01_STRUCTURING", "weight": 40},
            {"rule_id": "R02_THRESHOLD_AGGREGATION", "weight": 15},
            {"rule_id": "R03_PASS_THROUGH", "weight": 30},
            {"rule_id": "R04_HIGH_RISK_GEOGRAPHY", "weight": "10 per country, max 20"},
            {"rule_id": "R05_BASELINE_ANOMALY", "weight": 25},
            {"rule_id": "R06_RAPID_DISPERSAL", "weight": 20},
            {"rule_id": "R07_NEW_ACCOUNT_LARGE_WIRES", "weight": 15},
            {"rule_id": "R08_SANCTIONS_HIT", "weight": 50},
            {"rule_id": "R09_PEP", "weight": 10},
            {"rule_id": "R10_KYC_NOT_VERIFIED", "weight": 10},
        ],
    }


score_fraud_risk = StructuredTool.from_function(
    func=score_customer_risk,
    name="score_fraud_risk",
    description=(
        "Run the deterministic financial-crime rule engine against a customer's "
        "activity. Returns a 0-100 score, a risk level, and every triggered "
        "rule with the transactions that triggered it. The score is computed, "
        "not estimated — report it verbatim and explain the triggered rules."
    ),
    args_schema=TransactionQueryArgs,
)

describe_risk_rules = StructuredTool.from_function(
    func=_rule_catalogue,
    name="describe_risk_rules",
    description=(
        "List the rule catalogue and scoring thresholds used by the risk "
        "engine. Use when you need to explain how a score was derived."
    ),
)

FRAUD_TOOLS = [score_fraud_risk, describe_risk_rules]
