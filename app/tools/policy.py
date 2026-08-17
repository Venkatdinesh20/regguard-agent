"""Tools owned by the policy specialist — retrieval over the policy corpus.

This is deliberately a small, dependency-free lexical retriever (term overlap
across tags, title and body, with tag matches weighted higher). It is the seam
where a production deployment plugs in a real vector store and embeddings; the
tool contract the agent sees does not change.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import StructuredTool

from app.core.logging import get_logger
from app.schemas.tool_args import PolicySearchArgs
from app.tools.repository import all_policies

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "we",
        "what",
        "when",
        "which",
        "with",
        "do",
        "does",
    }
)

TAG_WEIGHT = 3
TITLE_WEIGHT = 2
BODY_WEIGHT = 1


def _tokenise(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def _score(policy: dict[str, Any], terms: set[str]) -> int:
    tags = _tokenise(" ".join(policy.get("tags", [])))
    title = _tokenise(policy.get("title", ""))
    body = _tokenise(policy.get("text", ""))
    return (
        TAG_WEIGHT * len(terms & tags)
        + TITLE_WEIGHT * len(terms & title)
        + BODY_WEIGHT * len(terms & body)
    )


def search_policies(topic: str, top_k: int = 3) -> dict[str, Any]:
    """Retrieve the policy documents most relevant to a topic."""
    terms = _tokenise(topic)
    scored = [
        (score, policy) for policy in all_policies() if (score := _score(policy, terms))
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]["policy_id"]))
    selected = scored[:top_k]

    logger.info(
        "policy.searched",
        extra={"topic": topic, "matches": len(scored), "returned": len(selected)},
    )

    return {
        "topic": topic,
        "match_count": len(scored),
        "results": [
            {
                "policy_id": policy["policy_id"],
                "title": policy["title"],
                "jurisdiction": policy["jurisdiction"],
                "relevance": score,
                "text": policy["text"],
            }
            for score, policy in selected
        ],
        "note": (
            "No policy matched the topic; try different terms."
            if not selected
            else "Cite policy_id values in the report."
        ),
    }


def _list_policies() -> dict[str, Any]:
    """List every policy document available, without their full text."""
    return {
        "policies": [
            {
                "policy_id": policy["policy_id"],
                "title": policy["title"],
                "jurisdiction": policy["jurisdiction"],
                "tags": policy["tags"],
            }
            for policy in all_policies()
        ]
    }


search_policy = StructuredTool.from_function(
    func=search_policies,
    name="search_policy",
    description=(
        "Search internal financial-crime policy and regulatory guidance by "
        "topic, for example 'structuring cash deposits', 'SAR filing deadline' "
        "or 'enhanced due diligence PEP'. Returns policy identifiers and text "
        "that must be cited in the report."
    ),
    args_schema=PolicySearchArgs,
)

list_policies = StructuredTool.from_function(
    func=_list_policies,
    name="list_policies",
    description=(
        "List the identifiers, titles and tags of every available policy "
        "document. Use to orient before searching."
    ),
)

POLICY_TOOLS = [search_policy, list_policies]
