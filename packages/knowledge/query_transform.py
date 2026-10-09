import re
from typing import List

from pydantic import BaseModel, Field


class QueryVariants(BaseModel):
    queries: List[str] = Field(min_length=3, max_length=3)


class HypotheticalDocument(BaseModel):
    document: str


def is_broad_query(query: str) -> bool:
    words = re.findall(r"\b[\w'-]+\b", query)
    lowered = query.casefold()
    return (
        len(words) > 12
        or re.search(r"\band\b", lowered) is not None
        or re.search(r"\boverall\b", lowered) is not None
    )


def transform_query(query: str, llm_client) -> List[str]:
    prompt = f"""
Create exactly three distinct repository-search queries that cover different
facets of the user's compound request. Keep each query concise and concrete.
Return only the structured query list.

Original query:
{query}
"""
    try:
        result = llm_client.complete_structured(
            prompt=prompt,
            schema=QueryVariants,
            system="You reformulate repository retrieval queries without adding unsupported assumptions.",
        )
        variants = []
        seen = set()
        for candidate in result.queries:
            normalized = candidate.strip()
            key = normalized.casefold()
            if normalized and key not in seen:
                variants.append(normalized)
                seen.add(key)
        return variants if len(variants) > 1 else [query]
    except Exception:
        return [query]


def generate_hypothetical_document(query: str, llm_client) -> str:
    prompt = f"""
Write a concise hypothetical repository document excerpt that would answer the
architecture question below. Do not invent specific file paths, symbols,
dependencies, or implementation facts. This text is used only to improve
semantic retrieval; it is not repository evidence.

Question:
{query}
"""
    try:
        result = llm_client.complete_structured(
            prompt=prompt,
            schema=HypotheticalDocument,
            system="Generate a generic hypothetical answer for semantic retrieval only.",
        )
        return result.document.strip() or query
    except Exception:
        return query