"""Relevance gating for repository Q&A (Phase 3).

A repository question must not be considered answered merely because top-k
documents were retrieved. This module implements a relevance/evidence gate that
classifies the relationship between a user's query and the evidence actually
retrieved from the repository index, distinguishing:

    HIGH_RELEVANCE        direct file/symbol match backed by repository evidence
    MEDIUM_RELEVANCE      meaningful term overlap / broad semantic support
    LOW_RELEVANCE         weak, incidental, or generic overlap
    NO_RELEVANT_EVIDENCE  the repository contains no evidence for the question

The gate is technology-agnostic: it never hardcodes names such as "PostgreSQL",
"Kubernetes", "Redis", or "Kafka". It instead measures whether the *distinctive*
terms of the question appear anywhere in the actual evidence (file path, symbol,
or content). A query whose distinctive terms have zero presence in the reported
evidence must be reported as unsupported rather than answered from unrelated
top-k documents.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Set

from packages.knowledge.vector_store import RetrievedEvidenceItem

# Generic English stopwords + repository-QA chatter words that carry no
# information about what the user is actually asking about.
_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "this", "that", "these", "those", "it", "its", "in", "on", "at", "for",
    "of", "to", "from", "with", "without", "about", "into", "onto", "by",
    "and", "or", "but", "not", "no", "be", "been", "being", "have", "has",
    "had", "having", "can", "could", "will", "would", "should", "shall",
    "may", "might", "must", "any", "all", "some", "each", "every", "both",
    "few", "more", "most", "other", "such", "than", "too", "very", "just",
    "repository", "repositories", "codebase", "project", "code", "source",
    "use", "uses", "using", "used", "provide", "available", "there",
    "make", "makes", "made", "give", "gives", "show", "shows", "tell",
    "list", "lists", "name", "names", "define", "defines", "defined",
    "find", "finds", "implement", "implements", "implemented",
    "describe", "describes", "explain", "explains", "contain", "contains",
    "containing", "within", "across", "current", "their", "them", "they",
}


class RelevanceLevel(str, Enum):
    HIGH_RELEVANCE = "HIGH_RELEVANCE"
    MEDIUM_RELEVANCE = "MEDIUM_RELEVANCE"
    LOW_RELEVANCE = "LOW_RELEVANCE"
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"


def tokenize(query: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop empty tokens."""
    return [t for t in re.split(r"[^0-9a-z_]+", query.lower()) if t]


def significant_terms(query: str) -> List[str]:
    """Return the distinctive terms of a question (stopwords removed)."""
    return [t for t in tokenize(query) if t not in _STOPWORDS and len(t) > 1]


def _query_files(query: str) -> List[str]:
    """Return file-like tokens named in the query (e.g. ``calculator.py``)."""
    return re.findall(r"([\w./\\-]+\.py\b)", query.lower())


def _query_symbols(query: str) -> List[str]:
    """Return likely symbol identifiers named in the query (dotted or PascalCase)."""
    matches = re.findall(
        r"\b[A-Za-z_][A-Za-z0-9_]*[.][A-Za-z_][A-Za-z0-9_]*\b"
        r"|\b[A-Z][A-Za-z0-9_]*\b",
        query,
    )
    return matches


def _extract_evidence_files(items: List[RetrievedEvidenceItem]) -> Set[str]:
    """Extract the set of file paths present in retrieved evidence."""
    return {item.file_path for item in items if item.file_path}


def _extract_evidence_symbols(items: List[RetrievedEvidenceItem]) -> Set[str]:
    """Extract the set of symbol names present in retrieved evidence."""
    return {item.symbol_name for item in items if item.symbol_name}


def _extract_evidence_terms(items: List[RetrievedEvidenceItem]) -> Set[str]:
    """Extract all meaningful terms from evidence file paths, symbols, and content."""
    terms: Set[str] = set()
    for item in items:
        if item.file_path:
            terms.update(tokenize(item.file_path))
        if item.symbol_name:
            # Split dotted symbols: "Calculator.add" -> {"Calculator", "add"}
            for part in re.split(r"[._]", item.symbol_name):
                terms.update(tokenize(part))
        if item.content:
            terms.update(tokenize(item.content))
    return terms - _STOPWORDS


class RelevanceDecision:
    """A structured relevance decision produced by :func:`evaluate_relevance`."""

    __slots__ = ("level", "reasoning", "direct_files", "direct_symbols",
                 "evidence_terms", "query_terms")

    def __init__(self, level: RelevanceLevel, reasoning: str,
                 direct_files: Optional[List[str]] = None,
                 direct_symbols: Optional[List[str]] = None,
                 evidence_terms: Optional[Set[str]] = None,
                 query_terms: Optional[List[str]] = None):
        self.level = level
        self.reasoning = reasoning
        self.direct_files = direct_files or []
        self.direct_symbols = direct_symbols or []
        self.evidence_terms = evidence_terms or set()
        self.query_terms = query_terms or []

    def to_dict(self) -> Dict[str, object]:
        return {
            "level": self.level.value,
            "reasoning": self.reasoning,
            "direct_files": self.direct_files,
            "direct_symbols": self.direct_symbols,
            "query_terms": self.query_terms,
            "evidence_terms": sorted(self.evidence_terms),
        }


def evaluate_relevance(query, retrieved_items, direct_file_lookup=False, repo_file_paths=None):
    """Evaluate how relevant the retrieved evidence is to the query."""
    items = retrieved_items or []
    terms = significant_terms(query)
    query_file_names = _query_files(query)
    query_symbol_names = _query_symbols(query)

    if not items:
        return RelevanceDecision(
            level=RelevanceLevel.NO_RELEVANT_EVIDENCE,
            reasoning="No evidence was retrieved for this query from the indexed repository.",
            evidence_terms=set(),
            query_terms=terms,
        )

    evidence_files = _extract_evidence_files(items)
    evidence_symbols = _extract_evidence_symbols(items)
    evidence_terms = _extract_evidence_terms(items)

    matched_files = []
    for qf in query_file_names:
        for ef in evidence_files:
            if ef == qf or ef.endswith("/" + qf) or ef.endswith("\\" + qf):
                matched_files.append(ef)
        if repo_file_paths:
            for rfp in repo_file_paths:
                if rfp == qf or rfp.endswith("/" + qf) or rfp.endswith("\\" + qf):
                    matched_files.append(rfp)

    matched_symbols = []
    for qs in query_symbol_names:
        for es in evidence_symbols:
            if es == qs:
                matched_symbols.append(es)
                break

    if matched_files or matched_symbols:
        parts = []
        if matched_files:
            parts.append("file")
        if matched_symbols:
            parts.append("symbol")
        names = matched_files + matched_symbols
        return RelevanceDecision(
            level=RelevanceLevel.HIGH_RELEVANCE,
            reasoning=(
                f"Direct {' and '.join(parts)} match(es) in repository evidence ({', '.join(names)})."
            ),
            direct_files=matched_files,
            direct_symbols=matched_symbols,
            evidence_terms=evidence_terms,
            query_terms=terms,
        )

    if not terms:
        return RelevanceDecision(
            level=RelevanceLevel.LOW_RELEVANCE,
            reasoning="Query contains no distinctive terms; cannot determine relevance.",
            evidence_terms=evidence_terms,
            query_terms=terms,
        )

    evidence_lower = " ".join(
        (item.file_path or "") + " " + (item.symbol_name or "") + " " + (item.content or "")
        for item in items
    ).lower()

    present_terms = {t for t in terms if t in evidence_lower}
    overlap_ratio = len(present_terms) / len(terms) if terms else 0.0

    if overlap_ratio == 0.0:
        return RelevanceDecision(
            level=RelevanceLevel.NO_RELEVANT_EVIDENCE,
            reasoning=(
                "The distinctive terms of the question do not appear in any "
                "retrieved evidence (file path, symbol, or content)."
            ),
            evidence_terms=evidence_terms,
            query_terms=terms,
        )

    if overlap_ratio >= 0.6:
        return RelevanceDecision(
            level=RelevanceLevel.MEDIUM_RELEVANCE,
            reasoning="Most distinctive question terms appear in the repository evidence.",
            evidence_terms=evidence_terms,
            query_terms=terms,
        )

    return RelevanceDecision(
        level=RelevanceLevel.LOW_RELEVANCE,
        reasoning=(
            "Only a weak or incidental overlap exists between the question and "
            "the retrieved evidence."
        ),
        evidence_terms=evidence_terms,
        query_terms=terms,
    )


from dataclasses import dataclass


@dataclass
class ConfidenceBreakdown:
    """Explainable confidence decomposition for a repository-intelligence answer.

    Each dimension is independently measurable and has a clear meaning.
    The overall ``label`` is derived from the combination — never from an
    opaque scalar with arbitrary weights.

    Dimensions
    ----------
    retrieval_confidence : float
        How well the retrieved evidence matches the query.
        Derived from RelevanceLevel: HIGH_RELEVANCE→1.0, MEDIUM→0.6, LOW→0.3, NONE→0.0.

    evidence_coverage : float
        Fraction of query's significant terms covered by evidence (0-1).
        Populated from the relevance gate's overlap_ratio calculation.

    source_reliability : float
        How authoritative the evidence sources are.
        AST/parsed symbols = 1.0; hybrid retrieval = 0.85; keyword-only = 0.7.

    freshness : float|None
        Evidence timeliness, measured from ingested Git history
        (exp(-age_days/180) of each cited file's last commit). ``None`` means
        freshness was not measured (no history data) — never reported as a
        perfect score.

    freshness_status : str
        "MEASURED" | "NO_HISTORY" | "NOT_MEASURED" — distinguishes a real
        measurement from missing data instead of silently substituting 1.0.

    conflicts : int
        Number of contradictory-evidence pairs detected among the retrieved
        items (same symbol asserted in different files). Any conflict caps the
        overall label at MEDIUM.

    label : str
        Human-readable summary: "HIGH" | "MEDIUM" | "LOW".

    reasoning : str
        One sentence explaining the label from the dimensional evidence.
    """

    retrieval_confidence: float
    evidence_coverage: float
    source_reliability: float
    freshness: Optional[float]
    label: str
    reasoning: str
    evidence_count: int = 0
    direct_match: bool = False
    freshness_status: str = "NOT_MEASURED"
    conflicts: int = 0

    def to_dict(self):
        return {
            "label": self.label,
            "retrieval_confidence": round(self.retrieval_confidence, 3),
            "evidence_coverage": round(self.evidence_coverage, 3),
            "source_reliability": round(self.source_reliability, 3),
            "freshness": round(self.freshness, 3) if self.freshness is not None else None,
            "freshness_status": self.freshness_status,
            "evidence_count": self.evidence_count,
            "direct_match": self.direct_match,
            "conflicts": self.conflicts,
            "reasoning": self.reasoning,
        }


def _evidence_conflicts(items: List[RetrievedEvidenceItem]) -> int:
    """Count contradictory evidence pairs among retrieved items.

    Two retrieved items conflict when they carry the same symbol name but
    resolve to different file paths (a duplicated definition) — a measurable,
    deterministic signal that the evidence set is internally inconsistent.
    """
    by_symbol: Dict[str, Set[str]] = {}
    for item in items or []:
        if item.symbol_name and item.file_path:
            by_symbol.setdefault(item.symbol_name, set()).add(item.file_path)
    return sum(1 for files in by_symbol.values() if len(files) > 1)


def _source_reliability(items) -> float:
    """Estimate source reliability from retrieval method metadata."""
    if not items:
        return 0.0
    scores = []
    for item in items:
        methods = getattr(item, "retrieval_method", []) or []
        src = getattr(item, "retrieval_source", "keyword") or "keyword"
        if "vector" in methods and "keyword" in methods:
            scores.append(0.85)   # hybrid = strong
        elif "vector" in methods:
            scores.append(0.80)
        elif src == "hybrid":
            scores.append(0.85)
        else:
            scores.append(0.70)   # keyword-only = adequate
        # Symbols from AST parse are maximally reliable
        if getattr(item, "symbol_name", None):
            scores[-1] = max(scores[-1], 0.90)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def confidence_breakdown(
    decision: RelevanceDecision,
    items,
    evidence_count: int,
    llm_available: bool,
    direct_file_lookup: bool = False,
    freshness_result: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ConfidenceBreakdown:
    """Compute a structured, explainable confidence breakdown.

    This is the authoritative confidence function. ``confidence_from_relevance``
    below delegates to it for backward compatibility.
    ``freshness_result`` is the optional output of
    ``facade.get_evidence_freshness(...)``; when supplied, the freshness
    dimension is a real Git-history measurement (MEASURED / NO_HISTORY per
    file) instead of an assumed perfect score.
    """
    direct_match = bool(decision.direct_files or decision.direct_symbols
                        or direct_file_lookup)

    # 1. Retrieval confidence from relevance level
    if direct_match or decision.level is RelevanceLevel.HIGH_RELEVANCE:
        ret_conf = 1.0
    elif decision.level is RelevanceLevel.MEDIUM_RELEVANCE:
        ret_conf = 0.6
    elif decision.level is RelevanceLevel.LOW_RELEVANCE:
        ret_conf = 0.3
    else:  # NO_RELEVANT_EVIDENCE
        ret_conf = 0.0

    # 2. Evidence coverage: ratio of query terms found in evidence
    ev_terms = decision.evidence_terms or set()
    q_terms = decision.query_terms or []
    if q_terms:
        ev_cov = min(1.0, len({t for t in q_terms if t in ev_terms}) / len(q_terms))
    else:
        ev_cov = 0.5 if evidence_count > 0 else 0.0

    # 3. Source reliability from retrieval method metadata
    src_rel = _source_reliability(items or [])
    if direct_match:
        src_rel = max(src_rel, 1.0)  # direct AST match is maximally reliable

    # 4. Freshness: measured from Git history when available (Phase G).
    freshness_status = "NOT_MEASURED"
    if freshness_result:
        measured = [v.get("freshness") for v in freshness_result.values()
                    if isinstance(v, dict) and v.get("status") == "MEASURED"]
        if measured:
            freshness = min(measured)  # oldest cited file governs the answer
            freshness_status = "MEASURED"
        else:
            freshness = None
            freshness_status = "NO_HISTORY"
    else:
        freshness = None

    # 5. Conflicting evidence among retrieved items (Phase G).
    conflicts = _evidence_conflicts(items or [])

    # Derive label from retrieval confidence + coverage.
    # Any detected conflict caps the answer at MEDIUM: the evidence set is
    # internally contradictory, so a HIGH answer would be dishonest.
    if conflicts:
        label = "MEDIUM" if ret_conf >= 0.6 and evidence_count >= 2 else "LOW"
        reason = (
            f"{conflicts} conflicting evidence pair(s) detected "
            "(same symbol resolved in different files); confidence capped."
        )
    elif direct_match or (ret_conf >= 1.0 and evidence_count >= 1):
        label = "HIGH"
        reason = (
            f"Direct {'file' if decision.direct_files else 'symbol'} match "
            f"in repository evidence ({', '.join((decision.direct_files + decision.direct_symbols)[:3])})."
            if (decision.direct_files or decision.direct_symbols)
            else "Direct file lookup confirmed by indexed repository."
        )
    elif ret_conf >= 0.6 and evidence_count >= 2 and ev_cov >= 0.4:
        label = "MEDIUM"
        reason = (
            f"Medium relevance: {evidence_count} evidence items, "
            f"{round(ev_cov * 100):.0f}% query-term coverage."
        )
    else:
        label = "LOW"
        if decision.level is RelevanceLevel.NO_RELEVANT_EVIDENCE:
            reason = "No evidence found in the repository for this query."
        elif ret_conf < 0.3:
            reason = "Evidence has weak overlap with query terms."
        else:
            reason = "Insufficient evidence depth to answer with confidence."

    return ConfidenceBreakdown(
        retrieval_confidence=ret_conf,
        evidence_coverage=ev_cov,
        source_reliability=src_rel,
        freshness=freshness,
        label=label,
        reasoning=reason,
        evidence_count=evidence_count,
        direct_match=direct_match,
        freshness_status=freshness_status,
        conflicts=conflicts,
    )


def confidence_from_relevance(
    decision: RelevanceDecision,
    evidence_count: int,
    llm_available: bool,
    direct_file_lookup: bool = False,
    freshness_result: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Map a relevance decision + evidence strength to a confidence label.

    Backward-compatible wrapper around :func:`confidence_breakdown`.
    Returns "HIGH" | "MEDIUM" | "LOW".
    """
    bd = confidence_breakdown(
        decision, [], evidence_count, llm_available, direct_file_lookup,
        freshness_result=freshness_result,
    )
    return bd.label

