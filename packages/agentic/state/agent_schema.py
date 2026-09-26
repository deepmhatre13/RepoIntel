"""Agentic Q&A state for the adaptive RepoIntel agent (Stage 6).

This state carries everything the adaptive planner / tool-selection /
reviewer / reflection loop needs. No hidden chain-of-thought is stored --
only concise reasoning summaries (tool selection rationales) and evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    """A single tool call selected by the planner (kept as reasoning trail)."""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class AgentEvidence(BaseModel):
    """A single evidence item obtained by a tool (with provenance)."""

    tool_name: str
    file_path: Optional[str] = None
    symbol: Optional[str] = None
    content: str = ""
    source: str = "deterministic"
    score: float = 1.0


class AgenticQAState(BaseModel):
    """Robust state for the adaptive repository-intelligence agent."""

    question: str
    repository_id: str
    retrieval_strategy: str = "auto"      # structural|architectural|historical|impact|security|default
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    selected_tools: List[AgentToolCall] = Field(default_factory=list)
    retrieval_results: List[Dict[str, Any]] = Field(default_factory=list)
    graph_context: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[AgentEvidence] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    confidence: str = "MEDIUM"            # HIGH | MEDIUM | LOW
    evidence_coverage: float = 0.0
    review_result: Optional[str] = None   # "accept" | "reflect"
    reflection_count: int = 0
    max_reflection_count: int = 2
    reasoning_summary: str = ""
    final_answer: Optional[str] = None
    degraded: bool = False
    errors: List[str] = Field(default_factory=list)


def infer_strategy(question: str) -> str:
    """Deterministic strategy selection (no LLM required, Stage 5).

    Recognises structural, architectural, historical, impact and security
    question intents purely from the query text. Architectural/structure style
    questions are checked before bare "why" (which could be either historical
    or architectural) so structure-heavy phrasing wins.
    """
    q = (question or "").lower()
    if any(k in q for k in ("security", "vulnerab", "secret", "unsafe", "risk")):
        return "security"
    if any(k in q for k in ("break", "affect", "impact", "happen if", "rename", "what if", "callers")):
        return "impact"
    # Structural/architectural intent is a stronger signal than bare "why".
    if any(k in q for k in ("architect", "component", "module", "structure", "structured",
                            "designed", "organize", "organized", "entrypoint", "package")):
        return "architectural"
    if any(k in q for k in ("function", "method", "class", "defined in", "symbol", "where is")):
        return "structural"
    if any(k in q for k in ("why", "history", "commit", "changed", "when", "evolv")):
        return "historical"
    return "default"