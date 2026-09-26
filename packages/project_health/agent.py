"""Project Health LangGraph agent (Stages 17-20).

A genuinely dynamic analyst graph: a planner selects only the analysts the
health question needs, analysts gather evidence, the health interpreter
narrates with Gemini (the only LLM), and a reviewer validates the conclusions.
On insufficient evidence the reviewer routes back to REFLECT to retrieve more,
bounded by ``max_reflection_iterations`` — no infinite loops.

The graph nodes are thin orchestrators: all deterministic analysis lives in
the health engine/services (no business logic duplicated inside nodes).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from packages.project_health.models import HealthAnalysis, HealthEvidence


class HealthAgentState(TypedDict, total=False):
    """TypedDict state for the LangGraph health analyst.

    Using a TypedDict (rather than a bare ``dict``) makes LangGraph treat each
    field as its own channel with a merge reducer, so a node returning only the
    keys it produces never drops unrelated state (e.g. ``analysis``) written by
    another node or the initial input.
    """

    analysis_id: str
    owner: str
    repository: str
    question: str
    selected_analysts: List[str]
    analysis: Optional[Dict[str, Any]]
    gemini_interpretation: Optional[str]
    review_notes: List[str]
    reflection_count: int
    max_reflections: int
    final: bool


ANALYSTS = {
    "maintenance": ["merge_rate", "median_merge_days", "median_resolution_days",
                    "closure_rate", "stale_open_ratio", "stale_issue_ratio"],
    "community": ["hhi", "contributor_count", "active_contributors_30d"],
    "release": ["median_interval_days", "frequency_per_90d"],
    "code": ["churn", "centrality", "hotspots"],
    "dependency": ["dependency_staleness", "dependency_vulnerabilities"],
    "security": ["security_findings"],
    "git": ["commits", "active_days_30d", "rolling_30d"],
    "trend": ["trends", "change_points"],
    "correlation": ["correlations"],
}


def _detect_analyst_from_question(question: str) -> List[str]:
    """Keyword-based analyst selection (the only dynamic routing rule).

    A lightweight planner that maps the natural-language question to the
    minimal set of analysts rather than invoking everything.
    """
    q = question.lower()
    selected: List[str] = []
    topics = {
        "dependency": ["depend", "package", "library", "stale", "vulnerab"],
        "security": ["secur", "vulnerab", "cve", "advis"],
        "release": ["release", "version", "interval", "tag"],
        "community": ["contributor", "community", "bus factor", "concentration"],
        "maintenance": ["maintain", "pull request", "issue", "merge", "response"],
        "code": ["code", "churn", "hotspot", "complex", "centrality", "coupling"],
        "git": ["commit", "activity", "pushed", "branch"],
        "trend": ["trend", "trajectory", "deteriorat", "improving", "declining", "change"],
        "correlation": ["correlat", "related", "move together"],
    }
    for analyst, keywords in topics.items():
        if any(k in q for k in keywords) and analyst not in selected:
            selected.append(analyst)
    if not selected:
        selected = ["maintenance", "community", "release", "git"]
    return selected


def _gather_evidence_for_analysis(analysis: HealthAnalysis,
                                  analysts: List[str]) -> List[HealthEvidence]:
    """Collect structured evidence for the selected analysts."""
    dim_map = {d.name.lower(): d for d in analysis.dimensions}
    evidence: List[HealthEvidence] = []
    for a in analysts:
        d = dim_map.get(a)
        if d and d.evidence:
            evidence.extend(d.evidence[:8])
    return evidence
class ProjectHealthAgent:
    """LangGraph analyst over the Project Health services."""

    def __init__(self, llm_client, max_reflections: int = 2):
        self.llm = llm_client
        self.max_reflections = max_reflections

    # -- node implementations -----------------------------------------------
    def _planner(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("question") or "Why is this project becoming unhealthy?"
        return {"selected_analysts": _detect_analyst_from_question(question)}

    def _gather(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # The analysis object is the result of the deterministic engine.
        return {}

    def _interpreter(self, state: Dict[str, Any]) -> Dict[str, Any]:
        analysis = state["analysis"]
        selected = state.get("selected_analysts") or []
        prompt = self._build_gemini_prompt(analysis, selected)
        if not self.llm.is_available():
            return {"gemini_interpretation": None,
                    "review_notes": ["Gemini unavailable (degraded deterministic mode)."]}
        try:
            text = self.llm.complete(prompt, system=HEALTH_INTERPRET_SYSTEM)
            return {"gemini_interpretation": text}
        except Exception as exc:
            return {"gemini_interpretation": None,
                    "review_notes": [f"Gemini unavailable: {exc}"]}

    def _reviewer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate conclusions against evidence; reject unsupported claims."""
        notes = list(state.get("review_notes") or [])
        analysis = state.get("analysis") or {}
        missing = []
        for d in analysis.get("dimensions") or []:
            if d.get("state") not in (None, "NOT_AVAILABLE") and not d.get("evidence"):
                missing.append(d.get("name"))
        if missing:
            notes.append(f"Missing evidence for dimensions: {', '.join(missing)}")
        return {"review_notes": notes, "final": bool(not missing)}

    def _reflect(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"reflection_count": state.get("reflection_count", 0) + 1}

    def _route(self, state: Dict[str, Any]) -> str:
        if state.get("final"):
            return "END"
        if state.get("reflection_count", 0) >= self.max_reflections:
            return "END"
        return "GATHER"

    # -- prompt construction ------------------------------------------------
    def _build_gemini_prompt(self, analysis: Dict[str, Any],
                             selected: List[str]) -> str:
        dims = []
        for d in (analysis or {}).get("dimensions") or []:
            if d.get("availability") != "NOT_AVAILABLE":
                dims.append(
                    f"- {d.get('name')}: state={d.get('state')}, trend={d.get('trend')}, "
                    f"score={d.get('score')}, confidence={d.get('confidence')}"
                )
        dim_text = "\n".join(dims) or "- no computed dimensions (insufficient data)"
        evidence_lines = []
        for d in (analysis or {}).get("dimensions") or []:
            for e in (d.get("evidence") or [])[:6]:
                evidence_lines.append(
                    f"- {e.get('metric')}: {e.get('value')} "
                    f"(prev {e.get('previous_value')}, delta {e.get('delta')})"
                )
        evidence_text = "\n".join(evidence_lines) or "- no quantitative evidence"
        changes = [c.get("metric") for c in ((analysis or {}).get("timeline") or [])[:6]]
        changes = [m for m in changes if isinstance(m, str)]
        return (
            f"Repository: {analysis.get('full_name')}\n"
            f"Project Health dimensions:\n{dim_text}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Significant changes:\n{', '.join(changes) if changes else '- none'}\n\n"
            f"Analysts active: {', '.join(selected)}\n\n"
            "Provide a concise, evidence-grounded interpretation. Do NOT add "
            "repository facts, scores, contributor counts, release statistics, "
            "dependency counts, trends, security findings or forecasts that are "
            "not present in the evidence above."
        )

    # -- build & run the graph ----------------------------------------------
    def build_graph(self):
        g = StateGraph(HealthAgentState)
        g.add_node("PLANNER", self._planner)
        g.add_node("GATHER", self._gather)
        g.add_node("INTERPRETER", self._interpreter)
        g.add_node("REVIEWER", self._reviewer)
        g.add_node("REFLECT", self._reflect)
        g.set_entry_point("PLANNER")
        g.add_edge("PLANNER", "GATHER")
        g.add_edge("GATHER", "INTERPRETER")
        g.add_edge("INTERPRETER", "REVIEWER")
        g.add_conditional_edges(
            "REVIEWER", self._route, {"GATHER": "GATHER", "END": END}
        )
        return g.compile()

    async def analyze(self, analysis: HealthAnalysis,
                      question: str) -> Dict[str, Any]:
        """Run the agent over a completed deterministic analysis."""
        graph = self.build_graph()
        state: Dict[str, Any] = {
            "analysis": analysis.model_dump(mode="json"),
            "question": question,
            "selected_analysts": [],
            "gemini_interpretation": None,
            "review_notes": [],
            "reflection_count": 0,
            "final": False,
        }
        return dict(await graph.ainvoke(state))


HEALTH_INTERPRET_SYSTEM = (
    "You are the Project Health Intelligence analyst for RepoIntel. You "
    "interpret deterministic statistical findings; you never invent metrics, "
    "scores, counts, trends, repository facts, or forecasts. Ground every "
    "statement in the supplied evidence. If evidence is insufficient, say so."
)