"""Adaptive LangGraph agent for RepoIntel (Stage 6).

The planner selects a deterministic retrieval strategy + concrete tool plan;
a tool-execution node dynamically invokes the chosen tools through the shared
:class:`RepoIntelTools` layer; evidence is assembled into a bounded context; a
synthesis node produces an answer (Gemini when available, deterministic
otherwise); and a reviewer decides accept-or-reflect with a strict budget.

The agent never exposes hidden chain-of-thought -- only concise reasoning.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from packages.agentic.state.agent_schema import (
    AgenticQAState, AgentEvidence, AgentToolCall, infer_strategy,
)
from packages.agentic.tools import RepoIntelTools, ToolError
from packages.agentic.llm_client import LLMClient
from packages.shared.tracing import span

_STRATEGY_PLANS: Dict[str, List[Dict[str, str]]] = {
    "structural": [
        {"tool": "search_repository", "arg": "query", "why": "locate relevant code"},
        {"tool": "get_file", "arg": "file", "why": "inspect target file symbols"},
        {"tool": "get_references", "arg": "symbol", "why": "cross-file references"},
    ],
    "architectural": [
        {"tool": "get_architecture", "arg": "", "why": "high-level structure"},
        {"tool": "search_repository", "arg": "query", "why": "find key modules"},
        {"tool": "get_dependencies", "arg": "symbol", "why": "dependency structure"},
    ],
    "historical": [
        {"tool": "get_git_history", "arg": "", "why": "recent commits"},
        {"tool": "get_symbol_history", "arg": "symbol", "why": "symbol change history"},
        {"tool": "search_repository", "arg": "query", "why": "find relevant symbols"},
    ],
    "impact": [
        {"tool": "search_symbol", "arg": "symbol", "why": "locate the target"},
        {"tool": "get_callers", "arg": "symbol", "why": "direct callers"},
        {"tool": "get_dependents", "arg": "symbol", "why": "dependents"},
        {"tool": "trace_call_chain", "arg": "symbol", "why": "call chain"},
        {"tool": "find_tests", "arg": "symbol", "why": "related tests"},
        {"tool": "analyze_impact", "arg": "symbol", "why": "deterministic impact"},
    ],
    "security": [
        {"tool": "get_security_findings", "arg": "", "why": "persist static findings"},
        {"tool": "search_repository", "arg": "query", "why": "locate risky code"},
    ],
    "default": [
        {"tool": "retrieve_evidence", "arg": "query", "why": "graph-aware hybrid retrieval"},
    ],
}


def _candidate_tokens(question: str) -> List[str]:
    cands = re.findall(r"[\w./\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cpp|h)\b", question)
    cands += re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\(\)\s*\b", question)
    cands += re.findall(r"`([a-zA-Z_][a-zA-Z0-9_.]*)`", question)
    return [c.replace("()", "") for c in cands]


def _arg_value(strategy, step, question) -> Dict[str, Any]:
    key = step["arg"]
    if not key:
        return {}
    if key == "query":
        return {"query": question}
    for tok in _candidate_tokens(question):
        if tok:
            return {key: tok}
    return {key: ""}
def planner_node(state: AgenticQAState) -> Dict[str, Any]:
    with span("planner", strategy=state.retrieval_strategy):
        strategy = (infer_strategy(state.question)
                    if state.retrieval_strategy in ("auto", "") else state.retrieval_strategy)
        plan_steps = _STRATEGY_PLANS.get(strategy, _STRATEGY_PLANS["default"])
        selected = [AgentToolCall(tool_name=s["tool"], arguments=_arg_value(strategy, s, state.question),
                                  reason=s["why"]) for s in plan_steps]
        reasoning = (f"Selected strategy '{strategy}' with {len(selected)} tool calls: "
                     + ", ".join(t.tool_name for t in selected))
        return {
            "retrieval_strategy": strategy,
            "plan": [dict(s) for s in plan_steps],
            "selected_tools": selected,
            "reasoning_summary": reasoning,
        }


async def tool_executor_node(state: AgenticQAState, tools: RepoIntelTools) -> Dict[str, Any]:
    evidence: List[AgentEvidence] = list(state.evidence)
    retrieval_results: List[Dict[str, Any]] = list(state.retrieval_results)
    graph_context: Dict[str, Any] = dict(state.graph_context)
    errors: List[str] = list(state.errors)

    for call in state.selected_tools:
        with span("tool.execution", tool_name=call.tool_name, strategy=state.retrieval_strategy):
            try:
                result = await tools.call(call.tool_name, state.repository_id, call.arguments)
            except ToolError as exc:
                errors.append(f"{call.tool_name}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{call.tool_name}: {type(exc).__name__}: {exc}")
                continue

        if isinstance(result, dict):
            for k, v in result.items():
                retrieval_results.append({"tool": call.tool_name, "kind": k, "data": _limit(v)})
                if call.tool_name in ("get_graph_neighbors", "trace_call_chain", "get_callers",
                                      "get_dependents", "get_dependencies", "get_references"):
                    graph_context[k] = _limit(v)
        if _nonempty(result):
            evidence.append(AgentEvidence(
                tool_name=call.tool_name,
                content=json.dumps(_limit(result), ensure_ascii=False)[:2000],
            ))

    return {"evidence": evidence, "retrieval_results": retrieval_results,
            "graph_context": graph_context, "errors": errors}


def evidence_assembler_node(state: AgenticQAState) -> Dict[str, Any]:
    with span("evidence_assembly", evidence_count=len(state.evidence), strategy=state.retrieval_strategy):
        coverage = min(1.0, len(state.evidence) / 3.0)
        confidence = "HIGH" if coverage >= 0.66 else ("MEDIUM" if coverage >= 0.33 else "LOW")
        return {
            "evidence_coverage": round(coverage, 3),
            "confidence": confidence,
            "claims": [e.tool_name for e in state.evidence[:8]],
            "unsupported_claims": [],
        }
def synthesis_node(state: AgenticQAState, llm_client: LLMClient) -> Dict[str, Any]:
    with span("synthesis", llm_available=llm_client.is_available(), strategy=state.retrieval_strategy):
        context = json.dumps(state.retrieval_results, ensure_ascii=False, indent=2)[:8000]
        if llm_client.is_available():
            answer = llm_client.complete(
                f"Question: {state.question}\n\n"
                f"Retrieval strategy: {state.retrieval_strategy}\n"
                f"Evidence:\n{context}\n\n"
                "Answer concisely, grounded ONLY in the evidence. If evidence is "
                "missing, say so explicitly. Do not invent code.",
                system=("You are RepoIntel, a repository-intelligence assistant. "
                        "Ground every claim in the provided evidence. Never fabricate."),
            )
            return {"final_answer": answer, "degraded": False}
        return {"final_answer": _deterministic_answer(state), "degraded": True}


def reviewer_node(state: AgenticQAState) -> Dict[str, Any]:
    with span("reviewer", evidence_count=len(state.evidence), strategy=state.retrieval_strategy):
        if state.evidence_coverage < 0.33 and state.reflection_count < state.max_reflection_count:
            return {"review_result": "reflect", "reflection_count": state.reflection_count + 1}
        return {"review_result": "accept"}


def reflection_node(state: AgenticQAState) -> Dict[str, Any]:
    with span("reflection", count=state.reflection_count):
        selected = list(state.selected_tools)
        if not any(t.tool_name == "retrieve_evidence" for t in selected):
            selected.append(AgentToolCall(
                tool_name="retrieve_evidence", arguments={"query": state.question},
                reason="Reflection: broaden evidence after low coverage",
            ))
        return {"selected_tools": selected}


def _deterministic_answer(state: AgenticQAState) -> str:
    lines = [
        "DEGRADED MODE: Gemini unavailable (no GOOGLE_API_KEY); this answer was "
        "assembled deterministically from extracted repository evidence.",
        "",
        f"Retrieval strategy: {state.retrieval_strategy}",
        f"Retrieved evidence items: {len(state.retrieval_results)}",
        f"Tool calls executed: {len(state.selected_tools)}",
    ]
    if state.retrieval_results:
        lines.append("")
        lines.append("Evidence summary:")
        seen = set()
        for r in state.retrieval_results[:10]:
            key = f"{r['tool']}:{json.dumps(r.get('data'))[:40]}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {r['tool']} -> {json.dumps(r.get('data'), ensure_ascii=False)[:160]}")
    if state.errors:
        lines.append("")
        lines.append("Tool errors (recorded, not fatal):")
        for e in state.errors[:5]:
            lines.append(f"  - {e}")
    lines.append("")
    lines.append("Confidence: " + state.confidence)
    return "\n".join(lines)
def _limit(value: Any, max_chars: int = 2000) -> Any:
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, list):
        return [_limit(v, max_chars) for v in value[:20]]
    if isinstance(value, dict):
        return {k: _limit(v, max_chars) for k, v in list(value.items())[:20]}
    return value


def _nonempty(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        return any(v not in (None, [], {}, "") for v in result.values())
    if isinstance(result, list):
        return len(result) > 0
    return bool(result)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_adaptive_agent(tools: RepoIntelTools, llm_client: LLMClient, repo_path: Optional[str] = None):
    graph = StateGraph(AgenticQAState)

    async def _planner(state):
        return planner_node(state)

    async def _executor(state):
        return await tool_executor_node(state, tools)

    async def _assembler(state):
        return evidence_assembler_node(state)

    async def _synthesis(state):
        return synthesis_node(state, llm_client)

    async def _reviewer(state):
        return reviewer_node(state)

    async def _reflect(state):
        return reflection_node(state)

    graph.add_node("Planner", _planner)
    graph.add_node("ToolExecutor", _executor)
    graph.add_node("EvidenceAssembler", _assembler)
    graph.add_node("Synthesis", _synthesis)
    graph.add_node("Reviewer", _reviewer)
    graph.add_node("Reflect", _reflect)

    graph.set_entry_point("Planner")
    graph.add_edge("Planner", "ToolExecutor")
    graph.add_edge("ToolExecutor", "EvidenceAssembler")
    graph.add_edge("EvidenceAssembler", "Synthesis")
    graph.add_edge("Synthesis", "Reviewer")

    def _route(state: AgenticQAState):
        if state.review_result == "reflect":
            return "Reflect"
        return END

    graph.add_conditional_edges("Reviewer", _route, {"Reflect": "Reflect", END: END})
    graph.add_edge("Reflect", "ToolExecutor")

    return graph.compile()