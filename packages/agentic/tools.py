"""Unified RepoIntel tool registry (Stage 6 / Stage 8 shared capability layer).

This module is the SINGLE definition of the agentic/intelligence tools available
over RepoIntel. The same underlying capabilities power:

    * LangGraph agent tools   (packages.agentic)
    * MCP server tools        (packages.mcp.server)
    * FastAPI chat / search   (apps.api)
    * CLI                     (apps.cli)

Tools are thin adapters that:

    * validate input strictly,
    * respect repository_id boundaries (a tool never crosses into another repo),
    * call the existing RepositoryFacade / service boundaries only,
    * preserve evidence and provenance,
    * NEVER execute arbitrary repository code or shell commands,
    * NEVER expose secrets.

Deterministic repository intelligence stays in the facade; the LLM (Gemini) is
used only by the agent orchestrator, never to author tool output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from packages.shared.tracing import span
from packages.shared.secret_redaction import redact_secrets

if TYPE_CHECKING:  # pragma: no cover
    from packages.agentic.llm_client import LLMClient


class ToolError(Exception):
    """Raised when a tool invocation is invalid or the underlying service fails."""


@dataclass
class ToolParameter:
    """Describes one tool input parameter (for MCP/agentic schemas)."""

    name: str
    type: str  # "string", "integer", "boolean", "array"
    description: str = ""
    required: bool = True


@dataclass
class RepoIntelTool:
    """Metadata + executor for a single RepoIntel tool."""

    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    # Handler receives (facade, repository_id, kwargs) and returns a dict.
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    category: str = "general"


def _repo_tool_decorator(name, description, parameters=None, category="general"):
    """Attach tool metadata to a ``tool_*`` method (factory for @repo_tool).

    Defined before the class because class-body decorators run at class
    creation time.
    """
    def deco(fn):
        fn._tool_meta = RepoIntelTool(
            name=name,
            description=description,
            parameters=parameters or [],
            handler=fn,
            category=category,
        )
        return fn
    return deco


# Public decorator alias (kept for readability / external use).
repo_tool = _repo_tool_decorator
class RepoIntelTools:
    """Registry + facade-binding executor used by API, CLI, MCP and the agent.

    A single instance is bound to one ``RepositoryFacade`` (and its session +
    vector store) for the duration of a request/run. All handlers only call the
    facade's public methods; no repository code is ever executed.
    """

    def __init__(self, facade, llm_client=None, repository_path: Optional[str] = None):
        self.facade = facade
        self.llm_client = llm_client
        self.repository_path = repository_path  # physical checkout (never executed)
        self._registry: Dict[str, RepoIntelTool] = self._build_registry()

    # ------------------------------------------------------------------
    # Registry construction
    # ------------------------------------------------------------------
    def _build_registry(self) -> Dict[str, RepoIntelTool]:
        reg: Dict[str, RepoIntelTool] = {}
        for name in dir(self):
            if name.startswith("tool_"):
                method = getattr(self, name)
                meta = getattr(method, "_tool_meta", None)
                if meta:
                    reg[meta.name] = meta
        return reg

    def list_tools(self) -> List[Dict[str, Any]]:
        """Expose tool descriptors (names, params) for discovery surfaces."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "parameters": [
                    {"name": p.name, "type": p.type, "description": p.description,
                     "required": p.required}
                    for p in t.parameters
                ],
            }
            for t in self._registry.values()
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def call(self, tool_name: str, repository_id: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a tool by name with repository_id boundary enforced."""
        tool = self._registry.get(tool_name)
        if tool is None:
            raise ToolError(f"Unknown tool: {tool_name}")
        if not tool.handler:
            raise ToolError(f"Tool {tool_name} has no handler")

        kwargs = dict(arguments or {})
        with span(f"tool.{tool_name}", repository_id=repository_id, tool_name=tool_name):
            # Handlers are captured unbound by the decorator at class-creation
            # time, so the instance must be passed explicitly.
            result = await tool.handler(self, self.facade, repository_id, kwargs)
        return _sanitize(result)

    def get_tool(self, name: str) -> Optional[RepoIntelTool]:
        return self._registry.get(name)
# ------------------------------------------------------------------
    # Tool implementations (each wraps an existing repository_facade boundary)
    # ------------------------------------------------------------------
    @_repo_tool_decorator("search_repository",
                          "Hybrid (lexical+semantic) search across the repository.",
                          [ToolParameter("query", "string", "Search query")],
                          category="search")
    async def tool_search_repository(self, facade, repository_id, kwargs):
        query = _require(kwargs, "query")
        items = await facade.search_code(repository_id, query, top_k=8)
        return {"results": items.get("results", items) if isinstance(items, dict) else items}

    @_repo_tool_decorator("search_symbol",
                          "Find symbols by name (exact or substring).",
                          [ToolParameter("symbol", "string", "Symbol name")],
                          category="search")
    async def tool_search_symbol(self, facade, repository_id, kwargs):
        sym = _require(kwargs, "symbol")
        return {"symbols": await facade.get_symbols_by_name(repository_id, sym)}

    @_repo_tool_decorator("get_symbol",
                          "Return structured details for a named symbol.",
                          [ToolParameter("symbol", "string", "Symbol name")],
                          category="search")
    async def tool_get_symbol(self, facade, repository_id, kwargs):
        sym = _require(kwargs, "symbol")
        return {"symbols": await facade.get_symbols_by_name(repository_id, sym)}

    @_repo_tool_decorator("get_file",
                          "Return metadata (file path, symbols) for a file.",
                          [ToolParameter("file_path", "string", "File path")],
                          category="search")
    async def tool_get_file(self, facade, repository_id, kwargs):
        fp = _require(kwargs, "file_path")
        meta = await facade.get_file_metadata(repository_id, fp)
        if meta is None:
            return {"found": False, "message": f"No indexed file matching '{fp}'"}
        return {"found": True, **meta}

    @_repo_tool_decorator("get_callers",
                          "Return direct callers (incoming call edges) of a symbol.",
                          [ToolParameter("symbol", "string", "Symbol name")],
                          category="graph")
    async def tool_get_callers(self, facade, repository_id, kwargs):
        sym = _require(kwargs, "symbol")
        edges = await facade.symbol_graph_neighbors(repository_id, sym)
        callers = [e for e in edges if e.get("target_symbol") == sym or e.get("target_file") == sym]
        return {"callers": callers}

    @_repo_tool_decorator("get_callees",
                          "Return direct callees (outgoing call edges) of a symbol.",
                          [ToolParameter("symbol", "string", "Symbol name")],
                          category="graph")
    async def tool_get_callees(self, facade, repository_id, kwargs):
        sym = _require(kwargs, "symbol")
        edges = await facade.symbol_graph_neighbors(repository_id, sym)
        callees = [e for e in edges if e.get("source_symbol") == sym or e.get("source_file") == sym]
        return {"callees": callees}

    @_repo_tool_decorator("get_dependents",
                          "Return entities that depend on (target) a symbol/file.",
                          [ToolParameter("target", "string", "Symbol or file"),
                           ToolParameter("max_results", "integer", "Max results", required=False)],
                          category="graph")
    async def tool_get_dependents(self, facade, repository_id, kwargs):
        target = _require(kwargs, "target")
        max_results = int(kwargs.get("max_results") or 50)
        return {"dependents": await facade.get_dependents(repository_id, target, max_results)}

    @_repo_tool_decorator("get_dependencies",
                          "Return dependencies of a target symbol/file.",
                          [ToolParameter("target", "string", "Symbol or file")],
                          category="graph")
    async def tool_get_dependencies(self, facade, repository_id, kwargs):
        target = _require(kwargs, "target")
        edges = await facade.symbol_graph_neighbors(repository_id, target)
        deps = [e for e in edges if e.get("source_symbol") == target or e.get("source_file") == target]
        return {"dependencies": deps}

    @_repo_tool_decorator("get_references",
                          "Return references to a symbol (both directions).",
                          [ToolParameter("symbol", "string", "Symbol name")],
                          category="graph")
    async def tool_get_references(self, facade, repository_id, kwargs):
        sym = _require(kwargs, "symbol")
        return {"references": await facade.symbol_graph_neighbors(repository_id, sym)}

    @_repo_tool_decorator("trace_call_chain",
                          "Trace a call chain starting from a symbol.",
                          [ToolParameter("symbol", "string", "Starting symbol"),
                           ToolParameter("direction", "string", "downstream|upstream", required=False),
                           ToolParameter("max_depth", "integer", "Max depth", required=False)],
                          category="graph")
    async def tool_trace_call_chain(self, facade, repository_id, kwargs):
        sym = _require(kwargs, "symbol")
        direction = kwargs.get("direction") or "downstream"
        max_depth = int(kwargs.get("max_depth") or 5)
        return {"chain": await facade.trace_call_chain(repository_id, sym, direction, max_depth)}

    @_repo_tool_decorator("get_graph_neighbors",
                          "Return neighbours of a node in the repository knowledge graph.",
                          [ToolParameter("node_id", "string", "Graph node id")],
                          category="graph")
    async def tool_get_graph_neighbors(self, facade, repository_id, kwargs):
        node = _require(kwargs, "node_id")
        kg = await facade.get_knowledge_graph(repository_id)
        return {"neighbors": kg.get_neighbors(node)}

    @_repo_tool_decorator("get_architecture",
                          "Return repository architecture analysis.",
                          [], category="architecture")
    async def tool_get_architecture(self, facade, repository_id, kwargs):
        return await facade.analyze_architecture(repository_id)

    @_repo_tool_decorator("get_hotspots",
                          "Return code hotspots computed from commit history.",
                          [], category="risk")
    async def tool_get_hotspots(self, facade, repository_id, kwargs):
        return {"hotspots": await facade.get_hotspots(repository_id)}

    @_repo_tool_decorator("get_coupling",
                          "Return coupling (fan-in/fan-out) for a file.",
                          [ToolParameter("file_path", "string", "File path")],
                          category="risk")
    async def tool_get_coupling(self, facade, repository_id, kwargs):
        fp = _require(kwargs, "file_path")
        return await facade.get_coupling_metrics(repository_id, fp)

    @_repo_tool_decorator("get_git_history",
                          "Return recent git commits.",
                          [ToolParameter("limit", "integer", "Max commits", required=False)],
                          category="git")
    async def tool_get_git_history(self, facade, repository_id, kwargs):
        limit = int(kwargs.get("limit") or 20)
        return {"commits": await facade.get_git_commits(repository_id, limit)}

    @_repo_tool_decorator("get_symbol_history",
                          "Return change history for a symbol.",
                          [ToolParameter("symbol", "string", "Symbol name"),
                           ToolParameter("limit", "integer", "Max records", required=False)],
                          category="git")
    async def tool_get_symbol_history(self, facade, repository_id, kwargs):
        sym = _require(kwargs, "symbol")
        limit = int(kwargs.get("limit") or 20)
        return {"history": await facade.get_symbol_history(repository_id, sym, limit)}

    @_repo_tool_decorator("find_tests",
                          "Find tests related to a symbol.",
                          [ToolParameter("symbol", "string", "Symbol name")],
                          category="testing")
    async def tool_find_tests(self, facade, repository_id, kwargs):
        sym = _require(kwargs, "symbol")
        return {"tests": await facade.get_tests_for_symbol(repository_id, sym)}

    @_repo_tool_decorator("analyze_impact",
                          "Analyse the deterministic impact of changing a symbol/file.",
                          [ToolParameter("target", "string", "Symbol or file")],
                          category="impact")
    async def tool_analyze_impact(self, facade, repository_id, kwargs):
        target = _require(kwargs, "target")
        return await facade.analyze_impact(repository_id, target)

    @_repo_tool_decorator("retrieve_evidence",
                          "Perform graph-aware hybrid retrieval over the repository.",
                          [ToolParameter("query", "string", "Search query")],
                          category="retrieval")
    async def tool_retrieve_evidence(self, facade, repository_id, kwargs):
        from packages.agentic.core.tool_loop import run_tool_selection_loop
        from packages.agentic.llm_client import LLMClient

        query = _require(kwargs, "query")
        llm = self.llm_client or LLMClient()
        result = await run_tool_selection_loop(
            llm,
            facade,
            objective=query,
            max_rounds=3,
            repo_id=repository_id,
            query=query,
            target_section="Repository evidence",
            task_id="adaptive-retrieval",
        )
        return {
            "results": [item.to_dict() for item in result.retrieved_items],
            "strategy": "shared_tool_loop",
            "reasoning": result.evidence.reasoning,
        }

    @_repo_tool_decorator("get_security_findings",
                          "Return persisted security findings for the repository.",
                          [], category="security")
    async def tool_get_security_findings(self, facade, repository_id, kwargs):
        findings = await facade.get_security_findings(repository_id)
        return {"findings": findings, "count": len(findings)}

    @_repo_tool_decorator("ask_repository",
                          "Repository Q&A using the shared retrieval core and grounded synthesis.",
                          [ToolParameter("question", "string", "Natural-language question")],
                          category="agent")
    async def tool_ask_repository(self, facade, repository_id, kwargs):
        from packages.agentic.llm_client import LLMClient
        from packages.agentic.agents.qa_analyst import answer_repository_query

        question = _require(kwargs, "question")
        llm = self.llm_client or LLMClient()
        answer = await answer_repository_query(
            repository_id,
            str(question),
            facade,
            llm,
        )
        return {
            "answer": answer.answer,
            "retrieval_strategy": "shared_tool_loop",
            "selected_tools": ["run_tool_selection_loop"],
            "evidence_count": len(answer.evidence),
            "confidence": answer.confidence,
            "reflection_count": 0,
            "degraded": answer.degraded,
            "llm_mode": answer.llm_mode,
            "confidence_breakdown": answer.confidence_breakdown,
            "audit": answer.audit,
        }
# ---------------------------------------------------------------------------
# Module-level helpers (validation/sanitisation).
# ---------------------------------------------------------------------------
def _require(kwargs: Dict[str, Any], key: str) -> Any:
    value = kwargs.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ToolError(f"Missing required argument: {key}")
    return value


def _sanitize(value: Any) -> Any:
    """Recursively scrub secrets out of tool output before it leaves the layer."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, str):
        return redact_secrets(value).text
    return value