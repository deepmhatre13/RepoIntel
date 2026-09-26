"""Project Health tools for MCP / agentic surfaces (Stage 4C).

A thin adapter over the EXISTING ``ProjectHealthService`` and persistence
layer -- the same service powers FastAPI and the LangGraph agent. No health
logic is duplicated here. Descriptors reuse the ``RepoIntelTool`` shape so the
MCP server can merge both registries without a second protocol path.

Health tools are keyed by ``owner/repository`` (not repository_id); every
handler validates and normalizes the input before touching the service.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from packages.project_health.github_provider import normalize_repository_input
from packages.project_health.service import ProjectHealthService

try:  # reuse the existing tool descriptor dataclasses (no duplicate schema)
    from packages.agentic.tools import RepoIntelTool, ToolParameter
except Exception:  # pragma: no cover
    RepoIntelTool = None  # type: ignore[misc]
    ToolParameter = None  # type: ignore[misc]


class ToolError(Exception):
    """Raised when a health tool invocation is invalid or fails."""


def _key(repository: str) -> str:
    """Normalize ``owner/repo`` or a GitHub URL to the canonical health key."""
    try:
        return normalize_repository_input(repository).lower()
    except ValueError as exc:
        raise ToolError(f"invalid repository: {exc}")


class ProjectHealthTools:
    """MCP/agentic adapter over ProjectHealthService (no duplicate engine)."""

    def __init__(self, llm_client=None, service: Optional[ProjectHealthService] = None):
        self._service = service
        self._llm_client = llm_client
        self._registry: Dict[str, RepoIntelTool] = self._build_registry()

    def _service_or_create(self) -> ProjectHealthService:
        if self._service is None:
            from packages.agentic.llm_client import LLMClient
            self._service = ProjectHealthService(
                llm_client=self._llm_client or LLMClient())
        return self._service

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
        return [
            {"name": t.name, "description": t.description, "category": t.category,
             "parameters": [
                 {"name": p.name, "type": p.type,
                  "description": p.description, "required": p.required}
                 for p in t.parameters]}
            for t in self._registry.values()
        ]

    def has_tool(self, name: str) -> bool:
        return name in self._registry

    async def call(self, tool_name: str,
                   arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tool = self._registry.get(tool_name)
        if tool is None or tool.handler is None:
            raise ToolError(f"Unknown health tools: {tool_name}")
        return await tool.handler(self, dict(arguments or {}))

    # ------------------------------------------------------------------
    # Tool implementations (each wraps an existing service boundary)
    # ------------------------------------------------------------------
    async def tool_analyze_project_health(self, kwargs):
        svc = self._service_or_create()
        key = _key(kwargs.get("repository", ""))
        owner, repo = key.split("/", 1)
        if kwargs.get("window_days"):
            svc.window_days = int(kwargs["window_days"])
        analysis = await svc.analyze_repository(
            owner, repo, question=kwargs.get("question") or "",
            include_source=bool(kwargs.get("include_source", True)))
        return analysis.model_dump(mode="json")

    async def tool_get_health_snapshot(self, kwargs):
        from packages.project_health import persistence
        key = _key(kwargs.get("repository", ""))
        async with persistence.persistence_session() as session:
            raw = await persistence.load_analysis(session, key)
        if not raw:
            raise ToolError(f"No analysis snapshot yet for {key}")
        return json.loads(raw)

    async def tool_get_health_history(self, kwargs):
        from packages.project_health import persistence
        key = _key(kwargs.get("repository", ""))
        limit = int(kwargs.get("limit") or 500)
        async with persistence.persistence_session() as session:
            data = await persistence.get_all_metric_history(session, key, limit)
        metrics = data.get("metrics") or {}
        count = sum(len(v) for v in metrics.values())
        return {"repository": key, "metrics": metrics,
                "metric_count": data.get("metric_count"), "count": count}

    async def tool_get_health_signals(self, kwargs):
        snap = await self.tool_get_health_snapshot(kwargs)
        return {"repository": _key(kwargs["repository"]),
                "signals": snap.get("signals", [])}

    async def tool_get_health_risks(self, kwargs):
        snap = await self.tool_get_health_snapshot(kwargs)
        return {"repository": _key(kwargs["repository"]),
                "risks": snap.get("risks", [])}

    async def tool_get_health_trends(self, kwargs):
        snap = await self.tool_get_health_snapshot(kwargs)
        return {"repository": _key(kwargs["repository"]),
                "trends": snap.get("trends", []),
                "change_points": snap.get("change_points", [])}

    async def tool_get_health_anomalies(self, kwargs):
        snap = await self.tool_get_health_snapshot(kwargs)
        return {"repository": _key(kwargs["repository"]),
                "anomalies": snap.get("anomalies", []),
                "change_points": snap.get("change_points", [])}

    async def tool_get_health_evidence(self, kwargs):
        snap = await self.tool_get_health_snapshot(kwargs)
        evidence = [{"dimension": d.get("name"), **e}
                    for d in snap.get("dimensions", [])
                    for e in d.get("evidence", [])]
        return {"repository": _key(kwargs["repository"]),
                "evidence": evidence,
                "why_changed": snap.get("why_changed", [])}

    async def tool_get_health_events(self, kwargs):
        snap = await self.tool_get_health_snapshot(kwargs)
        return {"repository": _key(kwargs["repository"]),
                "health_events": snap.get("health_events", [])}

    async def tool_explain_health_change(self, kwargs):
        svc = self._service_or_create()
        key = _key(kwargs.get("repository", ""))
        owner, repo = key.split("/", 1)
        analysis = await svc.analyze_repository(
            owner, repo, question=kwargs.get("question") or "Why did health change?",
            include_source=False, persist=False)
        return {
            "repository": key,
            "why_changed": analysis.why_changed,
            "explanation": analysis.gemini_analysis,
            "gemini_available": analysis.gemini_available,
            "trajectory": analysis.trajectory,
            "overall_score": analysis.overall_score,
            "score_state": analysis.score_state,
        }

    async def tool_refresh_project_health(self, kwargs):
        svc = self._service_or_create()
        analysis = await svc.refresh_project_health(
            _key(kwargs.get("repository", "")))
        return {"repository": _key(kwargs["repository"]),
                "health_events": [e.model_dump(mode="json")
                                  for e in analysis.health_events],
                "trajectory": analysis.trajectory,
                "overall_score": analysis.overall_score}

    async def tool_compare_project_health(self, kwargs):
        from packages.project_health import persistence
        from packages.project_health.comparison import compare_project_health
        key_a = _key(kwargs.get("repository_a", ""))
        key_b = _key(kwargs.get("repository_b", ""))
        async with persistence.persistence_session() as session:
            raw_a = await persistence.load_analysis(session, key_a)
            raw_b = await persistence.load_analysis(session, key_b)
        if not raw_a or not raw_b:
            missing = key_a if not raw_a else key_b
            raise ToolError(f"No analysis snapshot yet for {missing}")
        return compare_project_health(json.loads(raw_a), json.loads(raw_b))


def build_health_tools() -> Dict[str, RepoIntelTool]:
    """Register descriptor metadata onto the ``tool_*`` methods above.

    Kept as a single readable table: name -> (description, parameters).
    """
    spec = {
        "analyze_project_health": (
            "Run the full Project Health pipeline for a GitHub repository.",
            [ToolParameter("repository", "string",
                           "owner/repository or GitHub URL"),
             ToolParameter("window_days", "integer",
                           "Observation window in days", required=False),
             ToolParameter("include_source", "boolean",
                           "Acquire + analyze source tree when needed",
                           required=False)]),
        "get_health_snapshot": (
            "Latest persisted health snapshot for a repository.",
            [ToolParameter("repository", "string", "owner/repository")]),
        "get_health_history": (
            "Persisted historical observations for a repository.",
            [ToolParameter("repository", "string", "owner/repository"),
             ToolParameter("limit", "integer", "Max observations",
                           required=False)]),
        "get_health_signals": (
            "Current health signals (deteriorations, notable metric moves).",
            [ToolParameter("repository", "string", "owner/repository")]),
        "get_health_risks": (
            "Risk dimensions with level and trend.",
            [ToolParameter("repository", "string", "owner/repository")]),
        "get_health_trends": (
            "Per-metric trends and change points.",
            [ToolParameter("repository", "string", "owner/repository")]),
        "get_health_anomalies": (
            "Detected anomalies and change points.",
            [ToolParameter("repository", "string", "owner/repository")]),
        "get_health_evidence": (
            "Evidence records and the deterministic why-changed ranking.",
            [ToolParameter("repository", "string", "owner/repository")]),
        "get_health_events": (
            "Evidence-backed health events from the monitoring refresh.",
            [ToolParameter("repository", "string", "owner/repository")]),
        "explain_health_change": (
            "Deterministic why-changed ranking plus optional Gemini "
            "interpretation of that ranking.",
            [ToolParameter("repository", "string", "owner/repository"),
             ToolParameter("question", "string", "Focus question",
                           required=False)]),
        "refresh_project_health": (
            "Incremental monitoring refresh; emits evidence-backed events.",
            [ToolParameter("repository", "string", "owner/repository")]),
        "compare_project_health": (
            "Deterministic comparison of two repositories' health snapshots.",
            [ToolParameter("repository_a", "string", "owner/repository A"),
             ToolParameter("repository_b", "string", "owner/repository B")]),
    }
    out: Dict[str, RepoIntelTool] = {}
    for name, (desc, params) in spec.items():
        out[name] = RepoIntelTool(
            name=name, description=desc, parameters=params,
            handler=getattr(ProjectHealthTools, f"tool_{name}"),
            category="project-health")
    return out


# Bind descriptors once at import so _build_registry() picks them up.
for _name, _tool in build_health_tools().items():
    setattr(getattr(ProjectHealthTools, f"tool_{_name}"),
            "_tool_meta", _tool)

