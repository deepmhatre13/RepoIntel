"""Project Health API namespace (Stage 23, Stage 46).

Dedicated ``/project-health/*`` router with structured response schemas.

Endpoints follow the existing FastAPI conventions. Project Health is keyed by
``owner/repository`` (not a Repository Intelligence repository_id); the
per-analysis endpoints read the last persisted snapshot for that repository, so
they return real signals/trends/risks/anomalies/evidence rather than skeletons.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from packages.shared.config import settings
from packages.project_health.github_provider import (
    RepositoryNotFoundError,
    normalize_repository_input,
)
from packages.project_health.service import ProjectHealthService
from packages.project_health import persistence

# Import response models from the parent API module for OpenAPI schema consistency.
from apps.api import ProjectHealthAnalysisResponse, ProjectHealthCompareResponse

router = APIRouter(prefix="/project-health", tags=["project-health"])


class AnalyzeHealthRequest(BaseModel):
    repository: str
    window_days: int = 365
    question: Optional[str] = None
    include_source: bool = True


class AskHealthRequest(BaseModel):
    question: str


def _normalize_or_400(value: str) -> str:
    try:
        return normalize_repository_input(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _service() -> ProjectHealthService:
    from packages.agentic.llm_client import LLMClient
    settings.load_dotenv()
    return ProjectHealthService(llm_client=LLMClient())


async def _load_snapshot(owner_repo: str) -> Dict[str, Any]:
    """Return the latest persisted analysis snapshot for a repository key."""
    health_key = owner_repo.lower()
    async with persistence.persistence_session() as session:
        raw = await persistence.load_analysis(session, health_key)
        if not raw:
            raise HTTPException(status_code=404, detail=f"No analysis yet for {owner_repo}")
        return json.loads(raw)


@router.post("/analyze", response_model=ProjectHealthAnalysisResponse)
async def analyze_project_health(req: AnalyzeHealthRequest) -> Dict[str, Any]:
    """Run the full Project Health pipeline for a GitHub repository."""
    owner_repo = _normalize_or_400(req.repository)
    owner, repo = owner_repo.split("/", 1)
    service = _service()
    service.window_days = req.window_days
    try:
        analysis = await service.analyze_repository(
            owner, repo, question=req.question or "",
            include_source=req.include_source)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return analysis.model_dump(mode="json")


@router.get("/{repository__:path}/snapshot")
async def get_health_snapshot(repository__: str) -> Dict[str, Any]:
    """Latest persisted health snapshot for a repository key (owner/name)."""
    owner_repo = _normalize_or_400(repository__)
    return await _load_snapshot(owner_repo)


@router.get("/{repository__:path}/history")
async def get_health_history(repository__: str,
                             metric: Optional[str] = None,
                             limit: int = Query(200, ge=1, le=2000)) -> Dict[str, Any]:
    """Persisted historical observations for a repository key (owner/name)."""
    health_key = repository__.lower()
    async with persistence.persistence_session() as session:
        if metric:
            rows = await persistence.get_metric_history(session, health_key, metric, limit)
            return {"health_key": health_key, "metric": metric,
                    "observations": rows, "count": len(rows)}
        all_data = await persistence.get_all_metric_history(session, health_key, limit)
        return {"health_key": health_key, **all_data}


def _subsection(analysis: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Pull a section out of the persisted snapshot with a clear 404 if missing."""
    value = analysis.get(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"No '{key}' in analysis snapshot")
    return value


@router.get("/{repository__:path}/signals")
async def get_health_signals(repository__: str) -> Dict[str, Any]:
    owner_repo = _normalize_or_400(repository__)
    analysis = await _load_snapshot(owner_repo)
    return {"repository": owner_repo, "signals": _subsection(analysis, "signals")}


@router.get("/{repository__:path}/trends")
async def get_health_trends(repository__: str) -> Dict[str, Any]:
    owner_repo = _normalize_or_400(repository__)
    analysis = await _load_snapshot(owner_repo)
    return {"repository": owner_repo, "trends": _subsection(analysis, "trends"),
            "change_points": analysis.get("change_points", [])}


@router.get("/{repository__:path}/risks")
async def get_health_risks(repository__: str) -> Dict[str, Any]:
    owner_repo = _normalize_or_400(repository__)
    analysis = await _load_snapshot(owner_repo)
    return {"repository": owner_repo, "risks": _subsection(analysis, "risks")}


@router.get("/{repository__:path}/anomalies")
async def get_health_anomalies(repository__: str) -> Dict[str, Any]:
    owner_repo = _normalize_or_400(repository__)
    analysis = await _load_snapshot(owner_repo)
    return {"repository": owner_repo,
            "anomalies": _subsection(analysis, "anomalies"),
            "change_points": analysis.get("change_points", [])}


@router.get("/{repository__:path}/evidence")
async def get_health_evidence(repository__: str) -> Dict[str, Any]:
    owner_repo = _normalize_or_400(repository__)
    analysis = await _load_snapshot(owner_repo)
    evidence = []
    for d in analysis.get("dimensions", []):
        for e in d.get("evidence", []):
            evidence.append({"dimension": d.get("name"), **e})
    return {"repository": owner_repo, "evidence": evidence,
            "why_changed": analysis.get("why_changed", [])}


@router.post("/{repository__:path}/refresh")
async def refresh_project_health_route(repository__: str) -> Dict[str, Any]:
    """Incremental monitoring refresh boundary.

    Fetches changed GitHub data, recomputes health incrementally (re-parsing the
    source tree only when the pushed_at fingerprint changed) and emits
    evidence-backed health events versus the prior snapshot.
    """
    owner_repo = _normalize_or_400(repository__)
    owner, repo = owner_repo.split("/", 1)
    service = _service()
    try:
        analysis = await service.refresh_project_health(owner_repo)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {
        "repository": owner_repo,
        "health_events": analysis.model_dump(mode="json")["health_events"],
        "trajectory": analysis.trajectory,
        "overall_score": analysis.overall_score,
        "score_state": analysis.score_state,
    }


@router.get("/{repository__:path}/events")
async def get_health_events(repository__: str) -> Dict[str, Any]:
    """Evidence-backed health events from the latest persisted snapshot."""
    owner_repo = _normalize_or_400(repository__)
    analysis = await _load_snapshot(owner_repo)
    return {"repository": owner_repo,
            "health_events": analysis.get("health_events", [])}


class CompareRequest(BaseModel):
    repository_a: str
    repository_b: str


@router.post("/compare", response_model=ProjectHealthCompareResponse)
async def compare_project_health_route(req: CompareRequest) -> Dict[str, Any]:
    """Deterministic, evidence-backed comparison of two repositories.

    Reads both persisted snapshots and compares normalized metrics/dimensions
    using the same bounds as the health score. Missing data is explicit.
    """
    key_a = _normalize_or_400(req.repository_a)
    key_b = _normalize_or_400(req.repository_b)
    if key_a.lower() == key_b.lower():
        raise HTTPException(status_code=400, detail="Provide two different repositories")
    async with persistence.persistence_session() as session:
        raw_a = await persistence.load_analysis(session, key_a.lower())
        raw_b = await persistence.load_analysis(session, key_b.lower())
    if not raw_a or not raw_b:
        missing = key_a if not raw_a else key_b
        raise HTTPException(
            status_code=404,
            detail=f"No analysis snapshot yet for {missing}; analyze it first")
    from packages.project_health.comparison import compare_project_health
    return compare_project_health(json.loads(raw_a), json.loads(raw_b))


@router.post("/{repository__:path}/ask")
async def ask_health_question(repository__: str,
                              req: AskHealthRequest) -> Dict[str, Any]:
    """Ask an AI-interpreted question about a completed health analysis.

    Runs the LangGraph health agent; Gemini (the only LLM) interprets the
    deterministic evidence and never invents facts. In degraded mode returns a
    clear evidence-only answer.
    """
    owner_repo = _normalize_or_400(repository__)
    owner, repo = owner_repo.split("/", 1)
    service = _service()
    try:
        analysis = await service.analyze_repository(
            owner, repo, question=req.question, persist=False,
            include_source=False)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {
        "repository": owner_repo,
        "question": req.question,
        "answer": analysis.gemini_analysis or (
            "Gemini unavailable; deterministic evidence is provided in the "
            "analysis dimensions."),
        "ai_available": analysis.gemini_available,
        "why_changed": [c for c in analysis.why_changed],
        "evidence": [
            f"{d.name}: {e.metric}={e.value}"
            for d in analysis.dimensions for e in d.evidence[:4]
        ],
        "reviewed": analysis.reviewed,
    }