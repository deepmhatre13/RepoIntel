import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from apps.job_runner import job_runner
from packages.knowledge.db import init_db, AsyncSessionLocal
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.repository_facade import RepositoryFacade
from packages.agentic.llm_client import LLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize the schema on startup and release
    resources on shutdown. Uses the modern lifespan API (on_event is deprecated)."""
    await init_db()
    try:
        yield
    finally:
        # Cleanly release any engine/session resources on shutdown.
        from packages.knowledge.db import engine

        await engine.dispose()


app = FastAPI(title="RepoIntel API", version="1.0.0", lifespan=lifespan)

class CreateRepoRequest(BaseModel):
    repo_url: str

class ChatRequest(BaseModel):
    query: str

class CreateGroupRequest(BaseModel):
    name: str
    description: Optional[str] = None

class AddGroupMemberRequest(BaseModel):
    repository_id: str

class ImpactRequest(BaseModel):
    target: str
    requirement: Optional[str] = None

@app.post("/repository-groups")
async def create_repository_group(req: CreateGroupRequest):
    group_id = str(uuid.uuid4())
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        res = await facade.create_repository_group(group_id, req.name, req.description)
        return res

@app.post("/repository-groups/{group_id}/members")
async def add_group_member(group_id: str, req: AddGroupMemberRequest):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        await facade.add_repo_to_group(group_id, req.repository_id)
        return {"status": "SUCCESS", "group_id": group_id, "repository_id": req.repository_id}

@app.post("/repositories")
async def create_repository(req: CreateRepoRequest):
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, req.repo_url))
    return {"repository_id": repo_id, "repo_url": req.repo_url}

@app.post("/repositories/{repository_id}/analyze")
async def analyze_repository(repository_id: str, req: CreateRepoRequest, background_tasks: BackgroundTasks):
    run_id = job_runner.create_job(req.repo_url)
    background_tasks.add_task(job_runner.execute_job, run_id)
    return {"run_id": run_id, "status": "PENDING"}

@app.get("/runs/{run_id}/status")
async def get_run_status(run_id: str):
    if run_id not in job_runner.jobs:
        raise HTTPException(status_code=404, detail="Run ID not found")
    job = job_runner.jobs[run_id]
    return {"run_id": run_id, "status": job["status"], "error": job["error"]}

@app.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    return {"artifact_id": artifact_id, "content": "Artifact content preview"}

@app.post("/repositories/{repository_id}/chat")
async def chat_repository(repository_id: str, req: ChatRequest):
    from packages.agentic.agents.qa_analyst import answer_repository_query
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        llm = LLMClient()
        qa_ans = await answer_repository_query(repository_id, req.query, facade, llm)
        return qa_ans.model_dump()

@app.get("/repositories/{repository_id}/call-chain")
async def get_call_chain(repository_id: str, symbol: str, direction: str = "downstream", max_depth: int = 5):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        chain = await facade.trace_call_chain(repository_id, symbol, direction, max_depth)
        return {"repository_id": repository_id, "symbol": symbol, "call_chain": chain}

@app.get("/repositories/{repository_id}/commits")
async def get_commits(repository_id: str, limit: int = 20):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        commits = await facade.get_git_commits(repository_id, limit)
        return {"repository_id": repository_id, "commits": commits}

@app.get("/repositories/{repository_id}/symbols/{symbol}/history")
async def get_symbol_history_api(repository_id: str, symbol: str):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        history = await facade.get_symbol_history(repository_id, symbol)
        co_changes = await facade.get_co_change_relationships(repository_id, symbol)
        return {"repository_id": repository_id, "symbol": symbol, "history": history, "co_changes": co_changes}

@app.get("/repositories/{repository_id}/architecture")
async def get_architecture_snapshot_api(repository_id: str, commit_sha: str = "HEAD"):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        snapshot = await facade.create_architecture_snapshot(repository_id, commit_sha)
        return snapshot

@app.get("/repositories/{repository_id}/architecture/compare")
async def compare_architecture_api(repository_id: str, sha1: str, sha2: str):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        diff = await facade.compare_architecture_snapshots(repository_id, sha1, sha2)
        return {"repository_id": repository_id, "sha1": sha1, "sha2": sha2, "diff": diff}

@app.get("/repositories/{repository_id}/hotspots")
async def get_hotspots_api(repository_id: str):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        hotspots = await facade.get_hotspots(repository_id)
        return {"repository_id": repository_id, "hotspots": hotspots}

@app.get("/repositories/{repository_id}/coupling")
async def get_coupling_api(repository_id: str, file_path: str):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        coupling = await facade.get_coupling_metrics(repository_id, file_path)
        return coupling

@app.get("/repositories/{repository_id}/tests/{symbol}")
async def get_tests_for_symbol_api(repository_id: str, symbol: str):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        tests = await facade.get_tests_for_symbol(repository_id, symbol)
        return {"repository_id": repository_id, "symbol": symbol, "tests": tests}

@app.post("/repositories/{repository_id}/impact")
async def analyze_impact_api(repository_id: str, req: ImpactRequest):
    """Impact analysis for a target symbol/file (evidence-grounded blast radius)."""
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select as _select
        from packages.knowledge.models import RepositoryModel

        repo = (
            await session.execute(_select(RepositoryModel).where(RepositoryModel.id == repository_id))
        ).scalar_one_or_none()
        if repo is None:
            raise HTTPException(status_code=404, detail="Repository not found")

        facade = RepositoryFacade(session, vector_store)
        impact = await facade.analyze_impact(repository_id, req.target)
        return {
            "repository_id": repository_id,
            "target": req.target,
            "total_dependents": impact.get("total_dependents", 0),
            "affected_files": impact.get("affected_files", []),
            "affected_symbols": impact.get("affected_symbols", []),
            "llm_available": LLMClient().is_available(),
        }


@app.post("/repositories/{repository_id}/change-plan")
async def create_change_plan_api(repository_id: str, req: ImpactRequest):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        llm = LLMClient()
        from packages.agentic.agents.impact_planner import plan_code_change
        plan = await plan_code_change(repository_id, req.requirement or "Modify component", req.target, facade, llm)
        return plan.model_dump()

@app.get("/repositories/{repository_id}/symbols")
async def get_symbols(repository_id: str, file_path: str = ""):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        symbols = await facade.get_symbols_for_file(repository_id, file_path)
        return {"repository_id": repository_id, "symbols": symbols}

@app.get("/repositories/{repository_id}/dependencies")
async def get_dependencies(repository_id: str):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        deps = await facade.get_dependency_manifest(repository_id)
        return {"repository_id": repository_id, "dependencies": deps}

@app.get("/repositories/{repository_id}/graph")
async def get_graph(repository_id: str, symbol_or_file: str = ""):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        neighbors = await facade.symbol_graph_neighbors(repository_id, symbol_or_file)
        return {"repository_id": repository_id, "neighbors": neighbors}


@app.post("/repositories/{repository_id}/copilot")

async def repository_copilot_endpoint(repository_id: str, req_data: Dict[str, Any]):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        req_data["repository_id"] = repository_id
        res = await facade.execute_copilot_request(req_data)
        return res


@app.post("/repository-groups/{group_id}/copilot")
async def repository_group_copilot_endpoint(group_id: str, req_data: Dict[str, Any]):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        req_data["repository_group_id"] = group_id
        res = await facade.execute_copilot_request(req_data)
        return res


# ---------------------------------------------------------------------------
# Phase 40/41/43: lifecycle status, re-analysis, and cancellation endpoints
# ---------------------------------------------------------------------------

class ReanalyzeRequest(BaseModel):
    repo_url: Optional[str] = None
    incremental: bool = True
    force: bool = False
    generate_artifact: bool = False


@app.get("/repositories/{repository_id}/status")
async def get_repository_status(repository_id: str):
    """Repository lifecycle state + artifact counts (Phase 40.1)."""
    from sqlalchemy import select as _select
    from packages.knowledge.models import RepositoryModel, PipelineRunModel
    from packages.knowledge.persistence import count_repo_artifacts

    async with AsyncSessionLocal() as session:
        repo = (
            await session.execute(_select(RepositoryModel).where(RepositoryModel.id == repository_id))
        ).scalar_one_or_none()
        if repo is None:
            raise HTTPException(status_code=404, detail="Repository not found")
        counts = await count_repo_artifacts(session, repository_id)
        runs = (
            await session.execute(
                _select(PipelineRunModel)
                .where(PipelineRunModel.repository_id == repository_id)
                .order_by(PipelineRunModel.started_at.desc())
                .limit(5)
            )
        ).scalars().all()
        return {
            "repository_id": repository_id,
            "repo_url": repo.repo_url,
            "status": repo.status,
            "branch": repo.branch,
            "head_commit": repo.head_commit,
            "fingerprint_hash": repo.fingerprint_hash,
            "last_run_id": repo.last_run_id,
            "archetype": repo.archetype,
            "artifact_counts": counts,
            "recent_runs": [
                {"run_id": r.run_id, "status": r.status, "mode": r.mode,
                 "progress": r.progress, "files_parsed": r.files_parsed}
                for r in runs
            ],
        }


@app.post("/repositories/{repository_id}/reanalyze")
async def reanalyze_repository(repository_id: str, req: ReanalyzeRequest, background_tasks: BackgroundTasks):
    """Change-aware re-analysis (Phase 41.6): incremental=true processes only changed
    content; force=true rebuilds indexes regardless of fingerprint state."""
    repo_url = req.repo_url
    if not repo_url:
        async with AsyncSessionLocal() as session:
            from packages.knowledge.models import RepositoryModel
            from sqlalchemy import select as _select
            repo = (
                await session.execute(_select(RepositoryModel).where(RepositoryModel.id == repository_id))
            ).scalar_one_or_none()
            if repo is None:
                raise HTTPException(status_code=404, detail="Repository not found")
            repo_url = repo.repo_url

    run_id = job_runner.create_job(repo_url)
    mode = "INCREMENTAL" if (req.incremental and not req.force) else "FULL"
    background_tasks.add_task(
        job_runner.execute_job, run_id, mode, req.force, req.generate_artifact
    )
    return {
        "run_id": run_id,
        "repository_id": repository_id,
        "mode": mode,
        "force": req.force,
        "status": "PENDING",
    }


@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Graceful cancellation (Phase 43.5): stops new work, persists CANCELLED status."""
    ok = job_runner.cancel_job(run_id, reason="cancel requested via API")
    if not ok:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return {"run_id": run_id, "cancellation_requested": True}


@app.get("/runs/{run_id}")
async def get_run_detail(run_id: str):
    """Full pipeline run record incl. stage-level progress (Phase 40.2/40.3)."""
    detail = job_runner.get_job_status(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run ID not found")
    job = job_runner.jobs.get(run_id, {})
    result = job.get("result") or {}
    return {
        **detail,
        "lifecycle": result.get("lifecycle"),
        "stage_durations": result.get("stage_durations"),
        "incremental_report": result.get("incremental_report"),
        "artifact_counts": result.get("artifact_counts"),
    }




