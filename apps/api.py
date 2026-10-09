import os
import json
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, WebSocket, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy import select

from apps.job_runner import job_runner
from packages.knowledge.db import init_db, AsyncSessionLocal
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.repository_facade import RepositoryFacade
from packages.agentic.llm_client import LLMClient
from packages.shared.config import ensure_configured
from packages.ingestion.exceptions import (
    IngestionError,
    InvalidRepositoryError,
    CloneFailedError,
    RepoTooLargeError,
    FilteringError,
    MetadataExtractionError,
    UnsupportedRepoStructureError,
)
from packages.auth.jwt_handler import require_auth, get_current_user
from packages.auth.models import User, UserRole
from packages.auth.router import router as auth_router
from packages.websocket.event_stream import event_stream_manager
from packages.knowledge.models import RepositoryModel

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load configuration, then initialize the schema on
    startup and release resources on shutdown.

    Configuration (the project ``.env``) is loaded **before** ``init_db()`` and
    before any component reads environment-based configuration. This is the
    single place configuration initialization happens -- no other module calls
    ``load_dotenv``.
    """
    ensure_configured()
    await init_db()
    try:
        yield
    finally:
        # Cleanly release any engine/session resources on shutdown.
        from packages.knowledge.db import engine

        await engine.dispose()


cors_origins_env = os.getenv("REPOSITORY_CORS_ORIGINS", "")
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else [
    "http://localhost:5173", "http://127.0.0.1:5173"
]

app = FastAPI(title="RepoIntel API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)


# ---------------------------------------------------------------------------
# Structured error handling (Stage 21):
# API clients always receive JSON errors; raw internal tracebacks never leak.
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": True},
    )


@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, exc: IngestionError):
    """Map typed ingestion exceptions to the structured API error envelope."""
    status_code = 400
    if isinstance(exc, CloneFailedError):
        status_code = 502  # Upstream (git host) failure
    elif isinstance(exc, RepoTooLargeError):
        status_code = 413
    elif isinstance(exc, UnsupportedRepoStructureError):
        status_code = 422
    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc), "error": True, "error_type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Bearer token scheme
# ---------------------------------------------------------------------------
bearer_scheme = HTTPBearer()


# ---------------------------------------------------------------------------
# Access control dependencies
# ---------------------------------------------------------------------------

async def _require_auth(
    role: Optional[UserRole] = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> User:
    """Require authentication with optional role check.
    
    This is the primary dependency for protecting endpoints.
    """
    from packages.auth.jwt_handler import verify_token
    from packages.auth.service import get_user_by_id, user_to_pydantic
    
    payload = verify_token(credentials.credentials)
    async with AsyncSessionLocal() as session:
        user_model = await get_user_by_id(session, payload.sub)
    if user_model is None or not user_model.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await user_to_pydantic(user_model)
    if role and user.role != role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required role: {role.value}",
        )
    return user


async def _grant_ownership(session, user_id: str, repository_id: str) -> None:
    """Idempotently grant OWNER access for (user_id, repository_id).

    Ownership is a security fact: if it cannot be recorded, the repository must
    not be handed back to the caller as if it were theirs. This helper therefore
    propagates repository/transaction failures instead of swallowing them (the
    previous implementation logged an exception and returned success).

    ``RepositoryAccessModel.upsert`` performs a select-then-insert inside the
    caller's transaction, so a repeated request updates the existing row instead
    of violating the ``uq_repository_access_user_repo`` unique constraint.
    """
    from packages.auth.sqlalchemy_models import RepositoryAccessModel

    await RepositoryAccessModel.upsert(session, user_id, repository_id, access_level="OWNER")


def require_user_auth() -> User:
    """Dependency factory returning an async auth dependency for FastAPI."""
    async def _dependency(
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ) -> User:
        return await _require_auth(credentials=credentials)
    return _dependency


async def _access_check(
    repository_id: str,
    current_user: User = Depends(require_user_auth()),
) -> RepositoryModel:
    """Get a repository and verify the current user has access."""
    async with AsyncSessionLocal() as session:
        repo = (
            await session.execute(
                select(RepositoryModel).where(RepositoryModel.id == repository_id)
            )
        ).scalar_one_or_none()
        if repo is None:
            raise HTTPException(status_code=404, detail="Repository not found")
        # Authoritative access check -- shared with the WebSocket endpoint.
        from packages.auth.service import user_has_repository_access
        async with AsyncSessionLocal() as access_session:
            allowed = await user_has_repository_access(
                access_session, current_user.user_id, repository_id
            )
        if not allowed:
            # Not an owner or member — report as "not found" to avoid leaking
            # the existence of the repository to unauthorized users.
            raise HTTPException(status_code=404, detail="Repository not found")
        return repo

class CreateRepoRequest(BaseModel):
    repo_url: str
    force: bool = False

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


class SymbolsResponse(BaseModel):
    repository_id: str
    symbols: List[Dict[str, Any]]


class TemporalDriftResponse(BaseModel):
    repository_id: str
    drift: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    version: str
    llm_available: bool


class LLMStatusResponse(BaseModel):
    mode: str
    llm_available: bool
    provider: Optional[str] = None
    model: Optional[str] = None

# Public health probes (no auth required)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Production health probe with real infrastructure connectivity checks.

    Each dependency reports its *actual* state; the endpoint never fakes a healthy
    status just because a service (PostgreSQL, Redis) is unavailable. When any
    configured dependency is down the overall ``status`` is DEGRADED while the
    individual field still reports exactly what was observed.
    """
    db_status = "OK"
    try:
        from packages.knowledge.db import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await session.execute(select(1))
    except Exception:
        db_status = "UNREACHABLE"

    redis_status = "OK"
    try:
        import redis as redis_client
        from urllib.parse import urlparse

        broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        parsed = urlparse(broker_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        db = int((parsed.path or "/0").strip("/") or 0)
        password = parsed.password
        r = redis_client.Redis(host=host, port=port, db=db, password=password, socket_connect_timeout=2)
        r.ping()
    except Exception:
        redis_status = "UNREACHABLE"

    llm = LLMClient()
    healthy = db_status == "OK" and redis_status == "OK"
    return {
        "status": "OK" if healthy else "DEGRADED",
        "database": db_status,
        "redis": redis_status,
        "version": "1.0.0",
        "llm_available": llm.is_available(),
    }


@app.get("/llm-status", response_model=LLMStatusResponse)
async def llm_status():
    """Report the live LLM mode without making any network call."""
    llm = LLMClient()
    available = llm.is_available()
    return {
        "mode": "LIVE_LLM" if available else "DEGRADED_NO_LLM",
        "llm_available": available,
        "provider": "gemini" if available else None,
        "model": llm.model_name,
    }



@app.post("/repository-groups")
async def create_repository_group(
    req: CreateGroupRequest,
    current_user: User = Depends(require_user_auth()),
):
    group_id = str(uuid.uuid4())
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        res = await facade.create_repository_group(group_id, req.name, req.description)
        return res

@app.post("/repository-groups/{group_id}/members")
async def add_group_member(
    group_id: str,
    req: AddGroupMemberRequest,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        await facade.add_repo_to_group(group_id, req.repository_id)
        return {"status": "SUCCESS", "group_id": group_id, "repository_id": req.repository_id}

@app.post("/repositories")
async def create_repository(
    req: CreateRepoRequest,
    current_user: User = Depends(require_user_auth()),
):
    from packages.ingestion.security import validate_repo_url
    validate_repo_url(req.repo_url)
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, req.repo_url))
    # Repository creation is fully idempotent. Ownership assignment goes through
    # the single shared helper, which delegates to ``RepositoryAccessModel.upsert``
    # (an atomic ON CONFLICT write) rather than an INSERT whose IntegrityError the
    # call site had to guess about. The uniqueness rule stays enforced by the
    # database, and a failed ownership write aborts the request instead of
    # returning a repository the caller does not actually own.
    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(select(RepositoryModel).where(RepositoryModel.id == repo_id))
        ).scalar_one_or_none()
        if existing is None:
            repo = RepositoryModel(id=repo_id, repo_url=req.repo_url)
            session.add(repo)
            await session.flush()

        await _grant_ownership(session, current_user.user_id, repo_id)
        # Single commit for repository + ownership: either both become durable
        # or neither does, so a repository can never exist without an owner.
        await session.commit()
    return {"repository_id": repo_id, "repo_url": req.repo_url}

@app.post("/repositories/{repository_id}/analyze")
async def analyze_repository(
    repository_id: str,
    req: CreateRepoRequest,
    current_user: User = Depends(require_user_auth()),
):
    # Verify access to repository
    await _access_check(repository_id, current_user)
    return await _start_analysis(
        repository_id=repository_id,
        repo_url=req.repo_url,
        mode="FULL",
        force=req.force,
        generate_artifact=True,
        current_user=current_user,
    )


def _is_broker_connectivity_error(exc: Exception) -> bool:
    """Classify whether an exception represents genuine Celery/Redis broker unreachability."""
    from packages.worker.dispatch import DispatchError

    exc_type = type(exc).__name__
    broker_error_types = {
        "OperationalError", "BrokerConnectionError", "ConnectionError",
        "TimeoutError", "RedisError", "socket.error", "OSError"
    }
    if exc_type in broker_error_types:
        return True

    if exc.__cause__ and type(exc.__cause__).__name__ in broker_error_types:
        return True

    msg = str(exc).lower()
    if isinstance(exc, DispatchError) and ("unreachable" in msg or "connection refused" in msg or "connecting to" in msg):
        return True

    return False


async def _start_analysis(
    repository_id: str,
    repo_url: str,
    mode: str,
    force: bool,
    generate_artifact: bool,
    current_user: User,
) -> dict:
    """Persist the durable PipelineRunModel BEFORE dispatching to Celery, then
    submit the canonical analysis task. Returns the run_id the client should
    poll - which is the SAME identity the durable run and the Celery task use.

    Transaction ordering matters: the run row is committed before the task is
    published so the worker can always find the run it was handed. If the
    publish itself fails, the run is durably marked FAILED and the API returns
    a typed error - it must never report success for a run no worker received.
    """
    from packages.knowledge.models import RepositoryModel
    from packages.knowledge.models import PipelineRunModel
    from packages.worker.dispatch import DispatchError

    # Ensure the repository record exists (idempotent with POST /repositories).
    async with AsyncSessionLocal() as session:
        repo = (
            await session.execute(
                select(RepositoryModel).where(RepositoryModel.id == repository_id)
            )
        ).scalar_one_or_none()
        if repo is None:
            repo = RepositoryModel(id=repository_id, repo_url=repo_url, status="CREATED")
            session.add(repo)

        # Duplicate guard: if a non-terminal run already exists for this
        # repository, attach the caller to it instead of queueing a second full
        # pipeline. Re-dispatching would burn worker capacity and race two
        # writers over the same rows. ``force=True`` is an explicit override.
        if not force:
            existing = await job_runner.find_active_run(repository_id)
            if existing is not None:
                return {
                    "run_id": existing.run_id,
                    "status": existing.status,
                    "task_id": existing.task_id,
                    "deduplicated": True,
                }

        run_id = job_runner.create_job(repo_url)

        session.add(
            PipelineRunModel(
                run_id=run_id,
                repository_id=repository_id,
                mode=mode,
                status="CREATED",
                progress=0.0,
            )
        )
        # Commit the durable run row BEFORE the task is enqueued.
        await session.commit()


    celery_task_id = None
    try:
        # Dispatch is a pure function of the durable state committed above: the
        # run identity, the repository identity and the URL are all passed
        # explicitly, so no part of the worker's input lives only in API memory.
        # ``dispatch_durable`` publishes at most once per run_id and records the
        # transport id on this event loop, not on an ad-hoc thread.
        celery_task_id = await job_runner.dispatch_durable(
            run_id=run_id,
            repo_url=repo_url,
            repo_id=repository_id,
            mode=mode,
            force=force,
            generate_artifact=generate_artifact,
        )
    except Exception as exc:
        # Record a real terminal failure rather than leaving it in CREATED.
        from datetime import datetime, timezone
        from packages.shared.secret_redaction import redact_secrets

        exc_type = type(exc).__name__
        is_broker_err = _is_broker_connectivity_error(exc)
        safe_msg = redact_secrets(str(exc)).text

        logger.error(
            "Analysis dispatch failed run_id=%s task_id=%s dispatch_stage=dispatch exception_type=%s error=%s",
            run_id,
            celery_task_id,
            exc_type,
            safe_msg,
        )
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
                )
            ).scalar_one_or_none()
            if row is not None:
                row.status = "FAILED"
                row.current_stage = "DISPATCH"
                row.errors_json = json.dumps([{
                    "type": exc_type,
                    "stage": "dispatch",
                    "message": safe_msg,
                    "recoverable": is_broker_err,
                }])
                row.completed_at = datetime.now(timezone.utc)
                await session.commit()

        if is_broker_err:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Analysis could not be queued: the Celery broker is unreachable. "
                    f"run_id={run_id} has been marked FAILED."
                ),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"Analysis dispatch failed [{exc_type}]: {safe_msg}. "
                    f"run_id={run_id} has been marked FAILED."
                ),
            )

    logger.info("Started analysis run_id=%s task_id=%s", run_id, celery_task_id)
    return {"run_id": run_id, "status": "PENDING", "task_id": celery_task_id}

@app.get("/runs/{run_id}/status")
async def get_run_status(
    run_id: str,
    current_user: User = Depends(require_user_auth()),
):
    persisted = await job_runner.get_persisted_status(run_id)
    if persisted is None:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return {
        "run_id": run_id,
        "status": persisted.get("status"),
        "current_stage": persisted.get("current_stage"),
        "progress": persisted.get("progress"),
        "error": persisted.get("error"),
    }

@app.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    current_user: User = Depends(require_user_auth()),
):
    from packages.knowledge.models import ArtifactModel, AgentRunModel

    async with AsyncSessionLocal() as session:
        artifact = (
            await session.execute(
                select(ArtifactModel, AgentRunModel)
                .join(AgentRunModel, AgentRunModel.run_id == ArtifactModel.run_id)
                .where(ArtifactModel.id == int(artifact_id))
            )
        ).one_or_none()
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        artifact, agent_run = artifact
        await _access_check(agent_run.repo_id, current_user)
        return {
            "artifact_id": artifact.id,
            "run_id": artifact.run_id,
            "artifact_type": artifact.artifact_type,
            "version": artifact.version,
            "created_at": artifact.created_at,
            "content": artifact.content,
        }


@app.get("/repositories/{repository_id}/artifacts")
async def get_repository_artifacts(
    repository_id: str,
    artifact_type: str = "README",
    current_user: User = Depends(require_user_auth()),
):
    """Return the latest persisted artifacts for a repository."""
    from packages.knowledge.models import ArtifactModel, AgentRunModel

    await _access_check(repository_id, current_user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ArtifactModel, AgentRunModel)
            .join(AgentRunModel, AgentRunModel.run_id == ArtifactModel.run_id)
            .where(
                AgentRunModel.repo_id == repository_id,
                ArtifactModel.artifact_type == artifact_type,
            )
            .order_by(ArtifactModel.created_at.desc())
            .limit(10)
        )
        return {
            "repository_id": repository_id,
            "artifacts": [
                {
                    "artifact_id": artifact.id,
                    "run_id": artifact.run_id,
                    "artifact_type": artifact.artifact_type,
                    "version": artifact.version,
                    "created_at": artifact.created_at,
                    "content": artifact.content,
                    "run_status": run.status,
                }
                for artifact, run in result.all()
            ],
        }


@app.post("/repositories/{repository_id}/artifacts/generate")
async def generate_repository_artifact_endpoint(
    repository_id: str,
    artifact_type: str = "README",
    current_user: User = Depends(require_user_auth()),
):
    """Trigger real LangGraph artifact generation for a repository on demand."""
    repo = await _access_check(repository_id, current_user)
    from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline
    from packages.shared import IngestionManifest
    import uuid

    run_id = str(uuid.uuid4())
    pipeline = RepositoryAnalysisPipeline()
    manifest = IngestionManifest(archetype=repo.archetype or "Codebase")
    generated, warning = await pipeline._generate_artifact(
        repo_id=repository_id,
        repo_url=repo.repo_url,
        manifest=manifest,
        artifact_type=artifact_type,
        run_id=run_id,
    )
    return {
        "repository_id": repository_id,
        "run_id": run_id,
        "artifact_type": artifact_type,
        "generated": generated,
        "warning": warning,
    }


@app.post("/repositories/{repository_id}/chat")
async def chat_repository(
    repository_id: str,
    req: ChatRequest,
    current_user: User = Depends(require_user_auth()),
):
    # Verify access to repository
    await _access_check(repository_id, current_user)
    from packages.agentic.agents.qa_analyst import answer_repository_query
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        llm = LLMClient()
        qa_ans = await answer_repository_query(repository_id, req.query, facade, llm)
        return qa_ans.model_dump()

@app.get("/repositories/{repository_id}/call-chain")
async def get_call_chain(
    repository_id: str,
    symbol: str,
    direction: str = "downstream",
    max_depth: int = 5,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        chain = await facade.trace_call_chain(repository_id, symbol, direction, max_depth)
        return {"repository_id": repository_id, "symbol": symbol, "call_chain": chain}

@app.get("/repositories/{repository_id}/commits")
async def get_commits(
    repository_id: str,
    limit: int = 20,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        commits = await facade.get_git_commits(repository_id, limit)
        return {"repository_id": repository_id, "commits": commits}

@app.get("/repositories/{repository_id}/symbols/{symbol}/history")
async def get_symbol_history_api(
    repository_id: str,
    symbol: str,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        history = await facade.get_symbol_history(repository_id, symbol)
        co_changes = await facade.get_co_change_relationships(repository_id, symbol)
        return {"repository_id": repository_id, "symbol": symbol, "history": history, "co_changes": co_changes}

@app.get("/repositories/{repository_id}/architecture")
async def get_architecture_snapshot_api(
    repository_id: str,
    commit_sha: str = "HEAD",
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        snapshot = await facade.create_architecture_snapshot(repository_id, commit_sha)
        return snapshot

@app.get("/repositories/{repository_id}/architecture/compare")
async def compare_architecture_api(
    repository_id: str,
    sha1: str,
    sha2: str,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        diff = await facade.compare_architecture_snapshots(repository_id, sha1, sha2)
        return {"repository_id": repository_id, "sha1": sha1, "sha2": sha2, "diff": diff}

@app.get("/repositories/{repository_id}/hotspots")
async def get_hotspots_api(
    repository_id: str,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        hotspots = await facade.get_hotspots(repository_id)
        return {"repository_id": repository_id, "hotspots": hotspots}

@app.get("/repositories/{repository_id}/coupling")
async def get_coupling_api(
    repository_id: str,
    file_path: str,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        coupling = await facade.get_coupling_metrics(repository_id, file_path)
        return coupling

@app.get("/repositories/{repository_id}/tests/{symbol}")
async def get_tests_for_symbol_api(
    repository_id: str,
    symbol: str,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        tests = await facade.get_tests_for_symbol(repository_id, symbol)
        return {"repository_id": repository_id, "symbol": symbol, "tests": tests}

@app.post("/repositories/{repository_id}/impact")
async def analyze_impact_api(
    repository_id: str,
    req: ImpactRequest,
    current_user: User = Depends(require_user_auth()),
):
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
async def create_change_plan_api(
    repository_id: str,
    req: ImpactRequest,
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        llm = LLMClient()
        from packages.agentic.agents.impact_planner import plan_code_change
        plan = await plan_code_change(repository_id, req.requirement or "Modify component", req.target, facade, llm)
        return plan.model_dump()

@app.get("/repositories/{repository_id}/symbols")
async def get_symbols(
    repository_id: str,
    file_path: str = "",
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        symbols = await facade.get_symbols_for_file(repository_id, file_path)
        return {"repository_id": repository_id, "symbols": symbols}

@app.get("/repositories/{repository_id}/dependencies")
async def get_dependencies(
    repository_id: str,
    current_user: User = Depends(require_user_auth()),
):
    # Verify access to repository
    await _access_check(repository_id, current_user)
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
        deps = await facade.get_dependency_manifest(repository_id)
        return {"repository_id": repository_id, "dependencies": deps}

@app.get("/repositories/{repository_id}/graph")
async def get_graph(
    repository_id: str,
    symbol_or_file: str = "",
    current_user: User = Depends(require_user_auth()),
):
    # Verify access to repository
    await _access_check(repository_id, current_user)
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        neighbors = await facade.symbol_graph_neighbors(repository_id, symbol_or_file)
        return {"repository_id": repository_id, "neighbors": neighbors}


@app.post("/repositories/{repository_id}/copilot")

async def repository_copilot_endpoint(
    repository_id: str,
    req_data: Dict[str, Any],
    current_user: User = Depends(require_user_auth()),
):
    vector_store = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vector_store)
        req_data["repository_id"] = repository_id
        res = await facade.execute_copilot_request(req_data)
        return res


@app.post("/repository-groups/{group_id}/copilot")
async def repository_group_copilot_endpoint(
    group_id: str,
    req_data: Dict[str, Any],
    current_user: User = Depends(require_user_auth()),
):
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



# Register the project health router
try:
    from packages.project_health.api_router import router as project_health_router
    app.include_router(project_health_router)
except ImportError:
    pass

@app.get("/repositories/{repository_id}/status")
async def get_repository_status(
    repository_id: str,
    current_user: User = Depends(require_user_auth()),
):
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


@app.get("/repositories/{repository_id}/understanding")
async def get_repository_understanding_endpoint(
    repository_id: str,
    current_user: User = Depends(require_user_auth()),
):
    """Return real AI project understanding output for Task A."""
    await _access_check(repository_id, current_user)
    from packages.agentic.agents.repository_understanding import generate_repository_understanding
    return await generate_repository_understanding(repository_id)



@app.post("/repositories/{repository_id}/reanalyze")
async def reanalyze_repository(
    repository_id: str,
    req: ReanalyzeRequest,
    current_user: User = Depends(require_user_auth()),
):
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

    mode = "INCREMENTAL" if (req.incremental and not req.force) else "FULL"
    result = await _start_analysis(
        repository_id=repository_id,
        repo_url=repo_url,
        mode=mode,
        force=req.force,
        generate_artifact=req.generate_artifact,
        current_user=current_user,
    )
    return {
        "run_id": result["run_id"],
        "repository_id": repository_id,
        "mode": mode,
        "force": req.force,
        "status": "PENDING",
    }


@app.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    current_user: User = Depends(require_user_auth()),
):
    """Graceful cancellation (Phase 43.5): stops new work, persists CANCELLED status."""
    ok = job_runner.cancel_job(run_id, reason="cancel requested via API")
    if not ok:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return {"run_id": run_id, "cancellation_requested": True}


@app.get("/runs/{run_id}")
async def get_run_detail(
    run_id: str,
    current_user: User = Depends(require_user_auth()),
):
    """Full pipeline run record incl. stage-level progress (Phase 40.2/40.3)."""
    persisted = await job_runner.get_persisted_status(run_id)
    if persisted is None:
        raise HTTPException(status_code=404, detail="Run ID not found")
    # The orchestrator stores the full lifecycle report (incl. stage durations,
    # artifact counts) in PipelineRunModel.report_json.
    full = {
        "run_id": run_id,
        "repository_id": persisted.get("repository_id"),
        "status": persisted.get("status"),
        "current_stage": persisted.get("current_stage"),
        "progress": persisted.get("progress"),
        "error": persisted.get("error"),
    }
    try:
        report = json.loads(persisted.get("report_json", "{}"))
    except (TypeError, ValueError):
        report = {}
    full["lifecycle"] = report.get("lifecycle")
    full["stage_durations"] = report.get("stage_durations")
    full["incremental_report"] = report.get("incremental_report")
    full["artifact_counts"] = report.get("artifact_counts")
    return full




async def _resolve_repo_root(session, repository_id: str) -> Optional[str]:
    """Best-effort local working-tree resolution for local-repo analyses.

    Performs strict realpath resolution to prevent path traversal attempts.
    Returns None when no valid local directory tree can be found.
    """
    from packages.knowledge.models import RepositoryModel

    repo = (await session.execute(
        select(RepositoryModel).where(RepositoryModel.id == repository_id)
    )).scalar_one_or_none()
    if repo is None or not repo.repo_url:
        return None
    try:
        resolved = os.path.realpath(repo.repo_url)
        if os.path.isdir(resolved):
            return resolved
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# WebSocket run-event streaming: authenticated + authorized
# ---------------------------------------------------------------------------
async def _ws_authenticate(token: Optional[str]) -> User:
    """Authenticate a WebSocket client using the EXISTING JWT system.

    Transport note: browsers cannot set an ``Authorization`` header on the
    WebSocket handshake, so the short-lived access token is supplied via the
    ``token`` query parameter (``/ws/{run_id}?token=<jwt>``). This reuses the
    SAME verification and user-resolution path as REST (``verify_token`` +
    ``get_user_by_id``) — no separate auth mechanism exists, and no token
    material is ever logged here.
    """
    from packages.auth.jwt_handler import verify_token
    from packages.auth.service import get_user_by_id, user_to_pydantic

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    payload = verify_token(token)
    async with AsyncSessionLocal() as session:
        user_model = await get_user_by_id(session, payload.sub)
    if user_model is None or not user_model.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return await user_to_pydantic(user_model)


@app.websocket("/ws/{run_id}")
async def websocket_run_events(
    websocket: WebSocket,
    run_id: str,
    token: Optional[str] = None,
    last_seen_sequence: Optional[int] = None,
):
    """Stream typed run events over WebSocket (authenticated + authorized).

    Flow: extract credential -> verify JWT -> resolve user -> resolve run_id
    into PipelineRun -> resolve its repository -> verify repository access via
    the SAME authoritative service used by REST -> accept. Every failure mode
    rejects the connection BEFORE ``accept()`` with one uniform close code, so
    run/repository existence is never disclosed, no events stream, and no
    authorization details leak. Replay after reconnect is handled by the
    existing ``EventStreamManager`` via ``last_seen_sequence``.
    """
    from fastapi import WebSocketDisconnect

    try:
        user = await _ws_authenticate(token)
    except HTTPException:
        # Deliberately vague: same close for missing/invalid/expired tokens.
        await websocket.close(code=1008, reason="authentication required")
        return

    # run_id -> PipelineRun -> repository access. Performed before accept:
    # an unauthorized client learns nothing and receives no events.
    from packages.knowledge.models import PipelineRunModel
    from packages.auth.service import user_has_repository_access

    async with AsyncSessionLocal() as session:
        run = (
            await session.execute(
                select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
            )
        ).scalar_one_or_none()
        repository_id = run.repository_id if run is not None else None
        allowed = (
            await user_has_repository_access(session, user.user_id, repository_id)
            if repository_id is not None
            else False
        )

    if not allowed:
        # Identical close for "unknown run" and "no access" — the two are
        # indistinguishable from the client side.
        await websocket.close(code=1008, reason="not authorized")
        return

    try:
        await event_stream_manager.connect(websocket, run_id, last_seen_sequence)
        # Keep the connection open: server->client events flow through
        # EventStreamManager.emit(); client frames are drained until the
        # client disconnects.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        event_stream_manager.disconnect(websocket, run_id)


