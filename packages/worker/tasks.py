"""Celery tasks for RepoIntel background processing.

Core tasks matching the 4-week plan:
    ingest_repository    Clone + filter + metadata -> IngestionManifest
    parse_repository     AST + symbol graph + chunks
    index_repository     Embed chunks -> vector store
    run_repository_analysis Full API analysis through the canonical orchestrator

Runtime note:
    Celery workers run tasks in worker threads/processes without an event loop.
    The correct pattern for async tasks is to create a new event loop per task
    execution using asyncio.new_event_loop(), NOT asyncio.get_event_loop().
    The latter raises RuntimeError in worker threads.
"""

import asyncio
import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import select

from packages.worker.celery_app import celery_app
from packages.knowledge.db import AsyncSessionLocal
from packages.knowledge.models import (
    RepositoryModel, PipelineRunModel,
)
from packages.knowledge.embeddings import create_embedding_service
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.repository_facade import RepositoryFacade

logger = logging.getLogger(__name__)

_embedding_service = None


def _get_embedding_service():
    """Reuse the lazy embedding service for the lifetime of a worker process."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = create_embedding_service()
    return _embedding_service


# Local shim so tasks.py does not need to import from shared.config at
# decorator-evaluation time (keeps the celery_app import tree lean).
def _env_int(name: str, default: int) -> int:
    import os as _os
    try:
        return int(_os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Celery's time_limit is a HARD safety net (SIGKILL) that exists only to catch
# a runaway task the ResourceManager budget did not stop. It must therefore
# always exceed MAX_EXECUTION_SECONDS (the logical, application-level budget the
# orchestrator enforces) so a legitimately long repository is never killed by
# the transport before the app-layer deadline fires. We read the bound from the
# environment at task-definition time and add a 25% buffer so the app-layer
# budget always wins (Phase 43.7).
_APP_BUDGET = _env_int("MAX_EXECUTION_SECONDS", 3600)
_CELERY_HARD_LIMIT = int(_APP_BUDGET * 1.25)
_CELERY_SOFT_LIMIT = _APP_BUDGET


def _run_async(coro):
    """Run an async coroutine in a new event loop.

    Celery workers have no running event loop in worker threads. Creating a
    fresh loop per task is the correct pattern for async Celery tasks.
    The loop is always closed after execution to avoid resource leaks.

    Two execution modes must both work and both are exercised at runtime:

    * No loop pending in this thread (normal production worker). Drive it.
    * A loop already owning this thread -- Celery's *eager* in-process
      execution (``task_always_eager``) runs the task body from inside
      ``asyncio.run()``. ``run_until_complete`` would then raise
      ``RuntimeError: Cannot run the event loop while another loop is
      running`` and the coroutine would never be awaited, leaving the caller
      with a Celery ``PENDING`` result and no durable run state. Re-entering
      the running loop on the *same* thread (via a separate thread carrying
      the coroutine, then dispatching it back to the running loop) executes
      the task body instead of masking it behind an eager-only failure.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None and running.is_running():
        import threading

        outcome: Dict[str, Any] = {}

        async def _drive():
            try:
                outcome["result"] = await coro
            except BaseException as exc:  # noqa: BLE001 - re-raised in caller
                outcome["error"] = exc

        def _worker():
            # Runs without an event loop: the coroutine object is not executed
            # here, it is only handed to the already-running loop.
            running.call_soon_threadsafe(lambda: asyncio.ensure_future(_drive()))

        thread = threading.Thread(target=_worker, name="repointel-eager-run-async")
        thread.start()
        thread.join()
        # Let the scheduled task run to completion on the active loop.
        while "result" not in outcome and "error" not in outcome:
            time.sleep(0.001)
        if "error" in outcome:
            raise outcome["error"]
        return outcome["result"]

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        # Dispose the async engine's connection pool BEFORE the loop closes.
        #
        # Root cause of the second-dispatch failure ("asyncpg: cannot perform
        # operation: another operation is in progress"): packages.knowledge.db
        # keeps a module-global pooled async engine, and every pooled
        # connection is bound to the event loop that created it. Each Celery
        # task runs on a *fresh* loop here, so connections created by task N
        # became invalid the moment task N's loop closed; task N+1 then pulled
        # those dead connections out of the pool and any concurrent use of
        # them raised InterfaceError. Closing the pool while its owning loop
        # is still alive guarantees the next task creates connections on its
        # own loop. SQLite (aiosqlite) has the same loop-affinity property.
        try:
            from packages.knowledge.db import engine as _engine

            loop.run_until_complete(_engine.dispose())
        except Exception:  # never mask the task result with a teardown error
            logger.warning("engine pool dispose failed during task teardown", exc_info=True)
        loop.close()
        asyncio.set_event_loop(None)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _load_run(session, run_id: str) -> Optional[PipelineRunModel]:
    row = (
        await session.execute(
            select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
        )
    ).scalar_one_or_none()
    return row


async def _persist_run_state(
    session,
    run_id: str,
    *,
    repo_id: Optional[str] = None,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    result_json: Optional[Dict[str, Any]] = None,
):
    row = await _load_run(session, run_id)
    if row is None:
        row = PipelineRunModel(run_id=run_id)
        session.add(row)
    if repo_id is not None:
        row.repository_id = repo_id
    if status is not None:
        row.status = status
    if stage is not None:
        row.current_stage = stage
    if progress is not None:
        row.progress = progress
    if error is not None:
        row.errors_json = json.dumps([{"message": error, "type": "task_error"}])
    if result_json is not None:
        row.report_json = json.dumps(result_json, default=str)
    await session.commit()


# Task 1: Ingest repository

@celery_app.task(
    name="packages.worker.tasks.ingest_repository",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
)
def ingest_repository(self, run_id: str, repo_url: str, repo_id: str) -> Dict[str, Any]:
    """Clone, filter, and extract metadata from a repository."""
    return _run_async(_async_ingest(self, run_id, repo_url, repo_id))


async def _async_ingest(task, run_id: str, repo_url: str, repo_id: str) -> Dict[str, Any]:
    from packages.ingestion.pipeline import run_ingestion_pipeline

    started = time.monotonic()
    async with AsyncSessionLocal() as session:
        await _persist_run_state(
            session, run_id, repo_id=repo_id,
            status="RUNNING", stage="INGESTION", progress=10.0,
        )

    task.update_state(state="INGESTION", meta={"run_id": run_id, "progress": 10})

    try:
        manifest, filtered_files = run_ingestion_pipeline(repo_url)
        duration = round(time.monotonic() - started, 3)

        async with AsyncSessionLocal() as session:
            await _persist_run_state(
                session, run_id,
                stage="INGESTION_COMPLETE", progress=30.0,
                result_json={
                    "manifest": manifest.model_dump() if hasattr(manifest, "model_dump") else str(manifest),
                    "file_count": len(filtered_files) if filtered_files else 0,
                    "duration": duration,
                },
            )

        task.update_state(state="INGESTION_COMPLETE", meta={"run_id": run_id, "progress": 30})

        return {
            "local_path": manifest.local_path if hasattr(manifest, "local_path") else None,
            "repo_id": repo_id,
            "file_count": len(filtered_files) if filtered_files else 0,
            "duration": duration,
            "archetype": getattr(manifest, "archetype", "unknown"),
        }
    except Exception as exc:
        logger.exception("Ingestion failed for %s", run_id)
        async with AsyncSessionLocal() as session:
            await _persist_run_state(
                session, run_id, status="FAILED", error=str(exc),
            )
        raise


# Task 2: Parse repository

@celery_app.task(
    name="packages.worker.tasks.parse_repository",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
)
def parse_repository(self, run_id: str, repo_id: str, local_path: str) -> Dict[str, Any]:
    """Parse the repository into symbols, chunks, and graph edges."""
    return _run_async(_async_parse(self, run_id, repo_id, local_path))


async def _async_parse(task, run_id: str, repo_id: str, local_path: str) -> Dict[str, Any]:
    from packages.parsing.pipeline import run_parsing_pipeline
    from packages.ingestion.filtering import filter_repository

    started = time.monotonic()
    async with AsyncSessionLocal() as session:
        await _persist_run_state(
            session, run_id,
            status="RUNNING", stage="PARSING", progress=35.0,
        )

    task.update_state(state="PARSING", meta={"run_id": run_id, "progress": 35})

    try:
        filtered_files = filter_repository(local_path)
        parsing_result = run_parsing_pipeline(repo_id, local_path, filtered_files)

        symbols_count = len(getattr(parsing_result, "symbols", []))
        chunks_count = len(getattr(parsing_result, "chunks", []))
        edges_count = len(getattr(parsing_result, "symbol_edges", []))

        duration = round(time.monotonic() - started, 3)

        async with AsyncSessionLocal() as session:
            await _persist_run_state(
                session, run_id,
                stage="PARSING_COMPLETE", progress=55.0,
                result_json={
                    "symbols": symbols_count,
                    "chunks": chunks_count,
                    "edges": edges_count,
                    "duration": duration,
                },
            )

        task.update_state(state="PARSING_COMPLETE", meta={"run_id": run_id, "progress": 55})

        return {
            "repo_id": repo_id,
            "symbols_count": symbols_count,
            "chunks_count": chunks_count,
            "edges_count": edges_count,
            "duration": duration,
        }
    except Exception as exc:
        logger.exception("Parsing failed for %s", run_id)
        async with AsyncSessionLocal() as session:
            await _persist_run_state(
                session, run_id, status="FAILED", error=str(exc),
            )
        raise
# Task 3: Index repository (embeddings)

@celery_app.task(
    name="packages.worker.tasks.index_repository",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def index_repository(self, run_id: str, repo_id: str, local_path: str) -> Dict[str, Any]:
    """Embed all parsed chunks and upsert into the vector store."""
    return _run_async(_async_index(self, run_id, repo_id, local_path))


async def _async_index(task, run_id: str, repo_id: str, local_path: str) -> Dict[str, Any]:
    from packages.parsing.pipeline import run_parsing_pipeline
    from packages.ingestion.filtering import filter_repository
    from packages.knowledge.persistence import persist_parsing_results

    started = time.monotonic()
    async with AsyncSessionLocal() as session:
        await _persist_run_state(
            session, run_id,
            status="RUNNING", stage="INDEXING", progress=60.0,
        )

    task.update_state(state="INDEXING", meta={"run_id": run_id, "progress": 60})

    try:
        embedding_service = _get_embedding_service()
        vector_store = VectorStore(embedding_service)
        filtered_files = filter_repository(local_path)
        parsing_result = run_parsing_pipeline(repo_id, local_path, filtered_files)

        chunks = getattr(parsing_result, "chunks", [])
        if chunks:
            vector_store.upsert_chunks(chunks)

        await persist_parsing_results(
            session=session,
            manifest=None,
            parsing_result=parsing_result,
            embedding_service=embedding_service,
            embedding_model=embedding_service.model_name,
            embedding_provider=embedding_service.provider_name,
        )

        duration = round(time.monotonic() - started, 3)

        async with AsyncSessionLocal() as session:
            await _persist_run_state(
                session, run_id,
                stage="INDEXING_COMPLETE", progress=80.0,
                result_json={"chunks_embedded": len(chunks), "duration": duration},
            )

        task.update_state(state="INDEXING_COMPLETE", meta={"run_id": run_id, "progress": 80})

        return {
            "repo_id": repo_id,
            "chunks_embedded": len(chunks),
            "duration": duration,
        }
    except Exception as exc:
        logger.exception("Indexing failed for %s", run_id)
        async with AsyncSessionLocal() as session:
            await _persist_run_state(
                session, run_id, status="FAILED", error=str(exc),
            )
        raise


# ---------------------------------------------------------------------------
# Canonical production task: full analysis via the single pipeline orchestrator.
#
# This is the ONLY task dispatched by packages.worker.dispatch for normal
# API-triggered analysis. It reuses packages.runtime.pipeline_orchestrator
# (the same canonical path the CLI and benchmark use) so there is exactly ONE
# ingestion/parse/index/LangGraph implementation -- no duplicated code.
#
# The orchestrator handles the full lifecycle (CLONING -> INGESTING -> PARSING
# -> INDEXING -> ANALYZING -> READY/FAILED/PARTIAL), persists the durable
# PipelineRunModel AND RepositoryModel (status/last_run_id), cleanly removes the
# temp clone on success/failure, and -- when generate_artifact=True -- runs the
# real LangGraph agent pipeline (Planner -> Retriever -> Synthesis -> Reviewer
# -> bounded reflection) grounded on the RepositoryFacade and persists the
# resulting AgentRunModel + ArtifactModel rows.
# ---------------------------------------------------------------------------

@celery_app.task(
    name="packages.worker.tasks.run_repository_analysis",
    bind=True,
    max_retries=2,
    default_retry_delay=15,
    acks_late=True,
    time_limit=_CELERY_HARD_LIMIT,
    soft_time_limit=_CELERY_SOFT_LIMIT,
)
def run_repository_analysis(
    self,
    run_id: str,
    repo_id: str,
    repo_url: str,
    mode: str = "FULL",
    force: bool = False,
    generate_artifact: bool = True,
    artifact_type: str = "README",
) -> Dict[str, Any]:
    """Run the full repository analysis pipeline through the canonical
    orchestrator. Raises so acks_late + max_retries can retry on transient
    failure; durable lifecycle state is persisted by the orchestrator itself.
    """
    return _run_async(
        _async_run_repository_analysis(
            self, run_id, repo_id, repo_url, mode, force,
            generate_artifact, artifact_type,
        )
    )


async def _async_run_repository_analysis(
    task,
    run_id: str,
    repo_id: str,
    repo_url: str,
    mode: str,
    force: bool,
    generate_artifact: bool,
    artifact_type: str,
) -> Dict[str, Any]:
    from packages.knowledge.embeddings import create_embedding_service
    from packages.knowledge.vector_store import VectorStore
    from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline

    # Idempotency guard: the Celery task id equals the durable run id, and
    # ``acks_late=True`` means a message can legitimately be redelivered (worker
    # crash after side effects, broker redelivery, explicit retry). Re-running a
    # terminal run would reindex an already-indexed repository and create a
    # second artifact, so a redelivery of a run that already reached a terminal
    # state is reported as a no-op instead of being replayed.
    terminal = {"READY", "PARTIAL", "FAILED"}
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
            )
        ).scalar_one_or_none()
    if row is not None and row.status in terminal and not force:
        logger.info(
            "Skipping redelivered run_id=%s (already terminal status=%s)",
            run_id, row.status,
        )
        return {
            "run_id": run_id,
            "repo_id": repo_id,
            "status": row.status,
            "idempotent_replay": True,
        }

    task.update_state(
        state="STARTED", meta={"run_id": run_id, "stage": "INGESTION", "progress": 5}
    )

    # Construct the canonical pipeline with a real embedding service and a
    # process-local vector store that the orchestrator populates during
    # indexing and reuses when the LangGraph agent grounds itself on evidence.
    embedding_service = _get_embedding_service()
    vector_store = VectorStore(embedding_service)
    pipeline = RepositoryAnalysisPipeline(
        embedding_service=embedding_service,
    ).attach_vector_store(vector_store)

    outcome = await pipeline.analyze(
        repo_url=repo_url,
        repo_id=repo_id,
        mode=mode,
        force=force,
        generate_artifact=generate_artifact,
        artifact_type=artifact_type,
        run_id=run_id,
    )

    task.update_state(
        state="SUCCESS",
        meta={
            "run_id": run_id,
            "repo_id": repo_id,
            "status": outcome.get("status"),
        },
    )
    return outcome
