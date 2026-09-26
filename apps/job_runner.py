"""Job runner bridge — Celery-backed execution.

Historically (Phase 15) this held an in-process dict and dispatched via FastAPI
BackgroundTasks. The spec (§15) and plan (Day 22) require Celery workers so that
long-running work survives API restarts and runs outside the request path.

The authoritative run identity is the ``run_id`` that the API persists into
``PipelineRunModel`` *before* dispatch. The previous implementation additionally
stored run metadata in a process-local ``self.jobs`` dict and refused to dispatch
a run absent from it (``raise KeyError``). That made dispatch depend on
per-process memory: after an API restart, a worker retry, or with more than one
API replica, a perfectly valid durable run could not be enqueued. It also meant
dispatch silently re-derived ``repo_id`` instead of using the one the caller had
already persisted.

Dispatch is now a pure function of durable inputs:

    create_job(repo_url)                 -> run_id   (identity only, no memory)
    dispatch(run_id, repo_url, repo_id)  -> celery task id
    get_persisted_status(run_id)         -> durable PipelineRunModel view

``repo_id`` and ``repo_url`` are supplied explicitly by the caller, which has
already committed them, so nothing the worker depends on lives only in memory.
"""

import logging
import uuid
from typing import Dict, Any, Optional

from sqlalchemy import select

from packages.knowledge.db import AsyncSessionLocal
from packages.knowledge.models import PipelineRunModel
from packages.worker.dispatch import (
    dispatch_analysis,
    cancel_run as celery_cancel_run,
    get_task_id,
    get_celery_task_status,
)

logger = logging.getLogger(__name__)


class JobRunner:
    """Celery-backed job runner.

    There is deliberately NO in-process registry of live jobs. Authoritative
    state lives in ``PipelineRunModel`` (written by each Celery task) plus the
    Celery result backend for transient execution state. ``self.jobs`` is
    retained only as an empty attribute for backwards compatibility; it is never
    required for dispatch.
    """

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, repo_url: str) -> str:
        """Return a new durable run_id.

        Pure: allocates identity only. The caller persists the corresponding
        ``PipelineRunModel`` row in the transaction it is already using, so there
        is never a window where the API holds a run_id the database does not know.
        """
        return str(uuid.uuid4())

    def dispatch(
        self,
        run_id: str,
        repo_url: str,
        repo_id: str,
        mode: str = "FULL",
        force: bool = False,
        generate_artifact: bool = True,
        artifact_type: str = "README",
    ) -> str:
        """Submit the analysis run to Celery. Returns the Celery task id.

        Every argument the worker needs is passed explicitly from durable state
        the caller committed *before* this call, so no part of a run's inputs
        lives only in API process memory.

        Recording the returned task id is deliberately *not* done here: this is
        a synchronous call made from an async request handler, and persisting
        the id requires async database access. Use :meth:`dispatch_and_record`
        from async code so the write happens on the caller's event loop and
        transaction rather than on an ad-hoc thread.
        """
        return dispatch_analysis(
            repo_url=repo_url,
            run_id=run_id,
            repo_id=repo_id,
            mode=mode,
            force=force,
            generate_artifact=generate_artifact,
            artifact_type=artifact_type,
            idempotency_key=run_id,
        )

    async def dispatch_durable(
        self,
        run_id: str,
        repo_url: str,
        repo_id: str,
        mode: str = "FULL",
        force: bool = False,
        generate_artifact: bool = True,
        artifact_type: str = "README",
    ) -> str:
        """Publish ``run_id`` exactly once, using durable state as the guard.

        The API commits the ``PipelineRunModel`` row before calling this. The
        guard here is that same durable row: if it already carries a transport
        ``task_id``, the run has been published and is NOT published again. That
        matters because a client retry, a double click, or an API redeploy mid
        request must never start a second full pipeline over one run identity.

        The Celery task id equals the run id (see ``dispatch.dispatch_analysis``),
        so recording it is idempotent and makes revocation survive API restarts.
        """
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
                )
            ).scalar_one_or_none()
            if row is not None and row.task_id:
                logger.info(
                    "run_id=%s already published as task_id=%s; not re-dispatching",
                    run_id, row.task_id,
                )
                return row.task_id

        task_id = dispatch_analysis(
            repo_url=repo_url,
            run_id=run_id,
            repo_id=repo_id,
            mode=mode,
            force=force,
            generate_artifact=generate_artifact,
            artifact_type=artifact_type,
            idempotency_key=run_id,
        )
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
                )
            ).scalar_one_or_none()
            if row is not None:
                row.task_id = task_id
                await session.commit()
        return task_id

    async def dispatch_and_record(
        self,
        run_id: str,
        repo_url: str,
        repo_id: str,
        mode: str = "FULL",
        force: bool = False,
        generate_artifact: bool = True,
        artifact_type: str = "README",
        session=None,
    ) -> str:
        """Publish the task and persist its transport id on the caller's loop.

        The durable run row (``PipelineRunModel``) was committed by the caller
        before dispatch, so the follow-up ``task_id`` write is a small UPDATE on
        an existing row. A failure to record the id degrades cancellation only
        and must not undo a successfully queued run, so it is logged and
        swallowed *after* the run has genuinely been published.

        ``session`` lets an async caller pass the session that already owns the
        committed run row, avoiding a second connection acquisition and keeping
        the ``task_id`` write on the same identity map as the caller's objects.
        """
        task_id = dispatch_analysis(
            repo_url=repo_url,
            run_id=run_id,
            repo_id=repo_id,
            mode=mode,
            force=force,
            generate_artifact=generate_artifact,
            artifact_type=artifact_type,
            idempotency_key=run_id,
        )
        try:
            if session is not None:
                row = (
                    await session.execute(
                        select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
                    )
                ).scalar_one_or_none()
                if row is not None:
                    row.task_id = task_id
                    await session.commit()
            else:
                async with AsyncSessionLocal() as own_session:
                    row = (
                        await own_session.execute(
                            select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
                        )
                    ).scalar_one_or_none()
                    if row is not None:
                        row.task_id = task_id
                        await own_session.commit()
        except Exception as exc:  # noqa: BLE001 - cancellation degrades, run stands
            logger.warning("Could not persist task_id for run_id=%s: %s", run_id, exc)
        return task_id

    async def find_active_run(self, repo_id: str) -> Optional[PipelineRunModel]:
        """Return a non-terminal run for ``repo_id``, if one exists.

        Duplicate-analysis guard: the API consults this before creating another
        run so a double-click (or a retrying client) attaches to the run that is
        already queued/running instead of enqueueing a second full pipeline over
        the same repository. Terminal states are deliberately excluded; a run
        that failed or finished must not block a legitimate re-analysis.
        """
        terminal = {"COMPLETE", "FAILED", "PARTIAL", "CANCELLED", "READY"}
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(PipelineRunModel)
                    .where(PipelineRunModel.repository_id == repo_id)
                    .order_by(PipelineRunModel.started_at.desc())
                )
            ).scalars().all()
        for row in rows:
            if (row.status or "").upper() not in terminal:
                return row
        return None

    def _record_task_id(self, run_id: str, task_id: str) -> None:
        """Deprecated synchronous shim.

        Persisting a task id needs async database access, and this method used to
        open a *thread* with its own event loop when called from async code. That
        is exactly the kind of hidden async boundary this codebase avoids. The
        task id now equals the run id, so no separate recording step exists at
        all; this shim remains so any stale synchronous caller fails loudly
        rather than silently corrupting state.
        """
        raise RuntimeError(
            "JobRunner._record_task_id is not async-safe; the Celery task id now "
            "equals the run id, so no recording step is required. Use "
            "await job_runner.dispatch_and_record(...) from async code."
        )

    def cancel_job(self, run_id: str, reason: str = "", task_id: Optional[str] = None) -> bool:
        """Revoke a run's Celery task using its durable task id when available."""
        return celery_cancel_run(run_id, task_id=task_id)

    async def execute_job(
        self,
        run_id: str,
        repo_url: str,
        repo_id: str,
        mode: str = "FULL",
        force: bool = False,
        generate_artifact: bool = True,
    ) -> str:
        """Durability-first dispatch used by the API.

        Ensures a committed ``PipelineRunModel`` row exists for ``run_id`` and
        only then publishes the Celery task. Returns the Celery task id.
        """
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
                )
            ).scalar_one_or_none()
            if row is None:
                session.add(
                    PipelineRunModel(
                        run_id=run_id,
                        repository_id=repo_id,
                        mode=mode,
                        status="CREATED",
                        progress=0.0,
                    )
                )
                await session.commit()
        return await self.dispatch_and_record(
            run_id=run_id,
            repo_url=repo_url,
            repo_id=repo_id,
            mode=mode,
            force=force,
            generate_artifact=generate_artifact,
        )

    def get_job_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Transient Celery-side view. Use ``get_persisted_status`` for truth."""
        task_id = get_task_id(run_id)
        if task_id is None:
            return None
        celery_info = get_celery_task_status(task_id)
        return {
            "run_id": run_id,
            "task_id": task_id,
            "celery_state": celery_info.get("state"),
            "celery_info": celery_info.get("info"),
        }

    async def get_persisted_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Read authoritative status directly from PipelineRunModel."""
        async with AsyncSessionLocal() as session:
            row = (
                await session.execute(
                    select(PipelineRunModel).where(PipelineRunModel.run_id == run_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "run_id": row.run_id,
                "repository_id": row.repository_id,
                "task_id": row.task_id,
                "status": row.status,
                "current_stage": row.current_stage,
                "progress": row.progress,
                "error": row.errors_json,
                "errors_json": row.errors_json,
                "report_json": row.report_json,
            }


job_runner = JobRunner()
