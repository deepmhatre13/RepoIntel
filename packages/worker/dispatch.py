"""Celery dispatch bridge for the RepoIntel API.

This module is the ONLY place in the codebase that submits Celery tasks
on behalf of the API. The API (apps/api.py) calls into this module; it
never imports ``celery`` directly.

Dispatch submits a single canonical task
``packages.worker.tasks.run_repository_analysis`` which runs the full pipeline
through ``RepositoryAnalysisPipeline`` (the same canonical orchestrator the CLI
and benchmark use) - there is no parallel, duplicated ingestion implementation.

Why this exists:
    - Keeps Celery coupling out of the API layer (spec boundary rule).
    - Gives us a single place to translate run_id <-> Celery task_id.
    - Makes the dispatch path trivially testable with ``task_always_eager``.

The caller supplies the run_id/repo_id so the identity returned to the user is
the same durable identity used by PipelineRunModel (see start_repository_analysis).
"""

import logging
import uuid
from typing import Any, Dict, Optional

from packages.worker.celery_app import celery_app
from packages.worker import tasks as worker_tasks

logger = logging.getLogger(__name__)

# Maps our public run_id -> Celery task_id. This is a *cache*, not a source of
# truth: it exists only so a single API process can revoke a task it just
# published. Durable status always comes from PipelineRunModel; the mapping is
# empty after an API restart, which is why cancellation treats "unknown task id"
# as "revoke by durable run id is unavailable" rather than "cannot cancel".
_run_to_task: Dict[str, str] = {}


class DispatchError(RuntimeError):
    """Raised when a run cannot be handed to the durable Celery queue.

    A dispatch failure must NEVER be swallowed: callers translate it into a
    terminal FAILED run state (plus a typed HTTP error) so a run can never sit
    in CREATED while no worker will ever pick it up.
    """


def _new_run_id() -> str:
    return str(uuid.uuid4())


def verify_broker_reachable(timeout_seconds: float = 5.0) -> None:
    """Fail fast (typed) when the configured broker cannot accept the task.

    ``apply_async`` on a Kombu connection that cannot be established defaults to
    retrying in the background for tens of seconds and then reports the run as
    ``PENDING``. That makes an unreachable broker look like a queued run, which
    is exactly the failure mode the API must not present to the user. We probe
    the broker once, bounded, and raise :class:`DispatchError` instead.
    """
    conn = celery_app.connection(connect_timeout=timeout_seconds)
    try:
        conn.ensure_connection(max_retries=1, timeout=timeout_seconds)
    except Exception as exc:  # kombu/redis/OS level
        raise DispatchError(
            f"Celery broker {celery_app.conf.broker_url!r} is unreachable: {exc}"
        ) from exc
    finally:
        try:
            conn.release()
        except Exception:
            pass


def dispatch_analysis(
    repo_url: str,
    run_id: str,
    repo_id: str,
    mode: str = "FULL",
    force: bool = False,
    generate_artifact: bool = True,
    artifact_type: str = "README",
    idempotency_key: Optional[str] = None,
) -> str:
    """Enqueue full repository analysis and return the **Celery task id**.

    Identity contract:

    * ``run_id``   -- the durable ``PipelineRunModel`` primary key. Owned by the
      caller, which has already committed the row. It is the only identity the
      user ever sees.
    * ``task_id``  -- the Celery message id. Transport-level identity; used for
      revocation and for reading transient Celery state. It is NOT a run id.

    ``repo_url`` and ``repo_id`` are passed explicitly because the worker must be
    able to execute purely from the message payload plus committed database
    state -- no part of the run's inputs may live only in API process memory.

    Raises :class:`DispatchError` on an unreachable broker or a failed publish so
    an enqueue failure can never be presented as a queued run.
    """
    # A durable queue requires a reachable broker. Fail with a typed error
    # instead of returning a run whose task nobody will ever consume.
    verify_broker_reachable()

    # The Celery task id IS the run id. This makes redelivery idempotent at the
    # transport level as well: a duplicate publish of the same run_id produces
    # the same task id, so the worker can detect and skip a repeat delivery
    # instead of creating a second analysis pass over an already-indexed
    # repository. It also removes the need for a process-local run->task map,
    # which could not survive an API restart.
    task_id = run_id
    task_sig = worker_tasks.run_repository_analysis.s(
        run_id=run_id,
        repo_id=repo_id,
        repo_url=repo_url,
        mode=mode,
        force=force,
        generate_artifact=generate_artifact,
        artifact_type=artifact_type,
    )
    try:
        result = task_sig.apply_async(task_id=task_id)
    except DispatchError:
        raise
    except Exception as exc:  # publish failure must not surface as success
        raise DispatchError(f"Celery publish failed for run_id={run_id}: {exc}") from exc

    _run_to_task[run_id] = result.id
    logger.info(
        "Dispatched canonical analysis for run_id=%s repo_id=%s task_id=%s "
        "generate_artifact=%s mode=%s",
        run_id, repo_id, result.id, generate_artifact, mode,
    )
    return result.id


def start_repository_analysis(
    repo_url: str,
    run_id: Optional[str] = None,
    repo_id: Optional[str] = None,
    mode: str = "FULL",
    force: bool = False,
    generate_artifact: bool = True,
    artifact_type: str = "README",
) -> str:
    """Backwards-compatible wrapper that returns the **run_id**.

    Retained for callers that only need the durable identity (CLI, benchmark,
    tests). New API code uses :func:`dispatch_analysis` and persists the returned
    task id so revocation survives an API restart.
    """
    run_id = run_id or _new_run_id()
    repo_id = repo_id or _repo_id_from_url(repo_url)
    dispatch_analysis(
        repo_url=repo_url,
        run_id=run_id,
        repo_id=repo_id,
        mode=mode,
        force=force,
        generate_artifact=generate_artifact,
        artifact_type=artifact_type,
    )
    return run_id


def cancel_run(run_id: str, task_id: Optional[str] = None) -> bool:
    """Revoke a run's Celery task.

    ``task_id`` should be supplied by the caller from durable state (the API
    persists it on ``PipelineRunModel.task_id``). The in-process map is consulted
    only as a fallback. Returns False when no task id is known at all -- the
    caller must not report a successful cancellation in that case.
    """
    resolved = task_id or _run_to_task.get(run_id)
    if resolved is None:
        return False
    celery_app.control.revoke(resolved, terminate=True)
    return True


def get_task_id(run_id: str) -> Optional[str]:
    """Return the Celery task_id for a given run_id, if known."""
    return _run_to_task.get(run_id)


def get_celery_task_status(task_id: str) -> Dict[str, Any]:
    """Return the current Celery AsyncResult state and metadata."""
    from celery.result import AsyncResult
    res = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "state": res.state,
        "info": res.info,
        "result": res.result if res.ready() else None,
    }


def _repo_id_from_url(repo_url: str) -> str:
    import uuid as _uuid
    return str(_uuid.uuid5(_uuid.NAMESPACE_URL, repo_url))
