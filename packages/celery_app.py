"""Celery application factory for RepoIntel background workers.

This module creates the Celery app that powers the background worker processes.
It is the ONLY sanctioned entry point for long-running work -- the API never
runs analysis, parsing, indexing, or security scanning inside a request handler.

Architecture:
    API (FastAPI)  --dispatch-->  Redis broker  <--consume--  Celery worker
         |                                                      |
         |                                                      v
    PipelineRunModel (DB)  <--status updates---  packages.workers.repository_tasks

The worker creates its own DB session and pipeline per task -- no shared state
with the API process. Job status is persisted to PipelineRunModel so it survives
API restarts and is visible to any number of API instances.

Configuration is environment-driven:
    CELERY_BROKER_URL   Redis broker URL (default: redis://localhost:6379/0)
    CELERY_RESULT_BACKEND  Result backend (default: same as broker)
    CELERY_TASK_SOFT_TIME_LIMIT  Soft limit in seconds (default: 840 = 14min)
    CELERY_TASK_HARD_TIME_LIMIT  Hard limit in seconds (default: 900 = 15min)
"""
import os
import logging

logger = logging.getLogger(__name__)


def _get_broker_url() -> str:
    """Return the Redis broker URL.

    The default targets ``localhost`` so the worker runs against a native/local
    Redis without Docker. Docker Compose overrides this via ``CELERY_BROKER_URL``
    (redis://redis:6379/0) in its own environment, keeping the two workflows
    independent. The broker remains a real external Redis service -- it is never
    replaced by an in-memory stand-in.
    """
    return os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")


def _get_result_backend() -> str:
    """Return the result backend URL, defaulting to the broker URL."""
    return os.getenv("CELERY_RESULT_BACKEND", _get_broker_url())


def _get_soft_time_limit() -> int:
    """Soft time limit (seconds) -- task gets a SoftTimeLimitExceeded warning."""
    return int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "840"))


def _get_hard_time_limit() -> int:
    """Hard time limit (seconds) -- task is killed."""
    return int(os.getenv("CELERY_TASK_HARD_TIME_LIMIT", "900"))


def create_celery_app():
    """Create and configure the Celery application.

    This factory is called once per worker process. It is safe to call multiple
    times (e.g., in tests) -- each call returns a fresh app instance.
    """
    from celery import Celery

    app = Celery("repointel")

    # Use Redis as both broker and result backend.
    app.conf.broker_url = _get_broker_url()
    app.conf.result_backend = _get_result_backend()

    # Serialization: JSON only (no pickle -- security posture).
    app.conf.task_serializer = "json"
    app.conf.result_serializer = "json"
    app.conf.accept_content = ["json"]

    # Reliability: late acknowledgment (task is re-queued if worker dies mid-task).
    app.conf.task_acks_late = True
    app.conf.worker_prefetch_multiplier = 1

    # Time limits: prevent runaway tasks from consuming the worker forever.
    app.conf.task_soft_time_limit = _get_soft_time_limit()
    app.conf.task_time_limit = _get_hard_time_limit()

    # Task routing: all repository tasks go to the default queue.
    # (Future: separate queues for security, indexing, etc.)
    app.conf.task_default_queue = "repointel"

    # Autodiscover tasks from the workers package.
    app.autodiscover_tasks(["packages.workers"], force=True)

    logger.info(
        "Celery app created: broker=%s, soft_limit=%ds, hard_limit=%ds",
        app.conf.broker_url,
        app.conf.task_soft_time_limit,
        app.conf.task_time_limit,
    )
    return app


# Module-level singleton -- imported by the celery command-line:
#   celery -A packages.celery_app.celery_app worker
celery_app = create_celery_app()
