"""Celery application for RepoIntel worker.

This module creates and configures the Celery instance that powers all background
work. The API submits tasks through ``packages.worker.dispatch``; workers execute
them via ``packages.worker.tasks.*``.

Environment:
    CELERY_BROKER_URL  Redis broker URL (default redis://localhost:6379/0)
    CELERY_RESULT_BACKEND  Result backend (default redis://localhost:6379/1)
    WORKER_CONCURRENCY  Max concurrent worker processes (default 2)
"""

import os

from celery import Celery

DEFAULT_BROKER_URL = "redis://localhost:6379/0"
DEFAULT_RESULT_BACKEND = "redis://localhost:6379/1"


def broker_url() -> str:
    """Canonical broker URL resolution for BOTH the API process and the worker.

    A durable task queue only works if the producer (API) and the consumer
    (worker) agree on the broker. When the two sides resolve their broker from
    different sources they silently disagree: the API publishes to a broker no
    worker consumes and the run is stranded in its pre-dispatch state even
    though ``POST /repositories/{id}/analyze`` returned 200.

    Resolution order (identical for both sides):
        1. ``CELERY_BROKER_URL`` environment variable
        2. the loaded RepoIntel settings layer (the project ``.env``)
        3. ``DEFAULT_BROKER_URL``
    """
    value = os.getenv("CELERY_BROKER_URL")
    if value:
        return value
    try:
        from packages.shared.config import settings

        value = settings.get("CELERY_BROKER_URL")
    except Exception:  # configuration is optional for pure unit usage
        value = None
    return value or DEFAULT_BROKER_URL


def result_backend_url() -> str:
    """Result backend resolution (same order as :func:`broker_url`)."""
    value = os.getenv("CELERY_RESULT_BACKEND")
    if value:
        return value
    try:
        from packages.shared.config import settings

        value = settings.get("CELERY_RESULT_BACKEND")
    except Exception:  # configuration is optional for pure unit usage
        value = None
    return value or DEFAULT_RESULT_BACKEND


def _max_execution_seconds() -> int:
    """Application-layer execution budget from the environment."""
    try:
        return int(os.getenv("MAX_EXECUTION_SECONDS", "3600"))
    except (TypeError, ValueError):
        return 3600


def _celery_hard_limit() -> int:
    """Celery hard SIGKILL safety net: always exceeds the app budget (budget*1.25).

    The application layer enforces the *real* deadline via ResourceManager
    (Phase 42). Celery's time_limit is a last-resort backstop so a runaway
    task cannot escape the worker; it must therefore always be *larger* than
    the budget the orchestrator is willing to spend, otherwise Celery kills a
    repository the budget would still permit (the original E2E failure).
    """
    return int(_max_execution_seconds() * 1.25)


def _celery_soft_limit() -> int:
    """Soft limit == the app budget; triggers a graceful SoftTimeLimitExceeded
    the task can catch and persist as a terminal FAILED run."""
    return _max_execution_seconds()




celery_app = Celery(
    "repointel_worker",
    broker=broker_url(),
    backend=result_backend_url(),
    include=[
        "packages.worker.tasks",
    ],
)

celery_app.conf.update(
    task_always_eager=os.getenv("CELERY_TASK_ALWAYS_EAGER", "0").lower() in ("1", "true", "yes"),
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=_celery_hard_limit(),
    task_soft_time_limit=_celery_soft_limit(),
    worker_concurrency=int(os.getenv("WORKER_CONCURRENCY", "2")),
    worker_prefetch_multiplier=1,
    result_expires=3600 * 24,
    result_extended=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_queue="repointel",
    task_queues={
        "repointel": {"exchange": "repointel", "routing_key": "repointel"},
        "ingestion": {"exchange": "repointel", "routing_key": "ingestion"},
        "parsing": {"exchange": "repointel", "routing_key": "parsing"},
        "indexing": {"exchange": "repointel", "routing_key": "indexing"},
        "agent": {"exchange": "repointel", "routing_key": "agent"},
    },
    task_routes={
        "packages.worker.tasks.ingest_repository": {"queue": "ingestion"},
        "packages.worker.tasks.parse_repository": {"queue": "parsing"},
        "packages.worker.tasks.index_repository": {"queue": "indexing"},
        # Canonical production task for normal API-triggered analysis.
        "packages.worker.tasks.run_repository_analysis": {"queue": "repointel"},
    },
)


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    pass


if __name__ == "__main__":
    celery_app.start()

