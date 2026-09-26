"""CLI entrypoint for starting a Celery worker.

Usage:
    python -m packages.worker.start_worker
    # or
    celery -A packages.worker.celery_app worker --loglevel=info
"""

import sys

from packages.worker.celery_app import celery_app


def _build_worker_argv() -> list:
    """Build the worker CLI argument vector.

    NOTE: queue selection MUST use the long form ``--queues=<names>``.
    The short form ``-Q=<names>`` is parsed by Click as a SHORT option whose
    value is ``=repointel,...`` (leading '='), which makes the worker consume a
    phantom queue literally named ``=repointel`` while the real ``repointel``
    queue (where the canonical analysis task is routed) is never consumed.
    That silently stranded every API-triggered run in CREATED state.
    """
    concurrency = celery_app.conf.worker_concurrency
    queues = ",".join(celery_app.conf.task_queues.keys())

    argv = [
        "worker",
        "--loglevel=info",
        f"--concurrency={concurrency}",
        f"--queues={queues}",
        "--without-heartbeat",
        "--without-gossip",
        "--without-mingle",
    ]

    if sys.platform == "win32":
        # Windows needs the solo pool (no fork-based multiprocessing)
        argv.append("--pool=solo")

    return argv


def main():
    """Start the Celery worker with sensible defaults."""
    celery_app.worker_main(_build_worker_argv())


if __name__ == "__main__":
    main()

