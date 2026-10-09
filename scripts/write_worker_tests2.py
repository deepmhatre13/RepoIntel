"""Generate the Celery worker task test file."""
import os

base = r"D:\RepoIntel\tests"

TASKS_TESTS = '''"""Tests for Celery tasks (packages.worker.tasks)."""

import uuid
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "1")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")

import importlib
import packages.worker.celery_app as _celery_app_module
import packages.worker.tasks as _tasks_module

importlib.reload(_celery_app_module)
importlib.reload(_tasks_module)

celery_app = _celery_app_module.celery_app
tasks = _tasks_module


def test_ingest_repository_task_exists():
    assert hasattr(tasks, "ingest_repository")
    assert callable(tasks.ingest_repository)


def test_parse_repository_task_exists():
    assert hasattr(tasks, "parse_repository")
    assert callable(tasks.parse_repository)


def test_index_repository_task_exists():
    assert hasattr(tasks, "index_repository")
    assert callable(tasks.index_repository)


def test_all_tasks_registered():
    registered = [t.name for t in celery_app.tasks.values()]
    assert "packages.worker.tasks.ingest_repository" in registered
    assert "packages.worker.tasks.parse_repository" in registered
    assert "packages.worker.tasks.index_repository" in registered
    assert "packages.worker.tasks.run_repository_analysis" in registered


def test_tasks_have_retry_config():
    for name in [
        "packages.worker.tasks.ingest_repository",
        "packages.worker.tasks.parse_repository",
        "packages.worker.tasks.index_repository",
        "packages.worker.tasks.run_repository_analysis",
    ]:
        task = celery_app.tasks[name]
        assert task.max_retries >= 1


def test_tasks_are_json_serialized():
    conf = celery_app.conf
    assert "json" in conf.accept_content


def test_async_ingest_is_importable():
    assert hasattr(tasks, "_async_ingest")
    assert callable(tasks._async_ingest)


def test_async_parse_is_importable():
    assert hasattr(tasks, "_async_parse")
    assert callable(tasks._async_parse)


def test_async_index_is_importable():
    assert hasattr(tasks, "_async_index")
    assert callable(tasks._async_index)


def test_persist_run_state_is_importable():
    assert hasattr(tasks, "_persist_run_state")
    assert callable(tasks._persist_run_state)


def test_load_run_is_importable():
    assert hasattr(tasks, "_load_run")
    assert callable(tasks._load_run)
'''

with open(os.path.join(base, "test_worker_tasks.py"), "w") as f:
    f.write(TASKS_TESTS)

print("Task tests written.")
