"""Tests for the Celery dispatch bridge (packages.worker.dispatch)."""

import uuid
import os
from unittest.mock import MagicMock, patch, call
import pytest

os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "1")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")

import importlib
import packages.worker.celery_app as _celery_app_module
import packages.worker.dispatch as _dispatch_module

importlib.reload(_celery_app_module)
importlib.reload(_dispatch_module)

celery_app = _celery_app_module.celery_app
dispatch = _dispatch_module


@pytest.fixture(autouse=True)
def _reset_dispatch_state():
    dispatch._run_to_task.clear()
    yield
    dispatch._run_to_task.clear()


def test_celery_app_configured():
    assert celery_app.main == "repointel_worker"
    conf = celery_app.conf
    assert "repointel" in conf.task_queues
    assert "ingestion" in conf.task_queues
    assert "parsing" in conf.task_queues
    assert "indexing" in conf.task_queues
    assert "agent" in conf.task_queues


def test_task_routing():
    routes = celery_app.conf.task_routes
    assert routes["packages.worker.tasks.ingest_repository"]["queue"] == "ingestion"
    assert routes["packages.worker.tasks.parse_repository"]["queue"] == "parsing"
    assert routes["packages.worker.tasks.index_repository"]["queue"] == "indexing"
    assert routes["packages.worker.tasks.run_repository_analysis"]["queue"] == "repointel"


def test_start_repository_analysis_returns_run_id():
    run_id = dispatch.start_repository_analysis(
        repo_url="https://github.com/example/repo",
        mode="FULL",
        force=True,
        generate_artifact=True,
    )
    assert isinstance(run_id, str)
    assert len(run_id) == 36  # UUID4 string length


def test_start_repository_analysis_maps_run_to_task():
    run_id = dispatch.start_repository_analysis(
        repo_url="https://github.com/example/repo",
        generate_artifact=True,
    )
    task_id = dispatch.get_task_id(run_id)
    assert task_id is not None
    assert isinstance(task_id, str)


def test_start_analysis_without_artifact():
    run_id = dispatch.start_repository_analysis(
        repo_url="https://github.com/example/repo",
        generate_artifact=False,
    )
    assert run_id in dispatch._run_to_task


def test_cancel_run_returns_true_for_known_run():
    run_id = dispatch.start_repository_analysis(
        repo_url="https://github.com/example/repo",
        generate_artifact=False,
    )
    result = dispatch.cancel_run(run_id)
    assert result is True


def test_cancel_run_returns_false_for_unknown_run():
    result = dispatch.cancel_run(str(uuid.uuid4()))
    assert result is False


def test_get_task_id_returns_none_for_unknown():
    assert dispatch.get_task_id(str(uuid.uuid4())) is None


def test_get_celery_task_status_returns_dict():
    run_id = dispatch.start_repository_analysis(
        repo_url="https://github.com/example/repo",
        generate_artifact=False,
    )
    task_id = dispatch.get_task_id(run_id)
    status = dispatch.get_celery_task_status(task_id)
    assert "task_id" in status
    assert "state" in status
    assert "info" in status
    assert "result" in status


def test_repo_id_from_url_is_deterministic():
    url = "https://github.com/example/repo"
    id1 = dispatch._repo_id_from_url(url)
    id2 = dispatch._repo_id_from_url(url)
    assert id1 == id2
    assert id1 == str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def test_different_urls_produce_different_ids():
    id1 = dispatch._repo_id_from_url("https://github.com/example/repo1")
    id2 = dispatch._repo_id_from_url("https://github.com/example/repo2")
    assert id1 != id2
