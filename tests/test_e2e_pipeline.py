"""End-to-end pipeline integration tests.

Exercises the production analysis pipeline against a real filesystem
repository on a temporary migrated database:

- repeated FULL analysis is idempotent (no duplicate rows)
- repository lifecycle status persists (READY)
- INCREMENTAL analysis reprocesses only changed content
- forced reindex rebuilds derived knowledge while preserving repository identity
"""

import os
import shutil
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from packages.knowledge import migrations
from packages.knowledge.models import PipelineRunModel, RepositoryModel
from packages.knowledge.persistence import count_repo_artifacts
from packages.knowledge.vector_store import VectorStore
from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline

DATASET = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "python_library"


@pytest_asyncio.fixture
async def env(tmp_path):
    """Migrated temp database + a mutable copy of the sample repository."""
    db_file = tmp_path / "e2e.db"
    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    os.environ["DATABASE_URL"] = url
    migrations.upgrade_to_head()

    engine = create_async_engine(url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    repo_dir = tmp_path / "repo_under_test"
    shutil.copytree(DATASET, repo_dir)

    yield {"tmp_path": tmp_path, "repo_dir": repo_dir,
           "session_factory": session_factory, "engine": engine}

    await engine.dispose()


def _pipeline(env) -> RepositoryAnalysisPipeline:
    return RepositoryAnalysisPipeline(session_factory=env["session_factory"]).attach_vector_store(
        VectorStore()
    )


async def _counts(env, repo_id):
    async with env["session_factory"]() as session:
        return await count_repo_artifacts(session, repo_id)


@pytest.mark.asyncio
async def test_full_analysis_is_idempotent_and_persists_lifecycle(env):
    repo_url = str(env["repo_dir"])
    pipeline = _pipeline(env)

    first = await pipeline.analyze(repo_url=repo_url, mode="FULL")
    assert first["status"] == "READY", first.get("error")
    repo_id = first["repository_id"]
    counts_1 = await _counts(env, repo_id)
    assert counts_1["files"] > 0 and counts_1["symbols"] > 0 and counts_1["chunks"] > 0

    # Second identical analysis must not duplicate persistent entities.
    second = await pipeline.analyze(repo_url=repo_url, mode="FULL")
    assert second["status"] == "READY", second.get("error")
    counts_2 = await _counts(env, repo_id)
    assert counts_1 == counts_2, (counts_1, counts_2)

    # Lifecycle state persisted on both the run record and repository row.
    async with env["session_factory"]() as session:
        repo_row = (
            await session.execute(select(RepositoryModel).where(RepositoryModel.id == repo_id))
        ).scalar_one_or_none()
        assert repo_row is not None
        assert repo_row.status == "READY"
        assert repo_row.last_run_id is not None
        runs = (
            await session.execute(
                select(PipelineRunModel).where(PipelineRunModel.repository_id == repo_id)
            )
        ).scalars().all()
        assert len(runs) >= 2
        assert {r.status for r in runs} <= {"READY"}
        assert len({r.run_id for r in runs}) == len(runs)
@pytest.mark.asyncio
async def test_incremental_update_reprocesses_only_changed_content(env):
    repo_url = str(env["repo_dir"])
    pipeline = _pipeline(env)

    initial = await pipeline.analyze(repo_url=repo_url, mode="FULL")
    assert initial["status"] == "READY"
    repo_id = initial["repository_id"]
    before = await _counts(env, repo_id)

    # Modify exactly one source file.
    target = None
    for p in sorted(env["repo_dir"].rglob("*.py")):
        target = p
        break
    assert target is not None
    original = target.read_text(encoding="utf-8")
    target.write_text(original + "\n\n\ndef brand_new_function():\n    return 42\n", encoding="utf-8")

    outcome = await pipeline.analyze(repo_url=repo_url, mode="INCREMENTAL")
    assert outcome["status"] == "READY", outcome.get("error")
    inc = outcome.get("incremental_report")
    assert inc is not None, "incremental run must produce a diff report"

    rel_changed = target.relative_to(env["repo_dir"]).as_posix()
    assert rel_changed in inc["files_modified"], inc["files_modified"]
    # Unchanged content must not be flagged as modified.
    assert inc["files_unchanged_count"] >= 1

    after = await _counts(env, repo_id)
    # Only the changed file's derived rows were replaced; nothing duplicated.
    assert after["symbols"] > before["symbols"]
    assert after["files"] == before["files"]

    async with env["session_factory"]() as session:
        repo_row = (
            await session.execute(select(RepositoryModel).where(RepositoryModel.id == repo_id))
        ).scalar_one()
        assert repo_row.status == "READY"


@pytest.mark.asyncio
async def test_forced_reindex_rebuilds_without_duplicates(env):
    repo_url = str(env["repo_dir"])
    pipeline = _pipeline(env)

    first = await pipeline.analyze(repo_url=repo_url, mode="FULL")
    repo_id = first["repository_id"]
    before = await _counts(env, repo_id)

    rebuilt = await pipeline.analyze(repo_url=repo_url, mode="FULL", force=True)
    assert rebuilt["status"] == "READY", rebuilt.get("error")

    after = await _counts(env, repo_id)
    assert after == before, (before, after)

    async with env["session_factory"]() as session:
        repo_rows = (
            await session.execute(select(RepositoryModel).where(RepositoryModel.id == repo_id))
        ).scalars().all()
        assert len(repo_rows) == 1, "forced reindex must preserve repository identity"

