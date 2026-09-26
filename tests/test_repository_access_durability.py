"""Ownership/access durability tests for repository_access.

These exercise the real defect that previously surfaced as a PostgreSQL
``UNIQUE constraint failed: repository_access.user_id, repository_access.repository_id``
during repository creation. The root cause was never the constraint: the API
created an ownership row on *every* create/analyze call, so a repeated request
for the same repository attempted a second INSERT of an identical relationship.

The invariant these tests pin down:

    * ``(user_id, repository_id)`` is unique at the DATABASE level, so no code
      path can create a duplicate relationship even under concurrency.
    * ``RepositoryAccessModel.upsert`` is the sanctioned write path and is
      idempotent: the second call returns the same row, does not raise, and does
      not insert.
    * A duplicate relationship inserted out-of-band (simulating a double-click
      race that bypassed ``upsert``) is rejected by the database, not silently
      accepted.
"""

import asyncio
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from packages.auth.service import hash_password
from packages.knowledge.db import AsyncSessionLocal
from packages.auth.sqlalchemy_models import RepositoryAccessModel, UserModel
from packages.knowledge.models import RepositoryModel


def _run(coro):
    return asyncio.run(coro)


async def _make_user_and_repo(session) -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:10]
    user = UserModel(
        id=str(uuid.uuid4()),
        username=f"owner_{suffix}",
        email=f"owner_{suffix}@test.local",
        hashed_password=hash_password("testpass123"),
        role="analyst",
    )
    repo = RepositoryModel(
        id=str(uuid.uuid4()),
        repo_url=f"https://github.com/example/ownership-{suffix}",
        status="CREATED",
    )
    session.add(user)
    session.add(repo)
    await session.commit()
    return user.id, repo.id


@pytest.mark.integration
def test_upsert_is_idempotent_and_returns_single_row():
    async def scenario():
        async with AsyncSessionLocal() as session:
            user_id, repo_id = await _make_user_and_repo(session)
            try:
                first = await RepositoryAccessModel.upsert(session, user_id, repo_id, "OWNER")
                await session.commit()
                second = await RepositoryAccessModel.upsert(session, user_id, repo_id, "OWNER")
                await session.commit()

                # Same durable row, never a second relationship.
                assert first.id == second.id
                count = (
                    await session.execute(
                        select(func.count())
                        .select_from(RepositoryAccessModel)
                        .where(
                            RepositoryAccessModel.user_id == user_id,
                            RepositoryAccessModel.repository_id == repo_id,
                        )
                    )
                ).scalar_one()
                assert count == 1
            finally:
                await _cleanup(session, user_id, repo_id)

    _run(scenario())


@pytest.mark.integration
def test_duplicate_relationship_is_rejected_by_database_constraint():
    """A bypassed guard must fail loudly instead of duplicating the relationship.

    The database — not application code — is the last line of defence for
    ownership, which is why the UNIQUE constraint must exist in both the ORM and
    the migration.
    """

    async def scenario():
        async with AsyncSessionLocal() as session:
            user_id, repo_id = await _make_user_and_repo(session)
            try:
                session.add(
                    RepositoryAccessModel(user_id=user_id, repository_id=repo_id, access_level="OWNER")
                )
                await session.commit()

                session.add(
                    RepositoryAccessModel(user_id=user_id, repository_id=repo_id, access_level="OWNER")
                )
                with pytest.raises(IntegrityError):
                    await session.commit()
                await session.rollback()
            finally:
                await _cleanup(session, user_id, repo_id)

    _run(scenario())


async def _cleanup(session, user_id: str, repo_id: str) -> None:
    await session.rollback()
    for row in (
        await session.execute(
            select(RepositoryAccessModel).where(RepositoryAccessModel.user_id == user_id)
        )
    ).scalars().all():
        await session.delete(row)
    repo = (
        await session.execute(select(RepositoryModel).where(RepositoryModel.id == repo_id))
    ).scalar_one_or_none()
    if repo is not None:
        await session.delete(repo)
    user = (
        await session.execute(select(UserModel).where(UserModel.id == user_id))
    ).scalar_one_or_none()
    if user is not None:
        await session.delete(user)
    await session.commit()
