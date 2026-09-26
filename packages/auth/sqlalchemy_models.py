"""Database models for authentication (UserModel, RepositoryAccessModel).

Uses the existing SQLAlchemy Base from packages.knowledge.models so that
auth tables participate in the same Alembic migration infrastructure.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from sqlalchemy import String, Boolean, DateTime, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.knowledge.models import Base


class UserRole(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserModel(Base):
    __tablename__ = "auth_users"

    # Use a fixed UUID string as primary key
    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    email: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default=UserRole.VIEWER.value, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    repository_access: Mapped[List["RepositoryAccessModel"]] = relationship(
        "RepositoryAccessModel",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class RepositoryAccessModel(Base):
    """Repository ownership/access model.

    Maps a user to a repository with an access level.
    OWNER = full access (create, modify, delete)
    MEMBER = read/analyze access

    The composite (user_id, repository_id) is unique — a user can hold exactly
    one access level per repository. This constraint is also declared via a
    migration-created index, but declaring it on the ORM ensures
    ``create_all`` and any schema-driven tooling enforce it as well.
    """
    __tablename__ = "repository_access"

    __table_args__ = (
        UniqueConstraint("user_id", "repository_id",
                         name="uq_repository_access_user_repo"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("auth_users.id"), nullable=False, index=True)
    repository_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False, index=True)
    access_level: Mapped[str] = mapped_column(String, default="OWNER", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user: Mapped["UserModel"] = relationship("UserModel", back_populates="repository_access")
    repository: Mapped["RepositoryModel"] = relationship("RepositoryModel", back_populates="access")

    @classmethod
    async def upsert(
        cls,
        session,
        user_id: str,
        repo_id: str,
        access_level: str = "OWNER",
    ) -> "RepositoryAccessModel":
        """Idempotent, race-safe insert-or-update for a ``(user_id, repo_id)`` row.

        Ownership assignment must be idempotent: the same user legitimately
        creates or analyses the same repository repeatedly (double-click, HTTP
        retry, Celery re-delivery, a second API replica). A SELECT-then-INSERT
        is *not* idempotent under concurrency: two sessions can both observe
        "absent" and both INSERT, so the loser violates the uniqueness rule.

        This helper resolves that race inside a single atomic statement, so the
        *database* decides the winner and no ``IntegrityError`` ever escapes:

            INSERT ... ON CONFLICT (user_id, repository_id) DO UPDATE
                   SET access_level = excluded.access_level

        The ``uq_repository_access_user_repo`` constraint is therefore preserved
        -- it is precisely the mechanism that makes this correct. A savepoint is
        not needed because there is no failure path left to recover from.

        Must be awaited (it executes SQL). The caller owns the transaction and
        commits.
        """
        from sqlalchemy import select

        dialect = session.get_bind().dialect.name
        insert_fn = sqlite_insert if dialect == "sqlite" else pg_insert
        stmt = insert_fn(cls).values(
            user_id=user_id,
            repository_id=repo_id,
            access_level=access_level,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id", "repository_id"],
            set_={"access_level": access_level},
        )
        await session.execute(stmt)

        # Return the durable row so callers never inspect stale identity-map
        # state; the row is guaranteed to exist after the statement above.
        return (
            await session.execute(
                select(cls).where(cls.user_id == user_id, cls.repository_id == repo_id)
            )
        ).scalar_one()