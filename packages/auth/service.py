"""User persistence service for RepoIntel authentication.

Provides database-backed user lookup, creation, and password verification
using the existing SQLAlchemy async session infrastructure.
"""
import uuid
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


from packages.auth.models import User, UserRole
from packages.auth.sqlalchemy_models import UserModel

import bcrypt
import hashlib
import base64

def hash_password(password: str) -> str:
    """Hash a password using direct bcrypt."""
    pw_bytes = password.encode("utf-8")
    if len(pw_bytes) > 72:
        digest = hashlib.sha256(pw_bytes).digest()
        pw_bytes = base64.b64encode(digest)
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("ascii")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using direct bcrypt."""
    pw_bytes = plain_password.encode("utf-8")
    if len(pw_bytes) > 72:
        digest = hashlib.sha256(pw_bytes).digest()
        pw_bytes = base64.b64encode(digest)
    try:
        return bcrypt.checkpw(pw_bytes, hashed_password.encode("ascii"))
    except Exception:
        return False

logger = logging.getLogger(__name__)


async def get_user_by_id(session: AsyncSession, user_id: str) -> Optional[UserModel]:
    """Look up a user by their UUID string id."""
    result = await session.execute(
        select(UserModel).where(UserModel.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_username(session: AsyncSession, username: str) -> Optional[UserModel]:
    result = await session.execute(
        select(UserModel).where(UserModel.username == username)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(session: AsyncSession, email: str) -> Optional[UserModel]:
    result = await session.execute(
        select(UserModel).where(UserModel.email == email)
    )
    return result.scalar_one_or_none()


async def authenticate_user(
    session: AsyncSession,
    username_or_email: str,
    password: str,
) -> Optional[UserModel]:
    """Authenticate a user by username/email and password.

    Returns None if credentials are invalid (never raises for invalid creds).
    """
    user = await get_user_by_username(session, username_or_email)
    if user is None:
        user = await get_user_by_email(session, username_or_email)
    if user is None:
        return None

    if not user.is_active:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


async def create_user(
    session: AsyncSession,
    username: str,
    email: str,
    password: str,
    role: UserRole = UserRole.VIEWER,
) -> UserModel:
    """Create a new user. Raises ValueError if username or email already exists."""
    existing = await get_user_by_username(session, username)
    if existing is not None:
        raise ValueError(f"Username '{username}' already exists")

    if email:
        existing = await get_user_by_email(session, email)
        if existing is not None:
            raise ValueError(f"Email '{email}' already exists")

    user = UserModel(
        id=str(uuid.uuid4()),
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role.value,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    logger.info("Created user %s", user.username)
    return user


async def user_to_pydantic(user: UserModel) -> User:
    """Convert a UserModel to the pydantic User."""
    return User(
        user_id=user.id,
        username=user.username,
        email=user.email or "",
        role=UserRole(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
    )


async def user_has_repository_access(
    session: AsyncSession,
    user_id: str,
    repository_id: str,
) -> bool:
    """Authoritative repository access check.

    This is the SINGLE source of truth for repository-level authorization and
    is used by both the REST layer (``_access_check``) and the WebSocket
    endpoint (``/ws/{run_id}``). A row in ``repository_access`` for
    (user_id, repository_id) — OWNER or MEMBER — grants access; anything else
    (including no such repository, or no such user) denies.

    Callers must map a ``False`` result to a non-leaking 404/close so the
    existence of a repository or run is never revealed to unauthorized users.
    """
    from packages.auth.sqlalchemy_models import RepositoryAccessModel

    access = (
        await session.execute(
            select(RepositoryAccessModel).where(
                RepositoryAccessModel.repository_id == repository_id,
                RepositoryAccessModel.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    return access is not None