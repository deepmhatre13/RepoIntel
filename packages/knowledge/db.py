import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from packages.knowledge.models import Base  # noqa: F401  (central metadata import)
from packages.knowledge import migrations

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./repointel.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Initialize the database to the current schema via Alembic migrations.

    Production behavior: migrations are the single sanctioned path for schema
    creation AND evolution. ``create_all`` is intentionally NOT used because it
    creates missing tables but never migrates existing ones — that is the exact
    class of defect that left the persistent ``repositories`` table stale.

    Alembic's engine is synchronous, so the migration runs in a worker thread to
    stay non-blocking within the async application.
    """
    await asyncio.to_thread(migrations.upgrade_to_head)
