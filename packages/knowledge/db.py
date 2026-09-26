import asyncio
import os
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from packages.knowledge.models import Base  # noqa: F401  (central metadata import)
from packages.knowledge import migrations

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./repointel.db")

# SQLite concurrency hardening (production finding 2026-09-15):
# The canonical analysis task commits one long transaction that rewrites every
# chunk row including ~21 MB of embedding blobs. Under the default journal mode
# (`delete`) that transaction held an EXCLUSIVE lock on the whole database file
# for its full duration, so the API's reads failed instantly with
# `sqlite3.OperationalError: database is locked` and monitoring queries starved.
# WAL fixes both directions: writers append to the -wal file while readers keep
# reading the last committed snapshot, and busy_timeout makes competing writers
# wait (bounded) instead of failing on first contact. WAL recovery is also
# automatic on process kill -- no manual hot-journal rollback is needed.
SQLITE_BUSY_TIMEOUT_MS = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "30000"))

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"timeout": SQLITE_BUSY_TIMEOUT_MS / 1000.0} if DATABASE_URL.startswith("sqlite") else {},
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_wal_pragmas(dbapi_connection, _connection_record):
        """Apply per-connection pragmas on every new pooled SQLite connection.

        journal_mode=WAL is persistent (it converts the database file once), but
        it is re-asserted per connection because it is cheap and guarantees the
        mode is active even if another tool reset it. busy_timeout is strictly
        per-connection and must be set every time.
        """
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()


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
