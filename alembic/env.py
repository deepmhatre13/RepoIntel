"""Alembic migration environment for RepoIntel.

The project uses an async SQLAlchemy engine (aiosqlite / asyncpg). Alembic's
migration operations are internally synchronous against a sync-style connection,
so in this env we create a sync engine from the configured DATABASE_URL that is
compatible with Alembic's runner ("run_migrations`"). For sqlite URLs the async
prefix is stripped; for postgres URLs the async driver is swapped for the sync
psycopg2 driver so the same DSN works for both runtime and migrations.
"""

import asyncio
import os
import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, engine_from_config, pool

from packages.knowledge import models as knowledge_models

from packages.knowledge.checkpoint_store import checkpoint_metadata

# Separate infrastructure metadata prevents historical ORM-driven revisions
# from creating new checkpoint tables before their owning revision (0011).
target_metadata = [knowledge_models.Base.metadata, checkpoint_metadata]

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _sync_url(async_url: str) -> str:
    """Convert an async SQLAlchemy URL to one usable by Alembic's sync engine."""
    if not async_url:
        return async_url
    # sqlite+aiosqlite:// -> sqlite://
    url = async_url.replace("sqlite+aiosqlite://", "sqlite://")
    # asyncpg:// -> postgresql+psycopg2://
    url = re.sub(r"^postgresql\+asyncpg://", "postgresql+psycopg2://", url)
    # aiosqlite -> sqlite (for any remaining form)
    url = re.sub(r"\+aiosqlite", "", url)
    return url


def _resolve_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./repointel.db")
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        url = configured
    return _resolve_sync_url(url)


def _resolve_sync_url(async_url: str) -> str:
    parsed = _sync_url(async_url)
    return parsed


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode by emitting SQL to stdout."""
    url = _resolve_sync_url(
        os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./repointel.db")
    )
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with a real database connection."""
    url = _resolve_sync_url(os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./repointel.db"))
    # Build a sync engine directly from the resolved URL so sqlite batch mode works.
    engine = create_engine(url, poolclass=pool.NullPool)

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=url.startswith("sqlite"),
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


# Apply batch mode for sqlite always (required for ALTER TABLE support).
is_offline_mode = context.is_offline_mode()
if is_offline_mode:
    run_migrations_offline()
else:
    run_migrations_online()