"""Programmatic Alembic migration control for RepoIntel.

Wraps Alembic's scripting API so the application can migrate the persistent
database without requiring the user to run Alembic CLI directly. These helpers
are the single sanctioned path for schema evolution:

    upgrade() / upgrade_to_head()   -> run all pending migrations
    downgrade_base()                -> revert the most recent migration step
    current_revision()              -> read the applied revision, if any

The database URL is resolved from ``DATABASE_URL`` with the same async->sync
normalisation performed by ``alembic/env.py`` so migrations operate against the
same database the application uses at runtime.
"""

import os
import re
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ALEMBIC_INI = _PROJECT_ROOT / "alembic.ini"


def _sync_url(async_url: str) -> str:
    if not async_url:
        return async_url
    url = async_url.replace("sqlite+aiosqlite://", "sqlite://")
    url = re.sub(r"^postgresql\+asyncpg://", "postgresql+psycopg2://", url)
    url = re.sub(r"\+aiosqlite", "", url)
    return url


def _alembic_config() -> Config:
    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("script_location", str(_PROJECT_ROOT / "alembic"))
    url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./repointel.db")
    cfg.set_main_option("sqlalchemy.url", _sync_url(url))
    return cfg


def upgrade_to_head() -> None:
    """Run all pending migrations up to ``head``."""
    command.upgrade(_alembic_config(), "head")


def downgrade_one() -> Optional[str]:
    """Downgrade by one revision. Returns the new current revision or None."""
    cfg = _alembic_config()
    rev = current_revision(cfg=cfg)
    if not rev:
        return None
    command.downgrade(cfg, "-1")
    return current_revision(cfg=cfg)


def current_revision(cfg: Optional[Config] = None) -> Optional[str]:
    """Read the current deployed migration revision id (None if unknown/existing)."""
    cfg = cfg or _alembic_config()
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(cfg)
    url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./repointel.db")
    from sqlalchemy import create_engine

    engine = create_engine(_sync_url(url))
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current = context.get_current_revision()
    engine.dispose()
    if not current:
        return None
    # Look up the script for its (possibly short) revision id.
    for rev in script.walk_revisions():
        if rev.revision == current or rev.down_revision == current:
            return rev.revision
    return current


def head_revision() -> Optional[str]:
    """Return the latest revision id defined in the versions directory."""
    cfg = _alembic_config()
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    if not heads:
        return None
    head = script.get_revision(heads[0])
    return head.revision if head else None