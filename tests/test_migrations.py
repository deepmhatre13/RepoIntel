"""Migration & database lifecycle tests.

Covers the exact production failure mode that motivated the migration system:
an existing persistent database whose ``repositories`` table predates the Phase
40/41 ORM columns must migrate in place WITHOUT losing data, and a fresh
database must initialize directly to the current schema.
"""

import os
import sqlite3
from pathlib import Path

import pytest

from packages.knowledge import migrations
from packages.knowledge.database_validator import validate_database

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_legacy_db(db_path: Path) -> str:
    """Create a pre-Phase-40 style database with a stale repositories table.

    Mirrors the real defect: repositories only has the original seven columns,
    so ORM access to `status` fails with 'no such column: repositories.status'.
    """
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE repositories (
            id VARCHAR NOT NULL PRIMARY KEY,
            repo_url VARCHAR NOT NULL,
            archetype VARCHAR,
            size_mb FLOAT,
            file_count INTEGER,
            manifest_data TEXT,
            created_at DATETIME
        )
        """
    )
    c.execute(
        """
        INSERT INTO repositories (id, repo_url, archetype, size_mb, file_count,
                                  manifest_data, created_at)
        VALUES ('legacy-repo-1', 'https://example.com/legacy', 'LIBRARY', 1.5, 10,
                '{}', '2024-01-01 00:00:00.000000')
        """
    )
    conn.commit()
    conn.close()
    return str(db_path)


def _run_migrations(url_env: str):
    os.environ["DATABASE_URL"] = url_env
    migrations.upgrade_to_head()


def test_existing_database_migrates_in_place_and_preserves_rows(tmp_path):
    db_file = tmp_path / "legacy.db"
    _make_legacy_db(db_file)

    # Before migration the legacy table lacks the Phase 40/41 columns.
    conn = sqlite3.connect(str(db_file))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(repositories)").fetchall()}
    conn.close()
    assert not {"status", "branch", "head_commit", "fingerprint_hash", "last_run_id", "updated_at"} & cols

    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    _run_migrations(url)

    conn = sqlite3.connect(str(db_file))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(repositories)").fetchall()}
    rows = conn.execute("SELECT id, repo_url, status, branch, head_commit FROM repositories").fetchall()
    conn.close()

    # All migrated columns exist...
    for col in ("status", "branch", "head_commit", "fingerprint_hash", "last_run_id", "updated_at"):
        assert col in cols, col
    # ...with safe defaults applied to the preserved historical row.
    assert rows == [("legacy-repo-1", "https://example.com/legacy", "CREATED", "", None)]

    # The deployed revision is now the latest available one.
    assert migrations.current_revision() == migrations.head_revision()


def test_fresh_database_initializes_to_current_schema(tmp_path):
    db_file = tmp_path / "fresh.db"
    assert not db_file.exists()
    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    _run_migrations(url)

    assert db_file.exists()
    assert migrations.current_revision() == migrations.head_revision()
