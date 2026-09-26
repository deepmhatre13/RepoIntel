"""Migration & database lifecycle tests.

Covers the exact production failure mode that motivated the migration system:
an existing persistent database whose ``repositories`` table predates the Phase
40/41 ORM columns must migrate in place WITHOUT losing data, and a fresh
database must initialize directly to the current schema.
"""

import os
import sqlite3
from pathlib import Path


from packages.knowledge import migrations

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


def test_fresh_database_commit_identity_is_composite_repo_id_sha(tmp_path):
    """Regression test for the Docker startup failure.

    PostgreSQL rejects ``FOREIGN KEY (commit_sha) REFERENCES commits (sha)``
    because ``commits.sha`` alone is not unique: a commit's identity is
    ``(repo_id, sha)``. The fresh-install schema must therefore carry the
    composite UNIQUE constraint on commits and the composite FK from
    file_changes, or `alembic upgrade head` cannot run on PostgreSQL at all.
    """
    import sqlite3

    db_file = tmp_path / "fresh_constraints.db"
    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    _run_migrations(url)

    conn = sqlite3.connect(str(db_file))
    try:
        # 1) commits has a UNIQUE (repo_id, sha) constraint backing the FK.
        unique_cols = None
        for row in conn.execute("PRAGMA index_list(commits)").fetchall():
            name, is_unique = row[1], row[2]
            cols = [r[2] for r in conn.execute(f"PRAGMA index_info({name})").fetchall()]
            if is_unique and cols == ["repo_id", "sha"]:
                unique_cols = cols
                break
        assert unique_cols == ["repo_id", "sha"], (
            "commits must have a UNIQUE (repo_id, sha) constraint; "
            f"found unique indexes: {conn.execute('PRAGMA index_list(commits)').fetchall()}"
        )

        # 2) file_changes references the composite key (repo_id, commit_sha)
        #    -> commits (repo_id, sha) with ON DELETE CASCADE preserved.
        fk_rows = conn.execute("PRAGMA foreign_key_list(file_changes)").fetchall()
        # PRAGMA foreign_key_list -> (id, seq, table, from, to, on_update, on_delete, match)
        commit_fk = [r for r in fk_rows if r[2] == "commits"]
        assert len(commit_fk) == 2, f"expected composite FK to commits, got: {fk_rows}"
        by_from = {r[3]: (r[4], r[6]) for r in commit_fk}
        assert by_from["repo_id"] == ("repo_id", "CASCADE")
        assert by_from["commit_sha"] == ("sha", "CASCADE")
    finally:
        conn.close()


def test_downgrade_then_upgrade_round_trip(tmp_path):
    """The full migration history must be reversible down to base and replayable.

    Exercises 0001's downgrade (including its index handling) and every later
    migration's downgrade on a completely fresh database.
    """
    from alembic import command

    db_file = tmp_path / "roundtrip.db"
    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    _run_migrations(url)
    assert migrations.current_revision() == migrations.head_revision()

    cfg = migrations._alembic_config()  # noqa: SLF001 - test-only access
    command.downgrade(cfg, "base")
    assert migrations.current_revision() is None
    assert not db_file.exists() or db_file.stat().st_size == 0 or _is_empty_schema(db_file)

    _run_migrations(url)
    assert migrations.current_revision() == migrations.head_revision()


def _is_empty_schema(db_file) -> bool:
    import sqlite3

    conn = sqlite3.connect(str(db_file))
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return not [t for t in tables if t[0] not in ("sqlite_sequence", "alembic_version")]
