"""PostgreSQL fresh-install migration probe.

Runs ``alembic upgrade head`` against whatever ``DATABASE_URL`` points at and
then inspects the resulting PostgreSQL schema for the constraints the domain
model depends on. This is the check that a fresh PostgreSQL database can reach
head cleanly -- the failure mode that previously broke Docker startup:

    FOREIGN KEY(commit_sha) REFERENCES commits(sha)
    ... there is no unique constraint matching given keys for referenced table

Exit code 0 means the schema is valid for PostgreSQL; non-zero prints the
failing assertion. Run it explicitly, e.g.::

    $env:DATABASE_URL='postgresql+psycopg2://user:pass@localhost:5433/db'
    python scripts/pg_upgrade_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, inspect, text  # noqa: E402

from packages.knowledge import migrations  # noqa: E402


def _sync_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


def main() -> int:
    url = os.getenv("DATABASE_URL", "")
    if not url.startswith("postgresql"):
        print(f"SKIP: DATABASE_URL is not PostgreSQL ({url!r})")
        return 2

    sync_url = _sync_url(url)
    print(f"head revision      : {migrations.head_revision()}")
    migrations.upgrade_to_head()
    current = migrations.current_revision()
    print(f"current revision   : {current}")
    if current != migrations.head_revision():
        print("FAIL: database did not reach head")
        return 1

    engine = create_engine(sync_url)
    inspector = inspect(engine)
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if condition:
            print(f"  ok   {message}")
        else:
            print(f"  FAIL {message}")
            failures.append(message)

    # --- commit identity is (repo_id, sha), never sha alone -------------------
    unique_pairs = {
        tuple(u["column_names"])
        for u in inspector.get_unique_constraints("commits")
    }
    unique_pairs |= {
        tuple(i["column_names"])
        for i in inspector.get_indexes("commits")
        if i.get("unique")
    }
    check(
        ("repo_id", "sha") in unique_pairs,
        "commits has UNIQUE (repo_id, sha)",
    )

    fk_map: dict[str, list[tuple[str, str, str]]] = {}
    for fk in inspector.get_foreign_keys("file_changes"):
        for local, remote in zip(fk["constrained_columns"], fk["referred_columns"]):
            fk_map.setdefault(fk["referred_table"], []).append(
                (local, remote, fk.get("options", {}).get("ondelete", ""))
            )
    commit_fk = sorted(fk_map.get("commits", []))
    check(
        commit_fk == [("commit_sha", "sha", "CASCADE"), ("repo_id", "repo_id", "CASCADE")],
        f"file_changes -> commits composite FK with CASCADE (got {commit_fk})",
    )

    # --- repository_access ownership uniqueness ------------------------------
    access_unique = {
        tuple(u["column_names"]) for u in inspector.get_unique_constraints("repository_access")
    }
    access_unique |= {
        tuple(i["column_names"])
        for i in inspector.get_indexes("repository_access")
        if i.get("unique")
    }
    check(
        ("user_id", "repository_id") in access_unique,
        "repository_access has UNIQUE (user_id, repository_id)",
    )

    # --- auth + knowledge tables materialized --------------------------------
    tables = set(inspector.get_table_names())
    for required in (
        "repositories", "files", "symbols", "code_chunks", "commits",
        "file_changes", "agent_runs", "artifacts", "auth_users",
        "repository_access", "pipeline_runs", "review_feedback",
    ):
        check(required in tables, f"table {required} exists")

    # --- pgvector availability (embeddings must be storable) -----------------
    with engine.connect() as conn:
        has_vector = conn.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        chunk_col = conn.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name='code_chunks' AND column_name='embedding'"
            )
        ).scalar()
    check(bool(has_vector), "pgvector extension installed")
    check(chunk_col is not None, f"code_chunks.embedding column present ({chunk_col})")

    engine.dispose()

    if failures:
        print(f"\nPG PROBE FAILED ({len(failures)} checks)")
        return 1
    print("\nPG PROBE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
