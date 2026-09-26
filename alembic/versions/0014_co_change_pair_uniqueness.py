"""Add unique constraint on co_change_relations (repo_id, symbol_a, symbol_b).

Root cause of the Phase-1-audit runtime failure: the co-change aggregation
issued one SELECT per symbol pair per commit. A shallow clone yields a single
root commit whose fallback path marks every parsed symbol as changed, so the
O(k^2) pair loop executed on the order of 2 million database round-trips inside
one uncommitted transaction and the GIT stage appeared frozen for hours.

This migration enforces pair uniqueness at the schema level so retries or
concurrent runs can never duplicate a pair, and pairs with the aggregation's
set-based upsert (packages/ingestion/git_history.py). Idempotent on databases
that already carry the constraint (fresh installs get it from ORM metadata via
migration 0001).

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-20
"""
import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"
    if is_sqlite:
        # SQLite cannot add a UNIQUE constraint in place; the ORM metadata path
        # (fresh installs) already declares it, and legacy SQLite dev databases
        # are rebuilt rather than migrated in place.
        return

    existing = conn.execute(
        sa.text(
            "SELECT 1 FROM pg_constraint WHERE conname = 'uq_co_change_repo_pair'"
        )
    ).scalar()
    if existing:
        return

    # De-duplicate any pre-existing pairs before installing the constraint.
    conn.execute(
        sa.text(
            """
            DELETE FROM co_change_relations a
            USING co_change_relations b
            WHERE a.id > b.id
              AND a.repo_id = b.repo_id
              AND a.symbol_a = b.symbol_a
              AND a.symbol_b = b.symbol_b
            """
        )
    )
    op.create_unique_constraint(
        "uq_co_change_repo_pair",
        "co_change_relations",
        ["repo_id", "symbol_a", "symbol_b"],
    )
    op.create_index(
        "ix_co_change_repo_pair",
        "co_change_relations",
        ["repo_id", "symbol_a", "symbol_b"],
    )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return
    op.drop_index("ix_co_change_repo_pair", table_name="co_change_relations")
    op.drop_constraint("uq_co_change_repo_pair", "co_change_relations", type_="unique")
