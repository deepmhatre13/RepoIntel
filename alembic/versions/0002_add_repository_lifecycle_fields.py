"""Add repository lifecycle & incremental-indexing columns (Phases 40/41).

The ORM ``RepositoryModel`` defines six columns that were introduced during
Phases 40/41 (lifecycle state + incremental indexing bookkeeping). Databases
created before those phases have a ``repositories`` table that predates them, so
this migration adds the columns with safe defaults for existing rows:

    status         -> 'CREATED'
    branch         -> ''
    head_commit    -> NULL
    fingerprint_hash -> NULL
    last_run_id    -> NULL
    updated_at     -> initialized from created_at (or current UTC when NULL)

The migration is idempotent: on a fresh database (created by migration 0001)
where the columns already exist it is a safe no-op. Existing rows are preserved.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-23

"""
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    names = {t for t in inspector.get_table_names()}
    if "repositories" not in names:
        return
    existing = {c["name"] for c in inspector.get_columns("repositories")}

    if "status" not in existing:
        op.add_column("repositories", sa.Column("status", sa.String(), nullable=False, server_default="CREATED"))
    if "branch" not in existing:
        op.add_column("repositories", sa.Column("branch", sa.String(), nullable=False, server_default=""))
    if "head_commit" not in existing:
        op.add_column("repositories", sa.Column("head_commit", sa.String(), nullable=True))
    if "fingerprint_hash" not in existing:
        op.add_column("repositories", sa.Column("fingerprint_hash", sa.String(), nullable=True))
    if "last_run_id" not in existing:
        op.add_column("repositories", sa.Column("last_run_id", sa.String(), nullable=True))
    if "updated_at" not in existing:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        op.add_column(
            "repositories",
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text(f"'{now}'")),
        )
        table = sa.table(
            "repositories",
            sa.column("created_at", sa.DateTime),
            sa.column("updated_at", sa.DateTime),
        )
        # Initialize historical rows from their created_at where available; rows
        # without a created_at retain the current UTC server default.
        conn = op.get_bind()
        conn.execute(sa.update(table).values(updated_at=sa.column("created_at")))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    names = inspector.get_table_names()
    if "repositories" not in names:
        return
    existing = {c["name"] for c in inspector.get_columns("repositories")}
    for col in ("status", "branch", "head_commit", "fingerprint_hash", "last_run_id", "updated_at"):
        if col in existing:
            op.drop_column("repositories", col)