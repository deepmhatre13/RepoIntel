"""Add real embedding columns to code_chunks (Phase 45).

Adds the columns required to store real semantic embeddings alongside each
code chunk:

    embedding            Vector (pgvector) / JSON fallback
    embedding_model      model name used to produce the vector
    embedding_dimension  dimensional size of the vector

This is a forward, additive migration.  Existing rows keep NULL embeddings
(they have not yet been embedded with a real model).  The columns are added
idempotently so the migration is safe on fresh databases.

NOTE: For PostgreSQL the ``vector`` extension + pgvector is a project runtime
dependency (see pyproject) and must be installed in the target database
before this migration runs.  For SQLite the Vector type stores a JSON list.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-25
"""
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

from packages.knowledge import models as knowledge_models  # noqa: E402


def _dialect_name() -> str:
    return op.get_bind().dialect.name


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Ensure the pgvector extension is available for PostgreSQL databases.
    if dialect.startswith("postgresql"):
        existing = {t for t in set(sa.inspect(bind).get_table_names())}
        try:
            op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            pass

    # Add the ORM embedding columns to an existing (pre-Phase-45) code_chunks
    # table.  Fresh databases get them from migration 0001's metadata create.
    inspector = sa.inspect(bind)
    tables = {t for t in inspector.get_table_names()}
    if "code_chunks" not in tables:
        return

    present = {c["name"] for c in inspector.get_columns("code_chunks")}
    vector_type = "vector" if dialect.startswith("postgresql") else sa.Text
    additions = []
    if "embedding" not in present:
        additions.append(sa.Column("embedding", vector_type, nullable=True))
    if "embedding_model" not in present:
        additions.append(sa.Column("embedding_model", sa.String(), nullable=True))
    if "embedding_dimension" not in present:
        additions.append(sa.Column("embedding_dimension", sa.Integer(), nullable=True))

    if additions:
        # SQLite requires batch ALTER; PostgreSQL supports direct ADD COLUMN.
        if dialect.startswith("postgresql"):
            for col in additions:
                op.add_column("code_chunks", col)
        else:
            with op.batch_alter_table("code_chunks") as batch_op:
                for col in additions:
                    batch_op.add_column(col)


def downgrade() -> None:
    # Remove the embedding columns, restoring the pre-Phase-45 schema.
    drop_cols = ["embedding", "embedding_model", "embedding_dimension"]
    # SQLite requires batch ALTER; PostgreSQL can drop columns directly.
    if _dialect_name().startswith("postgresql"):
        import sqlalchemy as sa2

        md = sa2.MetaData()
        cols = [sa2.Column(c, sa2.Text) for c in drop_cols]
        for c in drop_cols:
            try:
                op.drop_column("code_chunks", c)
            except Exception:
                pass
    else:
        # Rebuild the table without embedding columns.
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        existing = {c["name"] for c in inspector.get_columns("code_chunks")}
        remaining = [c for c in existing if c not in set(drop_cols)]
        # SQLite batch mode is used by alembic env; use the batch API.
        with op.batch_alter_table("code_chunks") as batch_op:
            for c in drop_cols:
                if c in existing:
                    batch_op.drop_column(c)