"""Add embedding provenance columns (Stage 3: embedding versioning).

Extends ``code_chunks`` with two provenance columns that complete the
Stage 3 embedding-versioning record:

    embedding_provider   - provider name ("local", etc.)
    embedding_created_at - timezone-aware UTC timestamp of vector generation

Existing rows keep NULL (not yet re-embedded with provenance).  The migration
is idempotent: columns that already exist are left untouched.  Fresh databases
created by migration 0001 already declare these columns via the current ORM
metadata, so this step is a no-op for them.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-26
"""
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def _dialect_name() -> str:
    return op.get_bind().dialect.name


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    inspector = sa.inspect(bind)
    if "code_chunks" not in {t for t in inspector.get_table_names()}:
        return
    present = {c["name"] for c in inspector.get_columns("code_chunks")}
    additions = []
    if "embedding_provider" not in present:
        additions.append(sa.Column("embedding_provider", sa.String(), nullable=True))
    if "embedding_created_at" not in present:
        additions.append(sa.Column("embedding_created_at", sa.DateTime(), nullable=True))
    if not additions:
        return
    if dialect.startswith("postgresql"):
        for col in additions:
            op.add_column("code_chunks", col)
    else:
        with op.batch_alter_table("code_chunks") as batch_op:
            for col in additions:
                batch_op.add_column(col)


def downgrade() -> None:
    if not _dialect_name().startswith("postgresql"):
        with op.batch_alter_table("code_chunks") as batch_op:
            batch_op.drop_column("embedding_provider")
            batch_op.drop_column("embedding_created_at")
    else:
        for col in ("embedding_provider", "embedding_created_at"):
            try:
                op.drop_column("code_chunks", col)
            except Exception:
                pass
