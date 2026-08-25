"""Initial RepoIntel schema.

This migration materializes the full ORM schema (``target_metadata``) into the
database. It is written idempotently: any tables and indexes that ALREADY exist
in a pre-Alembic database are left untouched so existing installations upgrade
in place rather than being recreated. A fresh database gains the complete
current schema from this single migration.

Revision ID: 0001
Revises:
Create Date: 2026-08-23

"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

from packages.knowledge import models as knowledge_models  # noqa: E402


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    metadata = knowledge_models.Base.metadata
    for table in metadata.sorted_tables:
        if table.name in existing_tables:
            continue
        # Table.create emits the DDL for this table exactly as declared by the
        # ORM (columns + constraints) and is the supported non-copy API.
        table.create(bind=bind)
        for index in table.indexes:
            existing_indexes = {i["name"] for i in inspector.get_indexes(table.name)}
            if index.name and index.name not in existing_indexes:
                op.create_index(index.name, table.name, [c.name for c in index.columns])


def downgrade() -> None:
    # Safer reverse: drop the auto-created unique + named indexes first, then tables.
    metadata = knowledge_models.Base.metadata
    # Drop in reverse dependency order.
    for table in reversed(metadata.sorted_tables):
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        if table.name not in {t for t in inspector.get_table_names()}:
            continue
        for index in table.indexes:
            existing_indexes = {i["name"] for i in inspector.get_indexes(table.name)}
            if index.name and index.name in existing_indexes:
                op.drop_index(index.name, table_name=table.name)
        op.drop_table(table.name)