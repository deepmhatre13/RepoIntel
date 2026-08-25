"""Add hot-path query indexes (Step 21).

The facade and persistence layer constantly filter knowledge rows by
``repo_id`` plus ``file_path`` / ``symbol_name``, and history lookups resolve
commits per repository. These indexes are created only for measured query
patterns; no blanket indexing.

The migration is idempotent: any index that already exists is left untouched,
and indexes on tables that do not exist are skipped.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-23

"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

from packages.knowledge import models as knowledge_models  # noqa: E402


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    for table in knowledge_models.Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        existing_indexes = {i["name"] for i in inspector.get_indexes(table.name)}
        for index in table.indexes:
            if index.name and index.name not in existing_indexes:
                op.create_index(index.name, table.name, [c.name for c in index.columns])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    for table in knowledge_models.Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        existing_indexes = {i["name"] for i in inspector.get_indexes(table.name)}
        for index in table.indexes:
            if index.name and index.name in existing_indexes:
                op.drop_index(index.name, table_name=table.name)