"""Convert checkpoint timestamps already created without time zone.

Migration 0012 enumerated only the ORM metadata, so
``langgraph_checkpoints.created_at`` (declared in the knowledge layer's
standalone checkpoint metadata) kept its naive ``TIMESTAMP`` type on databases
migrated to 0012. This migration closes that drift; it is a no-op on SQLite and
on any database whose column is already ``timestamptz``.

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-19
"""
import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def _affected():
    """(table, column) pairs whose live PostgreSQL type is still naive."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return []
    from packages.knowledge.checkpoint_store import checkpoint_metadata

    inspector = sa.inspect(bind)
    live_tables = set(inspector.get_table_names())
    naive = []
    for table in checkpoint_metadata.sorted_tables:
        if table.name not in live_tables:
            continue
        live = {c["name"]: c for c in inspector.get_columns(table.name)}
        for column in table.columns:
            current = live.get(column.name)
            if current is None or not isinstance(column.type, sa.DateTime):
                continue
            if not getattr(current["type"], "timezone", False):
                naive.append((table.name, column.name))
    return naive


def upgrade() -> None:
    for table_name, column_name in _affected():
        op.execute(
            f'ALTER TABLE "{table_name}" ALTER COLUMN "{column_name}" '
            f"TYPE timestamptz USING \"{column_name}\" AT TIME ZONE 'UTC'"
        )


def downgrade() -> None:
    for table_name, column_name in _affected():
        op.execute(
            f'ALTER TABLE "{table_name}" ALTER COLUMN "{column_name}" '
            f'TYPE timestamp USING "{column_name}" AT TIME ZONE \'UTC\''
        )
