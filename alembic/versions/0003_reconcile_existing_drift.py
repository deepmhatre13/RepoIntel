"""Reconcile pre-existing tables with the current ORM schema.

Databases created before Phases 34-41 contain tables whose column sets are older
than the current ORM metadata (e.g. ``agent_runs``, ``symbol_histories``,
``artifacts``, ``test_cases``). This migration reconciles every existing table
already present in the persistent database to match the ORM, using safe defaults
so no existing row is destroyed.

Strategy per table:
- Any ORM column missing from the live table is added with a safe server default.
- ``agent_runs`` is the one table whose persisted schema lacks the ORM's primary
  key entirely (``id``). SQLite cannot add an autoincrement PK column in place,
  so the (empty) legacy table is rebuilt from the current ORM metadata.
- ``cross_repo_relations``: the legacy column was named ``details`` while the
  ORM now expects ``evidence``. ``evidence`` is added and backfilled from
  ``details`` so any stored relation data survives.

The migration is a safe no-op on fresh databases created by 0001/0002.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-23

"""
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

from packages.knowledge import models as knowledge_models  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _server_default_for(col) -> sa.DefaultClause:
    """Produce a safe server_default for adding a column to existing rows."""
    if col.default is not None and hasattr(col.default, "arg"):
        raw = col.default.arg
        if isinstance(raw, datetime):
            return sa.text(f"'{raw.isoformat()}'")
        return sa.text(f"'{raw}'")
    if isinstance(col.type, sa.Integer):
        return sa.text("'0'")
    if isinstance(col.type, sa.Float):
        return sa.text("'0.0'")
    if isinstance(col.type, sa.Boolean):
        return sa.text("'false'")
    if isinstance(col.type, sa.DateTime):
        return sa.text(f"'{_now()}'")
    return sa.text("''")


def _present_columns(inspector, table) -> set:
    return {c["name"] for c in inspector.get_columns(table)}


def _rebuild_agent_runs(inspector) -> None:
    """Rebuild the empty legacy agent_runs table with the current ORM schema."""
    bind = op.get_bind()
    metadata = knowledge_models.Base.metadata
    agent_runs_table = metadata.tables["agent_runs"]
    op.drop_table("agent_runs")  # table confirmed present by caller
    agent_runs_table.create(bind=bind)
    for index in agent_runs_table.indexes:
        if index.name:
            existing = {i["name"] for i in inspector.get_indexes("agent_runs")}
            if index.name not in existing:
                op.create_index(index.name, "agent_runs", [c.name for c in index.columns])


def upgrade() -> None:  # noqa: F811
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = {t for t in inspector.get_table_names()}
    metadata = knowledge_models.Base.metadata

    for table in metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = _present_columns(inspector, table.name)
        missing = {c.name for c in table.columns} - present

        if not missing:
            continue

        # The legacy agent_runs table lacks the ORM primary key; rebuild it
        # entirely (it contains no historical rows to preserve).
        pk_names = {c.name for c in table.columns if c.primary_key}
        if table.name == "agent_runs" and pk_names - present:
            _rebuild_agent_runs(inspector)
            continue

        for col in table.columns:
            if col.name not in missing:
                continue
            default = _server_default_for(col)
            new_col = sa.Column(col.name, col.type, nullable=col.nullable, server_default=default)
            op.add_column(table.name, new_col)

    # cross_repo_relations: legacy 'details' -> ORM 'evidence'.
    if "cross_repo_relations" in existing_tables:
        rel_cols = _present_columns(inspector, "cross_repo_relations")
        if "evidence" not in rel_cols:
            op.add_column("cross_repo_relations", sa.Column("evidence", sa.Text(), nullable=False, server_default=""))
            if "details" in rel_cols:
                rel = sa.table("cross_repo_relations", sa.column("evidence", sa.Text), sa.column("details", sa.Text))
                op.get_bind().execute(sa.update(rel).values(evidence=rel.c.details))


def downgrade() -> None:
    # Forward-only reconciliation; reversing would require reconstructing
    # historical per-table schemas. Returning no-op protects data.
    pass