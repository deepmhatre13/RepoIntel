"""Normalize timestamp columns to timezone-aware TIMESTAMPTZ (PostgreSQL).

The ORM invariant is timezone-aware UTC (``_utcnow()`` everywhere), but the
original declarations emitted ``TIMESTAMP WITHOUT TIME ZONE``. SQLite stored
ISO strings and silently tolerated aware values; PostgreSQL (asyncpg) rejects
them at bind time — which is how the defect surfaced in real worker execution:

    asyncpg.exceptions.DataError: invalid input for query argument $4:
    datetime.datetime(..., tzinfo=utc) ... (can't subtract offset-naive and
    offset-aware datetimes)

The declarations were wrong, not the data: every writer emits UTC. This
migration converts each affected PostgreSQL column in place with

    ALTER TYPE timestamptz USING <col> AT TIME ZONE 'UTC'

interpreting existing naive values as UTC — exactly what the writers meant.
Indexes on converted columns are dropped first and recreated from their stored
definitions, because PostgreSQL refuses ALTER TYPE under an existing index.

SQLite needs no conversion (it stores ISO strings); the migration is a no-op
there, matching how SQLite already behaved.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-19
"""
import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

# Live list of the columns the ORM now declares as DateTime(timezone=True),
# enumerated from metadata so this migration cannot drift from the ORM.
# The LangGraph checkpoint tables are included: their schema lives in the
# knowledge layer's checkpoint metadata (not the ORM Base), and enumerating
# only the ORM metadata is exactly how langgraph_checkpoints.created_at
# escaped this conversion.
from packages.knowledge import models as knowledge_models  # noqa: E402
from packages.auth import sqlalchemy_models as _auth_models  # noqa: E402,F401
from packages.project_health import orm as _health_orm  # noqa: E402,F401
from packages.knowledge.persistent_memory import CheckpointModel as _checkpoint_model  # noqa: E402,F401
from packages.knowledge.checkpoint_store import checkpoint_metadata  # noqa: E402


def _timestamp_columns():
    """[(table, column), ...] for every DateTime column in the shared metadata."""
    from sqlalchemy import DateTime as SADateTime

    pairs = []
    metadatas = (knowledge_models.Base.metadata, checkpoint_metadata)
    for metadata in metadatas:
        for table in metadata.sorted_tables:
            for column in table.columns:
                if isinstance(column.type, SADateTime):
                    pairs.append((table.name, column.name))
    return pairs


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite stores ISO strings; nothing to convert.

    inspector = sa.inspect(bind)
    live_tables = set(inspector.get_table_names())

    for table_name, column_name in _timestamp_columns():
        if table_name not in live_tables:
            continue

        live_columns = {c["name"]: c for c in inspector.get_columns(table_name)}
        column = live_columns.get(column_name)
        if column is None:
            continue

        # Already a tz-aware type: nothing to convert.
        if getattr(column["type"], "timezone", False):
            continue

        existing_indexes = inspector.get_indexes(table_name)
        affected = [
            index for index in existing_indexes
            if column_name in (index.get("column_names") or [])
            and not index.get("duplicates_constraint")
        ]
        definitions = {
            index["name"]: sa.schema.CreateIndex(
                sa.Index(
                    index["name"],
                    *[sa.column(c) for c in index["column_names"]],
                    unique=bool(index.get("unique")),
                )
            )
            for index in affected
            if index.get("name") and index.get("column_names")
        }

        for name in definitions:
            op.execute(sa.text(f'DROP INDEX IF EXISTS "{name}"'))

        op.execute(
            f'ALTER TABLE "{table_name}" ALTER COLUMN "{column_name}" '
            f"TYPE timestamptz USING \"{column_name}\" AT TIME ZONE 'UTC'"
        )

        for ddl in definitions.values():
            op.execute(ddl)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    inspector = sa.inspect(bind)
    live_tables = set(inspector.get_table_names())
    for table_name, column_name in _timestamp_columns():
        if table_name not in live_tables:
            continue
        live_columns = {c["name"]: c for c in inspector.get_columns(table_name)}
        if column_name not in live_columns:
            continue
        op.execute(
            f'ALTER TABLE "{table_name}" ALTER COLUMN "{column_name}" '
            f"TYPE timestamp WITHOUT TIME ZONE "
            f"USING \"{column_name}\" AT TIME ZONE 'UTC'"
        )
