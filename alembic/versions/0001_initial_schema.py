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
# Importing the auth models registers ``auth_users`` and ``repository_access``
# on the shared ``Base.metadata``. Without this import 0001 would happily create
# a schema that is missing the ownership tables, and every later migration would
# have to re-add them (which is exactly the drift this migration system exists
# to prevent).
from packages.auth import sqlalchemy_models as _auth_models  # noqa: E402,F401


def _ensure_table_indexes(bind, table) -> None:
    """Create any ORM-declared index on ``table`` that is not present yet."""
    inspector = sa.inspect(bind)
    existing_indexes = {i["name"] for i in inspector.get_indexes(table.name)}
    for index in table.indexes:
        if index.name and index.name not in existing_indexes:
            op.create_index(index.name, table.name, [c.name for c in index.columns])


def _ensure_vector_extension(bind) -> None:
    """Create the pgvector extension BEFORE any table DDL is emitted.

    ``code_chunks.embedding`` is declared as ``Vector`` by the ORM whenever the
    Python ``pgvector`` package is importable. That is a *client-side* fact and
    says nothing about whether the *server* has the ``vector`` extension, so on a
    PostgreSQL server without pgvector this migration previously emitted
    ``embedding VECTOR`` and aborted with::

        psycopg2.errors.UndefinedObject: type "vector" does not exist

    The extension is created here, in the migration that first needs it, rather
    than in a later migration (0005) that runs after the table already exists.
    If pgvector is genuinely unavailable the failure is raised loudly instead of
    being swallowed: a database that cannot honor the declared schema must not
    silently receive a different one.
    """
    if bind.dialect.name != "postgresql":
        return
    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:  # pragma: no cover - depends on server provisioning
        raise RuntimeError(
            "PostgreSQL target is missing the 'vector' (pgvector) extension, which "
            "the embedding columns require. Install pgvector on the server, or point "
            "DATABASE_URL at a database that has it. The migration refuses to create "
            "a schema whose declared column types the server cannot honor."
        ) from exc


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # Must happen before any CREATE TABLE that declares an embedding column.
    _ensure_vector_extension(bind)

    metadata = knowledge_models.Base.metadata
    for table in metadata.sorted_tables:
        if table.name in existing_tables:
            # Pre-Alembic database: the table exists, but its ORM-declared
            # indexes may not. Reconciling them here keeps a fresh database and
            # a migrated database structurally identical -- otherwise the
            # downgrade path (which drops ORM indexes) is not reversible.
            _ensure_table_indexes(bind, table)
            continue
        # Table.create emits the DDL for this table exactly as declared by the
        # ORM (columns + constraints) and is the supported non-copy API.
        table.create(bind=bind)
        _ensure_table_indexes(bind, table)


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