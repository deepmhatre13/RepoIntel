"""Add durable Celery dispatch identity to pipeline_runs.

``pipeline_runs`` is the durable authority for a repository analysis run. The
Celery task id used to live only in an in-process ``run_id -> task_id`` dict, so
an API restart (or a second replica) could no longer cancel its own runs, and
there was no persisted link from a run to the broker message that executes it.

This migration adds:

* ``task_id``        -- Celery message id for the run, indexed so cancellation
  resolves a run by durable identity.
* ``correlation_id`` -- request/correlation id for log correlation across API,
  broker, worker and LangGraph.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-16

"""
import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

_TABLE = "pipeline_runs"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _TABLE not in set(inspector.get_table_names()):
        # A pre-Alembic database without pipeline_runs: 0001/0003 create it.
        return
    present = {c["name"] for c in inspector.get_columns(_TABLE)}
    if "task_id" not in present:
        op.add_column(_TABLE, sa.Column("task_id", sa.String(), nullable=True))
    if "correlation_id" not in present:
        op.add_column(_TABLE, sa.Column("correlation_id", sa.String(), nullable=True))

    indexes = {i["name"] for i in inspector.get_indexes(_TABLE)}
    if "ix_pipeline_runs_task_id" not in indexes:
        op.create_index("ix_pipeline_runs_task_id", _TABLE, ["task_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _TABLE not in set(inspector.get_table_names()):
        return
    indexes = {i["name"] for i in inspector.get_indexes(_TABLE)}
    if "ix_pipeline_runs_task_id" in indexes:
        op.drop_index("ix_pipeline_runs_task_id", table_name=_TABLE)
    present = {c["name"] for c in inspector.get_columns(_TABLE)}
    if "correlation_id" in present:
        op.drop_column(_TABLE, "correlation_id")
    if "task_id" in present:
        op.drop_column(_TABLE, "task_id")