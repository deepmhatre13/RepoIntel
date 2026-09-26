"""Migration-owned LangGraph checkpoints, preserving legacy payloads.

Legacy rows have no recorded parent; nullable ancestry does not invent history.
The table definition is taken from the knowledge layer's live checkpoint
metadata, so the physical schema cannot drift from the columns the saver
actually reads and writes (created_at is timezone-aware to match every writer).
"""
import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

from packages.knowledge.checkpoint_store import checkpoints_tbl, writes_tbl  # noqa: E402


def upgrade():
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "langgraph_checkpoints" not in tables:
        checkpoints_tbl.create(bind=bind)
    else:
        columns = {c["name"] for c in sa.inspect(bind).get_columns("langgraph_checkpoints")}
        if "parent_checkpoint_id" not in columns:
            op.add_column("langgraph_checkpoints", sa.Column("parent_checkpoint_id", sa.String(64)))
    if "langgraph_checkpoint_writes" not in tables:
        writes_tbl.create(bind=bind)


def downgrade():
    op.drop_table("langgraph_checkpoint_writes")
    op.drop_table("langgraph_checkpoints")
