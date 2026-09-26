"""Add latest_analysis_json to Project Health repositories.

Stores the most recent Project Health analysis snapshot (dimensions, risks,
signals, correlations, why_changed, AI interpretation) so the API can serve
``GET /project-health/{owner}/{repo}/signals|trends|risks|anomalies|evidence``
from persisted data instead of returning empty skeletons, and so a monitoring
scheduler can diff consecutive snapshots.

Additive, guarded migration — safe to run even if partially applied.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-06
"""
import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("project_health_repositories")}
    if "latest_analysis_json" not in cols:
        op.add_column(
            "project_health_repositories",
            sa.Column("latest_analysis_json", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("project_health_repositories")}
    if "latest_analysis_json" in cols:
        op.drop_column("project_health_repositories", "latest_analysis_json")