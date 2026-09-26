"""Add Project Health Intelligence tables (Project Health product feature).

Creates two IFC-normalized tables:

    project_health_repositories - stable identity for a GitHub project under
                                  health analysis (keyed by ``owner/name``)
    project_health_observations - a single normalized historical observation
                                  table (``timestamp, metric, value, source,
                                  confidence, observation window, availability``)

There is deliberately *not* one table per metric: historical health data uses a
normalized time-series model so the feature can compare, trend and correlate
across metrics with one schema.

The migration is idempotent for sqlite (batch mode) and guarded for postgres.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-06
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def _dialect_name() -> str:
    return op.get_bind().dialect.name


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    inspector = sa.inspect(bind)
    present = {t for t in inspector.get_table_names()}

    if "project_health_observations" not in present:
        if dialect.startswith("postgresql"):
            op.create_table(
                "project_health_observations",
                sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
                sa.Column("health_key", sa.String(128), nullable=False),
                sa.Column("timestamp", sa.DateTime(), nullable=False),
                sa.Column("metric", sa.String(128), nullable=False),
                sa.Column("value", sa.Float(), nullable=True),
                sa.Column("source", sa.String(64), nullable=True),
                sa.Column("confidence", sa.Float(), nullable=True),
                sa.Column("observation_window", sa.String(32), nullable=True),
                sa.Column("availability", sa.String(32), nullable=True),
                sa.UniqueConstraint(
                    "health_key", "timestamp", "metric",
                    name="uq_health_obs_key_ts_metric",
                ),
            )
            op.create_index(
                "ix_project_health_observations_health_key",
                "project_health_observations", ["health_key"],
            )
            op.create_index(
                "ix_project_health_observations_timestamp",
                "project_health_observations", ["timestamp"],
            )
        else:
            # SQLite: same DDL works; define inline to stay explicit/portable.
            op.create_table(
                "project_health_observations",
                sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
                sa.Column("health_key", sa.String(128), nullable=False),
                sa.Column("timestamp", sa.DateTime(), nullable=False),
                sa.Column("metric", sa.String(128), nullable=False),
                sa.Column("value", sa.Float(), nullable=True),
                sa.Column("source", sa.String(64), nullable=True),
                sa.Column("confidence", sa.Float(), nullable=True),
                sa.Column("observation_window", sa.String(32), nullable=True),
                sa.Column("availability", sa.String(32), nullable=True),
                sa.UniqueConstraint(
                    "health_key", "timestamp", "metric",
                    name="uq_health_obs_key_ts_metric",
                ),
            )

    if "project_health_repositories" not in present:
        op.create_table(
            "project_health_repositories",
            sa.Column("id", sa.String(128), primary_key=True),
            sa.Column("owner", sa.String(128), nullable=True),
            sa.Column("name", sa.String(128), nullable=True),
            sa.Column("full_name", sa.String(256), nullable=True),
            sa.Column("url", sa.String(512), nullable=True),
            sa.Column("metadata_json", sa.Text(), nullable=True),
            sa.Column("analysis_count", sa.Integer(), nullable=True),
            sa.Column("last_analysis_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    op.drop_table("project_health_observations")
    op.drop_table("project_health_repositories")