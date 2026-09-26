"""Add repository access/ownership table + auth_users table.

Creates the repository_access table that maps users to repositories with
an access level (OWNER or MEMBER). Also creates the auth_users table that
was registered in the ORM metadata but not present in any migration.
"""
import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # auth_users table (user model used by JWT authentication)
    if "auth_users" not in existing_tables:
        op.create_table(
            "auth_users",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("username", sa.String(), nullable=False),
            sa.Column("email", sa.String(), nullable=True),
            sa.Column("hashed_password", sa.String(), nullable=False),
            sa.Column("role", sa.String(), nullable=False, server_default="viewer"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_auth_users_username", "auth_users", ["username"], unique=True)
        op.create_index("ix_auth_users_email", "auth_users", ["email"], unique=True)

    # repository_access ownership table
    if "repository_access" not in existing_tables:
        op.create_table(
            "repository_access",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("repository_id", sa.String(), nullable=False),
            sa.Column("access_level", sa.String(), nullable=False, server_default="OWNER"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_unique_constraint(
            "uq_repository_access_user_repo",
            "repository_access",
            ["user_id", "repository_id"],
        )
        op.create_index(
            "ix_repository_access_repository_id",
            "repository_access",
            ["repository_id"],
        )


def downgrade() -> None:
    # Ownership tables are NOT owned by this migration.
    #
    # ``0001`` materializes the complete ORM metadata, so on a fresh database
    # ``auth_users`` and ``repository_access`` already exist before ``0009``
    # runs, and this migration's ``upgrade`` is a pure no-op (it guards every
    # statement with ``if <table> not in existing_tables``). The previous
    # downgrade dropped them anyway, which broke the migration contract in two
    # ways: the tables vanished at revision 0008 (a state 0001 claims to
    # produce), and the following re-upgrade to 0009 re-created them with
    # *only* the columns named here -- silently discarding the ORM's
    # ``uq_repository_access_user_repo`` unique constraint and its FK
    # constraints. A ``downgrade`` -> ``upgrade`` round trip therefore corrupted
    # the schema instead of restoring it.
    #
    # Dropping them is 0001's responsibility (its own downgrade drops every
    # table in reverse dependency order), so this is deliberately a no-op.
    return None
