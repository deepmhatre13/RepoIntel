"""Add database-enforced identities for knowledge indexing rows.

Existing duplicate rows are compacted deterministically by retaining the lowest
surviving primary key before the unique constraints are created. The operation
is repository-scoped and does not delete rows from another repository.
"""
import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def _dedupe(bind, table: str, columns: list[str]) -> None:
    grouped = ", ".join(columns)
    op.execute(sa.text(
        f"DELETE FROM {table} WHERE id NOT IN "
        f"(SELECT MIN(id) FROM {table} GROUP BY {grouped})"
    ))


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    definitions = (
        ("file_fingerprints", ["repo_id", "file_path"], "uq_file_fingerprints_repo_path"),
        ("files", ["repo_id", "file_path"], "uq_files_repo_path"),
        (
            "code_chunks",
            ["repo_id", "file_path", "line_start", "line_end", "content_hash"],
            "uq_code_chunks_identity",
        ),
    )
    for table, columns, constraint in definitions:
        if table not in tables:
            continue
        _dedupe(bind, table, columns)
        existing = {
            item["name"]
            for item in sa.inspect(bind).get_unique_constraints(table)
        }
        if constraint not in existing:
            if bind.dialect.name == "sqlite":
                with op.batch_alter_table(table) as batch:
                    batch.create_unique_constraint(constraint, columns)
            else:
                op.create_unique_constraint(constraint, table, columns)


def downgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    definitions = (
        ("file_fingerprints", "uq_file_fingerprints_repo_path"),
        ("files", "uq_files_repo_path"),
        ("code_chunks", "uq_code_chunks_identity"),
    )
    for table, constraint in definitions:
        if table not in tables:
            continue
        if bind.dialect.name == "sqlite":
            with op.batch_alter_table(table) as batch:
                batch.drop_constraint(constraint, type_="unique")
        else:
            op.drop_constraint(constraint, table, type_="unique")
