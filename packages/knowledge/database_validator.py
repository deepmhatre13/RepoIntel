"""Database validation for RepoIntel.

Validates that the persistent database schema matches the ORM:

- required tables exist
- required columns exist per table
- expected indexes exist
- foreign keys resolve to existing tables / target columns
- migration revision is current

Used by the ``db-check`` CLI command and available as a programmatic utility so
tests and deployment tooling can assert a healthy database before running.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import sqlalchemy as sa

from packages.knowledge import models as knowledge_models
from packages.knowledge.migrations import head_revision, _sync_url, current_revision


@dataclass
class DatabaseValidationResult:
    """Result of a full database validation run."""
    database_ok: bool = False
    migration_ok: bool = False
    tables_ok: bool = True
    columns_ok: bool = True
    indexes_ok: bool = True
    foreign_keys_ok: bool = True
    current_revision: Optional[str] = None
    expected_revision: Optional[str] = None
    tables_missing: List[str] = field(default_factory=list)
    columns_missing: List[str] = field(default_factory=list)
    indexes_missing: List[str] = field(default_factory=list)
    foreign_key_errors: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.database_ok
            and self.migration_ok
            and self.tables_ok
            and self.columns_ok
            and self.indexes_ok
            and self.foreign_keys_ok
        )

    def summary_lines(self) -> List[str]:
        lines = [
            f"Database: {'OK' if self.database_ok else 'FAILED'}",
            f"Migration: {'CURRENT' if self.migration_ok else 'OUTDATED'}",
            f"Tables: {'OK' if self.tables_ok else 'MISSING'}",
            f"Columns: {'OK' if self.columns_ok else 'MISSING'}",
            f"Indexes: {'OK' if self.indexes_ok else 'MISSING'}",
            f"Foreign Keys: {'OK' if self.foreign_keys_ok else 'INVALID'}",
        ]
        return lines

    def problems(self) -> List[str]:
        problems = []
        if self.tables_missing:
            problems.append(f"Missing tables: {', '.join(self.tables_missing)}")
        if self.columns_missing:
            problems.append(f"Missing columns: {', '.join(self.columns_missing)}")
        if self.indexes_missing:
            problems.append(f"Missing indexes: {', '.join(self.indexes_missing)}")
        if self.foreign_key_errors:
            problems.append(f"Invalid foreign keys: {'; '.join(self.foreign_key_errors)}")
        if not self.migration_ok:
            problems.append(
                f"Migration out of date (applied={self.current_revision}, expected={self.expected_revision}); "
                "run `python -m repointel.cli db-upgrade`"
            )
        return problems
def validate_database(url: Optional[str] = None, engine=None) -> DatabaseValidationResult:
    """Inspect a database and compare its schema against the ORM metadata.

    ``url`` may be an async URL; it is normalised to a sync URL for inspection.
    """
    result = DatabaseValidationResult()
    owns_engine = engine is None
    try:
        if owns_engine:
            target_url = url or os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./repointel.db")
            engine = sa.create_engine(_sync_url(target_url))

        inspector = sa.inspect(engine)
        actual_tables = set(inspector.get_table_names())
        metadata = knowledge_models.Base.metadata

        # ---- Tables ----
        expected_tables = {tbl.name for tbl in metadata.sorted_tables}
        result.tables_missing = sorted(expected_tables - actual_tables)
        result.tables_ok = not result.tables_missing

        # ---- Columns, indexes, foreign keys for present tables ----
        for tbl in metadata.sorted_tables:
            if tbl.name not in actual_tables:
                continue
            cols = {c["name"] for c in inspector.get_columns(tbl.name)}
            missing = [c.name for c in tbl.columns if c.name not in cols]
            if missing:
                result.columns_missing.extend(f"{tbl.name}.{m}" for m in missing)
            idx = {i["name"] for i in inspector.get_indexes(tbl.name)}
            for index in tbl.indexes:
                if index.name and index.name not in idx:
                    result.indexes_missing.append(index.name)

            # Foreign keys: ensure referenced tables / columns exist.
            for fk in tbl.foreign_key_constraints:
                for element in fk.elements:
                    ref_table = element.column.table.name
                    if ref_table not in actual_tables:
                        result.foreign_key_errors.append(
                            f"{tbl.name}.{element.parent.name} -> {ref_table}.{element.column.name} (table missing)"
                        )
                        continue
                    ref_cols = {c["name"] for c in inspector.get_columns(ref_table)}
                    if element.column.name not in ref_cols:
                        result.foreign_key_errors.append(
                            f"{tbl.name}.{element.parent.name} -> {ref_table}.{element.column.name} (column missing)"
                        )

        result.columns_ok = not result.columns_missing
        result.indexes_ok = not result.indexes_missing
        result.foreign_keys_ok = not result.foreign_key_errors

        # ---- Migration version ----
        result.expected_revision = head_revision()
        result.current_revision = current_revision()
        result.migration_ok = (
            result.expected_revision is None
            or result.current_revision == result.expected_revision
        )

        result.database_ok = True
        if result.ok:
            result.message = (
                "Database: OK | Migration: CURRENT | Tables: OK | "
                "Columns: OK | Indexes: OK | Foreign Keys: OK"
            )
        else:
            result.message = "Database validation FAILED - " + "; ".join(result.problems())

    except Exception as exc:  # database unreachable or corrupt -> report exactly
        result.database_ok = False
        result.message = f"Database validation error: {type(exc).__name__}: {exc}"
    finally:
        if owns_engine and engine is not None:
            engine.dispose()
    return result