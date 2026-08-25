from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Integer, Float, Text, ForeignKey, DateTime, Boolean, JSON
from sqlalchemy import Index as sa_Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

try:
    from pgvector.sqlalchemy import Vector as PGVector
except ImportError:  # pragma: no cover
    PGVector = None  # type: ignore[assignment]

# Column type for embedding vectors: pgvector ``Vector`` when available,
# otherwise plain JSON (works for both SQLite and PostgreSQL).
def _embedding_column():
    if PGVector is not None:
        return PGVector(None)
    return JSON()


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (no deprecated utcnow)."""
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    pass

class RepositoryModel(Base):
    __tablename__ = "repositories"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    repo_url: Mapped[str] = mapped_column(String, nullable=False)
    archetype: Mapped[str] = mapped_column(String, default="UNKNOWN")
    size_mb: Mapped[float] = mapped_column(Float, default=0.0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    manifest_data: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    # --- Phase 40/41: production lifecycle + incremental indexing state ---
    status: Mapped[str] = mapped_column(String, default="CREATED")  # RepositoryLifecycleState value
    branch: Mapped[str] = mapped_column(String, default="")
    head_commit: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    fingerprint_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    files: Mapped[list["FileModel"]] = relationship(back_populates="repository", cascade="all, delete-orphan")
    symbols: Mapped[list["SymbolModel"]] = relationship(back_populates="repository", cascade="all, delete-orphan")
    dependencies: Mapped[list["DependencyModel"]] = relationship(back_populates="repository", cascade="all, delete-orphan")

class PipelineRunModel(Base):
    """Persisted pipeline run record (Phase 40.2)."""
    __tablename__ = "pipeline_runs"
    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    repository_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String, default="FULL")  # FULL | INCREMENTAL
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    current_stage: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="CREATED")  # lifecycle state value
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    files_seen: Mapped[int] = mapped_column(Integer, default=0)
    files_parsed: Mapped[int] = mapped_column(Integer, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, default=0)
    symbols_extracted: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    embeddings_created: Mapped[int] = mapped_column(Integer, default=0)
    graph_edges_created: Mapped[int] = mapped_column(Integer, default=0)
    errors_json: Mapped[str] = mapped_column(Text, default="[]")
    warnings_json: Mapped[str] = mapped_column(Text, default="[]")
    report_json: Mapped[str] = mapped_column(Text, default="{}")

class FileFingerprintModel(Base):
    """Per-file content fingerprints enabling change-aware reprocessing (Phase 41)."""
    __tablename__ = "file_fingerprints"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

class FileModel(Base):
    __tablename__ = "files"
    # Hot path: scoped invalidation and metadata lookups filter by repo + path.
    __table_args__ = (
        sa_Index("ix_files_repo_id_file_path", "repo_id", "file_path"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str] = mapped_column(String, default="Unknown")
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    repository: Mapped["RepositoryModel"] = relationship(back_populates="files")

class SymbolModel(Base):
    __tablename__ = "symbols"
    # Hot path: symbol lookup by name within a repository.
    __table_args__ = (
        sa_Index("ix_symbols_repo_id_symbol_name", "repo_id", "symbol_name"),
        sa_Index("ix_symbols_repo_id_file_path", "repo_id", "file_path"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    symbol_name: Mapped[str] = mapped_column(String, nullable=False)
    symbol_type: Mapped[str] = mapped_column(String, nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    signature: Mapped[str] = mapped_column(String, nullable=False)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    repository: Mapped["RepositoryModel"] = relationship(back_populates="symbols")

class SymbolEdgeModel(Base):
    __tablename__ = "symbol_edges"
    # Hot path: graph traversal / impact analysis filter edges by repo and endpoints.
    __table_args__ = (
        sa_Index("ix_symbol_edges_repo_id_source_file", "repo_id", "source_file"),
        sa_Index("ix_symbol_edges_repo_id_target_file", "repo_id", "target_file"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    source_symbol: Mapped[str] = mapped_column(String, nullable=False)
    source_file: Mapped[str] = mapped_column(String, nullable=False)
    edge_type: Mapped[str] = mapped_column(String, nullable=False)
    target_symbol: Mapped[str] = mapped_column(String, nullable=False)
    target_file: Mapped[Optional[str]] = mapped_column(String, nullable=True)

class DependencyModel(Base):
    __tablename__ = "dependencies"
    __table_args__ = (
        sa_Index("ix_dependencies_repo_id", "repo_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    package_name: Mapped[str] = mapped_column(String, nullable=False)
    ecosystem: Mapped[str] = mapped_column(String, nullable=False)
    version_spec: Mapped[str] = mapped_column(String, nullable=False)
    dependency_type: Mapped[str] = mapped_column(String, default="direct")
    repository: Mapped["RepositoryModel"] = relationship(back_populates="dependencies")


class CodeChunkModel(Base):
    __tablename__ = "code_chunks"
    # Hot path: retrieval + scoped invalidation filter chunks by repo and path.
    __table_args__ = (
        sa_Index("ix_code_chunks_repo_id_file_path", "repo_id", "file_path"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    symbol_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str] = mapped_column(String, default="Unknown")
    # --- Real embedding storage (Phase 45) ---
    embedding: Mapped[Optional[list]] = mapped_column(_embedding_column(), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    embedding_dimension: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

class AgentRunModel(Base):
    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    repo_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="RUNNING")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

class ArtifactModel(Base):
    __tablename__ = "artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("agent_runs.run_id"), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String, default="README")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

class ReviewFeedbackModel(Base):
    __tablename__ = "review_feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("agent_runs.run_id"), nullable=False)
    verdict: Mapped[str] = mapped_column(String, nullable=False)
    issues_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

class RepositoryGroupModel(Base):
    __tablename__ = "repository_groups"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

class RepositoryGroupMemberModel(Base):
    __tablename__ = "repository_group_members"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[str] = mapped_column(String, ForeignKey("repository_groups.id"), nullable=False)
    repo_id: Mapped[str] = mapped_column(String, nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

class CrossRepoRelationModel(Base):
    __tablename__ = "cross_repo_relations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_repo_id: Mapped[str] = mapped_column(String, nullable=False)
    target_repo_id: Mapped[str] = mapped_column(String, nullable=False)
    relation_type: Mapped[str] = mapped_column(String, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, default="")

class CommitModel(Base):
    __tablename__ = "commits"
    # Hot path: history lookups resolve commits per repository (never globally by sha).
    __table_args__ = (
        sa_Index("ix_commits_repo_id_sha", "repo_id", "sha"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    sha: Mapped[str] = mapped_column(String, nullable=False)
    parent_sha: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    author: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

class FileChangeModel(Base):
    __tablename__ = "file_changes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_sha: Mapped[str] = mapped_column(String, ForeignKey("commits.sha", ondelete="CASCADE"), nullable=False)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    change_type: Mapped[str] = mapped_column(String, default="MODIFIED")
    additions: Mapped[int] = mapped_column(Integer, default=0)
    deletions: Mapped[int] = mapped_column(Integer, default=0)
class SymbolHistoryModel(Base):
    __tablename__ = "symbol_histories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    symbol_name: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, nullable=False)
    author: Mapped[str] = mapped_column(String, nullable=False)
    change_type: Mapped[str] = mapped_column(String, default="MODIFIED")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    line_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    line_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    change_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
class CoChangeRelationModel(Base):
    __tablename__ = "co_change_relations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    symbol_a: Mapped[str] = mapped_column(String, nullable=False)
    symbol_b: Mapped[str] = mapped_column(String, nullable=False)
    co_change_count: Mapped[int] = mapped_column(Integer, default=1)
    supporting_commits: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    co_change_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
class SymbolChangeModel(Base):
    __tablename__ = "symbol_changes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    symbol_name: Mapped[str] = mapped_column(String, nullable=False)
    change_category: Mapped[str] = mapped_column(String, default="BODY_CHANGE")
    line_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    line_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    before_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    after_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[str] = mapped_column(String, default="HIGH")
    evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_index: Mapped[int] = mapped_column(Integer, default=0)
class APIChangeModel(Base):
    __tablename__ = "api_changes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, nullable=False)
    api_symbol: Mapped[str] = mapped_column(String, nullable=False)
    before_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    after_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    change_type: Mapped[str] = mapped_column(String, default="MODIFIED")
    confidence: Mapped[str] = mapped_column(String, default="HIGH")
    evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
class DependencyChangeModel(Base):
    __tablename__ = "dependency_changes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, nullable=False)
    package_name: Mapped[str] = mapped_column(String, nullable=False)
    ecosystem: Mapped[str] = mapped_column(String, nullable=False)
    change_type: Mapped[str] = mapped_column(String, nullable=False)
    old_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    new_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    confidence: Mapped[str] = mapped_column(String, default="HIGH")
    evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
class CommitAnalysisModel(Base):
    __tablename__ = "commit_analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, nullable=False)
    changed_files_json: Mapped[str] = mapped_column(Text, default="[]")
    changed_symbols_json: Mapped[str] = mapped_column(Text, default="[]")
    dependency_changes_json: Mapped[str] = mapped_column(Text, default="[]")
    api_changes_json: Mapped[str] = mapped_column(Text, default="[]")
    configuration_changes_json: Mapped[str] = mapped_column(Text, default="[]")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_signals_json: Mapped[str] = mapped_column(Text, default="[]")
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
class ArchitectureSnapshotModel(Base):
    __tablename__ = "architecture_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, nullable=False)
    components_json: Mapped[str] = mapped_column(Text, default="[]")
    edges_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    component_metadata_json: Mapped[str] = mapped_column(Text, default="{}")
class ArchitectureEdgeModel(Base):
    __tablename__ = "architecture_edges"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(Integer, ForeignKey("architecture_snapshots.id"), nullable=False)
    source_component: Mapped[str] = mapped_column(String, nullable=False)
    target_component: Mapped[str] = mapped_column(String, nullable=False)
    edge_type: Mapped[str] = mapped_column(String, nullable=False)
    evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
class TestCaseModel(Base):
    __tablename__ = "test_cases"
    __test__ = False  # pytest-safe: not a test class despite "Test" prefix
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    test_name: Mapped[str] = mapped_column(String, nullable=False)
    test_type: Mapped[str] = mapped_column(String, default="UNIT")
    framework: Mapped[Optional[str]] = mapped_column(String, nullable=True)
class TestToCodeRelationModel(Base):
    __tablename__ = "test_to_code_relations"
    __test__ = False  # pytest-safe: not a test class despite "Test" prefix
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    test_name: Mapped[str] = mapped_column(String, nullable=False)
    test_file: Mapped[str] = mapped_column(String, nullable=False)
    target_symbol: Mapped[str] = mapped_column(String, nullable=False)
    target_file: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    relation_type: Mapped[str] = mapped_column(String, default="IMPORTS")
    confidence: Mapped[str] = mapped_column(String, default="MEDIUM")
class CodeHotspotModel(Base):
    __tablename__ = "code_hotspots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    file_or_symbol: Mapped[str] = mapped_column(String, nullable=False)
    hotspot_score: Mapped[float] = mapped_column(Float, default=0.0)
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")
class ChangeRiskAssessmentModel(Base):
    __tablename__ = "change_risk_assessments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, ForeignKey("repositories.id"), nullable=False)
    target: Mapped[str] = mapped_column(String, nullable=False)
    target_type: Mapped[str] = mapped_column(String, default="symbol")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String, default="LOW")
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    calculated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
