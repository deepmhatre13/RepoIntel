"""Production-Grade Repository Lifecycle Manager for RepoIntel (Phase 40).

Deterministic lifecycle state machine:

    CREATED -> CLONING -> INGESTING -> PARSING -> INDEXING -> ANALYZING -> READY
        ^                                                                  |
        |                                                            RE-ANALYZE
        +---------------------------------------------------- UPDATING <-+

Any active state may transition to FAILED. A run that parsed some files but
recorded failures finalizes as PARTIAL instead of pretending success.

Contradictory transitions are rejected with InvalidLifecycleTransition.
"""

import uuid
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from packages.shared.telemetry import telemetry


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RepositoryLifecycleState(str, Enum):
    CREATED = "CREATED"
    CLONING = "CLONING"
    INGESTING = "INGESTING"
    PARSING = "PARSING"
    INDEXING = "INDEXING"
    ANALYZING = "ANALYZING"
    READY = "READY"
    UPDATING = "UPDATING"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class LifecycleStage(str, Enum):
    INGESTION = "INGESTION"
    PARSING = "PARSING"
    KNOWLEDGE = "KNOWLEDGE"
    EMBEDDING = "EMBEDDING"
    GRAPH = "GRAPH"
    GIT = "GIT"
    ANALYSIS = "ANALYSIS"


# Legal state transitions (Phase 40.1: no contradictory states).
ALLOWED_TRANSITIONS: Dict[RepositoryLifecycleState, set] = {
    RepositoryLifecycleState.CREATED: {RepositoryLifecycleState.CLONING, RepositoryLifecycleState.INGESTING, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.CLONING: {RepositoryLifecycleState.INGESTING, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.INGESTING: {RepositoryLifecycleState.PARSING, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.PARSING: {RepositoryLifecycleState.INDEXING, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.INDEXING: {RepositoryLifecycleState.ANALYZING, RepositoryLifecycleState.READY, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.ANALYZING: {RepositoryLifecycleState.READY, RepositoryLifecycleState.PARTIAL, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.READY: {RepositoryLifecycleState.UPDATING, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.UPDATING: {RepositoryLifecycleState.READY, RepositoryLifecycleState.PARTIAL, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.PARTIAL: {RepositoryLifecycleState.UPDATING, RepositoryLifecycleState.FAILED},
    RepositoryLifecycleState.FAILED: {RepositoryLifecycleState.UPDATING, RepositoryLifecycleState.CLONING, RepositoryLifecycleState.INGESTING},
}


class InvalidLifecycleTransition(Exception):
    pass


class StageStatus:
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class FailureDetail(BaseModel):
    """Phase 40.6 failure visibility record."""
    stage: str
    file: Optional[str] = None
    error_type: str
    message: str
    recoverable: bool = True


class StageProgress(BaseModel):
    stage: LifecycleStage
    status: str = StageStatus.PENDING
    duration_seconds: float = 0.0
    counts: Dict[str, int] = Field(default_factory=dict)
    errors: List[FailureDetail] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PipelineRunRecord(BaseModel):
    run_id: str
    repository_id: str
    mode: str = "FULL"  # FULL | INCREMENTAL
    lifecycle_state: RepositoryLifecycleState = RepositoryLifecycleState.CREATED
    current_stage: Optional[LifecycleStage] = None
    started_at: str = Field(default_factory=_utcnow)
    completed_at: Optional[str] = None
    progress_percentage: float = 0.0
    files_seen: int = 0
    files_parsed: int = 0
    files_failed: int = 0
    symbols_extracted: int = 0
    chunks_created: int = 0
    embeddings_created: int = 0
    graph_edges_created: int = 0
    errors: List[FailureDetail] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    stages: Dict[str, StageProgress] = Field(default_factory=dict)
    previous_commit: Optional[str] = None
    current_commit: Optional[str] = None


class RepositoryLifecycleManager:
    """Validated, observable lifecycle execution for one repository analysis run."""

    def __init__(self):
        self.runs: Dict[str, PipelineRunRecord] = {}

    def create_run(self, run_id: Optional[str] = None, repo_id: str = "", mode: str = "FULL") -> PipelineRunRecord:
        run_id = run_id or str(uuid.uuid4())
        record = PipelineRunRecord(run_id=run_id, repository_id=repo_id, mode=mode)
        for stage in LifecycleStage:
            record.stages[stage.value] = StageProgress(stage=stage)
        self.runs[run_id] = record
        return record

    def transition_state(self, run_id: str, new_state: RepositoryLifecycleState) -> PipelineRunRecord:
        if run_id not in self.runs:
            raise KeyError(f"Run {run_id} not found")
        run = self.runs[run_id]
        allowed = ALLOWED_TRANSITIONS.get(run.lifecycle_state, set())
        if new_state not in allowed:
            raise InvalidLifecycleTransition(
                f"Illegal lifecycle transition {run.lifecycle_state.value} -> {new_state.value} "
                f"(run {run_id})"
            )
        run.lifecycle_state = new_state
        if new_state in (RepositoryLifecycleState.READY, RepositoryLifecycleState.PARTIAL,
                         RepositoryLifecycleState.FAILED):
            run.completed_at = _utcnow()
        return run

    def start_stage(self, run_id: str, stage: LifecycleStage) -> float:
        """Mark stage IN_PROGRESS; returns monotonic start timestamp."""
        import time
        if run_id not in self.runs:
            raise KeyError(f"Run {run_id} not found")
        run = self.runs[run_id]
        run.current_stage = stage
        sp = run.stages[stage.value]
        sp.status = StageStatus.IN_PROGRESS
        stage_to_state = {
            LifecycleStage.INGESTION: RepositoryLifecycleState.INGESTING,
            LifecycleStage.PARSING: RepositoryLifecycleState.PARSING,
            LifecycleStage.KNOWLEDGE: RepositoryLifecycleState.INDEXING,
            LifecycleStage.EMBEDDING: RepositoryLifecycleState.INDEXING,
            LifecycleStage.GRAPH: RepositoryLifecycleState.INDEXING,
            LifecycleStage.GIT: RepositoryLifecycleState.INDEXING,
            LifecycleStage.ANALYSIS: RepositoryLifecycleState.ANALYZING,
        }
        target = stage_to_state[stage]
        if run.lifecycle_state != target and target in ALLOWED_TRANSITIONS.get(run.lifecycle_state, set()):
            self.transition_state(run_id, target)
        return time.monotonic()

    def complete_stage(
        self,
        run_id: str,
        stage: LifecycleStage,
        started_at_monotonic: float,
        counts: Optional[Dict[str, int]] = None,
        errors: Optional[List[FailureDetail]] = None,
        warnings: Optional[List[str]] = None,
    ):
        import time
        if run_id not in self.runs:
            raise KeyError(f"Run {run_id} not found")
        run = self.runs[run_id]
        sp = run.stages[stage.value]
        sp.duration_seconds = round(time.monotonic() - started_at_monotonic, 4)
        sp.counts.update(counts or {})
        sp.errors.extend(errors or [])
        sp.warnings.extend(warnings or [])
        if errors:
            sp.status = StageStatus.COMPLETED_WITH_ERRORS
            run.errors.extend(errors)
        else:
            sp.status = StageStatus.COMPLETED
        run.warnings.extend(warnings or [])
        telemetry.record_stage_duration(stage.value, sp.duration_seconds)

    def fail_stage(self, run_id: str, stage: LifecycleStage, started_at_monotonic: float,
                   error: FailureDetail):
        import time
        run = self.runs[run_id]
        sp = run.stages[stage.value]
        sp.duration_seconds = round(time.monotonic() - started_at_monotonic, 4)
        sp.status = StageStatus.FAILED
        sp.errors.append(error)
        run.errors.append(error)

    def record_counts(self, run_id: str, **counts: int):
        run = self.runs[run_id]
        for key, value in counts.items():
            if hasattr(run, key):
                setattr(run, key, value)

    def add_warning(self, run_id: str, message: str, stage: Optional[LifecycleStage] = None):
        run = self.runs[run_id]
        run.warnings.append(message)
        if stage:
            run.stages[stage.value].warnings.append(message)

    def update_progress(self, run_id: str):
        run = self.runs[run_id]
        total = len(LifecycleStage)
        done = sum(
            1 for s in run.stages.values()
            if s.status in (StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_ERRORS, StageStatus.SKIPPED)
        )
        run.progress_percentage = round(100.0 * done / total, 1)

    def finalize_run(self, run_id: str) -> PipelineRunRecord:
        """Determine READY / PARTIAL / FAILED from measured evidence (Phase 40.5)."""
        run = self.runs[run_id]
        run.completed_at = run.completed_at or _utcnow()

        hard_stage_failure = any(s.status == StageStatus.FAILED for s in run.stages.values())
        if run.files_parsed == 0 and (hard_stage_failure or run.files_failed > 0 or run.errors):
            final = RepositoryLifecycleState.FAILED
        elif run.files_failed > 0 or hard_stage_failure or run.errors:
            # Partial success: some data indexed, some failed -- never pretend full success.
            final = RepositoryLifecycleState.PARTIAL
        else:
            final = RepositoryLifecycleState.READY

        # FAILED is reachable from any state; others must be validated transitions.
        if final != RepositoryLifecycleState.FAILED:
            self.transition_state(run_id, final)
        else:
            run.lifecycle_state = RepositoryLifecycleState.FAILED
            run.completed_at = _utcnow()

        run.progress_percentage = 100.0
        return run
