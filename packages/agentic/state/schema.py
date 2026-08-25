from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class PlanTask(BaseModel):
    task_id: str
    objective: str
    retrieval_query: str
    evidence_type: str
    target_section: str
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING

class Plan(BaseModel):
    plan_id: str
    repo_id: str
    artifact_type: str = "README"
    tasks: List[PlanTask] = Field(default_factory=list)

class EvidenceCitation(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    symbol_name: Optional[str] = None
    content: str
    relevance_score: float = 1.0

class TaskEvidence(BaseModel):
    task_id: str
    query: str
    citations: List[EvidenceCitation] = Field(default_factory=list)
    reasoning: Optional[str] = None

class ArtifactSection(BaseModel):
    section_title: str
    content: str
    citations: List[EvidenceCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False

class ArtifactDraft(BaseModel):
    version: int
    artifact_type: str = "README"
    sections: List[ArtifactSection] = Field(default_factory=list)

class IssueType(str, Enum):
    WORDING = "WORDING"
    COVERAGE_GAP = "COVERAGE_GAP"
    GROUNDING_FAILURE = "GROUNDING_FAILURE"

class ReviewIssue(BaseModel):
    issue_type: IssueType
    description: str
    target_section: Optional[str] = None
    severity: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH"

class ReviewVerdict(str, Enum):
    APPROVE = "APPROVE"
    REVISE_SYNTHESIS = "REVISE_SYNTHESIS"
    REVISE_PLAN = "REVISE_PLAN"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"

class ReviewReport(BaseModel):
    verdict: ReviewVerdict
    issues: List[ReviewIssue] = Field(default_factory=list)
    groundedness_score: float = 1.0
    coverage_score: float = 1.0

class HumanFeedback(BaseModel):
    approved: bool
    decision: str = "APPROVE"  # "APPROVE", "REJECT", "REQUEST_REVISION"
    comments: Optional[str] = None
    selected_issues: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class RepoIntelState(BaseModel):
    run_id: str
    thread_id: str
    repo_id: str
    repo_url: str
    archetype: str
    artifact_type: str = "README"
    
    # Multi-Repository Intelligence (Phase 16)
    active_repository_ids: List[str] = Field(default_factory=list)
    repository_group_id: Optional[str] = None
    cross_repo_context: Dict[str, Any] = Field(default_factory=dict)
    
    plan: Optional[Plan] = None
    plan_history: List[Plan] = Field(default_factory=list)
    
    evidence_store: Dict[str, TaskEvidence] = Field(default_factory=dict)
    
    artifact_draft: Optional[ArtifactDraft] = None
    draft_history: List[ArtifactDraft] = Field(default_factory=list)
    
    review_report: Optional[ReviewReport] = None
    review_history: List[ReviewReport] = Field(default_factory=list)
    
    human_feedback: Optional[HumanFeedback] = None
    
    reflection_iteration_count: int = 0
    max_reflection_iterations: int = 3
    
    route_decision: Optional[str] = None
    errors: List[str] = Field(default_factory=list)

