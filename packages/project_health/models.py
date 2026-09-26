"""Pydantic data models for Project Health Intelligence.

Structured schemas returned by the ``/project-health/*`` API. Every conclusion
carries: ``metric``, ``value``, ``previous_value``, ``delta``, ``timestamp``,
``window``, ``source`` and ``confidence`` (evidence-first).

The missing-data ontology is explicit and never conflates ``0`` with
unavailable data:

* ``0``                    - a real measured zero
* ``NOT_AVAILABLE``        - the data source did not return a value
* ``INSUFFICIENT_DATA``    - too few observations for a reliable conclusion
* ``API_ERROR``            - the provider failed to retrieve the data
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DataAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    ZERO = "ZERO"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    API_ERROR = "API_ERROR"


class TrendDirection(str, Enum):
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class HealthState(str, Enum):
    GOOD = "GOOD"
    STABLE = "STABLE"
    DETERIORATING = "DETERIORATING"
    IMPROVING = "IMPROVING"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    NOT_AVAILABLE = "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# Repository metadata & provider-normalized records
# ---------------------------------------------------------------------------
class RepositoryMetadata(BaseModel):
    owner: str
    name: str
    full_name: str
    url: str
    stars: Optional[int] = None
    forks: Optional[int] = None
    watchers: Optional[int] = None
    open_issues: Optional[int] = None
    closed_issues: Optional[int] = None
    open_prs: Optional[int] = None
    merged_prs: Optional[int] = None
    contributors: Optional[int] = None
    created_at: Optional[datetime] = None
    pushed_at: Optional[datetime] = None
    default_branch: Optional[str] = None
    languages: Dict[str, int] = Field(default_factory=dict)
    license: Optional[str] = None
    archived: bool = False
    size_kb: Optional[int] = None
    description: Optional[str] = None


class CommitRecord(BaseModel):
    sha: str
    author: Optional[str] = None
    committed_at: Optional[datetime] = None
    additions: Optional[int] = None
    deletions: Optional[int] = None
    files_changed: Optional[int] = None
    message: str = ""


class PullRequestRecord(BaseModel):
    number: int
    author: Optional[str] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    state: str = "open"
    first_response_at: Optional[datetime] = None
    reviewer_count: int = 0
    comment_count: int = 0
    changed_files: Optional[int] = None
    additions: Optional[int] = None
    deletions: Optional[int] = None


class IssueRecord(BaseModel):
    number: int
    author: Optional[str] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    state: str = "open"
    comment_count: int = 0
    labels: List[str] = Field(default_factory=list)


class ReleaseRecord(BaseModel):
    tag: str
    published_at: Optional[datetime] = None
    prerelease: bool = False
    draft: bool = False


class ContributorRecord(BaseModel):
    login: str
    contributions: int = 0
    first_contribution: Optional[datetime] = None
    last_contribution: Optional[datetime] = None


class ProviderData(BaseModel):
    metadata: Optional[RepositoryMetadata] = None
    commits: List[CommitRecord] = Field(default_factory=list)
    pull_requests: List[PullRequestRecord] = Field(default_factory=list)
    issues: List[IssueRecord] = Field(default_factory=list)
    releases: List[ReleaseRecord] = Field(default_factory=list)
    contributors: List[ContributorRecord] = Field(default_factory=list)
    status: Dict[str, str] = Field(default_factory=dict)
    metadata_error: Optional[str] = None
# ---------------------------------------------------------------------------
# Temporal / statistical results
# ---------------------------------------------------------------------------
class MetricPoint(BaseModel):
    timestamp: str
    value: Optional[float] = None
    kind: str = "observed"  # observed | trend | forecast | prediction


class TrendResult(BaseModel):
    metric: str
    direction: TrendDirection = TrendDirection.NOT_AVAILABLE
    slope: Optional[float] = None
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    delta: Optional[float] = None
    window: Optional[str] = None
    observation_count: int = 0
    confidence: float = 0.0
    availability: str = DataAvailability.NOT_AVAILABLE.value
    points: List[MetricPoint] = Field(default_factory=list)


class CorrelationResult(BaseModel):
    metric_a: str
    metric_b: str
    correlation_coefficient: Optional[float] = None
    observation_window: Optional[str] = None
    sample_size: int = 0
    confidence: float = 0.0
    availability: str = DataAvailability.NOT_AVAILABLE.value
    note: str = ""


class AnomalyResult(BaseModel):
    metric: str
    kind: str = "anomaly"  # noise | persistent_change | anomaly
    timestamp: Optional[str] = None
    robust_zscore: Optional[float] = None
    value: Optional[float] = None
    baseline_median: Optional[float] = None
    baseline_mad: Optional[float] = None
    observation_count: int = 0
    description: str = ""


class ChangePointResult(BaseModel):
    metric: str
    index: Optional[int] = None
    timestamp: Optional[str] = None
    before_mean: Optional[float] = None
    after_mean: Optional[float] = None
    before_count: int = 0
    after_count: int = 0
    shift_magnitude: Optional[float] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Health dimensions & evidence
# ---------------------------------------------------------------------------
class HealthEvidence(BaseModel):
    metric: str
    value: Optional[float] = None
    previous_value: Optional[float] = None
    delta: Optional[float] = None
    timestamp: Optional[str] = None
    window: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    direction: str = TrendDirection.NOT_AVAILABLE.value
    availability: str = DataAvailability.NOT_AVAILABLE.value


class HealthDimension(BaseModel):
    name: str
    state: HealthState = HealthState.NOT_AVAILABLE
    trend: TrendDirection = TrendDirection.NOT_AVAILABLE
    confidence: float = 0.0
    score: Optional[float] = None
    availability: str = DataAvailability.NOT_AVAILABLE.value
    supporting_metrics: List[str] = Field(default_factory=list)
    evidence: List[HealthEvidence] = Field(default_factory=list)


class HealthRisk(BaseModel):
    name: str
    level: RiskLevel = RiskLevel.NOT_AVAILABLE
    trend: TrendDirection = TrendDirection.NOT_AVAILABLE
    confidence: float = 0.0
    supporting_evidence: List[HealthEvidence] = Field(default_factory=list)
    summary: str = ""


class HealthSignal(BaseModel):
    name: str
    dimension: str
    severity: str = "info"  # info | warning | critical
    direction: TrendDirection = TrendDirection.NOT_AVAILABLE
    description: str = ""
    evidence: List[HealthEvidence] = Field(default_factory=list)


class ScoreState(str, Enum):
    """Deterministic explainable state of the overall health score.

    The score is never invented / never faked precision. When too few
    dimensions are observed we say so explicitly rather than emitting a
    misleading 0-100 number.

    """

    COMPUTED = "COMPUTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_SCORABLE = "NOT_SCORABLE"


class HealthEvent(BaseModel):
    """An evidence-backed meaningful change emitted by the monitoring refresh.

    ``kind`` uses deterministic event names (e.g. HEALTH_DETERIORATED,
    HEALTH_IMPROVED, MAINTENANCE_RISK_INCREASED, CONTRIBUTOR_CONCENTRATION_INCREASED,
    RELEASE_STAGNATION_DETECTED, DEPENDENCY_RISK_INCREASED,
    SECURITY_SIGNAL_CHANGED, CODE_HEALTH_CHANGED).
    """

    kind: str
    dimension: str = ""
    severity: str = "info"   # info | warning
    summary: str = ""
    evidence: List[HealthEvidence] = Field(default_factory=list)
    observed_at: str = ""

# ---------------------------------------------------------------------------
# Top-level analysis result
# ---------------------------------------------------------------------------
class HealthAnalysis(BaseModel):
    analysis_id: str
    owner: str
    repository: str
    full_name: str
    url: str
    created_at: datetime
    availability: str = DataAvailability.AVAILABLE.value

    overall_score: Optional[float] = None
    score_state: str = ScoreState.NOT_SCORABLE.value
    missing_dimensions: List[str] = Field(default_factory=list)
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    trajectory: str = TrendDirection.NOT_AVAILABLE.value
    trajectory_confidence: float = 0.0

    dimensions: List[HealthDimension] = Field(default_factory=list)
    risks: List[HealthRisk] = Field(default_factory=list)
    signals: List[HealthSignal] = Field(default_factory=list)
    correlations: List[CorrelationResult] = Field(default_factory=list)
    anomalies: List[AnomalyResult] = Field(default_factory=list)
    change_points: List[ChangePointResult] = Field(default_factory=list)
    trends: List[TrendResult] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    why_changed: List[Dict[str, Any]] = Field(default_factory=list)
    changes: List[Dict[str, Any]] = Field(default_factory=list)

    contributors: List[ContributorRecord] = Field(default_factory=list)
    releases: List[ReleaseRecord] = Field(default_factory=list)
    repository_metadata: Optional[RepositoryMetadata] = None

    data_sufficiency: Dict[str, Any] = Field(default_factory=dict)
    ai_analysis: Optional[str] = None
    ai_available: bool = False

    gemini_analysis: Optional[str] = None
    gemini_available: bool = False

    reviewed: bool = False
    review_notes: List[str] = Field(default_factory=list)
    health_events: List[HealthEvent] = Field(default_factory=list)


class AnalyzeProjectHealthRequest(BaseModel):
    """Input for POST /project-health/analyze.

    Accepts either ``owner/repository`` or a full GitHub URL.
    """

    repository: str
    window_days: int = 365
    refresh: bool = True


class AskHealthQuestionRequest(BaseModel):
    question: str