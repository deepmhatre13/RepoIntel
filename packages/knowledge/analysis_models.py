from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class EntrypointTrace(BaseModel):
    entrypoint: str
    first_level_calls: List[str] = Field(default_factory=list)
    major_modules_reached: List[str] = Field(default_factory=list)
    config_dependencies: List[str] = Field(default_factory=list)

class ChangeSummary(BaseModel):
    commit_sha: str
    author: str
    message: str
    affected_modules: List[str] = Field(default_factory=list)
    affected_symbols: List[str] = Field(default_factory=list)
    change_type: str = "MODIFIED"
    summary: str = ""
    evidence: List[Dict[str, Any]] = Field(default_factory=list)

class ImpactReport(BaseModel):
    target: str
    target_type: str # file, symbol, dependency, config, endpoint
    affected_files: List[str] = Field(default_factory=list)
    affected_symbols: List[str] = Field(default_factory=list)
    dependency_impact: List[str] = Field(default_factory=list)
    configuration_impact: List[str] = Field(default_factory=list)
    test_impact: List[str] = Field(default_factory=list)
    documentation_impact: List[str] = Field(default_factory=list)
    risk_level: str = "LOW" # LOW, MEDIUM, HIGH, CRITICAL
    reasoning: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)

class ChangePlanStep(BaseModel):
    step_number: int
    component: str
    action: str
    risk_level: str = "LOW"
    risk_reasons: List[str] = Field(default_factory=list)
    prerequisite_steps: List[int] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)

class ChangeRequirement(BaseModel):
    objective: str
    target_entities: List[str] = Field(default_factory=list)
    likely_affected_domains: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)

class ChangePlan(BaseModel):
    requirement: str
    target: str
    requirement_analysis: Optional[ChangeRequirement] = None
    steps: List[ChangePlanStep] = Field(default_factory=list)
    recommended_tests_to_run: List[str] = Field(default_factory=list)
    rollout_considerations: List[str] = Field(default_factory=list)

class QAAnswer(BaseModel):
    query: str
    answer: str
    confidence: str = "HIGH" # HIGH, MEDIUM, LOW
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    related_files: List[str] = Field(default_factory=list)
    related_symbols: List[str] = Field(default_factory=list)
    related_commits: List[str] = Field(default_factory=list)

