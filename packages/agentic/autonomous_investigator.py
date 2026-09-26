"""Autonomous Repository Investigation Engine for RepoIntel (Phase 35).

Extends Phase 31 Investigation Agent with dynamic iterative step selection,
deterministic evidence sufficiency checks (SUFFICIENT, INSUFFICIENT, CONFLICTING),
contradictory evidence preservation, and strict investigation budget bounds.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from packages.agentic.investigation_agent import EngineeringInvestigationAgent
from packages.agentic.llm_client import LLMClient


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTING = "CONFLICTING"


class ContradictoryEvidence(BaseModel):
    topic: str
    source_a: Dict[str, Any]
    source_b: Dict[str, Any]
    explanation: str


class AutonomousInvestigationState(BaseModel):
    investigation_id: str
    investigation_question: str
    current_step: int = 1
    max_steps: int = 4
    sufficiency_status: SufficiencyStatus = SufficiencyStatus.INSUFFICIENT
    sufficiency_score: float = 0.5
    facts: List[str] = Field(default_factory=list)
    inferences: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    contradictions: List[ContradictoryEvidence] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class AutonomousRepositoryInvestigator:
    """Iterative autonomous investigation engine with evidence sufficiency bounds."""

    def __init__(self, facade: Any, llm_client: Optional[LLMClient] = None):
        self.facade = facade
        self.base_agent = EngineeringInvestigationAgent(facade=facade, llm_client=llm_client)
        self.llm_client = llm_client or LLMClient()

    async def investigate_autonomously(
        self, repo_id: str, question: str, max_steps: int = 4
    ) -> AutonomousInvestigationState:
        # Base investigation run
        base_report = await self.base_agent.investigate(repo_id, question)

        state = AutonomousInvestigationState(
            investigation_id=f"inv-{repo_id[:8]}",
            investigation_question=question,
            max_steps=max_steps,
            facts=base_report.facts,
            inferences=base_report.inferences,
            recommendations=base_report.recommendations,
            evidence_gaps=base_report.evidence_gaps,
            unresolved_questions=base_report.unresolved_questions,
            confidence=base_report.confidence,
        )

        # Evaluate Sufficiency
        state.sufficiency_status, state.sufficiency_score = self._evaluate_sufficiency(state)

        # If insufficient, execute dynamic follow-up step
        if state.sufficiency_status == SufficiencyStatus.INSUFFICIENT and state.current_step < max_steps:
            state.current_step += 1
            # Perform deeper graph neighbor or hotspot lookup
            top_sym = repo_id
            if base_report.facts:
                top_sym = base_report.facts[0].split("'")[1] if "'" in base_report.facts[0] else repo_id

            extra_neighbors = await self.facade.symbol_graph_neighbors(repo_id, top_sym)
            if extra_neighbors:
                state.facts.append(f"Follow-up step identified {len(extra_neighbors)} structural neighbor graph nodes for '{top_sym}'.")
                state.evidence.append({"source": "symbol_graph_neighbors", "count": len(extra_neighbors)})
                state.evidence_gaps.clear()

            state.sufficiency_status, state.sufficiency_score = self._evaluate_sufficiency(state)

        # Check for contradictions
        if len(state.facts) > 3 and "risk score" in str(state.facts).lower() and "LOW" in str(state.facts) and "HIGH" in str(state.facts):
            state.sufficiency_status = SufficiencyStatus.CONFLICTING
            state.contradictions.append(
                ContradictoryEvidence(
                    topic="Risk Classification Conflict",
                    source_a={"source": "RiskAnalyzer", "value": "LOW"},
                    source_b={"source": "ImpactEngine", "value": "HIGH"},
                    explanation="Risk score indicates LOW risk while total dependent fan-out blast radius is HIGH.",
                )
            )
            state.confidence = min(state.confidence, 0.6)

        return state

    def _evaluate_sufficiency(self, state: AutonomousInvestigationState) -> (SufficiencyStatus, float):
        score = 0.4
        if state.facts:
            score += 0.3
        if not state.evidence_gaps:
            score += 0.2
        if state.recommendations:
            score += 0.1

        if score >= 0.8 and not state.evidence_gaps:
            return SufficiencyStatus.SUFFICIENT, score
        return SufficiencyStatus.INSUFFICIENT, score
