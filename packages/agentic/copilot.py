"""Unified Engineering Copilot Backend for RepoIntel (Phase 39).

Single unified orchestration entrypoint integrating all repository intelligence capabilities:
Orchestrator, Search, Investigation, Architecture, Health, Root Cause Analysis, Impact,
Change Simulation, Decision Intelligence, and Report Generator.

Enforces standardized EngineeringCopilotRequest and EngineeringCopilotResponse schemas with provenance.
"""

import uuid
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from packages.agentic.query_orchestrator import UnifiedQueryOrchestrator
from packages.agentic.autonomous_investigator import AutonomousRepositoryInvestigator
from packages.knowledge.health_engine import RepositoryHealthEngine
from packages.knowledge.root_cause_analysis import RootCauseAnalysisEngine
from packages.knowledge.change_simulation import EngineeringChangeSimulator
from packages.agentic.llm_client import LLMClient
from packages.shared.llm_mode import LLMMode


class CopilotMode(str, Enum):
    ASK = "ASK"
    SEARCH = "SEARCH"
    INVESTIGATE = "INVESTIGATE"
    ARCHITECTURE = "ARCHITECTURE"
    HEALTH = "HEALTH"
    ROOT_CAUSE = "ROOT_CAUSE"
    IMPACT = "IMPACT"
    SIMULATE_CHANGE = "SIMULATE_CHANGE"
    DECISION = "DECISION"
    REPORT = "REPORT"


class EngineeringCopilotRequest(BaseModel):
    repository_id: Optional[str] = None
    repository_group_id: Optional[str] = None
    query: str
    mode: CopilotMode = CopilotMode.ASK
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)


class EngineeringCopilotResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    mode: CopilotMode
    answer: str
    summary: str
    facts: List[str] = Field(default_factory=list)
    inferences: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    affected_components: List[str] = Field(default_factory=list)
    related_files: List[str] = Field(default_factory=list)
    related_symbols: List[str] = Field(default_factory=list)
    related_commits: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    unresolved_questions: List[str] = Field(default_factory=list)
    human_review_required: bool = False
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)
    llm_mode: LLMMode = "LIVE_GEMINI"
    degraded: bool = False


class EngineeringCopilotBackend:
    """Master backend orchestrator for RepoIntel Copilot."""

    def __init__(self, facade: Any, llm_client: Optional[LLMClient] = None):
        self.facade = facade
        self.llm_client = llm_client or LLMClient()
        self.orchestrator = UnifiedQueryOrchestrator(facade, llm_client)
        self.investigator = AutonomousRepositoryInvestigator(facade, llm_client)
        self.health_engine = RepositoryHealthEngine(facade)
        self.rca_engine = RootCauseAnalysisEngine(facade, llm_client)
        self.simulator = EngineeringChangeSimulator(facade, llm_client)

    async def execute(self, req: EngineeringCopilotRequest) -> EngineeringCopilotResponse:
        response = await self._execute(req)
        available = self.llm_client.is_available()
        response.llm_mode = "LIVE_GEMINI" if available else "DEGRADED_FALLBACK"
        response.degraded = not available
        response.execution_metadata.update({
            "llm_mode": response.llm_mode,
            "degraded": response.degraded,
        })
        return response

    async def _execute(self, req: EngineeringCopilotRequest) -> EngineeringCopilotResponse:
        repo_id = req.repository_id or "default_repo"
        mode = req.mode

        if mode == CopilotMode.SEARCH:
            search_res = await self.facade.search_code(repo_id, req.query)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Found {len(search_res.get('results', []))} search results for query '{req.query}'.",
                summary=f"Search executed for '{req.query}'.",
                facts=[f"Matching symbols: {search_res.get('related_symbols', [])}"],
                evidence=search_res.get("evidence", []),
                related_files=search_res.get("related_files", []),
                related_symbols=search_res.get("related_symbols", []),
                confidence=search_res.get("confidence", 0.9),
            )

        elif mode == CopilotMode.INVESTIGATE:
            inv_state = await self.investigator.investigate_autonomously(repo_id, req.query)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Investigation completed with status {inv_state.sufficiency_status.value}.",
                summary=f"Autonomous investigation for '{req.query}'.",
                facts=inv_state.facts,
                inferences=inv_state.inferences,
                recommendations=inv_state.recommendations,
                evidence=inv_state.evidence,
                unresolved_questions=inv_state.unresolved_questions,
                confidence=inv_state.confidence,
                human_review_required=inv_state.sufficiency_status == "CONFLICTING",
            )

        elif mode == CopilotMode.ARCHITECTURE:
            arch = await self.facade.analyze_architecture(repo_id)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Architecture analysis completed for repository '{repo_id}'.",
                summary="System component architecture snapshot and signals.",
                facts=arch.get("facts", []),
                inferences=arch.get("inferences", []),
                recommendations=arch.get("recommendations", []),
                confidence=arch.get("confidence", 0.95),
            )

        elif mode == CopilotMode.HEALTH:
            health = await self.health_engine.compute_health(self.facade.session, repo_id)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Repository health status is {health.overall_status.value}.",
                summary=f"Health signals evaluated across {len(health.dimensions)} dimensions.",
                facts=health.facts,
                inferences=health.inferences,
                recommendations=health.recommendations,
                evidence=[s.to_dict() for s in health.signals],
                confidence=0.95,
                human_review_required=health.overall_status.value in ("RISK", "CRITICAL"),
            )

        elif mode == CopilotMode.ROOT_CAUSE:
            rca = await self.rca_engine.analyze_root_cause(self.facade.session, repo_id, req.query)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Root cause analysis identified {len(rca.candidate_causes)} candidate causes.",
                summary=f"RCA completed for symptom '{req.query}'.",
                facts=rca.facts,
                inferences=rca.inferences,
                recommendations=rca.recommendations,
                evidence=rca.strongest_evidence,
                affected_components=rca.affected_components,
                confidence=rca.confidence,
            )

        elif mode == CopilotMode.IMPACT:
            impact = await self.facade.analyze_impact(repo_id, req.query)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Impact analysis completed for target '{req.query}'.",
                summary=f"Total direct dependents: {impact.get('total_dependents', 0)}.",
                facts=[f"Affected files: {impact.get('affected_files', [])}"],
                related_files=impact.get("affected_files", []),
                related_symbols=impact.get("affected_symbols", []),
                confidence=0.9,
            )

        elif mode == CopilotMode.SIMULATE_CHANGE:
            sim = await self.simulator.simulate_change(self.facade.session, repo_id, req.query)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Change simulation predicted risk level {sim.risk_level}.",
                summary=f"Simulated consequences for proposed change '{req.query}'.",
                facts=sim.facts,
                inferences=sim.inferences,
                recommendations=sim.recommendations,
                affected_components=sim.affected_components,
                related_files=sim.affected_files,
                related_symbols=sim.affected_symbols,
                confidence=0.9,
                human_review_required=sim.risk_level in ("HIGH", "CRITICAL"),
            )

        elif mode == CopilotMode.DECISION:
            dec = await self.facade.evaluate_engineering_decision(repo_id, req.query)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Decision recommendation: {dec.get('decision')}.",
                summary=f"Evaluated {len(dec.get('alternatives', []))} decision alternatives.",
                facts=dec.get("facts", []),
                inferences=dec.get("inferences", []),
                recommendations=dec.get("recommendations", []),
                evidence=dec.get("evidence", []),
                confidence=dec.get("confidence", 0.9),
            )

        elif mode == CopilotMode.REPORT:
            rep = await self.facade.generate_engineering_report(repo_id, req.query or "ENGINEERING_OVERVIEW")
            return EngineeringCopilotResponse(
                mode=mode,
                answer=f"Generated {rep.get('report_type')} report for '{repo_id}'.",
                summary=rep.get("title", ""),
                facts=rep.get("facts", []),
                inferences=rep.get("inferences", []),
                recommendations=rep.get("recommendations", []),
                confidence=rep.get("confidence", 0.95),
            )

        else:  # ASK / DEFAULT
            plan = self.orchestrator.plan_query(req.query)
            orch_res = await self.orchestrator.execute_plan(repo_id, plan)
            return EngineeringCopilotResponse(
                mode=mode,
                answer=orch_res.get("orchestrated_answer", ""),
                summary=f"Orchestrated query execution for '{req.query}'.",
                facts=[f"Intent: {plan.intent.value}, Complexity: {plan.complexity.value}"],
                inferences=["Execution completed across planned repository evidence capabilities."],
                confidence=orch_res.get("confidence", 0.9),
            )
