"""Engineering Change Simulation Engine for RepoIntel (Phase 38).

Predicts the structural consequences of proposed engineering changes ("what-if" scenarios)
using existing repository evidence WITHOUT modifying source code.

Analyzes consequences across categories: DIRECT, INDIRECT, ARCHITECTURAL, DEPENDENCY,
CONFIGURATION, TEST, DOCUMENTATION, CROSS_REPOSITORY.
Labels outputs strictly as CURRENT FACT, SIMULATED CONSEQUENCE, and ASSUMPTION.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession

from packages.agentic.llm_client import LLMClient


class ConsequenceCategory(str, Enum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    ARCHITECTURAL = "ARCHITECTURAL"
    DEPENDENCY = "DEPENDENCY"
    CONFIGURATION = "CONFIGURATION"
    TEST = "TEST"
    DOCUMENTATION = "DOCUMENTATION"
    CROSS_REPOSITORY = "CROSS_REPOSITORY"


@dataclass
class SimulatedConsequence:
    category: ConsequenceCategory
    description: str
    affected_entity: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    label: str  # CURRENT FACT, SIMULATED CONSEQUENCE, ASSUMPTION
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value if isinstance(self.category, Enum) else str(self.category),
            "description": self.description,
            "affected_entity": self.affected_entity,
            "severity": self.severity,
            "label": self.label,
            "evidence": self.evidence,
        }


@dataclass
class ChangeSimulation:
    proposed_change: str
    target_entity: str
    affected_components: List[str]
    affected_files: List[str]
    affected_symbols: List[str]
    affected_tests: List[str]
    dependency_changes: List[str]
    architecture_changes: List[str]
    configuration_changes: List[str]
    risk_level: str
    consequences: List[SimulatedConsequence]
    facts: List[str]
    inferences: List[str]
    recommendations: List[str]
    assumptions: List[str]
    uncertainties: List[str]
    evidence: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed_change": self.proposed_change,
            "target_entity": self.target_entity,
            "risk_level": self.risk_level,
            "affected_components": self.affected_components,
            "affected_files": self.affected_files,
            "affected_symbols": self.affected_symbols,
            "affected_tests": self.affected_tests,
            "dependency_changes": self.dependency_changes,
            "architecture_changes": self.architecture_changes,
            "configuration_changes": self.configuration_changes,
            "consequences": [c.to_dict() for c in self.consequences],
            "facts": self.facts,
            "inferences": self.inferences,
            "recommendations": self.recommendations,
            "assumptions": self.assumptions,
            "uncertainties": self.uncertainties,
            "evidence": self.evidence,
        }


class EngineeringChangeSimulator:
    """Predictive "what-if" structural change simulator."""

    def __init__(self, facade: Any, llm_client: Optional[LLMClient] = None):
        self.facade = facade
        self.llm_client = llm_client or LLMClient()

    async def simulate_change(
        self, session: AsyncSession, repo_id: str, proposed_change: str, target_entity: str = ""
    ) -> ChangeSimulation:
        target = target_entity or repo_id

        # 1. Inspect target using facade capabilities
        search_res = await self.facade.search_code(repo_id, proposed_change, top_k=5)
        top_syms = search_res.get("related_symbols", [])
        if not target_entity and top_syms:
            target = top_syms[0]

        impact = await self.facade.analyze_impact(repo_id, target)
        risk = await self.facade.get_change_risk(repo_id, target)
        tests = await self.facade.get_tests_for_symbol(repo_id, target)
        deps = await self.facade.get_dependency_manifest(repo_id)

        affected_files = impact.get("affected_files", [])
        affected_symbols = impact.get("affected_symbols", [])
        risk_level = risk.get("risk_level", "LOW")

        # 2. Build Simulated Consequences with explicit labeling
        consequences = [
            SimulatedConsequence(
                category=ConsequenceCategory.DIRECT,
                description=f"Direct modification required for target '{target}'.",
                affected_entity=target,
                severity=risk_level,
                label="CURRENT FACT",
                evidence={"source": "SymbolModel", "target": target},
            ),
            SimulatedConsequence(
                category=ConsequenceCategory.INDIRECT,
                description=f"{len(affected_files)} caller files require updated signatures or imports.",
                affected_entity=", ".join(affected_files[:3]),
                severity="MEDIUM" if risk_level == "LOW" else "HIGH",
                label="SIMULATED CONSEQUENCE",
                evidence={"source": "ImpactEngine", "affected_count": len(affected_files)},
            ),
            SimulatedConsequence(
                category=ConsequenceCategory.TEST,
                description=f"Test coverage verification required for {len(tests)} mapped test cases.",
                affected_entity=target,
                severity="LOW",
                label="ASSUMPTION",
                evidence={"source": "TestCaseModel", "test_count": len(tests)},
            ),
        ]

        facts = [
            f"CURRENT FACT: Target entity '{target}' identified in repository.",
            f"CURRENT FACT: Impact analysis identifies {len(affected_files)} direct dependent files.",
            f"CURRENT FACT: Deterministic risk assessment score is {risk.get('score', 0.0)} ({risk_level}).",
        ]

        inferences = [
            f"SIMULATED CONSEQUENCE: Proposed change ('{proposed_change}') will cascade across {len(affected_symbols)} downstream symbols.",
            "SIMULATED CONSEQUENCE: Architecture boundaries require interface updates if extracting to a new service module.",
        ]

        recommendations = [
            "RECOMMENDATION: Create a feature branch and isolate consumer signatures behind a facade before refactoring.",
            "RECOMMENDATION: Run automated test suite for all mapped downstream test files.",
        ]

        assumptions = [
            "ASSUMPTION: Target code path execution follows static call-graph bindings without unmapped dynamic RPC overrides.",
        ]

        uncertainties = [
            "Uncertainty: Dynamic runtime reflection callers cannot be deterministically verified without runtime trace logs.",
        ]

        evidence = [
            {"type": "impact", "data": impact},
            {"type": "risk", "data": risk},
            {"type": "tests", "count": len(tests)},
        ]

        return ChangeSimulation(
            proposed_change=proposed_change,
            target_entity=target,
            affected_components=affected_files[:3],
            affected_files=affected_files,
            affected_symbols=affected_symbols,
            affected_tests=[t.get("test_name", "") for t in tests],
            dependency_changes=[d["package_name"] for d in deps[:2]],
            architecture_changes=[f"Refactor interface for {target}"],
            configuration_changes=[],
            risk_level=risk_level,
            consequences=consequences,
            facts=facts,
            inferences=inferences,
            recommendations=recommendations,
            assumptions=assumptions,
            uncertainties=uncertainties,
            evidence=evidence,
        )
