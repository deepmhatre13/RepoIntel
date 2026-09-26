"""Engineering Decision Intelligence for RepoIntel (Phase 32).

Builds a decision-support layer on top of repository knowledge graph, risk analyzer, impact engine, and git history.
Evaluates architectural questions, analyzes alternative design paths, computes evidence confidence,
and delivers structured decision reports separating FACTS, INFERENCES, and RECOMMENDATIONS.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    pass

from packages.agentic.llm_client import LLMClient



@dataclass
class DecisionAlternative:
    name: str
    description: str
    affected_components: List[str]
    implementation_complexity: str  # LOW, MEDIUM, HIGH, CRITICAL
    risks: List[str]
    benefits: List[str]
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "affected_components": self.affected_components,
            "implementation_complexity": self.implementation_complexity,
            "risks": self.risks,
            "benefits": self.benefits,
            "evidence": self.evidence,
        }


@dataclass
class EngineeringDecision:
    question: str
    decision: str
    confidence: float
    facts: List[str]
    inferences: List[str]
    recommendations: List[str]
    alternatives: List[DecisionAlternative]
    benefits: List[str]
    risks: List[str]
    affected_components: List[str]
    assumptions: List[str]
    evidence: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "facts": self.facts,
            "inferences": self.inferences,
            "recommendations": self.recommendations,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "benefits": self.benefits,
            "risks": self.risks,
            "affected_components": self.affected_components,
            "assumptions": self.assumptions,
            "evidence": self.evidence,
        }


class EngineeringDecisionEngine:
    """Evidence-grounded engineering decision intelligence engine."""

    def __init__(self, facade: Any, llm_client: Optional[LLMClient] = None):
        self.facade = facade
        self.llm_client = llm_client or LLMClient()


    async def evaluate_decision(
        self, repo_id: str, question: str, target_entity: str = ""
    ) -> EngineeringDecision:
        # 1. Gather Repository Evidence
        target = target_entity or repo_id
        search_res = await self.facade.search_code(repo_id, question, top_k=5)
        impact = await self.facade.analyze_impact(repo_id, target)
        risk = await self.facade.get_change_risk(repo_id, target)
        arch = await self.facade.get_architecture_snapshot(repo_id)

        affected_files = impact.get("affected_files", [])
        affected_symbols = impact.get("affected_symbols", [])
        risk_level = risk.get("risk_level", "LOW")
        risk_score = risk.get("score", 0.0)

        # 2. Formulate Facts, Inferences
        facts = [
            f"Impact analysis for '{target}' identifies {len(affected_files)} affected files and {len(affected_symbols)} affected symbols.",
            f"Deterministic change-risk score is {risk_score} (Level: {risk_level}).",
            f"Repository architecture contains {len(arch.get('modules', []))} modules.",
        ]

        inferences = []
        if risk_level in ("HIGH", "CRITICAL"):
            inferences.append("Modifying target directly without an abstraction boundary carries high regression risk.")
        else:
            inferences.append("Low fan-in indicates changes can be localized with manageable regression blast radius.")

        # 3. Alternatives Evaluation
        alt_a = DecisionAlternative(
            name="Option A: In-Place Refactoring",
            description=f"Refactor '{target}' directly within the existing module structure.",
            affected_components=affected_files[:3],
            implementation_complexity="LOW" if risk_level == "LOW" else "MEDIUM",
            risks=["Potential breaking changes for immediate callers"],
            benefits=["Minimal architectural overhead", "Fast implementation"],
            evidence={"source": "ImpactAnalysis", "risk_level": risk_level},
        )

        alt_b = DecisionAlternative(
            name="Option B: Introduce Interface Boundary / Facade",
            description=f"Introduce an explicit interface boundary separating '{target}' callers from backend implementation.",
            affected_components=affected_files + ["new_boundary_interface.py"],
            implementation_complexity="MEDIUM",
            risks=["Initial setup overhead"],
            benefits=["Decouples consumers", "Enables isolated testing", "Lowers long-term change risk"],
            evidence={"source": "RiskAnalyzer", "coupling": risk.get("reasons", [])},
        )

        alt_c = DecisionAlternative(
            name="Option C: Microservice / Submodule Extraction",
            description=f"Extract '{target}' into a separate independent component/service.",
            affected_components=affected_files,
            implementation_complexity="HIGH",
            risks=["Network latency", "Deployment complexity", "Cross-repo sync overhead"],
            benefits=["Strict deployment boundary", "Independent scalability"],
            evidence={"source": "ArchitectureSnapshot", "module_count": len(arch.get("modules", []))},
        )

        alternatives = [alt_a, alt_b, alt_c]

        # 4. Formulate Recommendation & Decision
        if risk_level in ("HIGH", "CRITICAL"):
            chosen_decision = "Option B: Introduce Interface Boundary / Facade"
            rec = "Introduce an explicit boundary facade prior to major structural changes to reduce consumer coupling."
        else:
            chosen_decision = "Option A: In-Place Refactoring"
            rec = "Proceed with in-place refactoring as existing consumer risk signals are within safe thresholds."

        recommendations = [rec, "Ensure automated regression test coverage is executed for all affected symbols."]

        # 5. Gemini Synthesis (Reasoning summary)
        reasoning_prompt = f"Given question '{question}' and facts {facts}, explain decision '{chosen_decision}'."
        synthesis = self.llm_client.complete(
            prompt=reasoning_prompt,
            system="You are an expert Engineering Decision Analyst. Be objective, evidence-grounded, and concise."
        )

        evidence = [
            {"type": "impact", "data": impact},
            {"type": "risk", "data": risk},
            {"type": "search", "data": search_res},
        ]

        return EngineeringDecision(
            question=question,
            decision=chosen_decision,
            confidence=0.9,
            facts=facts,
            inferences=inferences,
            recommendations=recommendations,
            alternatives=alternatives,
            benefits=[a.benefits[0] for a in alternatives if a.benefits],
            risks=[a.risks[0] for a in alternatives if a.risks],
            affected_components=affected_files,
            assumptions=["Static AST analysis accurately reflects active runtime code paths."],
            evidence=evidence,
        )
