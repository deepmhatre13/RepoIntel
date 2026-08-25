"""Impact capability mixin for RepositoryFacade.

Owns impact-related deterministic analysis: the Phase 20 impact function plus
change-requirement impact analysis. Implementation detail, not exposed to agents.
"""

from typing import List, Dict, Any

from sqlalchemy import select

from packages.knowledge.models import SymbolEdgeModel

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class ImpactCapabilities:
    """Impact and change-requirement analysis for RepositoryFacade."""

    async def analyze_impact(self, repo_id: str, target: str) -> Dict[str, Any]:
        """Analyze the deterministic impact of changing a symbol or file."""
        stmt = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id,
            (SymbolEdgeModel.target_symbol == target) | (SymbolEdgeModel.target_file == target),
        )
        res = await self.session.execute(stmt)
        edges = res.scalars().all()

        direct_symbols = list(set([e.source_symbol for e in edges]))
        direct_files = list(set([e.source_file for e in edges if e.source_file]))

        tests = [f for f in direct_files if "test" in f.lower()]
        configs = [
            f
            for f in direct_files
            if "config" in f.lower() or "setting" in f.lower() or f.endswith((".yaml", ".json", ".toml", ".env"))
        ]
        docs = [f for f in direct_files if f.endswith((".md", ".rst"))]

        total_dependents = len(direct_symbols) + len(direct_files)
        if total_dependents == 0:
            risk = "LOW"
        elif total_dependents < 5:
            risk = "MEDIUM"
        elif total_dependents < 15:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        return {
            "target": target,
            "affected_files": direct_files,
            "affected_symbols": direct_symbols,
            "test_impact": tests,
            "configuration_impact": configs,
            "documentation_impact": docs,
            "risk_level": risk,
            "total_dependents": total_dependents,
        }

    async def analyze_change_requirement(
        self, repo_id: str, objective: str, target_entities: List[str]
    ) -> Dict[str, Any]:
        """Perform deterministic change-requirement analysis for a set of targets.

        For each target entity this aggregates:
          - direct dependents / files / tests (Phase 20 impact analysis),
          - deterministic change risk (RiskCapabilities),
          - test coverage proxy (per-symbol tests).

        The result is a deterministic plan skeleton the caller (usually an agent)
        can use to sequence the change and select tests to run.
        """
        affected_symbols: List[str] = []
        affected_files: List[str] = []
        related_tests: List[str] = []
        per_target: List[Dict[str, Any]] = []
        risks: List[str] = []
        affected_domains: set = set()

        for entity in target_entities:
            impact = await self.analyze_impact(repo_id, entity)
            risk = impact.get("risk_level", "LOW") if isinstance(impact, dict) else "LOW"
            risks.append(risk)

            affected_symbols.extend(impact.get("affected_symbols", []))
            affected_files.extend(impact.get("affected_files", []))
            related_tests.extend(impact.get("test_impact", []))

            tests = await self.get_tests_for_symbol(repo_id, entity)
            related_tests.extend(t["test_name"] for t in tests)

            for f in impact.get("affected_files", []):
                parts = [p for p in f.replace("\\", "/").split("/") if p]
                if parts:
                    affected_domains.add(parts[0])

            change_risk = await self.get_change_risk(repo_id, entity)
            per_target.append(
                {
                    "target": entity,
                    "impact_risk_level": risk,
                    "change_risk_level": change_risk.get("risk_level", "LOW") if isinstance(change_risk, dict) else "LOW",
                    "change_risk_score": change_risk.get("score", 0.0) if isinstance(change_risk, dict) else 0.0,
                    "direct_dependents": impact.get("total_dependents", 0),
                    "risk_reasons": change_risk.get("reasons", []) if isinstance(change_risk, dict) else [],
                }
            )

        # Aggregate risk level across targets (max severity wins).
        aggregate_risk = max(risks, key=lambda r: _RISK_ORDER.get(r, 0)) if risks else "LOW"

        recommended_tests = sorted(set(related_tests))
        uncertainties: List[str] = []
        if not recommended_tests:
            uncertainties.append("No tests identified by static analysis; runtime verification recommended.")
        if not affected_domains:
            uncertainties.append("No affected source domains were inferred; verify module boundaries.")

        # Deterministic, risk-ordered plan skeleton.
        ordered_targets = sorted(
            per_target,
            key=lambda t: (_RISK_ORDER.get(t.get("change_risk_level", "LOW"), 0), -t.get("change_risk_score", 0.0)),
            reverse=True,
        )
        plan_steps = [
            {
                "step_number": idx + 1,
                "target": t["target"],
                "action": "modify",
                "risk_level": t.get("change_risk_level", "LOW"),
                "risk_reasons": t.get("risk_reasons", []),
            }
            for idx, t in enumerate(ordered_targets)
        ]

        return {
            "objective": objective,
            "requirement": objective,
            "target_entities": target_entities,
            "likely_affected_domains": sorted(affected_domains),
            "affected_files": sorted(set(affected_files)),
            "affected_symbols": sorted(set(affected_symbols)),
            "recommended_tests_to_run": recommended_tests,
            "aggregate_risk_level": aggregate_risk,
            "per_target_analysis": per_target,
            "uncertainties": uncertainties,
            "plan_steps": plan_steps,
        }