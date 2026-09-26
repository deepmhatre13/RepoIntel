"""Software Architecture Analyst for RepoIntel (Phase 30).

Analyzes architecture graphs, component boundaries, execution flows, static data flow
approximations, deterministic architectural signals, and potential structural issues.
Generates evidence-grounded reports separating FACTS, INFERENCES, and RECOMMENDATIONS.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.knowledge.models import (
    SymbolModel,
    SymbolEdgeModel,
)
from packages.knowledge.architecture_analysis import ArchitectureAnalyzer, CycleDetector
from packages.agentic.llm_client import LLMClient


@dataclass
class ArchitectureComponent:
    name: str
    category: str  # API, Controller, Service, Repository/Data, Models, Configuration, Worker, Queue, Database, CLI, External, Tests
    file_paths: List[str]
    symbol_names: List[str]
    confidence: float
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "file_count": len(self.file_paths),
            "symbol_count": len(self.symbol_names),
            "file_paths": self.file_paths,
            "symbol_names": self.symbol_names,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class ExecutionPath:
    entrypoint: str
    path: List[str]
    edge_types: List[str]
    confidence: float
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entrypoint": self.entrypoint,
            "path": self.path,
            "edge_types": self.edge_types,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class ArchitectureIssue:
    issue_type: str  # circular_dependency, high_coupling, layer_bypass, cross_repo_coupling, architecture_drift
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    affected_components: List[str]
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity,
            "confidence": self.confidence,
            "affected_components": self.affected_components,
            "evidence": self.evidence,
        }


class SoftwareArchitectureAnalyst:
    """Evidence-grounded software architecture analyzer."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.analyzer = ArchitectureAnalyzer()
        self.cycle_detector = CycleDetector()
        self.llm_client = llm_client or LLMClient()

    async def analyze(self, session: AsyncSession, repo_id: str) -> Dict[str, Any]:
        # 1. Base architecture snapshot
        snapshot = await self.analyzer.build_snapshot(session, repo_id, commit_sha=repo_id)
        modules = snapshot.get("modules", [])
        edges = snapshot.get("module_edges", [])

        # 2. Detect Components
        components = await self._detect_components(session, repo_id, modules)

        # 3. Execution Paths
        execution_paths = await self._trace_execution_paths(session, repo_id)

        # 4. Data Flow Approximation
        data_flow = self._approximate_data_flow(execution_paths, components)

        # 5. Detect Architectural Issues
        issues = await self._detect_issues(session, repo_id, edges, components)

        # 6. Architectural Signals
        fan_in_map = {}
        fan_out_map = {}
        for e in edges:
            src = e.get("source_component", "")
            tgt = e.get("target_component", "")
            fan_out_map[src] = fan_out_map.get(src, 0) + 1
            fan_in_map[tgt] = fan_in_map.get(tgt, 0) + 1

        signals = {
            "component_count": len(components),
            "module_count": len(modules),
            "edge_count": len(edges),
            "cycle_count": len([i for i in issues if i.issue_type == "circular_dependency"]),
            "max_fan_in": max(fan_in_map.values()) if fan_in_map else 0,
            "max_fan_out": max(fan_out_map.values()) if fan_out_map else 0,
        }

        # 7. Formulate Facts, Inferences, Recommendations
        facts = [
            f"Repository contains {len(modules)} architectural modules and {len(components)} detected logical components.",
            f"Detected {len(edges)} cross-module dependency relationships.",
            f"Found {len(execution_paths)} traced entrypoint execution flows.",
        ]

        inferences = []
        if signals["cycle_count"] > 0:
            inferences.append("Circular dependency detected indicating potential module coupling debt.")
        if signals["max_fan_in"] > 5:
            inferences.append("High central component fan-in suggests critical shared dependency points.")
        if not inferences:
            inferences.append("Architecture maintains clean unidirectional module dependency boundaries.")

        recommendations = []
        if signals["cycle_count"] > 0:
            recommendations.append("Break detected cyclic module relationships using dependency inversion.")
        if issues:
            recommendations.append("Review detected layer boundary warnings in potential architecture issues list.")
        if not recommendations:
            recommendations.append("Maintain existing modular separation during upcoming feature extensions.")

        # Optional LLM interpretation summary
        llm_summary = self.llm_client.complete(
            prompt=f"Summarize architecture for repo {repo_id}: {signals}",
            system="You are an expert Software Architecture Analyst. Be concise, evidence-grounded, and objective."
        )

        return {
            "repo_id": repo_id,
            "architecture_summary": llm_summary,
            "facts": facts,
            "inferences": inferences,
            "recommendations": recommendations,
            "components": [c.to_dict() for c in components],
            "execution_paths": [p.to_dict() for p in execution_paths],
            "data_flow_approximation": data_flow,
            "signals": signals,
            "potential_issues": [i.to_dict() for i in issues],
            "confidence": 0.95,
        }

    async def _detect_components(
        self, session: AsyncSession, repo_id: str, modules: List[str]
    ) -> List[ArchitectureComponent]:
        components = []
        for mod in modules:
            stmt = select(SymbolModel).where(
                SymbolModel.repo_id == repo_id, SymbolModel.file_path.like(f"%{mod}%")
            )
            syms = (await session.execute(stmt)).scalars().all()
            file_paths = list(set([s.file_path for s in syms]))
            symbol_names = [s.symbol_name for s in syms]

            cat = "Service"
            mod_low = mod.lower()
            if "api" in mod_low or "route" in mod_low or "endpoint" in mod_low:
                cat = "API"
            elif "controller" in mod_low:
                cat = "Controller"
            elif "repo" in mod_low or "db" in mod_low or "store" in mod_low:
                cat = "Repository/Data Layer"
            elif "model" in mod_low or "schema" in mod_low:
                cat = "Models"
            elif "config" in mod_low or "setting" in mod_low:
                cat = "Configuration"
            elif "test" in mod_low:
                cat = "Tests"
            elif "cli" in mod_low or "cmd" in mod_low:
                cat = "CLI"

            components.append(
                ArchitectureComponent(
                    name=mod,
                    category=cat,
                    file_paths=file_paths,
                    symbol_names=symbol_names,
                    confidence=0.9,
                    evidence={"source": "StaticASTAnalysis", "module": mod},
                )
            )
        return components

    async def _trace_execution_paths(
        self, session: AsyncSession, repo_id: str
    ) -> List[ExecutionPath]:
        stmt = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id, SymbolEdgeModel.edge_type == "calls"
        ).limit(10)
        edges = (await session.execute(stmt)).scalars().all()
        paths = []
        for e in edges:
            if e.source_symbol and e.target_symbol:
                paths.append(
                    ExecutionPath(
                        entrypoint=e.source_symbol,
                        path=[e.source_symbol, e.target_symbol],
                        edge_types=[e.edge_type],
                        confidence=0.95,
                        evidence={"source": "SymbolEdgeModel", "edge_id": e.id},
                    )
                )
        return paths

    def _approximate_data_flow(
        self, execution_paths: List[ExecutionPath], components: List[ArchitectureComponent]
    ) -> Dict[str, Any]:
        stages = {
            "input": [c.name for c in components if c.category in ("API", "CLI", "Controller")],
            "validation_and_processing": [c.name for c in components if c.category == "Service"],
            "persistence": [c.name for c in components if c.category == "Repository/Data Layer"],
        }
        return {
            "pipeline": ["input", "validation_and_processing", "persistence"],
            "stage_components": stages,
            "analysis_method": "static_data_flow_approximation",
        }

    async def _detect_issues(
        self,
        session: AsyncSession,
        repo_id: str,
        edges: List[Dict[str, Any]],
        components: List[ArchitectureComponent],
    ) -> List[ArchitectureIssue]:
        issues = []
        # Cycle detection
        cycles = self.cycle_detector.detect_cycles(edges)
        for cyc in cycles:
            issues.append(
                ArchitectureIssue(
                    issue_type="circular_dependency",
                    description=f"Circular dependency detected across modules: {' -> '.join(cyc)}",
                    severity="HIGH",
                    confidence=0.95,
                    affected_components=cyc,
                    evidence={"source": "CycleDetector", "cycle": cyc},
                )
            )

        # High coupling detection
        comp_counts = {}
        for e in edges:
            tgt = e.get("target_component", "")
            comp_counts[tgt] = comp_counts.get(tgt, 0) + 1
        for comp, count in comp_counts.items():
            if count >= 5:
                issues.append(
                    ArchitectureIssue(
                        issue_type="high_coupling",
                        description=f"Component '{comp}' has high fan-in ({count} dependent modules).",
                        severity="MEDIUM",
                        confidence=0.9,
                        affected_components=[comp],
                        evidence={"source": "CouplingAnalysis", "fan_in": count},
                    )
                )
        return issues
