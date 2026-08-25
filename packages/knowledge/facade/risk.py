"""Risk capability mixin for RepositoryFacade.

Owns deterministic engineering-risk intelligence: code hotspots, coupling
metrics, cycle detection, and change-risk scoring. Implementation detail, not
exposed to agents.
"""

from typing import List, Dict, Any

from sqlalchemy import select

from packages.knowledge.models import SymbolHistoryModel, SymbolEdgeModel
from packages.knowledge.architecture_analysis import CycleDetector as ArchCycleDetector
from packages.knowledge.risk_analysis import RiskScorer, CouplingAnalyzer


class RiskCapabilities:
    """Deterministic coupling / hotspot / risk intelligence for RepositoryFacade."""

    async def get_hotspots(self, repo_id: str) -> List[Dict[str, Any]]:
        """Compute code hotspots deterministically from commit history."""
        stmt = select(
            SymbolHistoryModel.symbol_name, SymbolHistoryModel.file_path
        ).where(SymbolHistoryModel.repo_id == repo_id)
        res = await self.session.execute(stmt)
        rows = res.all()

        counts = {}
        for r in rows:
            key = f"{r.file_path}:{r.symbol_name}"
            counts[key] = counts.get(key, 0) + 1

        sorted_hotspots = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return [
            {
                "file_or_symbol": item[0],
                "hotspot_score": float(item[1]),
                "reasons": [f"Modified in {item[1]} commits"],
            }
            for item in sorted_hotspots
        ]

    async def get_coupling_metrics(self, repo_id: str, file_path: str) -> Dict[str, Any]:
        """Compute coupling (fan-in/fan-out) metrics for a file."""
        stmt_out = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id, SymbolEdgeModel.source_file == file_path
        )
        stmt_in = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id, SymbolEdgeModel.target_file == file_path
        )

        edges_out = (await self.session.execute(stmt_out)).scalars().all()
        edges_in = (await self.session.execute(stmt_in)).scalars().all()

        fan_out = len(set([e.target_file for e in edges_out if e.target_file]))
        fan_in = len(set([e.source_file for e in edges_in if e.source_file]))

        return {
            "file_path": file_path,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "coupling_signal": "HIGH" if (fan_in + fan_out) > 10 else "NORMAL",
        }

    async def get_module_coupling(self, repo_id: str) -> Dict[str, Any]:
        """Compute module-level coupling (fan-in/fan-out matrix) across the repository.

        Delegates to the deterministic :class:`CouplingAnalyzer` using the shared
        ``self.session`` so that all graph/database logic stays out of the facade.
        """
        analyzer = CouplingAnalyzer()
        return await analyzer.get_module_coupling_matrix(self.session, repo_id)

    async def detect_cycles(self, repo_id: str) -> Dict[str, Any]:
        """Detect circular dependencies in the current architecture graph."""
        snapshot = await self.get_architecture_snapshot(repo_id)
        edges = snapshot.get("module_edges", [])
        detector = ArchCycleDetector()
        cycles = detector.detect_cycles(edges)
        return {
            "repo_id": repo_id,
            "cycle_count": len(cycles),
            "cycles": cycles,
            "drift_signals": len(cycles),
        }

    async def get_change_risk(self, repo_id: str, target: str) -> Dict[str, Any]:
        """Compute a deterministic change-risk assessment for a target symbol/file.

        Gathers signals from the composed facade methods and feeds them through
        the existing deterministic :class:`RiskScorer`. Does NOT persist the
        assessment -- persistence is left to the callers that need it.
        """
        scorer = RiskScorer()

        hist = await self.get_symbol_history(repo_id, target)
        commit_count = len(hist)

        dependents = await self.get_dependents(repo_id, target)
        co_change_rels = await self.get_co_change_relationships(repo_id, target)

        api_changes = await self.get_api_changes(repo_id, limit=50)
        is_public_api = any(
            ac.get("api_symbol") == target or ac.get("api_symbol", "").endswith(target)
            for ac in api_changes
        )

        tests = await self.get_tests_for_symbol(repo_id, target)
        cross_repo = await self.cross_repository_neighbors(repo_id)

        score, risk_level, reasons, evidence = scorer.compute_change_risk(
            target=target,
            target_type="symbol",
            direct_dependents=len(dependents),
            fan_in=len(dependents),
            fan_out=0,
            commit_frequency=commit_count,
            co_change_count=len(co_change_rels),
            is_public_api=is_public_api,
            cross_repo_consumers=len(cross_repo),
            test_relations_count=len(tests),
            has_dependency_changes=False,
        )

        return {
            "target": target,
            "target_type": "symbol",
            "score": score,
            "risk_level": risk_level,
            "reasons": reasons,
            "evidence": evidence,
        }