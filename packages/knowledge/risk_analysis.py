"""Phase 26: Coupling, Hotspot & Risk Intelligence.

Deterministic engineering risk analysis combining static graph, Git history,
symbol changes, co-change, test relationships, and dependency relationships.
"""
import json
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from packages.knowledge.models import (
    SymbolEdgeModel, SymbolHistoryModel, CoChangeRelationModel,
    TestToCodeRelationModel, ChangeRiskAssessmentModel
)

# Configuration constants
MAX_HISTORY_COMMITS = 50
MAX_GRAPH_HOPS = 3
MAX_GRAPH_NODES = 50
MAX_DIFF_SIZE = 100000
MAX_TEST_RELATIONS = 200
MAX_HOTSPOTS = 20


@dataclass
class HotspotSignal:
    """A single signal contributing to a hotspot score."""
    name: str
    value: float
    weight: float
    evidence: str


def compute_hotspot_score(signals: List[HotspotSignal]) -> float:
    """Compute a normalized hotspot score from signals."""
    if not signals:
        return 0.0
    weighted_sum = sum(s.value * s.weight for s in signals)
    total_weight = sum(s.weight for s in signals)
    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 4)


class HotspotAnalyzer:
    """Calculates code hotspot scores from multiple deterministic signals."""

    def __init__(self, max_hotspots: int = MAX_HOTSPOTS):
        self.max_hotspots = max_hotspots

    async def analyze(self, session: AsyncSession, repo_id: str) -> List[Dict[str, Any]]:
        """Analyze hotspots for a repository."""
        hotspots = []

        # 1. Get symbol change history
        stmt = select(
            SymbolHistoryModel.symbol_name,
            SymbolHistoryModel.file_path,
            SymbolHistoryModel.author,
            SymbolHistoryModel.timestamp
        ).where(SymbolHistoryModel.repo_id == repo_id)
        res = await session.execute(stmt)
        history_rows = res.all()

        # Aggregate per symbol
        symbol_stats = defaultdict(lambda: {
            "commit_count": 0,
            "authors": set(),
            "first_seen": None,
            "last_modified": None,
            "file_path": None
        })

        for row in history_rows:
            key = f"{row.file_path}:{row.symbol_name}"
            stats = symbol_stats[key]
            stats["commit_count"] += 1
            stats["authors"].add(row.author)
            stats["file_path"] = row.file_path
            ts = row.timestamp
            if ts:
                if stats["first_seen"] is None or ts < stats["first_seen"]:
                    stats["first_seen"] = ts
                if stats["last_modified"] is None or ts > stats["last_modified"]:
                    stats["last_modified"] = ts

        # 2. Get co-change counts
        coc_stmt = select(
            CoChangeRelationModel.symbol_a,
            CoChangeRelationModel.symbol_b,
            CoChangeRelationModel.co_change_count
        ).where(CoChangeRelationModel.repo_id == repo_id)
        coc_res = await session.execute(coc_stmt)
        coc_rows = coc_res.all()

        coc_count = defaultdict(int)
        for row in coc_rows:
            coc_count[row.symbol_a] += row.co_change_count
            coc_count[row.symbol_b] += row.co_change_count

        # 3. Get dependents (fan-in via SymbolEdgeModel)
        fanin_stmt = select(
            SymbolEdgeModel.target_symbol,
            func.count().label("cnt")
        ).where(
            and_(SymbolEdgeModel.repo_id == repo_id, SymbolEdgeModel.edge_type == "calls")
        ).group_by(SymbolEdgeModel.target_symbol)
        fanin_res = await session.execute(fanin_stmt)
        fanin_counts = {row.target_symbol: row.cnt for row in fanin_res.all()}

        # 4. Get test relationships
        test_stmt = select(
            TestToCodeRelationModel.target_symbol
        ).where(TestToCodeRelationModel.repo_id == repo_id)
        test_res = await session.execute(test_stmt)
        test_targets = set(row.target_symbol for row in test_res.all())

        # Build hotspots
        for key, stats in symbol_stats.items():
            signals = []

            # Change frequency
            if stats["commit_count"] > 0:
                change_freq = min(stats["commit_count"] / 10.0, 1.0)
                signals.append(HotspotSignal(
                    name="change_frequency",
                    value=change_freq,
                    weight=0.3,
                    evidence=f"Modified in {stats['commit_count']} commits"
                ))

            # Number of authors
            if stats["authors"]:
                author_diversity = min(len(stats["authors"]) / 5.0, 1.0)
                signals.append(HotspotSignal(
                    name="author_diversity",
                    value=author_diversity,
                    weight=0.15,
                    evidence=f"Changed by {len(stats['authors'])} authors"
                ))

            # Co-change count
            co_change_max = coc_count.get(stats["file_path"], 0)
            if co_change_max > 0:
                co_change_score = min(co_change_max / 20.0, 1.0)
                signals.append(HotspotSignal(
                    name="co_change_count",
                    value=co_change_score,
                    weight=0.2,
                    evidence=f"Co-changed with {co_change_max} other symbols"
                ))

            # Fan-in (dependents)
            fan_in = fanin_counts.get(stats["file_path"].split("/")[-1], 0)
            if fan_in > 0:
                fan_in_score = min(fan_in / 15.0, 1.0)
                signals.append(HotspotSignal(
                    name="fan_in",
                    value=fan_in_score,
                    weight=0.2,
                    evidence=f"{fan_in} direct dependents"
                ))

            # Test coverage proxy (lower coverage = higher risk)
            has_tests = key.split(":")[-1] in test_targets
            if not has_tests:
                signals.append(HotspotSignal(
                    name="no_test_coverage_proxy",
                    value=0.5,
                    weight=0.15,
                    evidence="No related tests detected by static analysis"
                ))

            score = compute_hotspot_score(signals)
            if score > 0:
                hotspots.append({
                    "file_or_symbol": key,
                    "hotspot_score": score,
                    "reasons": [s.name for s in signals if s.value > 0],
                    "signals": [{"name": s.name, "value": s.value, "weight": s.weight, "evidence": s.evidence} for s in signals if s.value > 0],
                    "affected_module": stats["file_path"].split("/")[-2] if "/" in stats["file_path"] else "root",
                    "commit_count": stats["commit_count"],
                    "authors_count": len(stats["authors"])
                })

        hotspots.sort(key=lambda x: x["hotspot_score"], reverse=True)
        return hotspots[:self.max_hotspots]


class CouplingAnalyzer:
    """Calculates module-level coupling metrics."""

    def _edge_modules(self, e) -> "tuple[str, str]":
        """Return (source_module, target_module) resolved for an edge.

        Module-level coupling must only ever be derived from actual module/file
        dependencies, never from symbol names.

        * ``imports`` edges reference a module. When ``target_file`` is present it
          wins; otherwise the module is the imported name itself (e.g. ``os``),
          which is legitimate for a stdlib/third-party import.
        * ``calls``/``inherits``/``references`` edges are *symbol-level*. A call
          edge whose ``target_file`` is absent refers to a symbol defined in the
          same source file (or an unresolved/builtin target) and MUST NOT be
          promoted to a module dependency.

        This prevents symbols such as ``Calculator``, ``add`` or ``print`` from
        leaking into the module coupling matrix.
        """
        src_module = self._extract_module(e.source_file)
        if not src_module:
            return "", ""

        if e.edge_type == "imports":
            if e.target_file:
                tgt_module = self._extract_module(e.target_file)
            else:
                tgt_module = self._extract_module(e.target_symbol)
        else:
            if not e.target_file:
                return "", ""
            tgt_module = self._extract_module(e.target_file)

        return src_module, tgt_module

    def _extract_module(self, path: str) -> str:
        """Extract module/component name from a file path.

        A bare symbol name (no path separators, not a ``.py`` file) cannot
        resolve to a module; it is treated as empty so symbols never become
        modules.
        """
        if not path:
            return ""
        parts = path.replace("\\", "/").split("/")
        skip_dirs = {"src", "lib", "pkg", "packages", "app", "apps"}
        for part in parts:
            if part not in skip_dirs and not part.startswith(".") and not part.endswith(".py"):
                return part
        return parts[-1].replace(".py", "") if parts else ""

    async def analyze_module_coupling(self, session: AsyncSession, repo_id: str, module: str) -> Dict[str, Any]:
        """Calculate coupling metrics for a module/directory."""
        # fan-out: number of distinct modules this module depends on
        # fan-in: number of distinct modules that depend on this module

        stmt = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id,
            SymbolEdgeModel.edge_type.in_(["calls", "imports", "inherits"])
        )
        res = await session.execute(stmt)
        edges = res.scalars().all()

        # Group by module directory
        module_edges_out = defaultdict(set)
        module_edges_in = defaultdict(set)

        for e in edges:
            src_module, tgt_module = self._edge_modules(e)
            if src_module and tgt_module and src_module != tgt_module:
                module_edges_out[src_module].add(tgt_module)
                module_edges_in[tgt_module].add(src_module)

        fan_out = len(module_edges_out.get(module, set()))
        fan_in = len(module_edges_in.get(module, set()))
        total_direct = len(module_edges_out.get(module, set())) + len(module_edges_in.get(module, set()))

        signals = []
        if total_direct > 10:
            signals.append("HIGH coupling signal: module has more than 10 cross-module edges")
        if fan_in > 5:
            signals.append("High fan-in: many modules depend on this component")
        if fan_out > 5:
            signals.append("High fan-out: component depends on many other modules")

        return {
            "module": module,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "total_cross_module_edges": total_direct,
            "coupling_signal": "HIGH" if total_direct > 10 else "NORMAL",
            "coupling_issues": signals
        }

    async def get_module_coupling_matrix(self, session: AsyncSession, repo_id: str) -> Dict[str, Any]:
        """Get coupling matrix across all modules."""
        stmt = select(SymbolEdgeModel).where(SymbolEdgeModel.repo_id == repo_id)
        res = await session.execute(stmt)
        edges = res.scalars().all()

        modules = defaultdict(lambda: {"fan_in": set(), "fan_out": set()})
        for e in edges:
            src_mod, tgt_mod = self._edge_modules(e)
            if src_mod and tgt_mod and src_mod != tgt_mod:
                modules[src_mod]["fan_out"].add(tgt_mod)
                modules[tgt_mod]["fan_in"].add(src_mod)

        result = {}
        for mod, data in modules.items():
            result[mod] = {
                "fan_in": len(data["fan_in"]),
                "fan_out": len(data["fan_out"]),
                "total": len(data["fan_in"]) + len(data["fan_out"])
            }

        # Sort by coupling
        sorted_modules = sorted(result.items(), key=lambda x: x[1]["total"], reverse=True)
        return {
            "modules": {k: v for k, v in sorted_modules},
            "highly_coupled_modules": [
                {"module": k, "fan_in": v["fan_in"], "fan_out": v["fan_out"], "total": v["total"]}
                for k, v in sorted_modules if v["total"] > 10
            ]
        }


class CycleDetector:
    """Detects circular dependencies in dependency graphs."""

    def detect_cycles_in_graph(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect cycles using DFS. Input edges have 'source' and 'target' keys."""
        adj = defaultdict(list)
        for e in edges:
            adj[e["source"]].append(e["target"])

        cycles = []
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        path = []

        def dfs(node: str):
            color[node] = GRAY
            path.append(node)

            for neighbor in adj.get(node, []):
                if color[neighbor] == GRAY:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle_nodes = path[cycle_start:]
                    cycles.append({
                        "cycle": cycle_nodes + [neighbor],
                        "involved_nodes": list(set(cycle_nodes)),
                        "evidence": "Circular dependency detected via DFS traversal"
                    })
                elif color[neighbor] == WHITE:
                    dfs(neighbor)

            path.pop()
            color[node] = BLACK

        for node in adj:
            if color[node] == WHITE:
                dfs(node)

        # Deduplicate
        unique = []
        seen = set()
        for c in cycles:
            key = frozenset(c["involved_nodes"])
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique


class RiskScorer:
    """Computes deterministic change risk scores."""

    def compute_change_risk(
        self,
        target: str,
        target_type: str,
        direct_dependents: int,
        fan_in: int,
        fan_out: int,
        commit_frequency: int,
        co_change_count: int,
        is_public_api: bool,
        cross_repo_consumers: int,
        test_relations_count: int,
        has_dependency_changes: bool
    ) -> Tuple[float, str, List[str], List[Dict[str, Any]]]:
        """Compute a deterministic risk score for changing a target.

        Returns: (score, risk_level, reasons, evidence)
        """
        score = 0.0
        reasons = []
        evidence = []

        # 1. Direct dependents (fan-in)
        if direct_dependents > 0:
            contrib = min(direct_dependents / 20.0, 1.0) * 0.25
            score += contrib
            if direct_dependents > 5:
                reasons.append(f"{direct_dependents} direct dependents")
                evidence.append({"signal": "fan_in", "value": direct_dependents, "weight": 0.25, "threshold": ">5"})

        # 2. Public API involvement
        if is_public_api:
            score += 0.15
            reasons.append("public API symbol")
            evidence.append({"signal": "public_api", "value": True, "weight": 0.15})

        # 3. Historical change frequency
        if commit_frequency > 0:
            freq_score = min(commit_frequency / 15.0, 1.0)
            contrib = freq_score * 0.2
            score += contrib
            if commit_frequency > 5:
                reasons.append(f"modified in {commit_frequency} commits")
                evidence.append({"signal": "change_frequency", "value": commit_frequency, "weight": 0.2, "threshold": ">5"})

        # 4. Co-change count
        if co_change_count > 0:
            contrib = min(co_change_count / 30.0, 1.0) * 0.1
            score += contrib
            if co_change_count > 5:
                reasons.append(f"frequently co-changed ({co_change_count} co-changes)")
                evidence.append({"signal": "co_change", "value": co_change_count, "weight": 0.1, "threshold": ">5"})

        # 5. Cross-repository consumers
        if cross_repo_consumers > 0:
            contrib = min(cross_repo_consumers / 10.0, 1.0) * 0.15
            score += contrib
            reasons.append(f"cross-repository consumers ({cross_repo_consumers})")
            evidence.append({"signal": "cross_repo", "value": cross_repo_consumers, "weight": 0.15})

        # 6. Test coverage proxy
        if test_relations_count == 0:
            score += 0.1
            reasons.append("no related tests detected by static analysis")
            evidence.append({"signal": "test_coverage_proxy", "value": 0, "weight": 0.1})
        elif test_relations_count < 3:
            score += 0.05
            reasons.append(f"limited related tests ({test_relations_count})")
            evidence.append({"signal": "test_coverage_proxy", "value": test_relations_count, "weight": 0.05})

        # 7. Dependency changes
        if has_dependency_changes:
            score += 0.05
            reasons.append("associated dependency changes detected")
            evidence.append({"signal": "dependency_changes", "value": True, "weight": 0.05})

        # Clamp score
        score = round(min(score, 1.0), 4)

        # Map to risk level
        if score >= 0.7:
            risk_level = "CRITICAL"
        elif score >= 0.5:
            risk_level = "HIGH"
        elif score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return score, risk_level, reasons, evidence


async def compute_and_store_risk(
    session: AsyncSession,
    repo_id: str,
    target: str,
    facade: "RepositoryFacade"
) -> Dict[str, Any]:
    """Compute risk assessment for a target and persist it."""
    scorer = RiskScorer()

    # Gather signals
    hist = await facade.get_symbol_history(repo_id, target)
    commit_count = len(hist)

    direct_deps = await facade.trace_call_chain(repo_id, target, max_depth=1)
    direct_dependents = len(direct_deps) if isinstance(direct_deps, list) else 0

    dependents = await facade.get_dependents(repo_id, target)
    co_change_rels = await facade.get_co_change_relationships(repo_id, target)

    api_changes = await facade.get_api_changes(repo_id, limit=50)
    is_public_api = any(
        ac.get("api_symbol") == target or ac.get("api_symbol", "").endswith(target)
        for ac in api_changes
    )

    tests = await facade.get_tests_for_symbol(repo_id, target)

    cross_repo = await facade.cross_repository_neighbors(repo_id)

    score, risk_level, reasons, evidence = scorer.compute_change_risk(
        target=target,
        target_type="symbol",
        direct_dependents=len(dependents),
        fan_in=direct_dependents,
        fan_out=0,
        commit_frequency=commit_count,
        co_change_count=len(co_change_rels),
        is_public_api=is_public_api,
        cross_repo_consumers=len(cross_repo),
        test_relations_count=len(tests),
        has_dependency_changes=False
    )

    # Persist
    assessment = ChangeRiskAssessmentModel(
        repo_id=repo_id,
        target=target,
        target_type="symbol",
        score=score,
        risk_level=risk_level,
        reasons_json=json.dumps(reasons),
        evidence_json=json.dumps(evidence),
        calculated_at=datetime.now(timezone.utc)
    )
    session.add(assessment)
    await session.commit()

    return {
        "target": target,
        "target_type": "symbol",
        "score": score,
        "risk_level": risk_level,
        "reasons": reasons,
        "evidence": evidence
    }
