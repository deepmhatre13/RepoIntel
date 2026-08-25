"""RepositoryFacade -- the single knowledge-access boundary for RepoIntel.

This facade is intentionally orchestration-oriented. The Phase 22-27 capability
implementations live in focused mixin modules under ``packages.knowledge.facade``
(history, architecture, testing, risk, impact) and are composed here so that
agents and services depend only on ``RepositoryFacade``. The internal capability
modules are implementation details and must NOT be imported by agents or services.
"""

import json
from typing import List, Dict, Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.knowledge.models import (
    RepositoryModel,
    DependencyModel,
    SymbolModel,
    SymbolEdgeModel,
    ArtifactModel,
    RepositoryGroupModel,
    RepositoryGroupMemberModel,
    CrossRepoRelationModel,
)
from packages.knowledge.vector_store import VectorStore, RetrievedEvidenceItem
from packages.knowledge.reranker import EvidenceReranker
from packages.knowledge.context_assembler import ContextAssembler

from packages.knowledge.facade.history import HistoryCapabilities
from packages.knowledge.facade.architecture import ArchitectureCapabilities
from packages.knowledge.facade.testing import TestCapabilities
from packages.knowledge.facade.risk import RiskCapabilities
from packages.knowledge.facade.impact import ImpactCapabilities
from packages.knowledge.facade.graph import GraphCapabilities
from packages.knowledge.facade.search import SearchCapabilities
from packages.knowledge.facade.decision import DecisionCapabilities
from packages.knowledge.facade.report import ReportCapabilities
from packages.knowledge.facade.health import HealthCapabilities
from packages.knowledge.facade.root_cause import RootCauseCapabilities
from packages.knowledge.facade.simulation import SimulationCapabilities
from packages.knowledge.facade.copilot import CopilotCapabilities


class RepositoryFacade(
    HistoryCapabilities,
    ArchitectureCapabilities,
    TestCapabilities,
    RiskCapabilities,
    ImpactCapabilities,
    GraphCapabilities,
    SearchCapabilities,
    DecisionCapabilities,
    ReportCapabilities,
    HealthCapabilities,
    RootCauseCapabilities,
    SimulationCapabilities,
    CopilotCapabilities,
):
    """Public knowledge-access boundary composing focused capability mixins."""



    def __init__(
        self,
        session: AsyncSession,
        vector_store: VectorStore,
        max_graph_hops: int = 2,
        max_graph_nodes: int = 10,
    ):
        self.session = session
        self.vector_store = vector_store
        self.max_graph_hops = max_graph_hops
        self.max_graph_nodes = max_graph_nodes
        self.reranker = EvidenceReranker()
        self.context_assembler = ContextAssembler()

    # ------------------------------------------------------------------
    # Phase 1-21 core retrieval (stable facade orchestration stays here)
    # ------------------------------------------------------------------

    async def get_ingestion_manifest(self, repo_id: str) -> Optional[Dict[str, Any]]:
        stmt = select(RepositoryModel).where(RepositoryModel.id == repo_id)
        res = await self.session.execute(stmt)
        repo = res.scalar_one_or_none()
        if not repo:
            return None
        return {
            "repo_id": repo.id,
            "repo_url": repo.repo_url,
            "archetype": repo.archetype,
            "size_mb": repo.size_mb,
            "file_count": repo.file_count,
            "manifest_data": json.loads(repo.manifest_data),
        }

    async def get_dependency_manifest(self, repo_id: str) -> List[Dict[str, Any]]:
        stmt = select(DependencyModel).where(DependencyModel.repo_id == repo_id)
        res = await self.session.execute(stmt)
        deps = res.scalars().all()
        return [
            {
                "package_name": d.package_name,
                "ecosystem": d.ecosystem,
                "version_spec": d.version_spec,
                "dependency_type": d.dependency_type,
            }
            for d in deps
        ]
    async def get_file_metadata(self, repo_id: str, file_path: str) -> Optional[Dict[str, Any]]:
        stmt = select(SymbolModel).where(
            SymbolModel.repo_id == repo_id, SymbolModel.file_path == file_path
        )
        res = await self.session.execute(stmt)
        symbols = res.scalars().all()
        return {
            "file_path": file_path,
            "symbol_count": len(symbols),
            "symbols": [s.symbol_name for s in symbols],
        }

    def vector_search(self, repo_id: str, query: str, top_k: int = 10) -> List[RetrievedEvidenceItem]:
        return self.vector_store.vector_search(repo_id, query, top_k)

    def keyword_search(self, repo_id: str, query: str, top_k: int = 10) -> List[RetrievedEvidenceItem]:
        return self.vector_store.keyword_search(repo_id, query, top_k)

    def hybrid_search(self, repo_id: str, query: str, top_k: int = 10) -> List[RetrievedEvidenceItem]:
        candidates = self.vector_store.hybrid_search(repo_id, query, top_k=top_k * 2)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        return self.context_assembler.assemble(reranked)

    async def symbol_graph_neighbors(
        self, repo_id: str, symbol_or_file: str, max_neighbors: int = 10
    ) -> List[Dict[str, Any]]:
        stmt = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id,
            (SymbolEdgeModel.source_symbol == symbol_or_file)
            | (SymbolEdgeModel.target_symbol == symbol_or_file)
            | (SymbolEdgeModel.source_file == symbol_or_file)
            | (SymbolEdgeModel.target_file == symbol_or_file),
        ).limit(max_neighbors or self.max_graph_nodes)
        res = await self.session.execute(stmt)
        edges = res.scalars().all()
        return [
            {
                "source_symbol": e.source_symbol,
                "source_file": e.source_file,
                "edge_type": e.edge_type,
                "target_symbol": e.target_symbol,
                "target_file": e.target_file,
            }
            for e in edges
        ]
    async def get_symbols_for_file(self, repo_id: str, file_path: str) -> List[Dict[str, Any]]:
        stmt = select(SymbolModel).where(
            SymbolModel.repo_id == repo_id, SymbolModel.file_path == file_path
        )
        res = await self.session.execute(stmt)
        symbols = res.scalars().all()
        return [
            {
                "symbol_name": s.symbol_name,
                "symbol_type": s.symbol_type,
                "line_start": s.line_start,
                "line_end": s.line_end,
                "signature": s.signature,
                "docstring": s.docstring,
            }
            for s in symbols
        ]

    async def get_episodic_history(self, repo_id: str) -> List[Dict[str, Any]]:
        stmt = select(ArtifactModel).order_by(ArtifactModel.created_at.desc()).limit(5)
        res = await self.session.execute(stmt)
        artifacts = res.scalars().all()
        return [
            {
                "run_id": a.run_id,
                "artifact_type": a.artifact_type,
                "content_preview": a.content[:200],
            }
            for a in artifacts
        ]

    async def create_repository_group(
        self, group_id: str, name: str, description: Optional[str] = None
    ) -> Dict[str, Any]:
        group = RepositoryGroupModel(id=group_id, name=name, description=description)
        self.session.add(group)
        await self.session.commit()
        return {"id": group_id, "name": name}

    async def add_repo_to_group(self, group_id: str, repo_id: str):
        member = RepositoryGroupMemberModel(group_id=group_id, repo_id=repo_id)
        self.session.add(member)
        await self.session.commit()

    async def add_cross_repo_relation(
        self,
        source_repo_id: str,
        target_repo_id: str,
        relation_type: str,
        details: Optional[str] = None,
    ):
        rel = CrossRepoRelationModel(
            source_repo_id=source_repo_id,
            target_repo_id=target_repo_id,
            relation_type=relation_type,
            evidence=details,
        )
        self.session.add(rel)
        await self.session.commit()

    async def cross_repository_neighbors(self, repo_id: str) -> List[Dict[str, Any]]:
        stmt = select(CrossRepoRelationModel).where(
            (CrossRepoRelationModel.source_repo_id == repo_id)
            | (CrossRepoRelationModel.target_repo_id == repo_id)
        )
        res = await self.session.execute(stmt)
        rels = res.scalars().all()
        return [
            {
                "source_repo_id": r.source_repo_id,
                "target_repo_id": r.target_repo_id,
                "relation_type": r.relation_type,
                "details": r.evidence,
            }
            for r in rels
        ]
    def multi_repository_hybrid_search(
        self, repo_ids: List[str], query: str, top_k: int = 10
    ) -> List[RetrievedEvidenceItem]:
        all_evidence = []
        for r_id in repo_ids:
            all_evidence.extend(self.hybrid_search(r_id, query, top_k=top_k))
        all_evidence.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_evidence[:top_k]

    async def trace_call_chain(
        self,
        repo_id: str,
        starting_symbol: str,
        direction: str = "downstream",
        max_depth: int = 5,
        max_nodes: int = 20,
    ) -> List[Dict[str, Any]]:
        visited = set()
        queue = [(starting_symbol, 0)]
        chain = []

        while queue and len(chain) < max_nodes:
            curr_symbol, depth = queue.pop(0)
            if curr_symbol in visited or depth >= max_depth:
                continue
            visited.add(curr_symbol)

            stmt = select(SymbolEdgeModel).where(
                SymbolEdgeModel.repo_id == repo_id,
                (
                    SymbolEdgeModel.source_symbol == curr_symbol
                    if direction == "downstream"
                    else SymbolEdgeModel.target_symbol == curr_symbol
                ),
            )
            res = await self.session.execute(stmt)
            edges = res.scalars().all()
            for e in edges:
                next_sym = e.target_symbol if direction == "downstream" else e.source_symbol
                chain.append(
                    {
                        "from_symbol": e.source_symbol,
                        "to_symbol": e.target_symbol,
                        "edge_type": e.edge_type,
                        "depth": depth + 1,
                    }
                )
                if next_sym not in visited:
                    queue.append((next_sym, depth + 1))
        return chain

    async def get_module_dependencies(self, repo_id: str, module_path: str) -> List[Dict[str, Any]]:
        stmt = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id,
            SymbolEdgeModel.source_file.like(f"%{module_path}%"),
        )
        res = await self.session.execute(stmt)
        edges = res.scalars().all()
        deps = list(set([e.target_file for e in edges if e.target_file]))
        return [{"module": module_path, "depends_on": d} for d in deps]

    async def generate_repository_summary(self, repo_id: str) -> Dict[str, Any]:
        manifest = await self.get_ingestion_manifest(repo_id)
        deps = await self.get_dependency_manifest(repo_id)
        if not manifest:
            return {}
        return {
            "repo_id": repo_id,
            "archetype": manifest.get("archetype"),
            "file_count": manifest.get("file_count"),
            "languages": manifest.get("manifest_data", {}).get("languages", {}),
            "entrypoints": manifest.get("manifest_data", {}).get("entrypoints", []),
            "dependencies_count": len(deps),
            "top_dependencies": [d["package_name"] for d in deps[:5]],
        }