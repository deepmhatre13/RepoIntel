"""RepositoryFacade -- the single knowledge-access boundary for RepoIntel.

This facade is intentionally orchestration-oriented. The Phase 22-27 capability
implementations live in focused mixin modules under ``packages.knowledge.facade``
(history, architecture, testing, risk, impact) and are composed here so that
agents and services depend only on ``RepositoryFacade``. The internal capability
modules are implementation details and must NOT be imported by agents or services.
"""

import json
import asyncio
import copy
import hashlib
import os
import unicodedata
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.knowledge.models import (
    RepositoryModel,
    FileModel,
    CodeChunkModel,
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
        self._retrieval_cache = None
        self._retrieval_cache_initialized = False

    def _redis_cache(self):
        if self._retrieval_cache_initialized:
            return self._retrieval_cache
        self._retrieval_cache_initialized = True
        try:
            import redis

            self._retrieval_cache = redis.Redis.from_url(
                os.getenv(
                    "REPOINTEL_RETRIEVAL_CACHE_URL",
                    "redis://localhost:6379/2",
                ),
                socket_connect_timeout=0.05,
                socket_timeout=0.05,
                decode_responses=True,
            )
        except Exception:
            self._retrieval_cache = None
        return self._retrieval_cache

    @staticmethod
    def _normalize_cache_value(value):
        if isinstance(value, str):
            return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        if isinstance(value, list):
            return [RepositoryFacade._normalize_cache_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: RepositoryFacade._normalize_cache_value(item)
                for key, item in sorted(value.items())
            }
        return value

    def _retrieval_cache_key(self, repo_id, mode, query, top_k, options):
        content_hash = self.vector_store.repository_content_hash(repo_id)
        key_data = {
            "repo_id": repo_id,
            "content_hash": content_hash,
            "mode": mode,
            "query": self._normalize_cache_value(query),
            "top_k": top_k,
            "options": self._normalize_cache_value(options),
        }
        encoded = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
        return "repointel:retrieval:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _cached_retrieval(self, repo_id, mode, query, top_k, options):
        cache = self._redis_cache()
        if cache is None:
            return None
        try:
            key = self._retrieval_cache_key(repo_id, mode, query, top_k, options)
            payload = cache.get(key)
            if payload is None:
                return None
            return [
                RetrievedEvidenceItem(**item)
                for item in json.loads(payload)
            ]
        except Exception:
            return None

    def _cache_retrieval(self, repo_id, mode, query, top_k, options, results):
        cache = self._redis_cache()
        if cache is None:
            return
        try:
            key = self._retrieval_cache_key(repo_id, mode, query, top_k, options)
            ttl = max(1, int(os.getenv("REPOINTEL_RETRIEVAL_CACHE_TTL", "3600")))
            cache.setex(
                key,
                ttl,
                json.dumps([asdict(item) for item in results]),
            )
        except Exception:
            return

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

    async def get_quickstart_facts(self, repo_id: str) -> Dict[str, Any]:
        manifest = await self.get_ingestion_manifest(repo_id)
        if not manifest:
            return {}
        manifest_data = manifest.get("manifest_data", {})
        file_result = await self.session.execute(
            select(FileModel.file_path).where(FileModel.repo_id == repo_id)
        )
        file_paths = sorted(set(file_result.scalars().all()))
        manifest_paths = set(manifest_data.get("manifest_files", []))
        entrypoints = set(manifest_data.get("entrypoints", []))
        candidate_paths = [
            path for path in file_paths
            if path in manifest_paths
            or path in entrypoints
            or path.replace("\\", "/").rsplit("/", 1)[-1].casefold().startswith((
                "license", "copying", "pytest.ini", "tox.ini", "makefile", "justfile",
            ))
            or path.casefold().endswith("pyproject.toml")
            or ".github/workflows/" in path.replace("\\", "/").casefold()
        ][:100]
        if not candidate_paths:
            return {
                "file_paths": file_paths,
                "entrypoints": list(entrypoints),
                "files": {},
            }

        chunk_result = await self.session.execute(
            select(CodeChunkModel)
            .where(
                CodeChunkModel.repo_id == repo_id,
                CodeChunkModel.file_path.in_(candidate_paths),
            )
            .order_by(CodeChunkModel.file_path, CodeChunkModel.line_start)
            .limit(500)
        )
        files: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunk_result.scalars().all():
            files.setdefault(chunk.file_path, []).append({
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "content": chunk.content,
            })
        return {
            "file_paths": file_paths,
            "entrypoints": list(entrypoints),
            "files": files,
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

    async def get_security_findings(self, repo_id: str) -> List[Dict[str, Any]]:
        from packages.knowledge.models import SecurityFindingModel

        stmt = select(SecurityFindingModel).where(SecurityFindingModel.repo_id == repo_id)
        result = await self.session.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "rule_id": row.rule_id,
                "severity": row.severity,
                "message": row.message,
                "file_path": row.file_path,
                "line_start": row.line_start,
                "line_end": row.line_end,
                "finding_type": row.finding_type,
                "source_endpoint": row.source_endpoint,
            }
            for row in rows
        ]

    async def get_file_metadata(self, repo_id: str, file_path: str) -> Optional[Dict[str, Any]]:
        # Suffix match so a query naming "calculator.py" resolves to the
        # indexed "src/calculator.py" (mirrors get_symbols_for_file).
        stmt = select(SymbolModel).where(
            SymbolModel.repo_id == repo_id, SymbolModel.file_path.endswith(file_path)
        )
        res = await self.session.execute(stmt)
        symbols = res.scalars().all()
        if not symbols:
            return None
        actual_path = symbols[0].file_path
        return {
            "file_path": actual_path,
            "symbol_count": len(symbols),
            "symbols": [s.symbol_name for s in symbols],
        }

    def vector_search(self, repo_id: str, query: str, top_k: int = 10) -> List[RetrievedEvidenceItem]:
        return self.vector_store.vector_search(repo_id, query, top_k)

    def keyword_search(self, repo_id: str, query: str, top_k: int = 10) -> List[RetrievedEvidenceItem]:
        return self.vector_store.keyword_search(repo_id, query, top_k)

    def hybrid_search(self, repo_id: str, query: str, top_k: int = 10) -> List[RetrievedEvidenceItem]:
        cached = self._cached_retrieval(repo_id, "hybrid", query, top_k, {})
        if cached is not None:
            return cached
        candidates = self.vector_store.hybrid_search(repo_id, query, top_k=top_k * 2)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        results = self.context_assembler.assemble(reranked)
        self._cache_retrieval(repo_id, "hybrid", query, top_k, {}, results)
        return results

    def hyde_search(
        self,
        repo_id: str,
        query: str,
        hypothetical_document: str,
        top_k: int = 10,
    ) -> List[RetrievedEvidenceItem]:
        options = {"hypothetical_document": hypothetical_document}
        cached = self._cached_retrieval(repo_id, "hyde", query, top_k, options)
        if cached is not None:
            return cached
        candidates = self.vector_store.hybrid_search(
            repo_id,
            query,
            top_k=top_k * 2,
            vector_query=hypothetical_document,
            keyword_query=query,
        )
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        results = self.context_assembler.assemble(reranked)
        self._cache_retrieval(repo_id, "hyde", query, top_k, options, results)
        return results

    async def multi_query_search(
        self,
        repo_id: str,
        queries: List[str],
        top_k: int = 10,
        ranking_query: Optional[str] = None,
    ) -> List[RetrievedEvidenceItem]:
        options = {"queries": queries, "ranking_query": ranking_query}
        cached = self._cached_retrieval(repo_id, "multi_query", queries, top_k, options)
        if cached is not None:
            return cached
        searches = await asyncio.gather(*(
            asyncio.to_thread(
                self.vector_store.hybrid_search,
                repo_id,
                query,
                top_k=top_k * 2,
            )
            for query in queries
        ))

        merged: Dict[str, RetrievedEvidenceItem] = {}
        for items in searches:
            for item in items:
                key = item.chunk_id or (
                    f"{item.file_path}:{item.line_start}:{item.line_end}:{item.content}"
                )
                existing = merged.get(key)
                if existing is None:
                    merged[key] = copy.copy(item)
                    merged[key].retrieval_method = list(item.retrieval_method)
                    continue
                existing.relevance_score += item.relevance_score
                existing.retrieval_method = sorted(
                    set(existing.retrieval_method) | set(item.retrieval_method)
                )

        candidates = list(merged.values())
        reranked = self.reranker.rerank(
            ranking_query or (queries[0] if queries else ""),
            candidates,
            top_k=top_k,
        )
        results = self.context_assembler.assemble(reranked)
        self._cache_retrieval(repo_id, "multi_query", queries, top_k, options, results)
        return results

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
        # Suffix match so a user query naming "calculator.py" resolves to
        # "src/calculator.py" (or any other directory prefix) in the index.
        stmt = select(SymbolModel).where(
            SymbolModel.repo_id == repo_id, SymbolModel.file_path.endswith(file_path)
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
                "file_path": s.file_path,
            }
            for s in symbols
        ]

    async def get_symbols_by_name(self, repo_id: str, symbol_name: str) -> List[Dict[str, Any]]:
        """Look up repository symbols whose name matches (exact or dotted suffix)."""
        stmt = select(SymbolModel).where(
            SymbolModel.repo_id == repo_id,
            SymbolModel.symbol_name == symbol_name,
        )
        res = await self.session.execute(stmt)
        symbols = res.scalars().all()
        return [
            {
                "symbol_name": s.symbol_name,
                "symbol_type": s.symbol_type,
                "file_path": s.file_path,
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