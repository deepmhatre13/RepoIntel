"""Semantic Code Search Engine for RepoIntel (Phase 29).

Upgrades repository search into an engineering-oriented search system.
Supports search intents (symbol, behavior, architecture, dependency, config, execution_flow, history, test, impact)
and search modes (exact, semantic, hybrid, graph, historical, architecture).
Preserves provenance and provides explicit explanations for search results.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.knowledge.models import (
    SymbolModel,
    SymbolEdgeModel,
    SymbolHistoryModel,
)
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.reranker import EvidenceReranker
from packages.knowledge.query_understanding import QueryUnderstandingEngine


class SearchIntent(str, Enum):
    SYMBOL = "symbol"
    BEHAVIOR = "behavior"
    ARCHITECTURE = "architecture"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    EXECUTION_FLOW = "execution_flow"
    HISTORICAL = "historical"
    TEST = "test"
    IMPACT = "impact"


class SearchMode(str, Enum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    GRAPH = "graph"
    HISTORICAL = "historical"
    ARCHITECTURE = "architecture"


@dataclass
class SearchResultItem:
    repository: str
    file_path: str
    symbol: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]
    score: float
    retrieval_method: str
    content_preview: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository": self.repository,
            "file_path": self.file_path,
            "symbol": self.symbol,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "score": round(self.score, 4),
            "retrieval_method": self.retrieval_method,
            "content_preview": self.content_preview,
            "evidence": self.evidence,
            "reasons": self.reasons,
        }


@dataclass
class CodeSearchResult:
    query: str
    intent: SearchIntent
    mode: SearchMode
    confidence: float
    results: List[SearchResultItem] = field(default_factory=list)
    related_symbols: List[str] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)
    related_modules: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.value if isinstance(self.intent, Enum) else str(self.intent),
            "mode": self.mode.value if isinstance(self.mode, Enum) else str(self.mode),
            "confidence": round(self.confidence, 4),
            "result_count": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "related_symbols": self.related_symbols,
            "related_files": self.related_files,
            "related_modules": self.related_modules,
            "evidence": self.evidence,
        }


class SemanticCodeSearchEngine:
    """Multi-stage engineering-oriented code search pipeline."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.reranker = EvidenceReranker()
        self.query_analyzer = QueryUnderstandingEngine()

    async def search(
        self,
        session: AsyncSession,
        repo_id: str,
        query: str,
        intent: Optional[str] = None,
        mode: Optional[str] = None,
        top_k: int = 10,
    ) -> CodeSearchResult:
        # 1. Query Understanding
        q_analysis = self.query_analyzer.analyze(query)
        inferred_intent = intent or self._infer_intent(query, q_analysis)
        inferred_mode = mode or self._infer_mode(query, inferred_intent)

        enum_intent = SearchIntent(inferred_intent) if inferred_intent in SearchIntent._value2member_map_ else SearchIntent.BEHAVIOR
        enum_mode = SearchMode(inferred_mode) if inferred_mode in SearchMode._value2member_map_ else SearchMode.HYBRID

        results: List[SearchResultItem] = []
        related_symbols: Set[str] = set()
        related_files: Set[str] = set()
        evidence_list: List[Dict[str, Any]] = []

        # 2. Multi-retrieval based on mode & intent
        if enum_mode == SearchMode.EXACT:
            exact_res = await self._exact_symbol_search(session, repo_id, query)
            results.extend(exact_res)

        elif enum_mode == SearchMode.GRAPH:
            graph_res = await self._graph_search(session, repo_id, query)
            results.extend(graph_res)

        elif enum_mode == SearchMode.HISTORICAL:
            hist_res = await self._historical_search(session, repo_id, query)
            results.extend(hist_res)

        elif enum_mode == SearchMode.ARCHITECTURE:
            arch_res = await self._architecture_search(session, repo_id, query)
            results.extend(arch_res)

        else:  # HYBRID / SEMANTIC
            # Vector / hybrid retrieval (real semantic + lexical fusion).
            hyb_items = self.vector_store.hybrid_search(repo_id, query, top_k=top_k * 2)
            reranked = self.reranker.rerank(query, hyb_items, top_k=top_k)

            for item in reranked:
                method = "hybrid"
                if item.retrieval_method:
                    method = "+".join(sorted(item.retrieval_method))
                results.append(
                    SearchResultItem(
                        repository=repo_id,
                        file_path=item.file_path,
                        symbol=item.symbol_name,
                        line_start=item.line_start,
                        line_end=item.line_end,
                        score=item.relevance_score,
                        retrieval_method=method,
                        content_preview=item.snippet,
                        evidence=item.to_dict(),
                        reasons=["semantic similarity", "lexical keyword overlap"],
                    )
                )

            # Supplement with exact database symbol lookup if available
            exact_syms = await self._exact_symbol_search(session, repo_id, query)
            results = self._merge_results(results, exact_syms)

        # 3. Extract related files & symbols
        for r in results:
            if r.symbol:
                related_symbols.add(r.symbol)
            if r.file_path:
                related_files.add(r.file_path)
            evidence_list.append({"file": r.file_path, "symbol": r.symbol, "score": r.score, "reasons": r.reasons})

        modules = list(set(f.split("/")[0] for f in related_files if "/" in f))

        confidence = 0.9 if results else 0.2

        return CodeSearchResult(
            query=query,
            intent=enum_intent,
            mode=enum_mode,
            confidence=confidence,
            results=results[:top_k],
            related_symbols=sorted(related_symbols),
            related_files=sorted(related_files),
            related_modules=sorted(modules),
            evidence=evidence_list[:top_k],
        )

    def _infer_intent(self, query: str, analysis: Dict[str, Any]) -> str:
        q = query.lower()
        if "test" in q:
            return "test"
        if "config" in q or "setting" in q or "redis" in q:
            return "configuration"
        if "depend" in q or "package" in q:
            return "dependency"
        if "caller" in q or "call" in q or "flow" in q:
            return "execution_flow"
        if "commit" in q or "change" in q or "history" in q:
            return "historical"
        if "architect" in q or "component" in q or "module" in q:
            return "architecture"
        if "impact" in q or "affect" in q:
            return "impact"
        if "_" in query or any(c.isupper() for c in query):
            return "symbol"
        return "behavior"

    def _infer_mode(self, query: str, intent: str) -> str:
        q = query.lower()
        if "caller" in q or "call chain" in q:
            return "graph"
        if intent == "historical" or "history" in q:
            return "historical"
        if intent == "architecture" or "architecture" in q:
            return "architecture"
        if intent == "symbol" and (" " not in query.strip()):
            return "exact"
        return "hybrid"

    async def _exact_symbol_search(
        self, session: AsyncSession, repo_id: str, query: str
    ) -> List[SearchResultItem]:
        stmt = select(SymbolModel).where(
            SymbolModel.repo_id == repo_id, SymbolModel.symbol_name.like(f"%{query}%")
        ).limit(10)
        res = await session.execute(stmt)
        symbols = res.scalars().all()
        return [
            SearchResultItem(
                repository=repo_id,
                file_path=s.file_path,
                symbol=s.symbol_name,
                line_start=s.line_start,
                line_end=s.line_end,
                score=1.0 if s.symbol_name == query else 0.85,
                retrieval_method="exact_database_symbol",
                content_preview=f"{s.symbol_type} {s.symbol_name}: {s.signature or ''}",
                evidence={"source": "SymbolModel", "symbol_type": s.symbol_type},
                reasons=["exact or substring symbol name match"],
            )
            for s in symbols
        ]

    async def _graph_search(
        self, session: AsyncSession, repo_id: str, query: str
    ) -> List[SearchResultItem]:
        stmt = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id,
            (SymbolEdgeModel.source_symbol.like(f"%{query}%"))
            | (SymbolEdgeModel.target_symbol.like(f"%{query}%")),
        ).limit(10)
        res = await session.execute(stmt)
        edges = res.scalars().all()
        results = []
        for e in edges:
            results.append(
                SearchResultItem(
                    repository=repo_id,
                    file_path=e.source_file or e.target_file or "",
                    symbol=e.source_symbol or e.target_symbol,
                    line_start=None,
                    line_end=None,
                    score=0.9,
                    retrieval_method="graph_edge_traversal",
                    content_preview=f"{e.source_symbol} --[{e.edge_type}]--> {e.target_symbol}",
                    evidence={"source": "SymbolEdgeModel", "edge_type": e.edge_type},
                    reasons=[f"graph call-edge match ({e.edge_type})"],
                )
            )
        return results

    async def _historical_search(
        self, session: AsyncSession, repo_id: str, query: str
    ) -> List[SearchResultItem]:
        stmt = select(SymbolHistoryModel).where(
            SymbolHistoryModel.repo_id == repo_id,
            (SymbolHistoryModel.symbol_name.like(f"%{query}%"))
            | (SymbolHistoryModel.change_summary.like(f"%{query}%")),
        ).limit(10)
        res = await session.execute(stmt)
        rows = res.scalars().all()
        return [
            SearchResultItem(
                repository=repo_id,
                file_path=r.file_path,
                symbol=r.symbol_name,
                line_start=r.line_start,
                line_end=r.line_end,
                score=0.88,
                retrieval_method="git_history_analysis",
                content_preview=f"Commit {r.commit_sha[:8]} by {r.author}: {r.change_summary or r.change_type}",
                evidence={"source": "SymbolHistoryModel", "commit_sha": r.commit_sha},
                reasons=["git commit history match"],
            )
            for r in rows
        ]

    async def _architecture_search(
        self, session: AsyncSession, repo_id: str, query: str
    ) -> List[SearchResultItem]:
        stmt = select(SymbolModel).where(SymbolModel.repo_id == repo_id).limit(15)
        res = await session.execute(stmt)
        syms = res.scalars().all()
        matched = [s for s in syms if query.lower() in s.file_path.lower()]
        return [
            SearchResultItem(
                repository=repo_id,
                file_path=s.file_path,
                symbol=s.symbol_name,
                line_start=s.line_start,
                line_end=s.line_end,
                score=0.82,
                retrieval_method="architecture_module_match",
                content_preview=f"Module path match: {s.file_path}",
                evidence={"source": "SymbolModel", "file_path": s.file_path},
                reasons=["architectural module path match"],
            )
            for s in matched
        ]

    def _merge_results(
        self, primary: List[SearchResultItem], secondary: List[SearchResultItem]
    ) -> List[SearchResultItem]:
        seen = set((r.file_path, r.symbol) for r in primary)
        merged = list(primary)
        for s in secondary:
            if (s.file_path, s.symbol) not in seen:
                seen.add((s.file_path, s.symbol))
                merged.append(s)
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged
