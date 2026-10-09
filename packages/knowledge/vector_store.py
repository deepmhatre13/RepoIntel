"""Retrieval store for RepoIntel: lexical + semantic + hybrid retrieval.

This module replaces the previous placeholder behaviour (where "vector" search
was simply token-overlap scoring against the in-memory chunk list) with a real,
measureable retrieval stack:

    * ``vector_search``      -> cosine similarity over real embeddings
    * ``keyword_search``     -> BM25 lexical scoring (rank_bm25)
    * ``hybrid_search``      -> reciprocal-rank fusion of vector + lexical

Provenance is preserved on every result via ``RetrievedEvidenceItem`` (which
now also carries ``repo_id``, ``chunk_id``, ``retrieval_source`` and a stable
dedup key).

Design notes
------------
The store is process-local and fed by the ingestion pipeline.  Persistent
retrieval (surviving restarts) is delegated to the knowledge layer via
``CodeChunkModel``; the in-process store is the fast path for a live session.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from packages.parsing.chunking import CodeChunk

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None


@dataclass
class RetrievedEvidenceItem:
    repo_id: str
    file_path: str
    symbol_name: Optional[str]
    line_start: int
    line_end: int
    content: str
    retrieval_source: str
    relevance_score: float
    snippet: str = ""
    chunk_id: Optional[str] = None
    # --- Phase 45 provenance fields ---
    repository_id: Optional[str] = None
    file_id: Optional[str] = None
    symbol_id: Optional[str] = None
    chunk_ref: Optional[str] = None
    retrieval_method: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.repository_id is None:
            self.repository_id = self.repo_id
        if not self.retrieval_method:
            self.retrieval_method = [self.retrieval_source]
        if not self.snippet:
            preview = self.content
            self.snippet = preview[:400] + "..." if len(preview) > 400 else preview
        if self.chunk_id is None and self.content:
            self.chunk_id = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    @property
    def content_hash_prop(self) -> Optional[str]:
        return self.chunk_id

    def to_dict(self) -> Dict[str, object]:
        return {
            "repo_id": self.repository_id or self.repo_id,
            "repository_id": self.repository_id or self.repo_id,
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content": self.content,
            "retrieval_source": self.retrieval_source,
            "retrieval_method": self.retrieval_method,
            "relevance_score": round(self.relevance_score, 4),
            "chunk_id": self.chunk_id,
            "content_hash": self.chunk_id,
        }


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length, L2-normalised vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


class VectorStore:
    """Real retrieval store: semantic + lexical + hybrid (RRF)."""

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service
        self.chunks: List[CodeChunk] = []
        self._bm25_dirty = True

    def upsert_chunks(self, chunks: List[CodeChunk]):
        """Append chunks, skipping exact duplicates (idempotent upsert)."""
        if not chunks:
            return
        existing = {
            (c.repo_id, c.file_path, c.line_start, c.line_end, c.content_hash)
            for c in self.chunks
        }
        for chunk in chunks:
            key = (chunk.repo_id, chunk.file_path, chunk.line_start, chunk.line_end, chunk.content_hash)
            if key not in existing:
                self.chunks.append(chunk)
                existing.add(key)

    def delete_repo(self, repo_id: str):
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.repo_id != repo_id]
        if len(self.chunks) != before:
            self._bm25_dirty = True

    def invalidate_paths(self, repo_id: str, file_paths: List[str]) -> int:
        """Remove indexed chunks for specific files.

        Returns the number of chunks removed so invalidation is observable.
        """
        paths = set(file_paths)
        if not paths:
            return 0
        before = len(self.chunks)
        self.chunks = [
            c for c in self.chunks
            if not (c.repo_id == repo_id and c.file_path in paths)
        ]
        removed = before - len(self.chunks)
        if removed:
            self._bm25_dirty = True
        return removed

    def repository_content_hash(self, repo_id: str) -> str:
        """Return a stable fingerprint for the currently indexed repo chunks."""
        descriptors = sorted(
            f"{chunk.file_path}:{chunk.line_start}:{chunk.line_end}:{chunk.content_hash}"
            for chunk in self._repo_chunks(repo_id)
        )
        return hashlib.sha256("\n".join(descriptors).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _repo_chunks(self, repo_id: str) -> List[CodeChunk]:
        return [c for c in self.chunks if c.repo_id == repo_id]

    def _get_embedding(self, chunk: CodeChunk) -> Optional[List[float]]:
        if self.embedding_service is None:
            return None
        try:
            return self.embedding_service.embed_query(chunk.content)
        except Exception:
            return None

    def _bm25_for(self, repo_id: str):
        if BM25Okapi is None:
            return None
        chunks = self._repo_chunks(repo_id)
        corpus = [c.content.split() for c in chunks]
        if not corpus:
            return None
        try:
            return BM25Okapi(corpus)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Retrieval APIs
    # ------------------------------------------------------------------

    def vector_search(self, repo_id: str, query: str, top_k: int = 10) -> List[RetrievedEvidenceItem]:
        """Semantic (embedding) search — cosine similarity over real vectors."""
        chunks = self._repo_chunks(repo_id)
        if not chunks:
            return []
        if self.embedding_service is None:
            # No embedding service: degrade clearly to lexical, never fake semantic.
            return self.keyword_search(repo_id, query, top_k)
        q_emb = self.embedding_service.embed_query(query)
        scored: List[Tuple[float, CodeChunk]] = []
        for chunk in chunks:
            emb = self._get_embedding(chunk)
            if emb is None:
                continue
            scored.append((_cosine(q_emb, emb), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            self._to_evidence(chunk, "vector", score)
            for score, chunk in scored[:top_k]
        ]

    def keyword_search(self, repo_id: str, query: str, top_k: int = 10) -> List[RetrievedEvidenceItem]:
        """Lexical retrieval via BM25 over chunk content (Jaccard fallback)."""
        chunks = self._repo_chunks(repo_id)
        if not chunks:
            return []
        tokens = query.lower().split()
        if not tokens:
            return []

        scored: List[Tuple[float, int]] = []  # (score, chunk_index)
        bm = self._bm25_for(repo_id) if BM25Okapi is not None else None
        if bm is not None:
            try:
                # Corpus was tokenized via content.split(); the query must be
                # tokenized the same way or BM25 returns ~0 for every chunk.
                bm_scores = bm.get_scores(tokens)
                for i in range(len(chunks)):
                    scored.append((float(bm_scores[i]), i))
            except Exception:
                bm = None

        if not scored:
            # Jaccard fallback when BM25 is unavailable/failed.
            q_set = set(tokens)
            for i, chunk in enumerate(chunks):
                c_set = set(chunk.content.lower().split())
                inter = len(q_set & c_set)
                scored.append((inter / max(1, len(q_set)), i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            self._to_evidence(chunks[i], "keyword", score)
            for score, i in scored[:top_k]
        ]

    def hybrid_search(
        self,
        repo_id: str,
        query: str,
        top_k: int = 10,
        *,
        vector_query: Optional[str] = None,
        keyword_query: Optional[str] = None,
    ) -> List[RetrievedEvidenceItem]:
        """Combine lexical + semantic retrieval via reciprocal-rank fusion (RRF)."""
        chunks = self._repo_chunks(repo_id)
        if not chunks:
            return []

        vector_hits = self.vector_search(
            repo_id, vector_query or query, top_k=top_k * 2
        )
        keyword_hits = self.keyword_search(
            repo_id, keyword_query or query, top_k=top_k * 2
        )

        fused: Dict[str, Dict[str, object]] = {}
        k = 60.0

        def add_hits(hits: List[RetrievedEvidenceItem], source: str):
            for rank, item in enumerate(hits):
                key = item.chunk_id or item.content_hash_prop or item.content
                entry = fused.setdefault(key, {"item": item, "rrf": 0.0, "sources": set()})
                entry["rrf"] += 1.0 / (k + rank + 1)
                entry["sources"].add(source)

        add_hits(vector_hits, "vector")
        add_hits(keyword_hits, "keyword")

        ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
        results: List[RetrievedEvidenceItem] = []
        for entry in ranked[:top_k]:
            item = entry["item"]
            item.retrieval_source = "hybrid"
            item.retrieval_method = sorted(entry["sources"])  # type: ignore[arg-type]
            item.relevance_score = round(float(entry["rrf"]), 4)
            results.append(item)
        return results

    def _to_evidence(self, chunk: CodeChunk, source: str, score: float) -> RetrievedEvidenceItem:
        return RetrievedEvidenceItem(
            repo_id=chunk.repo_id,
            repository_id=chunk.repo_id,
            file_path=chunk.file_path,
            symbol_name=chunk.symbol_name,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            content=chunk.content,
            retrieval_source=source,
            retrieval_method=[source],
            relevance_score=round(score, 4),
            chunk_id=chunk.content_hash[:16] if chunk.content_hash else None,
        )
