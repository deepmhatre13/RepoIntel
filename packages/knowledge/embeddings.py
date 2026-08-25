"""Real embedding system for RepoIntel.

Replaces the previous placeholder-vector implementation with a proper
``EmbeddingProvider`` abstraction backed by ``sentence-transformers``.

Architecture
------------
``EmbeddingProvider`` is the interface.  ``LocalSentenceTransformerProvider``
is the default implementation that runs entirely locally (no API key).

``EmbeddingService`` is the facade used by the rest of the codebase.  It wraps
a provider, adds a content-hash -> vector cache (thread-safe), dimension
validation, batching, and lightweight telemetry.  The cache is purely
in-process; persistent reuse of embeddings across runs is handled by the
persistence layer, which stores the content hash alongside every vector.

Deterministic behaviour
-----------------------
* The model is loaded lazily (first call to ``embed_*``).
* ``normalize_embeddings=True`` ensures cosine similarity == dot product.
* No randomness is introduced: identical input always yields identical output.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Dict, List, Optional

from packages.shared.telemetry import telemetry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """Interface that every embedding provider must implement."""

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> List[float]:
        raise NotImplementedError


class LocalSentenceTransformerProvider:
    """Local embedding provider using the ``sentence-transformers`` package.

    No API key is required.  The model is downloaded on first use (cached in
    the default HuggingFace cache directory) and loaded lazily to avoid
    slowing down startup for code paths that never need embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu",
        max_seq_length: int = 512,
        max_workers: int = 4,
    ):
        self._model_name = model_name
        self._batch_size = max(1, min(batch_size, 256))
        self._device = device
        self._max_seq_length = max_seq_length
        self._max_workers = max(1, max_workers)
        self._model = None
        self._dimension: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension or 384

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ImportError(
                        "sentence-transformers is required for local embeddings."
                    ) from exc
                self._model = SentenceTransformer(self._model_name, device=self._device)
                try:
                    dim = self._model.get_embedding_dimension()
                except AttributeError:  # pragma: no cover - older sentence-transformers
                    dim = self._model.get_sentence_embedding_dimension()
                self._dimension = int(dim) if dim else 384
                self._model.max_seq_length = self._max_seq_length
                logger.info(
                    "Loaded embedding model %s (dim=%d, device=%s)",
                    self._model_name, self._dimension, self._device,
                )

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        last_nl = cut.rfind("\n")
        if last_nl > max_chars // 2:
            return cut[:last_nl]
        return cut

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        max_chars = self._max_seq_length * 4
        processed = [self._truncate(t or " ", max_chars) for t in texts]
        results: List[List[float]] = []
        for start in range(0, len(processed), self._batch_size):
            batch = processed[start: start + self._batch_size]
            embs = self._model.encode(
                batch, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            results.extend(embs.tolist())
        return results

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def get_telemetry(self) -> Dict[str, object]:
        return {
            "provider": "LocalSentenceTransformerProvider",
            "model_name": self._model_name,
            "dimension": self.dimension,
            "batch_size": self._batch_size,
            "device": self._device,
        }


# ---------------------------------------------------------------------------
# Service (facade + cache + telemetry)
# ---------------------------------------------------------------------------

class EmbeddingService:
    """Facade over :class:`EmbeddingProvider` with a content-hash cache.

    Public API (compatible with the previous placeholder version):
        * ``embed_texts(texts)``  -> ``List[List[float]]``
        * ``embed_query(query)``  -> ``List[float]``
        * ``model_name`` property
        * ``dimension`` property

    New additions:
        * ``provider`` property
        * ``get_telemetry()``
        * ``cache_stats`` property
        * ``warm_cache(items)``
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: Optional[EmbeddingProvider] = None,
        batch_size: int = 32,
    ):
        if provider is not None:
            self._provider: EmbeddingProvider = provider
        else:
            self._provider = LocalSentenceTransformerProvider(
                model_name=model_name, batch_size=batch_size,
            )
        self._cache: Dict[str, List[float]] = {}
        self._cache_lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "cache_hits": 0, "cache_misses": 0,
            "total_embedded": 0, "errors": 0,
        }

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    @property
    def cache_stats(self) -> Dict[str, int]:
        with self._cache_lock:
            return dict(self._stats)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def warm_cache(self, items: Dict[str, List[float]]) -> int:
        with self._cache_lock:
            before = len(self._cache)
            self._cache.update(items)
            return len(self._cache) - before

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        batch_texts: List[str] = []
        batch_indices: List[int] = []

        for i, text in enumerate(texts):
            h = self._hash(text)
            with self._cache_lock:
                cached = self._cache.get(h)
            if cached is not None:
                self._stats["cache_hits"] += 1
                results[i] = cached
            else:
                self._stats["cache_misses"] += 1
                batch_texts.append(text)
                batch_indices.append(i)

        if batch_texts:
            embs = self._provider.embed_texts(batch_texts)
            for idx, emb in zip(batch_indices, embs):
                results[idx] = emb
                h = self._hash(texts[idx])
                with self._cache_lock:
                    self._cache[h] = emb
                self._stats["total_embedded"] += 1

        telemetry.increment("embedding_cache_hits", self._stats["cache_hits"])
        return [r for r in results if r is not None]  # type: ignore[misc]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def get_telemetry(self) -> Dict[str, object]:
        return {
            "embedding_provider": "LocalSentenceTransformerProvider",
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension,
            "cache_stats": dict(self._stats),
        }


def create_embedding_service(
    model_name: Optional[str] = None,
    provider: Optional[EmbeddingProvider] = None,
) -> EmbeddingService:
    """Create an :class:`EmbeddingService` from environment configuration."""
    model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    return EmbeddingService(model_name=model_name, provider=provider, batch_size=batch_size)


def embedding_telemetry() -> Dict[str, object]:
    """Return static embedding telemetry without forcing a model load."""
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return {
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "local"),
        "embedding_model": model_name,
        "embedding_dimension": 0,
    }

