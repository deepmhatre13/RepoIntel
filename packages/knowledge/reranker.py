from __future__ import annotations

import os
from dataclasses import replace
from typing import List, Optional

from packages.knowledge.vector_store import RetrievedEvidenceItem

class EvidenceReranker:
    """Cross-encoder reranker with an offline lexical fallback.

    A local BGE cross-encoder is used when its model is cached or downloads are
    explicitly enabled. Otherwise, retrieval score and query-term overlap are
    blended without making network access a requirement.

    The cross-encoder and retrieval scores are normalized within the candidate
    set before blending, so RRF scores remain comparable to model scores.
    """

    def __init__(self, model_name: Optional[str] = None, cross_encoder=None):
        self.model_name = model_name or os.getenv(
            "REPOINTEL_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.device = os.getenv("REPOINTEL_RERANKER_DEVICE", "cpu")
        self._cross_encoder = cross_encoder
        self._load_attempted = cross_encoder is not None

    def _load_cross_encoder(self):
        if self._load_attempted:
            return self._cross_encoder
        self._load_attempted = True
        local_files_only = os.getenv(
            "REPOINTEL_RERANKER_ALLOW_DOWNLOAD", "0"
        ).casefold() not in {"1", "true", "yes"}
        try:
            if local_files_only and not os.path.isdir(self.model_name):
                from huggingface_hub import try_to_load_from_cache

                if try_to_load_from_cache(self.model_name, "config.json") is None:
                    return None

            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(
                self.model_name,
                device=self.device,
                local_files_only=local_files_only,
                max_length=512,
            )
        except Exception:
            self._cross_encoder = None
        return self._cross_encoder

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return [1.0] * len(scores)
        return [(score - minimum) / (maximum - minimum) for score in scores]

    def rerank(
        self,
        query: str,
        candidate_evidence: List[RetrievedEvidenceItem],
        top_k: int = 10,
    ) -> List[RetrievedEvidenceItem]:
        if not candidate_evidence:
            return []

        cross_encoder_scores = None
        model = self._load_cross_encoder()
        if model is not None:
            try:
                pairs = [
                    (query, item.content or item.snippet or "")
                    for item in candidate_evidence
                ]
                raw_scores = model.predict(pairs, show_progress_bar=False)
                cross_encoder_scores = [float(score) for score in raw_scores]
                if len(cross_encoder_scores) != len(candidate_evidence):
                    cross_encoder_scores = None
            except Exception:
                cross_encoder_scores = None

        q_terms = set(query.lower().split())
        n = len(candidate_evidence)

        # Normalise retrieval scores to [0, 1] within this candidate set.
        max_retrieval = max((c.relevance_score for c in candidate_evidence), default=0.0)
        max_retrieval = max(max_retrieval, 1e-9)
        retrieval_scores = self._normalize([
            c.relevance_score for c in candidate_evidence
        ])
        if cross_encoder_scores is not None:
            cross_encoder_scores = self._normalize(cross_encoder_scores)

        scored: List[tuple[float, RetrievedEvidenceItem]] = []

        for index, item in enumerate(candidate_evidence):
            c_terms = set(item.content.lower().split())
            if q_terms:
                overlap = len(q_terms & c_terms) / len(q_terms)
            else:
                overlap = 0.0

            norm_retrieval = retrieval_scores[index]
            if cross_encoder_scores is not None:
                combined_score = round(
                    0.75 * cross_encoder_scores[index] + 0.25 * norm_retrieval,
                    4,
                )
            else:
                norm_retrieval = item.relevance_score / max_retrieval
                combined_score = round(0.5 * norm_retrieval + 0.5 * overlap, 4)

            # Preserve ALL provenance fields via dataclasses.replace so that
            # chunk_id, retrieval_method, file_id, symbol_id, etc. survive
            # reranking — citations must still resolve to indexed evidence.
            reranked_item = replace(
                item,
                retrieval_source=f"{item.retrieval_source}+reranked",
                relevance_score=combined_score,
            )
            scored.append((combined_score, reranked_item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
