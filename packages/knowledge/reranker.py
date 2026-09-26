from __future__ import annotations

from dataclasses import replace
from typing import List

from packages.knowledge.vector_store import RetrievedEvidenceItem

class EvidenceReranker:
    """Lightweight evidence reranker.

    Combines the original retrieval score (from hybrid/RRF or vector search)
    with a query-term-overlap signal into a single blended score.

    The original score and the overlap score are first normalised to [0, 1]
    within the candidate set so that neither signal dominates simply because
    of its native scale.  RRF scores (e.g. 1/(k+rank) ≈ 0.016) are therefore
    comparable to overlap scores in [0, 1] instead of being drowned out.
    """

    def __init__(self, model_name: str = "lightweight-cross-encoder"):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidate_evidence: List[RetrievedEvidenceItem],
        top_k: int = 10,
    ) -> List[RetrievedEvidenceItem]:
        if not candidate_evidence:
            return []

        q_terms = set(query.lower().split())
        n = len(candidate_evidence)

        # Normalise retrieval scores to [0, 1] within this candidate set.
        max_retrieval = max((c.relevance_score for c in candidate_evidence), default=0.0)
        max_retrieval = max(max_retrieval, 1e-9)

        scored: List[tuple[float, RetrievedEvidenceItem]] = []

        for item in candidate_evidence:
            c_terms = set(item.content.lower().split())
            if q_terms:
                overlap = len(q_terms & c_terms) / len(q_terms)
            else:
                overlap = 0.0

            norm_retrieval = item.relevance_score / max_retrieval  # [0, 1]
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
