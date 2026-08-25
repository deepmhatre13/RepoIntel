from typing import List
from packages.knowledge.vector_store import RetrievedEvidenceItem

class EvidenceReranker:
    def __init__(self, model_name: str = "lightweight-cross-encoder"):
        self.model_name = model_name

    def rerank(self, query: str, candidate_evidence: List[RetrievedEvidenceItem], top_k: int = 10) -> List[RetrievedEvidenceItem]:
        if not candidate_evidence:
            return []

        q_terms = set(query.lower().split())
        scored = []

        for item in candidate_evidence:
            c_terms = set(item.content.lower().split())
            overlap = len(q_terms.intersection(c_terms)) / max(1, len(q_terms))
            
            # Combine original retrieval score with term-overlap heuristic score
            combined_score = round(0.5 * item.relevance_score + 0.5 * overlap, 4)
            
            reranked_item = RetrievedEvidenceItem(
                repo_id=item.repo_id,
                file_path=item.file_path,
                symbol_name=item.symbol_name,
                line_start=item.line_start,
                line_end=item.line_end,
                content=item.content,
                retrieval_source=f"{item.retrieval_source}+reranked",
                relevance_score=combined_score
            )
            scored.append(reranked_item)

        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored[:top_k]
