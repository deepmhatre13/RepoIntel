"""Search capability mixin for RepositoryFacade (Phase 29).

Exposes engineering-oriented semantic code search capabilities on RepositoryFacade.
"""

from typing import Dict, Any, Optional
from packages.knowledge.semantic_search import SemanticCodeSearchEngine


class SearchCapabilities:
    """Semantic code search intelligence for RepositoryFacade."""

    async def search_code(
        self,
        repo_id: str,
        query: str,
        intent: Optional[str] = None,
        mode: Optional[str] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Perform intent-aware multi-stage code search returning provenance and explanations."""
        engine = SemanticCodeSearchEngine(vector_store=self.vector_store)
        res = await engine.search(
            session=self.session,
            repo_id=repo_id,
            query=query,
            intent=intent,
            mode=mode,
            top_k=top_k,
        )
        return res.to_dict()
