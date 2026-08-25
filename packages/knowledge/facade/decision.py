"""Decision capability mixin for RepositoryFacade (Phase 32).

Exposes engineering decision intelligence capabilities on RepositoryFacade.
"""

from typing import Dict, Any


class DecisionCapabilities:
    """Engineering decision support intelligence for RepositoryFacade."""

    async def evaluate_engineering_decision(
        self, repo_id: str, question: str, target_entity: str = ""
    ) -> Dict[str, Any]:
        """Evaluate architectural or refactoring decisions with alternatives and evidence grounding."""
        from packages.knowledge.decision_intelligence import EngineeringDecisionEngine
        engine = EngineeringDecisionEngine(facade=self)
        res = await engine.evaluate_decision(repo_id, question, target_entity)
        return res.to_dict()

