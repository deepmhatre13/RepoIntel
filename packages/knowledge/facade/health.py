"""Health capability mixin for RepositoryFacade (Phase 36).

Exposes explainable repository health intelligence capabilities on RepositoryFacade.
"""

from typing import Dict, Any
from packages.knowledge.health_engine import RepositoryHealthEngine


class HealthCapabilities:
    """Repository health intelligence for RepositoryFacade."""

    async def get_repository_health(self, repo_id: str) -> Dict[str, Any]:
        """Compute explainable repository health signals across key dimensions."""
        engine = RepositoryHealthEngine(facade=self)
        report = await engine.compute_health(self.session, repo_id)
        return report.to_dict()
