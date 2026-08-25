"""Root Cause capability mixin for RepositoryFacade (Phase 37).

Exposes evidence-grounded root cause analysis capabilities on RepositoryFacade.
"""

from typing import Dict, Any, Optional
from packages.knowledge.root_cause_analysis import RootCauseAnalysisEngine


class RootCauseCapabilities:
    """Root cause investigation intelligence for RepositoryFacade."""

    async def analyze_root_cause(
        self, repo_id: str, problem_description: str, problem_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Investigate engineering symptoms with evidence for/against balance."""
        engine = RootCauseAnalysisEngine(facade=self)
        report = await engine.analyze_root_cause(self.session, repo_id, problem_description, problem_type)
        return report.to_dict()
