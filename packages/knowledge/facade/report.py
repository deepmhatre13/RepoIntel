"""Report capability mixin for RepositoryFacade (Phase 33).

Exposes engineering intelligence report generation on RepositoryFacade.
"""

from typing import Dict, Any, Optional


class ReportCapabilities:
    """Engineering intelligence report generation for RepositoryFacade."""

    async def generate_engineering_report(
        self,
        repo_id: str,
        report_type: str,
        target_entity: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive evidence-grounded engineering intelligence reports."""
        from packages.agentic.report_generator import EngineeringReportGenerator
        generator = EngineeringReportGenerator(facade=self)
        report = await generator.generate_report(
            repo_id=repo_id,
            report_type=report_type,
            target_entity=target_entity,
            question=question,
        )
        return report.model_dump()

