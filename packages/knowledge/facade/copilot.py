"""Copilot capability mixin for RepositoryFacade (Phase 39).

Exposes unified Copilot backend request processing on RepositoryFacade.
"""

from typing import Dict, Any


class CopilotCapabilities:
    """Unified Copilot backend processing for RepositoryFacade."""

    async def execute_copilot_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process unified Copilot requests across all repository intelligence modes."""
        from packages.agentic.copilot import EngineeringCopilotBackend, EngineeringCopilotRequest
        req = EngineeringCopilotRequest.model_validate(request_data)
        backend = EngineeringCopilotBackend(facade=self)
        resp = await backend.execute(req)
        return resp.model_dump()
