"""Architecture capability mixin for RepositoryFacade.

Owns architecture evolution intelligence: creating/retrieving architecture
snapshots, comparing snapshots over time, and cycle detection. Implementation
detail, not exposed to agents.
"""

from typing import Dict, Any
from sqlalchemy import select

from packages.knowledge.models import CommitModel
from packages.knowledge.architecture_analysis import ArchitectureAnalyzer


class ArchitectureCapabilities:
    """Architecture evolution intelligence for RepositoryFacade."""

    async def create_architecture_snapshot(
        self, repo_id: str, commit_sha: str = ""
    ) -> Dict[str, Any]:
        """Create an architecture snapshot from current repository knowledge."""
        if not commit_sha:
            latest = (
                await self.session.execute(
                    select(CommitModel)
                    .where(CommitModel.repo_id == repo_id)
                    .order_by(CommitModel.timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            commit_sha = latest.sha if latest else repo_id

        analyzer = ArchitectureAnalyzer()
        snapshot_data = await analyzer.build_snapshot(self.session, repo_id, commit_sha)
        return snapshot_data

    async def get_architecture_snapshot(
        self, repo_id: str, commit_sha: str = ""
    ) -> Dict[str, Any]:
        """Retrieve a stored architecture snapshot or build a new one."""
        return await self.create_architecture_snapshot(repo_id, commit_sha)

    async def compare_architecture(self, repo_id: str, sha1: str, sha2: str) -> Dict[str, Any]:
        """Compare two architecture snapshots."""
        snap1 = await self.create_architecture_snapshot(repo_id, sha1)
        snap2 = await self.create_architecture_snapshot(repo_id, sha2)

        comps1 = set(snap1.get("modules", []))
        comps2 = set(snap2.get("modules", []))
        edges1 = set(
            (e["source_component"], e["target_component"], e.get("edge_type", ""))
            for e in snap1.get("module_edges", [])
        )
        edges2 = set(
            (e["source_component"], e["target_component"], e.get("edge_type", ""))
            for e in snap2.get("module_edges", [])
        )

        return {
            "sha1": sha1,
            "sha2": sha2,
            "components_added": sorted(comps2 - comps1),
            "components_removed": sorted(comps1 - comps2),
            "components_retained": sorted(comps1 & comps2),
            "relationships_added": sorted(edges2 - edges1),
            "relationships_removed": sorted(edges1 - edges2),
        }

    async def compare_architecture_snapshots(self, repo_id: str, sha1: str, sha2: str) -> Dict[str, Any]:
        """Alias for compare_architecture."""
        return await self.compare_architecture(repo_id, sha1, sha2)

    async def analyze_architecture(self, repo_id: str) -> Dict[str, Any]:
        """Perform comprehensive architecture analysis including components, execution paths, signals, and issues."""
        from packages.knowledge.architecture_analyst import SoftwareArchitectureAnalyst
        analyst = SoftwareArchitectureAnalyst()
        return await analyst.analyze(self.session, repo_id)