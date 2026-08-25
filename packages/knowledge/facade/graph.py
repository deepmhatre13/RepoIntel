"""Graph capability mixin for RepositoryFacade (Phase 28).

Exposes unified knowledge graph construction, traversal, path finding, and evidence lookup
to RepositoryFacade without leaking graph implementation details.
"""

from typing import List, Dict, Any, Optional
from packages.knowledge.unified_graph import RepositoryKnowledgeGraph


class GraphCapabilities:
    """Unified repository knowledge graph intelligence for RepositoryFacade."""

    async def get_knowledge_graph(
        self, repo_id: str, max_hops: int = 3, max_nodes: int = 50
    ) -> RepositoryKnowledgeGraph:
        """Build and return a unified RepositoryKnowledgeGraph for a given repository."""
        kg = RepositoryKnowledgeGraph(max_hops=max_hops, max_nodes=max_nodes)
        await kg.build_from_session(self.session, repo_id)
        return kg

    async def query_knowledge_graph(
        self,
        repo_id: str,
        start_node_id: str,
        action: str = "traverse",
        target_node_id: Optional[str] = None,
        edge_types: Optional[List[str]] = None,
        max_hops: int = 3,
    ) -> Dict[str, Any]:
        """Perform graph query operations (traverse, neighbors, shortest_path, evidence)."""
        kg = await self.get_knowledge_graph(repo_id, max_hops=max_hops)

        if action == "neighbors":
            neighbors = kg.get_neighbors(start_node_id, edge_types=edge_types)
            return {"start_node": start_node_id, "neighbors": neighbors, "count": len(neighbors)}

        elif action == "shortest_path" and target_node_id:
            path = kg.shortest_path(start_node_id, target_node_id, max_hops=max_hops)
            return {
                "start_node": start_node_id,
                "target_node": target_node_id,
                "path": path,
                "found": path is not None,
            }

        elif action == "evidence":
            return kg.get_node_evidence(start_node_id)

        else:
            # Default to traverse
            res = kg.traverse(start_node_id, max_hops=max_hops, edge_types=edge_types)
            return {
                "start_node": start_node_id,
                "visited_nodes": res["visited_nodes"],
                "traversed_edges": res["traversed_edges"],
                "node_count": len(res["visited_nodes"]),
                "edge_count": len(res["traversed_edges"]),
            }

    async def find_graph_path(
        self, repo_id: str, start_symbol_or_file: str, target_symbol_or_file: str
    ) -> Dict[str, Any]:
        """Find relationship path between two entities in the unified graph."""
        kg = await self.get_knowledge_graph(repo_id)

        # Resolve candidate node IDs
        start_candidates = [
            nid for nid in kg._nodes if start_symbol_or_file in nid
        ]
        target_candidates = [
            nid for nid in kg._nodes if target_symbol_or_file in nid
        ]

        if not start_candidates or not target_candidates:
            return {
                "start": start_symbol_or_file,
                "target": target_symbol_or_file,
                "path_found": False,
                "message": "One or both entities not found in repository graph.",
            }

        start_id = start_candidates[0]
        target_id = target_candidates[0]

        path = kg.shortest_path(start_id, target_id, max_hops=5)
        return {
            "start": start_id,
            "target": target_id,
            "path_found": path is not None,
            "path": path or [],
        }
