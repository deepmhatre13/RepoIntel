"""Graph-aware retrieval: bounded expansion over the repository knowledge graph (Stage 5).

Takes a set of seed nodes/evidence items discovered via hybrid retrieval and expands
them into a minimal relevant evidence subgraph. All expansion is:

  1. Bounded: max traversal depth, max nodes, edge-type filtering, relevance thresholds
  2. Cycle-protected: visited-set tracking prevents infinite loops
  3. Evidence-preserving: every expanded edge retains its provenance

The retriever constructs a minimal relevant evidence subgraph; it never dumps
the entire graph into the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from packages.knowledge.unified_graph import (
    RepositoryKnowledgeGraph,
    GraphNode,
    GraphEdge,
    EdgeType,
)


# Edge types that carry strong evidence for impact / architecture questions.
_IMPACT_RELEVANT_EDGES = {
    EdgeType.CALLS,
    EdgeType.DEPENDS_ON,
    EdgeType.REFERENCES,
    EdgeType.IMPLEMENTS,
    EdgeType.INHERITS,
    EdgeType.TESTS,
    EdgeType.CONTAINS,
}

# Edge types relevant to security questions.
_SECURITY_RELEVANT_EDGES = {
    EdgeType.SECURITY_FINDING_FOR,
    EdgeType.DEPENDS_ON,
    EdgeType.REFERENCES,
    EdgeType.CALLS,
}


@dataclass
class ExpansionConfig:
    """Controls graph-aware retrieval expansion behaviour."""

    max_depth: int = 2
    max_nodes: int = 30
    max_edges: int = 120
    edge_types: Optional[Set[EdgeType]] = None
    min_confidence: float = 0.0
    include_incoming: bool = True
    include_outgoing: bool = True


@dataclass
class ExpandedEvidence:
    """Result of a graph-aware expansion step."""

    seed_node_ids: List[str] = field(default_factory=list)
    expanded_node_ids: List[str] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    truncated: bool = False

    def nodes_and_edges(self, graph: RepositoryKnowledgeGraph) -> Dict[str, Any]:
        node_ids = list(dict.fromkeys(self.seed_node_ids + self.expanded_node_ids))
        nodes = [graph._nodes[nid].to_dict() for nid in node_ids if nid in graph._nodes]
        edge_dicts = [e.to_dict() for e in self.edges]
        return {"nodes": nodes, "edges": edge_dicts}
class GraphExpander:
    """Performs bounded, cycle-protected, provenance-preserving graph expansion."""

    def __init__(self, graph: RepositoryKnowledgeGraph, config: Optional[ExpansionConfig] = None):
        self.graph = graph
        self.config = config or ExpansionConfig()

    def resolve_seed(self, seed: str) -> List[str]:
        """Match a raw seed (symbol name or id) against graph node ids.

        If the seed is already a full node id, return it directly.
        Otherwise, attempt an exact and then a substring match on symbol names.
        """
        # Already a node id?
        if seed in self.graph._nodes:
            return [seed]

        # Not a node id - attempt symbol-name resolution.
        matches = []
        seed_lower = seed.lower()
        for nid in self.graph._nodes:
            node = self.graph._nodes[nid]
            name = (node.name or "").lower()
            if name == seed_lower or name.endswith(f".{seed_lower}"):
                matches.append(nid)
        if matches:
            return matches

        # Fall back to substring match (bounded to keep evidence focused).
        for nid in self.graph._nodes:
            node = self.graph._nodes[nid]
            name = (node.name or "").lower()
            if seed_lower in name:
                matches.append(nid)
        return matches[:5]

    def expand(self, seeds: List[str], config: Optional[ExpansionConfig] = None) -> ExpandedEvidence:
        """Bounded expansion starting from seed node ids / query terms.

        Performs a BFS traversal rooted at the seed nodes, honouring the
        depth, node-count, edge-count, and edge-type limits in the config.
        Records every traversed edge so callers can preserve provenance.
        """
        cfg = config or self.config
        edge_types = cfg.edge_types
        result = ExpandedEvidence()

        for seed in seeds:
            resolved = self.resolve_seed(seed)
            for nid in resolved:
                if nid not in result.seed_node_ids:
                    result.seed_node_ids.append(nid)

        if not result.seed_node_ids:
            return result

        visited: Set[str] = set()
        visited_edges: List[GraphEdge] = []
        queue: List[tuple] = [(nid, 0) for nid in result.seed_node_ids]
        node_count = 0

        while queue and node_count < cfg.max_nodes:
            curr_id, depth = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            node_count += 1

            if depth >= cfg.max_depth:
                continue

            candidates: List[GraphEdge] = []
            if cfg.include_outgoing:
                candidates.extend(self.graph._adj_out.get(curr_id, []))
            if cfg.include_incoming:
                candidates.extend(self.graph._adj_in.get(curr_id, []))

            for edge in candidates:
                if len(visited_edges) >= cfg.max_edges:
                    result.truncated = True
                    break

                et = edge.edge_type
                if edge_types is not None and et not in edge_types:
                    continue
                if edge.confidence < cfg.min_confidence:
                    continue

                visited_edges.append(edge)
                nxt_id = edge.target_id if edge.source_id == curr_id else edge.source_id
                if nxt_id not in visited and nxt_id not in {q[0] for q in queue}:
                    queue.append((nxt_id, depth + 1))

            if len(visited_edges) >= cfg.max_edges:
                result.truncated = True
                break

        result.expanded_node_ids = [nid for nid in visited]
        result.edges = visited_edges
        return result

    def expand_for_question(self, seeds: List[str], question_kind: str) -> ExpandedEvidence:
        """Pick sensible default edge-type filters based on the question kind."""
        kind = (question_kind or "").lower()

        if "security" in kind or "risk" in kind:
            cfg = ExpansionConfig(
                max_depth=2,
                max_nodes=25,
                edge_types=_SECURITY_RELEVANT_EDGES,
            )
        elif "impact" in kind or "break" in kind or "affect" in kind:
            cfg = ExpansionConfig(
                max_depth=3,
                max_nodes=40,
                edge_types=_IMPACT_RELEVANT_EDGES,
            )
        else:
            cfg = ExpansionConfig(
                max_depth=2,
                max_nodes=30,
                edge_types={
                    EdgeType.CONTAINS,
                    EdgeType.DEFINES,
                    EdgeType.CALLS,
                    EdgeType.REFERENCES,
                    EdgeType.INHERITS,
                    EdgeType.IMPLEMENTS,
                },
            )

        return self.expand(seeds, cfg)


def build_evidence_subgraph(
    graph: RepositoryKnowledgeGraph,
    seed_terms: List[str],
    question_kind: str = "",
    max_depth: int = 2,
    max_nodes: int = 30,
) -> Dict[str, Any]:
    """Convenience entry point: expand seeds and return a minimal evidence subgraph dict."""
    expander = GraphExpander(graph)
    result = expander.expand_for_question(seed_terms, question_kind)
    return result.nodes_and_edges(graph)