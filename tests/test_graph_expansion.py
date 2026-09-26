"""Tests for the rich repository knowledge graph expansion (Stage 4/5)."""

import pytest

from packages.knowledge.unified_graph import (
    RepositoryKnowledgeGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)
from packages.knowledge.graph_expansion import (
    GraphExpander,
    ExpansionConfig,
    ExpandedEvidence,
    build_evidence_subgraph,
)


def build_test_graph() -> RepositoryKnowledgeGraph:
    """Build a small graph with symbols, inheritance, references, and a security edge."""
    g = RepositoryKnowledgeGraph(max_hops=4, max_nodes=100)

    g.add_node(GraphNode(id="file:auth.py", node_type=NodeType.FILE, name="auth.py"))
    g.add_node(GraphNode(id="file:user.py", node_type=NodeType.FILE, name="user.py"))
    g.add_node(GraphNode(id="file:jwt.py", node_type=NodeType.FILE, name="jwt.py"))
    for nid, name in [
        ("symbol:auth.py:authenticate", "authenticate"),
        ("symbol:auth.py:AuthService", "AuthService"),
        ("symbol:auth.py:JWTValidator", "JWTValidator"),
        ("symbol:user.py:UserService", "UserService"),
        ("symbol:user.py:UserRepository", "UserRepository"),
        ("symbol:jwt.py:JWTValidator", "JWTValidator"),
    ]:
        g.add_node(GraphNode(id=nid, node_type=NodeType.SYMBOL, name=name,
                             properties={"symbol_type": "mixed", "file_path": nid.split(":")[1]}))

    # FILE -> DEFINES -> SYMBOL
    for f in ["auth.py", "user.py", "jwt.py"]:
        prefix = f"file:{f}"
        for nid in list(g._nodes):
            if nid.startswith(f"symbol:{f}:"):
                g.add_edge(GraphEdge(source_id=prefix, target_id=nid,
                                     edge_type=EdgeType.DEFINES, evidence={"source": "test"}))

    # Symbol relationships
    g.add_edge(GraphEdge(source_id="symbol:auth.py:AuthService", target_id="symbol:auth.py:authenticate",
                         edge_type=EdgeType.CALLS, evidence={"source": "extracted"}))
    g.add_edge(GraphEdge(source_id="symbol:auth.py:JWTValidator", target_id="symbol:jwt.py:JWTValidator",
                         edge_type=EdgeType.REFERENCES, evidence={"source": "extracted"}))
    g.add_edge(GraphEdge(source_id="symbol:auth.py:AuthService", target_id="symbol:user.py:UserService",
                         edge_type=EdgeType.CALLS, evidence={"source": "extracted"}))
    g.add_edge(GraphEdge(source_id="symbol:user.py:UserService", target_id="symbol:user.py:UserRepository",
                         edge_type=EdgeType.CALLS, evidence={"source": "extracted"}))
    g.add_edge(GraphEdge(source_id="symbol:auth.py:JWTValidator", target_id="symbol:auth.py:authenticate",
                         edge_type=EdgeType.INHERITS, evidence={"source": "extracted"}))

    # Security finding (add node first, then edge)
    g.add_node(GraphNode(id="finding:auth.py:hardcoded_secret", node_type=NodeType.SYMBOL,
                         name="hardcoded_secret", properties={"severity": "HIGH"}))
    g.add_edge(GraphEdge(source_id="finding:auth.py:hardcoded_secret",
                         target_id="symbol:auth.py:authenticate",
                         edge_type=EdgeType.SECURITY_FINDING_FOR,
                         evidence={"source": "semgrep"}))
    return g


class TestEdgeTypeExpansion:
    def test_new_edge_types_available(self):
        assert EdgeType.INHERITS in EdgeType
        assert EdgeType.REFERENCES in EdgeType
        assert EdgeType.SECURITY_FINDING_FOR in EdgeType
        assert EdgeType.EXPOSES in EdgeType
        assert EdgeType.INTRODUCED_BY in EdgeType
        assert EdgeType.MODIFIED_BY in EdgeType
        assert EdgeType.DELETED_BY in EdgeType
        assert EdgeType.AFFECTS in EdgeType
        assert EdgeType.RELATED_TO in EdgeType
        assert EdgeType.CHANGED_BY in EdgeType

    def test_legacy_edge_types_preserved(self):
        assert EdgeType.CONTAINS in EdgeType
        assert EdgeType.CALLS in EdgeType
        assert EdgeType.DEPENDS_ON in EdgeType
        assert EdgeType.DEFINES in EdgeType
        assert EdgeType.TESTS in EdgeType
        assert EdgeType.IMPLEMENTS in EdgeType


class TestGraphExpander:
    def test_resolve_seed_exact_match(self):
        g = build_test_graph()
        expander = GraphExpander(g)
        resolved = expander.resolve_seed("authenticate")
        assert len(resolved) == 1
        assert resolved[0] == "symbol:auth.py:authenticate"

    def test_resolve_seed_by_id(self):
        g = build_test_graph()
        expander = GraphExpander(g)
        resolved = expander.resolve_seed("symbol:auth.py:AuthService")
        assert resolved == ["symbol:auth.py:AuthService"]

    def test_resolve_seed_no_match(self):
        g = build_test_graph()
        expander = GraphExpander(g)
        resolved = expander.resolve_seed("nonexistent_symbol")
        assert resolved == []

    def test_expand_bounded_depth(self):
        g = build_test_graph()
        config = ExpansionConfig(max_depth=1, max_nodes=50)
        expander = GraphExpander(g, config)
        result = expander.expand(["authenticate"], config)
        assert len(result.seed_node_ids) == 1
        assert len(result.expanded_node_ids) >= 2
        assert len(result.edges) >= 1

    def test_expand_cycle_protection(self):
        g = build_test_graph()
        # Add a cycle: authenticate -> AuthService (already AuthService -> authenticate)
        g.add_edge(GraphEdge(source_id="symbol:auth.py:authenticate",
                             target_id="symbol:auth.py:AuthService",
                             edge_type=EdgeType.CALLS))
        config = ExpansionConfig(max_depth=3, max_nodes=100)
        expander = GraphExpander(g, config)
        result = expander.expand(["authenticate"], config)
        assert result.expanded_node_ids is not None
        assert len(result.expanded_node_ids) >= 1

    def test_expand_max_nodes_bounded(self):
        g = build_test_graph()
        config = ExpansionConfig(max_depth=5, max_nodes=3, edge_types=None)
        expander = GraphExpander(g, config)
        result = expander.expand(["authenticate"], config)
        assert len(result.expanded_node_ids) <= config.max_nodes

    def test_expand_edge_type_filter(self):
        g = build_test_graph()
        config = ExpansionConfig(max_depth=3, max_nodes=50, edge_types={EdgeType.INHERITS})
        expander = GraphExpander(g, config)
        result = expander.expand(["authenticate"], config)
        for edge in result.edges:
            assert edge.edge_type == EdgeType.INHERITS

    def test_expand_provenance_preserved(self):
        g = build_test_graph()
        config = ExpansionConfig(max_depth=2, max_nodes=50)
        expander = GraphExpander(g, config)
        result = expander.expand(["authenticate"], config)
        for edge in result.edges:
            assert edge.evidence, "Every edge must have provenance"
            assert isinstance(edge.evidence, dict)

    def test_expand_for_question_impact(self):
        g = build_test_graph()
        expander = GraphExpander(g)
        result = expander.expand_for_question(["authenticate"], "What breaks if authenticate changes?")
        assert result.edges
        for edge in result.edges:
            assert edge.edge_type in {
                EdgeType.CALLS, EdgeType.DEPENDS_ON, EdgeType.REFERENCES,
                EdgeType.IMPLEMENTS, EdgeType.INHERITS, EdgeType.TESTS, EdgeType.CONTAINS,
            }

    def test_expand_for_question_security(self):
        g = build_test_graph()
        expander = GraphExpander(g)
        result = expander.expand_for_question(["authenticate"], "What security risks exist for auth?")
        security_edges = [e for e in result.edges if e.edge_type == EdgeType.SECURITY_FINDING_FOR]
        assert security_edges


class TestBuildEvidenceSubgraph:
    def test_build_subgraph_returns_dict(self):
        g = build_test_graph()
        subgraph = build_evidence_subgraph(g, ["authenticate"], question_kind="impact")
        assert isinstance(subgraph, dict)
        assert "nodes" in subgraph
        assert "edges" in subgraph
        assert isinstance(subgraph["nodes"], list)
        assert isinstance(subgraph["edges"], list)

    def test_build_subgraph_contains_seed(self):
        g = build_test_graph()
        subgraph = build_evidence_subgraph(g, ["authenticate"], question_kind="")
        node_names = [n.get("name") for n in subgraph["nodes"]]
        assert "authenticate" in node_names

    def test_build_subgraph_empty_for_no_match(self):
        g = build_test_graph()
        subgraph = build_evidence_subgraph(g, ["does_not_exist"], question_kind="")
        assert isinstance(subgraph, dict)
        assert subgraph["nodes"] == []