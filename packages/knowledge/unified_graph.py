"""Unified Repository Knowledge Graph for RepoIntel (Phase 28).

Provides a coherent graph abstraction over existing persistent knowledge:
symbols, dependencies, architecture, Git history, test relationships, and cross-repo links.
Enforces bounded traversal and provenance tracking on all nodes and edges.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.knowledge.models import (
    RepositoryModel,
    SymbolModel,
    SymbolEdgeModel,
    DependencyModel,
    CommitModel,
    TestCaseModel,
    TestToCodeRelationModel,
    ArtifactModel,
    CrossRepoRelationModel,
)


class NodeType(str, Enum):
    REPOSITORY = "Repository"
    REPOSITORY_GROUP = "RepositoryGroup"
    FILE = "File"
    MODULE = "Module"
    SYMBOL = "Symbol"
    DEPENDENCY = "Dependency"
    COMMIT = "Commit"
    TEST = "Test"
    TEST_CASE = "TestCase"
    ARCHITECTURE_COMPONENT = "ArchitectureComponent"
    CONFIGURATION = "Configuration"
    ARTIFACT = "Artifact"


class EdgeType(str, Enum):
    CONTAINS = "CONTAINS"
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    DEPENDS_ON = "DEPENDS_ON"
    DEFINES = "DEFINES"
    TESTS = "TESTS"
    CONFIGURES = "CONFIGURES"
    CHANGED_IN = "CHANGED_IN"
    CO_CHANGED_WITH = "CO_CHANGED_WITH"
    API_CALLS = "API_CALLS"
    MEMBER_OF = "MEMBER_OF"
    PART_OF = "PART_OF"
    IMPLEMENTS = "IMPLEMENTS"
    DOCUMENTS = "DOCUMENTS"


@dataclass
class GraphNode:
    id: str
    node_type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value if isinstance(self.node_type, Enum) else str(self.node_type),
            "name": self.name,
            "properties": self.properties,
            "evidence": self.evidence,
        }


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value if isinstance(self.edge_type, Enum) else str(self.edge_type),
            "confidence": self.confidence,
            "properties": self.properties,
            "evidence": self.evidence,
        }


class RepositoryKnowledgeGraph:
    """Unified graph model representing repository entities and relationships."""

    def __init__(
        self,
        max_hops: int = 3,
        max_nodes: int = 50,
        max_paths: int = 10,
    ):
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.max_paths = max_paths

        self._nodes: Dict[str, GraphNode] = {}
        self._adj_out: Dict[str, List[GraphEdge]] = {}
        self._adj_in: Dict[str, List[GraphEdge]] = {}

    def add_node(self, node: GraphNode):
        self._nodes[node.id] = node
        if node.id not in self._adj_out:
            self._adj_out[node.id] = []
        if node.id not in self._adj_in:
            self._adj_in[node.id] = []

    def add_edge(self, edge: GraphEdge):
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            return
        self._adj_out[edge.source_id].append(edge)
        self._adj_in[edge.target_id].append(edge)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        n = self._nodes.get(node_id)
        return n.to_dict() if n else None

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        edge_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if node_id not in self._nodes:
            return []

        neighbors: List[GraphNode] = []
        target_types = set(edge_types) if edge_types else None

        if direction in ("out", "both"):
            for edge in self._adj_out.get(node_id, []):
                et = edge.edge_type.value if isinstance(edge.edge_type, Enum) else str(edge.edge_type)
                if target_types is None or et in target_types:
                    if edge.target_id in self._nodes:
                        neighbors.append(self._nodes[edge.target_id])

        if direction in ("in", "both"):
            for edge in self._adj_in.get(node_id, []):
                et = edge.edge_type.value if isinstance(edge.edge_type, Enum) else str(edge.edge_type)
                if target_types is None or et in target_types:
                    if edge.source_id in self._nodes:
                        neighbors.append(self._nodes[edge.source_id])

        seen = set()
        res = []
        for n in neighbors:
            if n.id not in seen:
                seen.add(n.id)
                res.append(n.to_dict())
        return res

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        edges: List[GraphEdge] = []
        target_types = set(edge_types) if edge_types else None

        if source_id and source_id in self._adj_out:
            candidates = self._adj_out[source_id]
        else:
            candidates = [e for edge_list in self._adj_out.values() for e in edge_list]

        for e in candidates:
            if target_id and e.target_id != target_id:
                continue
            et = e.edge_type.value if isinstance(e.edge_type, Enum) else str(e.edge_type)
            if target_types and et not in target_types:
                continue
            edges.append(e)

        return [e.to_dict() for e in edges]

    def traverse(
        self,
        start_node_id: str,
        max_hops: Optional[int] = None,
        edge_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        hops_limit = min(max_hops or self.max_hops, self.max_hops)
        if start_node_id not in self._nodes:
            return {"visited_nodes": [], "traversed_edges": []}

        visited_nodes: Set[str] = {start_node_id}
        traversed_edges: List[GraphEdge] = []
        queue = [(start_node_id, 0)]
        target_types = set(edge_types) if edge_types else None

        while queue and len(visited_nodes) < self.max_nodes:
            curr_id, depth = queue.pop(0)
            if depth >= hops_limit:
                continue

            for edge in self._adj_out.get(curr_id, []) + self._adj_in.get(curr_id, []):
                et = edge.edge_type.value if isinstance(edge.edge_type, Enum) else str(edge.edge_type)
                if target_types and et not in target_types:
                    continue

                nxt_id = edge.target_id if edge.source_id == curr_id else edge.source_id
                traversed_edges.append(edge)
                if nxt_id not in visited_nodes:
                    visited_nodes.add(nxt_id)
                    queue.append((nxt_id, depth + 1))

        return {
            "visited_nodes": [self._nodes[nid].to_dict() for nid in visited_nodes if nid in self._nodes],
            "traversed_edges": [e.to_dict() for e in traversed_edges],
        }

    def shortest_path(
        self, start_node_id: str, target_node_id: str, max_hops: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        if start_node_id not in self._nodes or target_node_id not in self._nodes:
            return None

        queue = [[start_node_id]]
        visited = {start_node_id}

        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == target_node_id:
                return [self._nodes[nid].to_dict() for nid in path]

            if len(path) > max_hops:
                continue

            for edge in self._adj_out.get(node, []):
                nxt = edge.target_id
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])

        return None

    def subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        node_set = set(node_ids)
        sub_nodes = [self._nodes[nid].to_dict() for nid in node_ids if nid in self._nodes]
        sub_edges = []
        for nid in node_ids:
            for e in self._adj_out.get(nid, []):
                if e.target_id in node_set:
                    sub_edges.append(e.to_dict())
        return {"nodes": sub_nodes, "edges": sub_edges}

    def find_related_nodes(
        self, start_node_id: str, relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        return self.get_neighbors(start_node_id, direction="both", edge_types=relation_types)

    def find_paths(
        self, start_node_id: str, end_node_id: str, max_hops: int = 5
    ) -> List[List[Dict[str, Any]]]:
        if start_node_id not in self._nodes or end_node_id not in self._nodes:
            return []

        paths: List[List[str]] = []
        queue = [[start_node_id]]

        while queue and len(paths) < self.max_paths:
            path = queue.pop(0)
            last = path[-1]
            if last == end_node_id:
                paths.append(path)
                continue

            if len(path) > max_hops:
                continue

            for edge in self._adj_out.get(last, []):
                nxt = edge.target_id
                if nxt not in path:
                    queue.append(path + [nxt])

        return [[self._nodes[nid].to_dict() for nid in p] for p in paths]

    def get_node_evidence(self, node_id: str) -> Dict[str, Any]:
        node = self._nodes.get(node_id)
        if not node:
            return {}
        incoming = [e.to_dict() for e in self._adj_in.get(node_id, [])]
        outgoing = [e.to_dict() for e in self._adj_out.get(node_id, [])]
        return {
            "node": node.to_dict(),
            "evidence": node.evidence,
            "incoming_edges": incoming,
            "outgoing_edges": outgoing,
        }

    async def build_from_session(self, session: AsyncSession, repo_id: str):
        """Populate graph from stored relational entities for a given repository."""
        # 1. Repository node
        repo_stmt = select(RepositoryModel).where(RepositoryModel.id == repo_id)
        repo_res = await session.execute(repo_stmt)
        repo = repo_res.scalar_one_or_none()
        if repo:
            self.add_node(
                GraphNode(
                    id=f"repo:{repo.id}",
                    node_type=NodeType.REPOSITORY,
                    name=repo.id,
                    properties={"repo_url": repo.repo_url, "archetype": repo.archetype},
                    evidence={"source": "RepositoryModel", "repo_id": repo.id},
                )
            )

        # 2. Symbols & Files
        sym_stmt = select(SymbolModel).where(SymbolModel.repo_id == repo_id)
        sym_res = await session.execute(sym_stmt)
        symbols = sym_res.scalars().all()
        for s in symbols:
            file_node_id = f"file:{s.file_path}"
            if file_node_id not in self._nodes:
                self.add_node(
                    GraphNode(
                        id=file_node_id,
                        node_type=NodeType.FILE,
                        name=s.file_path,
                        properties={"file_path": s.file_path},
                        evidence={"source": "SymbolModel", "repo_id": repo_id},
                    )
                )
                if repo:
                    self.add_edge(
                        GraphEdge(
                            source_id=f"repo:{repo.id}",
                            target_id=file_node_id,
                            edge_type=EdgeType.CONTAINS,
                            evidence={"source": "RepositoryModel"},
                        )
                    )

            sym_node_id = f"symbol:{s.file_path}:{s.symbol_name}"
            self.add_node(
                GraphNode(
                    id=sym_node_id,
                    node_type=NodeType.SYMBOL,
                    name=s.symbol_name,
                    properties={
                        "symbol_type": s.symbol_type,
                        "line_start": s.line_start,
                        "line_end": s.line_end,
                        "signature": s.signature,
                    },
                    evidence={"source": "SymbolModel", "file_path": s.file_path, "symbol": s.symbol_name},
                )
            )

            self.add_edge(
                GraphEdge(
                    source_id=file_node_id,
                    target_id=sym_node_id,
                    edge_type=EdgeType.DEFINES,
                    evidence={"source": "SymbolModel"},
                )
            )

        # 3. Symbol Edges (Calls, Imports, Dependencies)
        edge_stmt = select(SymbolEdgeModel).where(SymbolEdgeModel.repo_id == repo_id)
        edge_res = await session.execute(edge_stmt)
        edges = edge_res.scalars().all()
        for e in edges:
            src_id = f"symbol:{e.source_file}:{e.source_symbol}" if e.source_symbol and e.source_file else f"file:{e.source_file}"
            tgt_id = f"symbol:{e.target_file}:{e.target_symbol}" if e.target_symbol and e.target_file else f"file:{e.target_file}"
            if src_id in self._nodes and tgt_id in self._nodes:
                edge_type = EdgeType.CALLS if e.edge_type == "calls" else EdgeType.DEPENDS_ON
                self.add_edge(
                    GraphEdge(
                        source_id=src_id,
                        target_id=tgt_id,
                        edge_type=edge_type,
                        evidence={"source": "SymbolEdgeModel", "edge_type": e.edge_type},
                    )
                )

        # 4. Dependencies
        dep_stmt = select(DependencyModel).where(DependencyModel.repo_id == repo_id)
        dep_res = await session.execute(dep_stmt)
        deps = dep_res.scalars().all()
        for d in deps:
            dep_id = f"dep:{d.package_name}"
            self.add_node(
                GraphNode(
                    id=dep_id,
                    node_type=NodeType.DEPENDENCY,
                    name=d.package_name,
                    properties={"ecosystem": d.ecosystem, "version_spec": d.version_spec},
                    evidence={"source": "DependencyModel", "package_name": d.package_name},
                )
            )
            if repo:
                self.add_edge(
                    GraphEdge(
                        source_id=f"repo:{repo.id}",
                        target_id=dep_id,
                        edge_type=EdgeType.DEPENDS_ON,
                        evidence={"source": "DependencyModel"},
                    )
                )

        # 5. Tests
        test_stmt = select(TestCaseModel).where(TestCaseModel.repo_id == repo_id)
        test_res = await session.execute(test_stmt)
        tests = test_res.scalars().all()
        for t in tests:
            test_id = f"test:{t.file_path}:{t.test_name}"
            self.add_node(
                GraphNode(
                    id=test_id,
                    node_type=NodeType.TEST_CASE,
                    name=t.test_name,
                    properties={"test_type": t.test_type, "framework": t.framework},
                    evidence={"source": "TestCaseModel", "file_path": t.file_path},
                )
            )

        rel_stmt = select(TestToCodeRelationModel).where(TestToCodeRelationModel.repo_id == repo_id)
        rel_res = await session.execute(rel_stmt)
        rels = rel_res.scalars().all()
        for r in rels:
            test_id = f"test:{r.test_file}:{r.test_name}"
            tgt_sym_id = None
            for nid, node in self._nodes.items():
                if node.node_type == NodeType.SYMBOL and node.name == r.target_symbol:
                    tgt_sym_id = nid
                    break
            if test_id in self._nodes and tgt_sym_id in self._nodes:
                self.add_edge(
                    GraphEdge(
                        source_id=test_id,
                        target_id=tgt_sym_id,
                        edge_type=EdgeType.TESTS,
                        confidence=r.confidence or 1.0,
                        evidence={"source": "TestToCodeRelationModel", "relation_type": r.relation_type},
                    )
                )
