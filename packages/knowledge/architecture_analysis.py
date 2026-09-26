"""Phase 24: Architecture Evolution Intelligence.

Detects components from file structure, builds component-level graphs,
creates architecture snapshots, and compares them over time.
"""
from collections import defaultdict
from typing import List, Dict, Any, Set
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from packages.knowledge.models import (
    SymbolModel, SymbolEdgeModel, CommitModel
)

# Component type definitions based on naming conventions
COMPONENT_TYPE_BY_DIR = {
    "test": "Tests",
    "tests": "Tests",
    "spec": "Tests",
    "specs": "Tests",
    "config": "Configuration",
    "configs": "Configuration",
    "infrastructure": "Infrastructure",
    "infra": "Infrastructure",
    "migrations": "Database",
    "db": "Database",
    "docker": "Infrastructure",
    "ci": "Infrastructure",
    "github": "Infrastructure",
    "gitlab": "Infrastructure",
}

CONFIG_FILENAMES = {
    "docker-compose.yml", "docker-compose.yaml", "Dockerfile", ".env", ".env.local",
    "config.yaml", "config.yml", "settings.py", "pyproject.toml", "setup.cfg"
}

ENTRYPOINT_NAMES = {"main.py", "app.py", "wsgi.py", "asgi.py", "server.py", "index.js", "server.js"}


def detect_component_type(dir_parts: List[str], filename: str, file_path: str) -> str:
    """Determine the component type based on directory structure and conventions."""
    dir_lower = [d.lower() for d in dir_parts]
    fname_lower = filename.lower()

    # Check for test files
    if "test" in fname_lower or any(d in ("test", "tests", "spec", "specs") for d in dir_lower):
        return "Tests"
    if fname_lower.endswith(".test.") or fname_lower.endswith(".spec."):
        return "Tests"

    # Check for config files
    if fname_lower in CONFIG_FILENAMES or any(d in ("config", "configs") for d in dir_lower):
        return "Configuration"

    # Check for database/migrations
    if "migration" in fname_lower or any(d in ("migration", "migrations", "db", "database") for d in dir_lower):
        return "Database"

    # Check for infrastructure
    if any(d in ("infra", "infrastructure", "docker", "ci", ".github") for d in dir_lower):
        return "Infrastructure"

    # Check for entrypoints (API/CLI)
    if fname_lower in ENTRYPOINT_NAMES:
        return "API" if "api" in file_path.lower() else "Entry Point"

    # Check for API layer
    if any("api" in d for d in dir_lower) or fname_lower.startswith("api"):
        return "API"

    # Check for services layer
    if any(d in ("service", "services", "srv") for d in dir_lower):
        return "Services"

    # Check for models layer
    if any(d in ("model", "models", "schema", "schemas", "dto") for d in dir_lower):
        return "Models"

    # Check for repositories
    if any(d in ("repo", "repos", "repository", "repositories", "dao") for d in dir_lower):
        return "Repository/Data Layer"

    # Check for workers
    if any(d in ("worker", "workers", "job", "jobs", "task") for d in dir_lower):
        return "Workers"

    # Default: infer from top-level directory
    if len(dir_parts) > 0:
        return f"Module ({dir_parts[0]})"
    return "Root"


def extract_component_name(file_path: str) -> str:
    """Extract the component name from a file path (e.g., src/auth/service.py -> auth)."""
    parts = file_path.replace("\\", "/").split("/")
    # Skip common root dirs
    skip_dirs = {"src", "lib", "pkg", "packages", "app", "apps"}
    filtered = [p for p in parts if p not in skip_dirs]
    if len(filtered) >= 2:
        # Use the first non-skip directory as component name
        return filtered[0] if not filtered[0].endswith(".py") else "root"
    elif len(filtered) == 1:
        f = filtered[0]
        return f.replace(".py", "") if not f.startswith(".") else "root"
    return "root"


def infer_component_from_file(file_path: str, file_set: Set[str]) -> str:
    """Infer component name from file path, using the most specific directory grouping."""
    parts = file_path.replace("\\", "/").split("/")
    # Find the first directory with multiple files (indicating a component)
    skip_dirs = {"src", "lib", "pkg", "packages", "app"}
    best_dir = "root"
    for i in range(1, len(parts)):
        dir_part = "/".join(parts[:i])
        if dir_part in skip_dirs or parts[i-1] in skip_dirs:
            continue
        # Count files in this directory level
        sibling_count = sum(1 for f in file_set if "/".join(f.split("/")[:i]) == dir_part)
        if sibling_count >= 1:
            best_dir = parts[i-1]
    return best_dir


class ArchitectureAnalyzer:
    """Analyzes and builds architecture snapshots from stored repository data."""

    def __init__(self, max_modules: int = 100):
        self.max_modules = max_modules

    async def build_snapshot(
        self,
        session: AsyncSession,
        repo_id: str,
        commit_sha: str
    ) -> Dict[str, Any]:
        """Build an architecture snapshot for a repository at a given commit."""
        from packages.knowledge.models import FileModel

        # Get all symbols for this repo
        sym_stmt = select(SymbolModel).where(SymbolModel.repo_id == repo_id)
        sym_res = await session.execute(sym_stmt)
        symbols = sym_res.scalars().all()

        # Get all symbol edges
        edge_stmt = select(SymbolEdgeModel).where(SymbolEdgeModel.repo_id == repo_id)
        edge_res = await session.execute(edge_stmt)
        edges = edge_res.scalars().all()

        # Get all files for this repo
        file_stmt = select(FileModel).where(FileModel.repo_id == repo_id)
        file_res = await session.execute(file_stmt)
        files = file_res.scalars().all()

        # Get commit info (for timestamp)
        commit_stmt = select(CommitModel).where(CommitModel.repo_id == repo_id, CommitModel.sha == commit_sha)
        commit_res = await session.execute(commit_stmt)
        commit = commit_res.scalar_one_or_none()

        # Group files into components
        file_paths = [f.file_path for f in files]
        file_set = set(file_paths)
        components = self._detect_components(file_paths, file_set)

        # Build module-level edges from symbol-level edges
        component_edges = self._build_component_edges(edges, components)

        # Identify entrypoints
        entrypoints = self._detect_entrypoints(file_paths)

        # Identify major components (those with most symbols)
        component_symbol_counts = defaultdict(int)
        for s in symbols:
            comp = components.get(s.file_path, "root")
            component_symbol_counts[comp] += 1

        major_components = [
            {"name": name, "symbol_count": count, "file_count": len([fp for fp in file_paths if components.get(fp) == name])}
            for name, count in sorted(component_symbol_counts.items(), key=lambda x: x[1], reverse=True)
        ][:self.max_modules]

        # Dependency groups: clusters of components that depend on each other
        dep_groups = self._detect_dependency_groups(component_edges)

        # Component metadata
        component_metadata = {}
        for comp_name in components.values():
            if comp_name not in component_metadata:
                component_metadata[comp_name] = {
                    "type": self._detect_component_type(comp_name, file_paths),
                    "file_count": len([fp for fp in file_paths if components.get(fp) == comp_name])
                }

        snapshot_data = {
            "repo_id": repo_id,
            "commit_sha": commit_sha,
            "modules": list(set(components.values())),
            "module_edges": component_edges,
            "major_components": major_components,
            "entrypoints": entrypoints,
            "dependency_groups": dep_groups,
            "component_metadata": component_metadata,
            "total_symbols": len(symbols),
            "total_files": len(files),
            "commit_timestamp": commit.timestamp.isoformat() if commit else None
        }

        return snapshot_data

    def _detect_components(self, file_paths: List[str], file_set: Set[str]) -> Dict[str, str]:
        """Map each file path to its component name."""
        components = {}
        for fp in file_paths:
            parts = fp.replace("\\", "/").split("/")
            skip_dirs = {"src", "lib", "pkg", "packages", "app", "apps"}

            # Find the best component boundary
            best_component = "root"
            for i, part in enumerate(parts):
                if part in skip_dirs or part.startswith("."):
                    continue
                # Check if there are sibling files in this directory
                dir_prefix = "/".join(parts[:i]) if i > 0 else ""
                if dir_prefix:
                    siblings = sum(1 for f in file_set if "/".join(f.split("/")[:i]) == dir_prefix)
                else:
                    siblings = len(file_paths)

                if not part.endswith(".py") and siblings > 0:
                    best_component = part
                    break

            # If no directory component found, try file-level grouping
            if best_component == "root" and len(parts) >= 2:
                # Use parent directory as component if not a skip dir
                for part in reversed(parts[:-1]):
                    if part not in skip_dirs and not part.startswith("."):
                        best_component = part
                        break

            components[fp] = best_component
        return components

    def _build_component_edges(self, edges: List[SymbolEdgeModel], components: Dict[str, str]) -> List[Dict[str, Any]]:
        """Build module-level edges from symbol-level edges."""
        component_edges = []
        seen_edges = set()

        for e in edges:
            src_comp = components.get(e.source_file, "root")
            tgt_comp = components.get(e.target_file, "root") if e.target_file else components.get(e.target_symbol, "root")

            if src_comp == tgt_comp:
                continue

            edge_key = (src_comp, tgt_comp, e.edge_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            component_edges.append({
                "source_component": src_comp,
                "target_component": tgt_comp,
                "edge_type": e.edge_type.upper() if e.edge_type else "DEPENDS_ON",
                "target_symbol": e.target_symbol
            })

        return component_edges

    def _detect_entrypoints(self, file_paths: List[str]) -> List[str]:
        """Detect entrypoint files from file paths."""
        entrypoints = []
        for fp in file_paths:
            basename = fp.split("/")[-1]
            if basename in ENTRYPOINT_NAMES:
                entrypoints.append(fp)
        return entrypoints

    def _detect_component_type(self, comp_name: str, file_paths: List[str]) -> str:
        """Detect component type from component name and its files."""
        comp_files = [f for f in file_paths if self._get_component(f) == comp_name]
        for f in comp_files:
            dir_parts = f.replace("\\", "/").split("/")
            fname = f.split("/")[-1]
            ctype = detect_component_type(dir_parts, fname, f)
            if ctype != "Root":
                return ctype
        return "Module"

    def _get_component(self, file_path: str) -> str:
        parts = file_path.replace("\\", "/").split("/")
        skip_dirs = {"src", "lib", "pkg", "packages", "app", "apps"}
        for part in parts:
            if part not in skip_dirs and not part.startswith(".") and not part.endswith(".py"):
                return part
        return "root"

    def _detect_dependency_groups(self, component_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect clusters of components that depend on each other."""
        adj = defaultdict(set)
        for edge in component_edges:
            adj[edge["source_component"]].add(edge["target_component"])

        groups = []
        visited = set()
        for comp in adj:
            if comp in visited:
                continue
            group = self._strongly_connected_component(comp, adj, visited)
            if len(group) > 1:
                groups.append({"components": list(group), "type": "cyclic_dependency_risk"})
            visited.update(group)
        return groups

    def _strongly_connected_component(self, start: str, adj: Dict[str, Set[str]], visited: Set[str]) -> Set[str]:
        """Find SCC containing start node using Tarjan's algorithm (simplified)."""
        result = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in result:
                continue
            result.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in result:
                    stack.append(neighbor)
        return result


class CycleDetector:
    """Detects circular dependencies in module/component graphs."""

    def detect_cycles(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect cycles in a directed graph using DFS."""
        adj = defaultdict(list)
        for e in edges:
            src = e.get("source_component") or e.get("source")
            tgt = e.get("target_component") or e.get("target")
            if src and tgt:
                adj[src].append(tgt)

        cycles = []
        visited = set()
        rec_stack = []
        rec_set = set()

        def dfs(node: str):
            visited.add(node)
            rec_stack.append(node)
            rec_set.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_set:
                    # Found a cycle
                    cycle_start = rec_stack.index(neighbor)
                    cycle = rec_stack[cycle_start:] + [neighbor]
                    cycles.append({
                        "cycle": cycle,
                        "nodes": set(cycle[:-1]),
                        "edges": [{"source": cycle[i], "target": cycle[i+1]} for i in range(len(cycle)-1)]
                    })

            rec_stack.pop()
            rec_set.discard(node)

        for node in adj:
            if node not in visited:
                dfs(node)

        # Deduplicate cycles
        unique_cycles = []
        seen = set()
        for c in cycles:
            key = frozenset(c["nodes"])
            if key not in seen:
                seen.add(key)
                unique_cycles.append(c)

        return unique_cycles
