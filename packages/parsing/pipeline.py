import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any
from packages.parsing.python_ast import parse_python_file, SymbolNode, CallEdge
from packages.parsing.js_ts_ast import parse_js_ts_file
from packages.parsing.generic_fallback import parse_generic_fallback, GenericFileMetadata
from packages.parsing.manifest_builder import extract_dependencies_from_manifest, extract_env_vars_from_code, ParsedDependency
from packages.parsing.chunking import create_ast_aware_chunks, CodeChunk
from packages.parsing.symbol_graph import SymbolGraph

@dataclass
class ParsingPipelineResult:
    repo_id: str
    symbols: List[SymbolNode] = field(default_factory=list)
    edges: List[CallEdge] = field(default_factory=list)
    dependencies: List[ParsedDependency] = field(default_factory=list)
    env_vars: List[str] = field(default_factory=list)
    chunks: List[CodeChunk] = field(default_factory=list)
    symbol_graph: SymbolGraph = field(default_factory=SymbolGraph)
    # --- Phase 40.6 failure visibility: no parsing failure is silently dropped ---
    files_seen: int = 0
    files_parsed: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)  # {file, error_type, message, recoverable}
    parsed_file_hashes: Dict[str, tuple] = field(default_factory=dict)  # path -> (sha256, size_bytes)

    def record_failure(self, file_path: str, error: Exception, recoverable: bool = True):
        self.failures.append({
            "file": file_path,
            "error_type": type(error).__name__,
            "message": str(error)[:500],
            "recoverable": recoverable,
        })

def run_parsing_pipeline(repo_id: str, repo_path: str, filtered_files: List[str]) -> ParsingPipelineResult:
    result = ParsingPipelineResult(repo_id=repo_id)
    env_vars_set = set()
    result.files_seen = len(filtered_files)

    for rel_path in filtered_files:
        full_path = os.path.join(repo_path, rel_path)
        if not os.path.isfile(full_path):
            result.record_failure(rel_path, FileNotFoundError(f"listed file missing on disk: {rel_path}"))
            continue

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as exc:
            result.record_failure(rel_path, exc, recoverable=True)
            continue

        import hashlib
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        result.parsed_file_hashes[rel_path] = (content_hash, len(content.encode("utf-8")))

        ext = os.path.splitext(rel_path)[1].lower()
        filename = os.path.basename(rel_path)

        # 1. Dependency manifest parsing
        deps = extract_dependencies_from_manifest(rel_path, content)
        result.dependencies.extend(deps)

        # 2. Extract environment variables
        envs = extract_env_vars_from_code(content)
        env_vars_set.update(envs)

        # 3. AST / Structural parsing
        file_symbols = []
        file_edges = []
        language = "Unknown"

        try:
            if ext == ".py":
                language = "Python"
                py_res = parse_python_file(rel_path, content)
                file_symbols = py_res.symbols
                file_edges = py_res.edges
            elif ext in (".js", ".ts", ".jsx", ".tsx"):
                language = "TypeScript" if "ts" in ext else "JavaScript"
                js_res = parse_js_ts_file(rel_path, content)
                file_symbols = js_res.symbols
                file_edges = js_res.edges
            else:
                language = ext.lstrip(".").upper() if ext else "Text"
                parse_generic_fallback(rel_path, content, language)
        except Exception as exc:
            # Syntax errors etc. are permanent for the file but must be visible (Phase 40.6/43.1).
            result.record_failure(rel_path, exc, recoverable=False)
            continue

        result.files_parsed += 1
        result.symbols.extend(file_symbols)
        result.edges.extend(file_edges)

        for e in file_edges:
            result.symbol_graph.add_edge(e)

        # 4. AST-aware chunking
        file_chunks = create_ast_aware_chunks(repo_id, rel_path, content, language, file_symbols)
        result.chunks.extend(file_chunks)

    result.env_vars = list(env_vars_set)
    return result
