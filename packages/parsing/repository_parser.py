"""RepositoryParser — unified multi-language parsing boundary.

Routes each source file to the most appropriate parser:

* Python (.py)       → Python AST (richer semantic info: decorators, docstrings,
                       keyword args, async/await, imports with aliases)
* JS/TS/Java/Go/Rust/C/C++/etc. → Tree-sitter structural parser
* Unsupported        → heuristic fallback (generic_fallback)

Both paths emit :class:`NormalizedFile` so downstream consumers
(chunking, graph building, persistence) remain language-agnostic.

The parser does NOT mutate repository content or execute any code.
"""

from __future__ import annotations

import os
from typing import Optional

from packages.parsing.normalized_model import NormalizedFile
from packages.parsing.tree_sitter_parser import (
    parse_file_with_tree_sitter,
    is_language_supported,
)


def _get_ext(file_path: str) -> str:
    """Return the lowercase extension including the dot."""
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def parse_python_source(repository_id: str, file_path: str,
                        content: str) -> NormalizedFile:
    """Parse Python source using the richer Python-AST parser.

    Falls back to tree-sitter if AST fails (e.g., syntax errors).
    """
    try:
        from packages.parsing.python_ast import parse_python_file
        result = parse_python_file(file_path, content)
        if result and result.parse_succeeded:
            return _python_ast_to_normalized(repository_id, file_path, content, result)
    except Exception:
        pass
    # Fallback to tree-sitter
    return parse_file_with_tree_sitter(repository_id, file_path, content, ".py")


def _python_ast_to_normalized(repository_id: str, file_path: str,
                              content: str, ast_result) -> NormalizedFile:
    """Convert Python AST output to NormalizedFile."""
    from packages.parsing.normalized_model import (
        NormalizedSymbol, NormalizedEdge, NormalizedFile,
        CLASS, FUNCTION, METHOD, VARIABLE, IMPORT, MODULE,
        CONTAINS, INHERITS, CALLS, FROM_PYTHON_AST,
    )

    symbols = []
    edges = []

    for cls in ast_result.symbols:
        if cls.symbol_type != "class":
            continue
        symbols.append(NormalizedSymbol(
            repository_id=repository_id,
            file_path=file_path,
            language="Python",
            symbol_name=cls.symbol_name,
            symbol_type=CLASS,
            line_start=cls.line_start,
            line_end=cls.line_end,
            signature=cls.signature,
            docstring=cls.docstring,
            extraction_method=FROM_PYTHON_AST,
            confidence=1.0,
        ))
        edges.append(NormalizedEdge(
            repository_id=repository_id,
            source_symbol=file_path,
            source_file=file_path,
            edge_type=CONTAINS,
            target_symbol=cls.symbol_name,
            target_file=file_path,
            extraction_method=FROM_PYTHON_AST,
            confidence=1.0,
        ))

    for sym in ast_result.symbols:
        if sym.symbol_type in ("function", "method"):
            sym_type = FUNCTION if sym.symbol_type == "function" else METHOD
            symbols.append(NormalizedSymbol(
                repository_id=repository_id,
                file_path=file_path,
                language="Python",
                symbol_name=sym.symbol_name,
                symbol_type=sym_type,
                line_start=sym.line_start,
                line_end=sym.line_end,
                signature=sym.signature,
                docstring=sym.docstring,
                parent_class=sym.parent_class,
                extraction_method=FROM_PYTHON_AST,
                confidence=1.0,
            ))
            edges.append(NormalizedEdge(
                repository_id=repository_id,
                source_symbol=file_path,
                source_file=file_path,
                edge_type=CONTAINS,
                target_symbol=sym.symbol_name,
                target_file=file_path,
                extraction_method=FROM_PYTHON_AST,
                confidence=1.0,
            ))

    for edge in ast_result.edges:
        if edge.edge_type == "inherits":
            edges.append(NormalizedEdge(
                repository_id=repository_id,
                source_symbol=edge.source_symbol,
                source_file=edge.source_file,
                edge_type=INHERITS,
                target_symbol=edge.target_symbol,
                extraction_method=FROM_PYTHON_AST,
                confidence=0.95,
            ))
        elif edge.edge_type == "imports":
            symbols.append(NormalizedSymbol(
                repository_id=repository_id,
                file_path=file_path,
                language="Python",
                symbol_name=edge.target_symbol,
                symbol_type=IMPORT,
                line_start=1,
                line_end=1,
                extraction_method=FROM_PYTHON_AST,
                confidence=1.0,
            ))
        elif edge.edge_type == "calls":
            edges.append(NormalizedEdge(
                repository_id=repository_id,
                source_symbol=edge.source_symbol,
                source_file=edge.source_file,
                edge_type=CALLS,
                target_symbol=edge.target_symbol,
                extraction_method=FROM_PYTHON_AST,
                confidence=0.9,
            ))

    return NormalizedFile(
        repository_id=repository_id,
        file_path=file_path,
        language="Python",
        parse_succeeded=True,
        symbols=symbols,
        edges=edges,
        classes=sum(1 for s in symbols if s.symbol_type == CLASS),
        functions=sum(1 for s in symbols if s.symbol_type == FUNCTION),
        methods=sum(1 for s in symbols if s.symbol_type == METHOD),
        imports=sum(1 for s in symbols if s.symbol_type == IMPORT),
    )


def parse_file(repository_id: str, file_path: str, content: str) -> NormalizedFile:
    """Parse a single source file, routing to the appropriate parser.

    Returns a :class:`NormalizedFile`. Never raises; on failure returns
    a NormalizedFile with ``parse_succeeded=False``.
    """
    ext = _get_ext(file_path)

    # Python: prefer AST
    if ext == ".py":
        return parse_python_source(repository_id, file_path, content)

    # All other supported languages: tree-sitter
    if is_language_supported(ext):
        return parse_file_with_tree_sitter(repository_id, file_path, content, ext)

    # Final fallback: minimal metadata
    return NormalizedFile(
        repository_id=repository_id,
        file_path=file_path,
        language="Unknown",
        parse_succeeded=False,
        parse_error=f"unsupported file type: {ext}",
    )
