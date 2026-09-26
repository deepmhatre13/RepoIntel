"""Tree-sitter backed structural parser for multiple programming languages.

Provides a unified parse interface over tree-sitter for grammars that have
a tree-sitter grammar package installed. Emits
:class:`NormalizedSymbol` / :class:`NormalizedEdge` so the rest of the
pipeline is language-agnostic.

Grammars currently wired in:
    Python, JavaScript, TypeScript, TSX, Java, Go, Rust, C, C++

Python parsing still uses the richer Python-AST path in ``python_ast.py``
when invoked through the pipeline; this module is used for all other
supported grammars and as an optional alternative for Python when AST
fails.

Design notes
-----------
* Each grammar is loaded lazily on first use (with thread-safe caching).
* If a grammar package is unavailable, ``is_language_supported`` returns
  False and the pipeline falls back to the generic handler.
* All extracted entities retain source locations and a source excerpt so
  they can be cited as evidence.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from packages.parsing.normalized_model import (
    NormalizedSymbol,
    NormalizedEdge,
    NormalizedFile,
    CLASS,
    INTERFACE,
    FUNCTION,
    METHOD,
    VARIABLE,
    CONSTANT,
    STRUCT,
    ENUM,
    TYPE_ALIAS,
    TRAIT,
    IMPORT,
    MODULE,
    NAMESPACE,
    UNKNOWN_SYMBOL,
    CALLS,
    INHERITS,
    IMPLEMENTS,
    IMPORTS,
    DEFINES,
    CONTAINS,
    FROM_TREE_SITTER,
)


# ---------------------------------------------------------------------------
# Grammar registry
# ---------------------------------------------------------------------------
# Newer tree-sitter packages (0.26+) expose a `language()` function for most
# grammars. TypeScript is the exception with `language_typescript()` and
# `language_tsx()`.
_GRAMMAR_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    ".py":   ("Python",     "tree_sitter_python",   "language"),
    ".js":   ("JavaScript", "tree_sitter_javascript","language"),
    ".jsx":  ("JavaScript", "tree_sitter_javascript","language"),
    ".ts":   ("TypeScript", "tree_sitter_typescript","language_typescript"),
    ".tsx":  ("TypeScript", "tree_sitter_typescript","language_tsx"),
    ".java": ("Java",       "tree_sitter_java",     "language"),
    ".go":   ("Go",         "tree_sitter_go",       "language"),
    ".rs":   ("Rust",       "tree_sitter_rust",     "language"),
    ".c":    ("C",          "tree_sitter_c",        "language"),
    ".h":    ("C",          "tree_sitter_c",        "language"),
    ".cpp":  ("C++",        "tree_sitter_cpp",      "language"),
    ".cc":   ("C++",        "tree_sitter_cpp",      "language"),
    ".hpp":  ("C++",        "tree_sitter_cpp",      "language"),
}

_language_cache: Dict[str, object] = {}
_language_lock = threading.Lock()


def supported_extensions() -> List[str]:
    """Return the list of file extensions this parser can handle."""
    return list(_GRAMMAR_REGISTRY.keys())


def _normalize_ext(ext: str) -> Optional[str]:
    ext_lower = ext.lower()
    if ext_lower in _GRAMMAR_REGISTRY:
        return ext_lower
    if not ext_lower.startswith("."):
        dotted = "." + ext_lower
        if dotted in _GRAMMAR_REGISTRY:
            return dotted
    for key, (lang, _, _) in _GRAMMAR_REGISTRY.items():
        if lang.lower() == ext_lower:
            return key
    return None


def is_language_supported(ext: str) -> bool:
    """Return True if the tree-sitter grammar for ``ext`` is importable."""
    normalized = _normalize_ext(ext)
    if not normalized:
        return False
    pkg = _GRAMMAR_REGISTRY[normalized][1]
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def _get_language(ext: str) -> Optional[object]:
    """Load (and cache) the tree-sitter Language for ``ext``."""
    normalized = _normalize_ext(ext)
    if not normalized:
        return None
    lang_name = _GRAMMAR_REGISTRY[normalized][0]
    if lang_name in _language_cache:
        return _language_cache[lang_name]
    with _language_lock:
        if lang_name in _language_cache:
            return _language_cache[lang_name]
        pkg_name, attr = _GRAMMAR_REGISTRY[normalized][1], _GRAMMAR_REGISTRY[normalized][2]
        try:
            pkg = __import__(pkg_name, fromlist=[attr])
            lang_factory = getattr(pkg, attr)
            # tree-sitter 0.26+ returns a PyCapsule that must be wrapped
            raw = lang_factory()
            import tree_sitter
            if isinstance(raw, tree_sitter.Language):
                lang = raw
            else:
                lang = tree_sitter.Language(raw)
            _language_cache[lang_name] = lang
            return lang
        except (ImportError, AttributeError):
            return None


# ---------------------------------------------------------------------------
# Node type maps per language
# ---------------------------------------------------------------------------
_PY_NODE_TYPES = {
    "class_definition": CLASS,
    "function_definition": FUNCTION,
    "decorated_definition": FUNCTION,
}

_JS_NODE_TYPES = {
    "class_declaration": CLASS,
    "function_declaration": FUNCTION,
    "arrow_function": FUNCTION,
    "method_definition": METHOD,
    "lexical_declaration": VARIABLE,
    "variable_declaration": VARIABLE,
    "import_statement": IMPORT,
}

_GO_NODE_TYPES = {
    "type_spec": TYPE_ALIAS,
    "type_declaration": TYPE_ALIAS,
    "function_declaration": FUNCTION,
    "method_declaration": METHOD,
    "struct_type": STRUCT,
    "interface_type": INTERFACE,
}

_RUST_NODE_TYPES = {
    "struct_item": STRUCT,
    "enum_item": ENUM,
    "trait_item": TRAIT,
    "impl_item": IMPLEMENTS,
    "function_item": FUNCTION,
    "function_signature_item": FUNCTION,
    "let_declaration": VARIABLE,
}

_JAVA_NODE_TYPES = {
    "class_declaration": CLASS,
    "interface_declaration": INTERFACE,
    "method_declaration": METHOD,
    "function_definition": FUNCTION,
    "field_declaration": VARIABLE,
}

_C_CPP_NODE_TYPES = {
    "struct_specifier": STRUCT,
    "class_specifier": CLASS,
    "function_definition": FUNCTION,
    "declaration": VARIABLE,
    "preproc_include": IMPORT,
    "type_definition": TYPE_ALIAS,
    "enum_specifier": ENUM,
}

_TS_NODE_TYPES = {
    "class_declaration": CLASS,
    "function_declaration": FUNCTION,
    "method_definition": METHOD,
    "arrow_function": FUNCTION,
    "interface_declaration": INTERFACE,
    "type_alias_declaration": TYPE_ALIAS,
    "lexical_declaration": VARIABLE,
    "variable_declaration": VARIABLE,
    "import_statement": IMPORT,
}


def _node_types_for(language: str) -> Dict[str, str]:
    """Return the node-type map for the given language."""
    mapping = {
        "Python": _PY_NODE_TYPES,
        "JavaScript": _JS_NODE_TYPES,
        "TypeScript": _TS_NODE_TYPES,
        "Java": _JAVA_NODE_TYPES,
        "Go": _GO_NODE_TYPES,
        "Rust": _RUST_NODE_TYPES,
        "C": _C_CPP_NODE_TYPES,
        "C++": _C_CPP_NODE_TYPES,
    }
    return mapping.get(language, {})


def _extract_name(node, code_bytes: bytes) -> Optional[str]:
    """Extract the identifier name from a tree-sitter declaration node."""
    # Priority order matters: prefer name/identifier over type_identifier
    # (e.g., Go methods have both a field_identifier (name) and type_identifier (return type))
    for child in node.children:
        if child.type in ("name", "identifier", "field_identifier", "property_identifier"):
            try:
                return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            except Exception:
                pass
    # For C/C++ function_definition, the name is inside function_declarator
    if node.type == "function_definition":
        for child in node.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type in ("identifier", "field_identifier"):
                        try:
                            return code_bytes[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                        except Exception:
                            pass
    # Fallback: type_identifier is used in some languages (e.g., Go type_spec)
    for child in node.children:
        if child.type in ("type_identifier",):
            try:
                return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            except Exception:
                pass
    return None


def _extract_excerpt(node, code_bytes: bytes, max_lines: int = 6) -> str:
    """Return a short source excerpt for evidence purposes."""
    start_row = node.start_point[0]
    all_lines = code_bytes.decode("utf-8", errors="replace").splitlines()
    end_row = min(start_row + max_lines, len(all_lines))
    return "\n".join(all_lines[start_row:end_row])


def _build_signature(sym_type: str, name: str) -> str:
    """Build a minimal human-readable signature string."""
    sig_map = {
        CLASS: f"class {name}",
        INTERFACE: f"interface {name}",
        FUNCTION: f"function {name}()",
        METHOD: f"method {name}()",
        VARIABLE: f"var {name}",
        CONSTANT: f"const {name}",
        STRUCT: f"struct {name}",
        ENUM: f"enum {name}",
        TYPE_ALIAS: f"type {name}",
        TRAIT: f"trait {name}",
        IMPORT: f"import {name}",
        MODULE: f"module {name}",
    }
    return sig_map.get(sym_type, f"{sym_type} {name}")


def _extract_docstring(node, code_bytes: bytes) -> Optional[str]:
    """Look for an immediately-following comment node as docstring (best-effort)."""
    try:
        parent = node.parent
        if parent is None:
            return None
        children = parent.children
        idx = children.index(node)
        if idx + 1 < len(children):
            nxt = children[idx + 1]
            if nxt.type in ("comment", "line_comment", "block_comment", "documentation_comment"):
                text = code_bytes[nxt.start_byte:nxt.end_byte].decode("utf-8", errors="replace")
                text = text.lstrip("/#*<!- \t").rstrip("*/- \t")
                text = "\n".join(line.strip(" *#/-") for line in text.splitlines())
                return text.strip() or None
    except Exception:
        return None
    return None


def _extract_inheritance(node, code_bytes: bytes, class_name: str,
                         edges: list, file_meta: dict) -> None:
    """Extract superclass / interface relationships."""
    super_types = ["superclass", "class_heritage", "extends_clause",
                   "super_interfaces", "type_arguments", "superclass_type",
                   "argument_list"]  # Python: class Dog(Animal) uses argument_list
    for child in node.children:
        if child.type in super_types:
            for sub in child.children:
                if sub.type in ("identifier", "type_identifier", "scoped_identifier"):
                    try:
                        base = code_bytes[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                        # Skip non-identifier content like parentheses or commas
                        if base in "(),:":
                            continue
                        edges.append(NormalizedEdge(
                            repository_id=file_meta["repository_id"],
                            source_symbol=class_name,
                            source_file=file_meta["file_path"],
                            edge_type=INHERITS,
                            target_symbol=base,
                            line_start=file_meta.get("line_start"),
                            extraction_method=FROM_TREE_SITTER,
                            confidence=0.85,
                        ))
                    except Exception:
                        pass


def _walk_tree(node, code_bytes: bytes, symbols: list, edges: list,
               file_meta: dict, language: str) -> None:
    """Recursively walk the tree-sitter AST and collect symbols/edges."""
    node_types = _node_types_for(language)
    ntype = node.type

    if ntype in node_types:
        sym_type = node_types[ntype]
        name = _extract_name(node, code_bytes)
        if name:
            start_row, end_row = node.start_point[0], node.end_point[0]
            excerpt = _extract_excerpt(node, code_bytes)

            symbols.append(NormalizedSymbol(
                repository_id=file_meta["repository_id"],
                file_path=file_meta["file_path"],
                language=language,
                symbol_name=name,
                symbol_type=sym_type,
                line_start=start_row + 1,
                line_end=end_row + 1,
                signature=_build_signature(sym_type, name),
                docstring=_extract_docstring(node, code_bytes),
                extraction_method=FROM_TREE_SITTER,
                confidence=0.95 if language in ("Go", "Rust", "Java") else 0.9,
                source_excerpt=excerpt,
            ))

            edges.append(NormalizedEdge(
                repository_id=file_meta["repository_id"],
                source_symbol=file_meta["file_path"],
                source_file=file_meta["file_path"],
                edge_type=CONTAINS,
                target_symbol=name,
                target_file=file_meta["file_path"],
                line_start=start_row + 1,
                line_end=end_row + 1,
                extraction_method=FROM_TREE_SITTER,
                confidence=0.95,
            ))

            _extract_inheritance(node, code_bytes, name, edges, file_meta)

    for child in node.children:
        _walk_tree(child, code_bytes, symbols, edges, file_meta, language)


def parse_file_with_tree_sitter(repository_id: str, file_path: str,
                                content: str, ext: str) -> NormalizedFile:
    """Parse ``content`` using tree-sitter for the given extension."""
    ext = ext.lower()
    language = _GRAMMAR_REGISTRY.get(ext, ("Unknown", None, None))[0]
    lang_obj = _get_language(ext)

    meta = {
        "repository_id": repository_id,
        "file_path": file_path,
        "language": language,
    }

    if lang_obj is None:
        return NormalizedFile(
            repository_id=repository_id,
            file_path=file_path,
            language=language,
            parse_succeeded=False,
            parse_error=f"tree-sitter grammar for {language} not available",
        )

    try:
        from tree_sitter import Parser, Tree
        parser = Parser(lang_obj)
        code_bytes = content.encode("utf-8")
        tree = parser.parse(code_bytes)
    except Exception as exc:
        return NormalizedFile(
            repository_id=repository_id,
            file_path=file_path,
            language=language,
            parse_succeeded=False,
            parse_error=str(exc)[:500],
        )

    symbols: List[NormalizedSymbol] = []
    edges: List[NormalizedEdge] = []
    _walk_tree(tree.root_node, code_bytes, symbols, edges, meta, language)

    return NormalizedFile(
        repository_id=repository_id,
        file_path=file_path,
        language=language,
        parse_succeeded=True,
        symbols=symbols,
        edges=edges,
        classes=sum(1 for s in symbols if s.symbol_type == CLASS),
        functions=sum(1 for s in symbols if s.symbol_type == FUNCTION),
        methods=sum(1 for s in symbols if s.symbol_type == METHOD),
        imports=sum(1 for s in symbols if s.symbol_type == IMPORT),
    )
