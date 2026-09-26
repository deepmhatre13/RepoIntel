"""Normalized code intelligence model for RepoIntel.

Provides a language-agnostic representation of extracted code entities
(symbols) and their relationships (edges). All parsers -- Python AST,
Tree-sitter structural parsers, SCIP adapter -- emit instances of these
dataclasses so the rest of the pipeline operates on a single, technology-
independent model.

Every entity retains:
    repository_id / file_path / language / symbol_name / symbol_type
    line_start / line_end / signature (where available) / evidence reference

Edges carry provenance: the extraction method, source/target files,
and a confidence indicator.
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ---------------------------------------------------------------------------
# Normalized symbol types (language-agnostic)
# ---------------------------------------------------------------------------
MODULE = "module"
FILE = "file"
CLASS = "class"
INTERFACE = "interface"
FUNCTION = "function"
METHOD = "method"
VARIABLE = "variable"
CONSTANT = "constant"
STRUCT = "struct"
ENUM = "enum"
TYPE_ALIAS = "type_alias"
DECORATOR = "decorator"
ANNOTATION = "annotation"
TRAIT = "trait"
IMPORT = "import"
NAMESPACE = "namespace"
PACKAGE = "package"
UNKNOWN_SYMBOL = "unknown"


# ---------------------------------------------------------------------------
# Normalized edge types (relationship provenance)
# ---------------------------------------------------------------------------
CALLS = "calls"
REFERENCES = "references"
INHERITS = "inherits"
IMPLEMENTS = "implements"
IMPORTS = "imports"
DEFINES = "defines"
CONTAINS = "contains"
EXPOSES = "exposes"
DEPENDS_ON = "depends_on"
TESTS = "tests"
CONFIGURES = "configures"
RELATED_TO = "related_to"
UNKNOWN_EDGE = "unknown_edge"


# ---------------------------------------------------------------------------
# Extraction provenance
# ---------------------------------------------------------------------------
FROM_PYTHON_AST = "python_ast"
FROM_TREE_SITTER = "tree_sitter"
FROM_SCIP = "scip"
FROM_GIT = "git_history"
FROM_STATIC_ANALYSIS = "static_analysis"
FROM_HEURISTIC = "heuristic"


@dataclass
class NormalizedSymbol:
    """A technology-independent code entity."""

    repository_id: str
    file_path: str
    language: str
    symbol_name: str
    symbol_type: str  # one of the *_SYMBOL constants above
    line_start: int
    line_end: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    parent_module: Optional[str] = None
    decorators: Optional[List[str]] = None
    extraction_method: str = FROM_TREE_SITTER
    confidence: float = 1.0
    # Source excerpt used as evidence (first non-empty lines of the symbol)
    source_excerpt: Optional[str] = None


@dataclass
class NormalizedEdge:
    """A provenance-carrying relationship between two code entities."""

    repository_id: str
    source_symbol: str
    source_file: str
    edge_type: str  # one of the *_EDGE constants above
    target_symbol: str
    target_file: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    extraction_method: str = FROM_TREE_SITTER
    confidence: float = 1.0
    evidence_reference: Optional[str] = None  # chunk-id / content-hash


@dataclass
class NormalizedFile:
    """Language-agnostic representation of a parsed source file."""

    repository_id: str
    file_path: str
    language: str
    parse_succeeded: bool = True
    parse_error: Optional[str] = None
    symbols: List[NormalizedSymbol] = field(default_factory=list)
    edges: List[NormalizedEdge] = field(default_factory=list)
    classes: int = 0
    functions: int = 0
    methods: int = 0
    imports: int = 0

    def __post_init__(self):
        """Auto-compute symbol counts from the symbols list."""
        if self.symbols:
            self.classes = sum(1 for s in self.symbols if s.symbol_type == CLASS)
            self.functions = sum(1 for s in self.symbols if s.symbol_type == FUNCTION)
            self.methods = sum(1 for s in self.symbols if s.symbol_type == METHOD)
            self.imports = sum(1 for s in self.symbols if s.symbol_type == IMPORT)
