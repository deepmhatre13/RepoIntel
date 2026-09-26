"""SCIP adapter for enriched code intelligence.

SCIP (Source Code Intelligence Protocol) provides richer semantic information
than structural parsing alone: precise definitions, references, implementations,
and symbol relationships across the entire compilation unit.

This adapter:
1. Detects whether a SCIP index (.scip file) is available for a repository
2. Parses SCIP output into the same NormalizedSymbol/NormalizedEdge model
3. Gracefully reports UNAVAILABLE when SCIP tooling is absent
4. Enriches rather than replaces the existing knowledge graph

The SCIP adapter is OPTIONAL. Tree-sitter/native analysis remains the
foundation. SCIP enriches when available; absence does not degrade functionality.

Security note: SCIP indexes are treated as untrusted data. No SCIP tooling
is executed against repository source code at ingestion time -- SCIP indexes
must be pre-built (CI integration) or built via controlled local invocation
that never executes repository code.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple
from enum import Enum

from packages.parsing.normalized_model import (
    NormalizedSymbol,
    NormalizedEdge,
    NormalizedFile,
    CLASS,
    FUNCTION,
    METHOD,
    VARIABLE,
    STRUCT,
    INTERFACE,
    ENUM as ENUM_TYPE,
    TYPE_ALIAS,
    TRAIT,
    IMPORT,
    CALLS,
    REFERENCES,
    INHERITS,
    IMPLEMENTS,
    DEFINES,
    CONTAINS,
    FROM_SCIP,
)


class SCIPAvailability(str, Enum):
    """Internal availability states tracked for SCIP (Stage 3).

    The values map to the conceptual states required by the SCIP fallback
    contract: SCIP_AVAILABLE, SCIP_UNAVAILABLE, SCIP_INDEX_NOT_FOUND,
    SCIP_INDEX_INVALID.
    """
    AVAILABLE = "SCIP_AVAILABLE"
    NOT_INSTALLED = "SCIP_UNAVAILABLE"
    NO_INDEX = "SCIP_INDEX_NOT_FOUND"
    PARSE_ERROR = "SCIP_INDEX_INVALID"


# Module-level cache for SCIP availability (checked once per process)
_scip_availability: Optional[SCIPAvailability] = None
_scip_dependencies_checked: bool = False
_has_protobuf: bool = False
_has_scip_cli: bool = False


def _check_scip_dependencies() -> None:
    """Check SCIP dependency availability (once per process)."""
    global _scip_dependencies_checked, _has_protobuf, _has_scip_cli
    if _scip_dependencies_checked:
        return
    _scip_dependencies_checked = True

    try:
        import google.protobuf  # noqa: F401
        _has_protobuf = True
    except ImportError:
        _has_protobuf = False

    # Check if scip CLI is available
    try:
        result = subprocess.run(
            ["scip", "--version"],
            capture_output=True,
            timeout=5,
        )
        _has_scip_cli = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _has_scip_cli = False


def get_scip_availability(repo_path: Optional[str] = None) -> SCIPAvailability:
    """Report SCIP availability status without requiring network access.

    The returned value reflects the *internal* availability contract:

    * SCIP_AVAILABLE         -- tooling present and a valid index was parsed
    * SCIP_UNAVAILABLE       -- SCIP tooling not installed
    * SCIP_INDEX_NOT_FOUND   -- tooling present but no ``index.scip`` found
    * SCIP_INDEX_INVALID     -- an index exists but could not be parsed

    When ``repo_path`` is omitted (e.g. a module-level status probe) we cannot
    claim AVAILABLE because no index has been produced yet.
    """
    global _scip_availability
    _check_scip_dependencies()

    if not _has_scip_cli and not _has_protobuf:
        _scip_availability = SCIPAvailability.NOT_INSTALLED
        return _scip_availability

    # No explicit index location -> we cannot verify a usable index exists.
    if not repo_path:
        return SCIPAvailability.NO_INDEX

    index_path = find_scip_index(repo_path)
    if index_path is None:
        _scip_availability = SCIPAvailability.NO_INDEX
        return _scip_availability

    symbols, edges = parse_scip_index(index_path, "probe")
    if not symbols and not edges:
        _scip_availability = SCIPAvailability.PARSE_ERROR
    else:
        _scip_availability = SCIPAvailability.AVAILABLE
    return _scip_availability


def get_scip_status(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """Structured SCIP status summary for APIs/CLI (never fabricates usage)."""
    availability = get_scip_availability(repo_path)

    # Reset the cached "no index" state so repeated calls with a repo_path can
    # legitimately transition to AVAILABLE when an index appears later.
    global _scip_availability
    if availability == SCIPAvailability.NO_INDEX:
        _scip_availability = None

    return {
        "status": availability.value,
        "available": availability == SCIPAvailability.AVAILABLE,
        "tooling_present": _has_scip_cli or _has_protobuf,
        "protobuf_available": _has_protobuf,
        "cli_available": _has_scip_cli,
        "index_found": availability in (SCIPAvailability.AVAILABLE,
                                        SCIPAvailability.PARSE_ERROR),
    }


def is_scip_available() -> bool:
    """Return True if SCIP tooling is usable (CLI or protobuf)."""
    _check_scip_dependencies()
    return _has_scip_cli or _has_protobuf


def find_scip_index(repo_path: str) -> Optional[str]:
    """Look for a pre-built SCIP index file in the repository."""
    if not repo_path or not os.path.isdir(repo_path):
        return None
    candidates = [
        os.path.join(repo_path, "index.scip"),
        os.path.join(repo_path, ".scip", "index.scip"),
        os.path.join(repo_path, "target", "scip", "index.scip"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def parse_scip_index(index_path: str, repository_id: str) -> Tuple[List[NormalizedSymbol], List[NormalizedEdge]]:
    """Parse a SCIP index file into normalized symbols and edges.

    Returns empty lists if SCIP is unavailable or the index is unparseable.
    This function never raises.
    """
    _check_scip_dependencies()
    if not _has_protobuf:
        return [], []
    try:
        return _parse_scip_protobuf(index_path, repository_id)
    except Exception:
        return [], []


def _parse_scip_protobuf(index_path: str, repository_id: str) -> Tuple[List[NormalizedSymbol], List[NormalizedEdge]]:
    """Parse SCIP protobuf index file."""
    symbols: List[NormalizedSymbol] = []
    edges: List[NormalizedEdge] = []
    try:
        from scip import scip_pb2
        index = scip_pb2.Index()
        with open(index_path, "rb") as f:
            index.ParseFromString(f.read())
        for doc in index.documents:
            file_path = doc.relative_path
            for sym in doc.symbols:
                sym_type = _scip_symbol_kind_to_type(sym.kind)
                symbols.append(NormalizedSymbol(
                    repository_id=repository_id,
                    file_path=file_path,
                    language=doc.language or "Unknown",
                    symbol_name=sym.symbol,
                    symbol_type=sym_type,
                    line_start=1,
                    line_end=1,
                    extraction_method=FROM_SCIP,
                    confidence=0.98,
                ))
            for occ in doc.occurrences:
                if occ.symbol_roles & 1:
                    start_line = occ.range[0] if len(occ.range) > 0 else 1
                    end_line = occ.range[2] if len(occ.range) > 2 else start_line
                    sym_type = _scip_symbol_kind_to_type(occ.symbol_kind)
                    symbols.append(NormalizedSymbol(
                        repository_id=repository_id,
                        file_path=file_path,
                        language=doc.language or "Unknown",
                        symbol_name=occ.symbol,
                        symbol_type=sym_type,
                        line_start=start_line + 1,
                        line_end=end_line + 1,
                        extraction_method=FROM_SCIP,
                        confidence=0.98,
                    ))
    except ImportError:
        symbols, edges = _parse_scip_text_format(index_path, repository_id)
    return symbols, edges


def _parse_scip_text_format(index_path: str, repository_id: str) -> Tuple[List[NormalizedSymbol], List[NormalizedEdge]]:
    """Fallback: parse SCIP text-format output (from `scip print`)."""
    symbols: List[NormalizedSymbol] = []
    edges: List[NormalizedEdge] = []
    try:
        result = subprocess.run(
            ["scip", "print", index_path, "--format=json"],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return [], []
        data = json.loads(result.stdout)
        for doc in data.get("documents", []):
            file_path = doc.get("relativePath", "unknown")
            for sym in doc.get("symbols", []):
                symbols.append(NormalizedSymbol(
                    repository_id=repository_id,
                    file_path=file_path,
                    language=doc.get("language", "Unknown"),
                    symbol_name=sym.get("symbol", "unknown"),
                    symbol_type=_scip_symbol_kind_to_type(sym.get("kind", "")),
                    line_start=1,
                    line_end=1,
                    extraction_method=FROM_SCIP,
                    confidence=0.95,
                ))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, OSError):
        pass
    return symbols, edges


def _scip_symbol_kind_to_type(kind) -> str:
    """Map SCIP symbol kind to our normalized symbol type."""
    kind_str = str(kind).lower() if kind else ""
    if "class" in kind_str or "struct" in kind_str:
        return CLASS
    if "interface" in kind_str or "trait" in kind_str:
        return INTERFACE
    if "method" in kind_str:
        return METHOD
    if "function" in kind_str:
        return FUNCTION
    if "variable" in kind_str or "field" in kind_str:
        return VARIABLE
    if "enum" in kind_str:
        return ENUM_TYPE
    if "type" in kind_str:
        return TYPE_ALIAS
    return FUNCTION


def reconcile_scip_symbols(
    primary: List[NormalizedSymbol],
    scip_symbols: List[NormalizedSymbol],
) -> Tuple[List[NormalizedSymbol], List[str]]:
    """Deterministically merge Tree-sitter/native symbols with SCIP symbols.

    ``primary`` holds the Tree-sitter/native result (the source of truth for
    structure); ``scip_symbols`` is the optional SCIP enrichment for the same
    repository. When both discover the same symbol, we do NOT create a second
    symbol. Instead we keep the native symbol and tag its provenance to reflect
    combined extraction (extraction_method lists both sources), preserving the
    SCIP confidence/source identifier on the existing entity.

    Returns ``(merged, enriched_keys)`` where ``enriched_keys`` are the
    ``(file_path, symbol_name)`` keys whose provenance was combined.
    """
    merged: List[NormalizedSymbol] = []
    enriched_keys: List[str] = []
    by_key: Dict[Tuple[str, str], NormalizedSymbol] = {}
    order: List[Tuple[str, str]] = []

    for sym in primary:
        key = (sym.file_path, sym.symbol_name)
        by_key[key] = sym
        order.append(key)

    for sym in scip_symbols:
        key = (sym.file_path, sym.symbol_name)
        existing = by_key.get(key)
        if existing is not None:
            # Combine provenance on the existing entity: keep native extraction,
            # but record that SCIP also observed it and note the stronger source.
            existing.extraction_method = (
                f"{existing.extraction_method}+scip"
                if "scip" not in existing.extraction_method
                else existing.extraction_method
            )
            existing.confidence = max(existing.confidence, sym.confidence)
            if key not in enriched_keys:
                enriched_keys.append(f"{key[1]}@@{key[0]}")
            continue
        # SCIP-only symbol: adopt it (SCIP couldn't have discovered this without
        # having indexed the file, so it is trustworthy enrichment, never invented).
        by_key[key] = sym
        order.append(key)

    for key in order:
        merged.append(by_key[key])
    return merged, enriched_keys


def enrich_with_scip(repository_id: str, repo_path: str,
                     existing_symbols: List[NormalizedSymbol]) -> Tuple[List[NormalizedSymbol], List[NormalizedEdge], Dict[str, Any]]:
    """Enrich existing symbols with SCIP data when available (Stage 3).

    Returns ``(merged_symbols, scip_edges, status)`` where ``status`` reports
    whether SCIP actually contributed. If SCIP is unavailable or yields no
    index, the existing symbols are returned unchanged and ``status["used"]``
    is False -- we never pretend SCIP was used when it wasn't.
    """
    no_usage = {
        "used": False,
        "status": get_scip_availability(repo_path).value,
        "added_symbols": 0,
        "enriched_symbols": 0,
    }

    availability = get_scip_availability(repo_path)
    if availability not in (SCIPAvailability.AVAILABLE, SCIPAvailability.PARSE_ERROR):
        return list(existing_symbols), [], no_usage

    index_path = find_scip_index(repo_path)
    if index_path is None:
        return list(existing_symbols), [], no_usage

    scip_symbols, scip_edges = parse_scip_index(index_path, repository_id)
    if not scip_symbols:
        return list(existing_symbols), [], no_usage

    merged, enriched_keys = reconcile_scip_symbols(list(existing_symbols), scip_symbols)
    status = {
        "used": True,
        "status": SCIPAvailability.AVAILABLE.value,
        "index_path": index_path,
        "scip_symbols": len(scip_symbols),
        "scip_edges": len(scip_edges),
        "enriched_symbols": len(enriched_keys),
        "deduplicated_symbols": len(scip_symbols) - len(enriched_keys),
        "total_symbols": len(merged),
    }
    return merged, scip_edges, status
