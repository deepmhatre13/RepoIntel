"""Source intelligence adapter for Project Health Intelligence.

Enables the standalone Project Health workflow to acquire code-level
intelligence (dependencies, security, architecture) without requiring the
user to first run the Repository Intelligence pipeline.

This module is an ADAPTER, not a duplicate implementation. It composes:

    packages.ingestion.cloning       -> safe repository snapshot
    packages.ingestion.filtering     -> admissible file filtering
    packages.parsing.pipeline        -> AST / Tree-sitter / SCIP parsing
    packages.security.semgrep_security -> static security analysis

The repository is treated as untrusted source material. No repository code
is ever executed; only static analysis is performed.

Availability semantics:
    SOURCE_ANALYSIS_AVAILABLE   -> source intelligence was computed
    SOURCE_ANALYSIS_UNAVAILABLE -> cloning/parsing failed (explicit, not 0)
    NOT_REQUESTED               -> source analysis was not triggered
"""
from __future__ import annotations

import os
import logging
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class DependencyInfo:
    """Structured dependency intelligence from repository manifests."""
    total_count: int = 0
    direct_count: int = 0
    manifest_files: List[str] = field(default_factory=list)
    ecosystems: Dict[str, int] = field(default_factory=dict)
    availability: str = "NOT_AVAILABLE"
    error: Optional[str] = None


@dataclass
class SecurityInfo:
    """Structured security intelligence from static analysis."""
    status: str = "NOT_AVAILABLE"  # available | not_available | no_findings | error
    findings_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    findings_by_rule: Dict[str, int] = field(default_factory=dict)
    scanner_available: bool = True
    error: Optional[str] = None


@dataclass
class CodeHealthInfo:
    """Structured code health intelligence from parsing."""
    files_parsed: int = 0
    files_seen: int = 0
    symbol_count: int = 0
    edge_count: int = 0
    chunk_count: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    env_vars_detected: List[str] = field(default_factory=list)
    parse_failures: int = 0
    availability: str = "NOT_AVAILABLE"
    error: Optional[str] = None


@dataclass
class SourceIntelligence:
    """Combined source intelligence from the repository snapshot."""
    availability: str = "NOT_REQUESTED"
    repository_path: Optional[str] = None
    is_temp_clone: bool = False
    dependencies: DependencyInfo = field(default_factory=DependencyInfo)
    security: SecurityInfo = field(default_factory=SecurityInfo)
    code_health: CodeHealthInfo = field(default_factory=CodeHealthInfo)


def acquire_source_intelligence(
    owner: str,
    repo: str,
    *,
    max_repo_size_mb: float = 500.0,
    clone_timeout_seconds: int = 180,
    run_security_scan: bool = True,
) -> SourceIntelligence:
    """Acquire source intelligence for a public GitHub repository.

    Safely clones the repository, filters admissible files, runs the parsing
    pipeline, and optionally performs static security analysis.

    Returns a SourceIntelligence with explicit availability flags. Never raises
    for analysis failures — the availability field captures the outcome.

    Args:
        owner: GitHub owner/organization
        repo: Repository name
        max_repo_size_mb: Maximum allowed repository size in MB
        clone_timeout_seconds: Timeout for git clone operation
        run_security_scan: Whether to run Semgrep security analysis

    Returns:
        SourceIntelligence with availability flags set appropriately
    """
    result = SourceIntelligence(availability="NOT_AVAILABLE")
    local_path = None
    is_temp = False

    try:
        # Step 1: Clone the repository safely
        from packages.ingestion.cloning import clone_repository

        repo_url = f"https://github.com/{owner}/{repo}"
        local_path = clone_repository(
            repo_url,
            max_repo_size_mb=max_repo_size_mb,
            clone_timeout_seconds=clone_timeout_seconds,
        )
        is_temp = True
        result.repository_path = local_path
        result.is_temp_clone = is_temp

        # Step 2: Filter admissible files
        from packages.ingestion.filtering import filter_repository
        admitted_files = filter_repository(local_path)

        if not admitted_files:
            result.availability = "SOURCE_ANALYSIS_UNAVAILABLE"
            result.code_health.error = "No admissible files found in repository"
            return result

        # Step 3: Run parsing pipeline
        repo_id = f"{owner}/{repo}"
        from packages.parsing.pipeline import run_parsing_pipeline
        parse_result = run_parsing_pipeline(repo_id, local_path, admitted_files)

        # Step 4: Extract code health info
        result.code_health = _extract_code_health(parse_result)

        # Step 5: Extract dependency info
        result.dependencies = _extract_dependency_info(parse_result)

        # Step 6: Run security analysis if requested
        if run_security_scan:
            result.security = _run_security_analysis(local_path, admitted_files)
        else:
            result.security.status = "NOT_REQUESTED"

        result.availability = "SOURCE_ANALYSIS_AVAILABLE"

    except Exception as exc:
        logger.warning("Source intelligence acquisition failed for %s/%s: %s", owner, repo, exc)
        result.availability = "SOURCE_ANALYSIS_UNAVAILABLE"
        result.code_health.error = str(exc)[:500]

    finally:
        # Clean up temporary clone
        if is_temp and local_path and os.path.isdir(local_path):
            shutil.rmtree(local_path, ignore_errors=True)

    return result


def _extract_code_health(parse_result) -> CodeHealthInfo:
    """Extract code health metrics from a parsing pipeline result."""
    info = CodeHealthInfo(availability="NOT_AVAILABLE")
    try:
        # Handle both object and dict-like parse results
        if hasattr(parse_result, 'files_parsed'):
            info.files_parsed = parse_result.files_parsed
            info.files_seen = getattr(parse_result, 'files_seen', info.files_parsed)
            info.symbol_count = len(getattr(parse_result, 'symbols', []))
            info.edge_count = len(getattr(parse_result, 'edges', []))
            info.chunk_count = len(getattr(parse_result, 'chunks', []))
            info.parse_failures = len(getattr(parse_result, 'failures', []))
        elif isinstance(parse_result, dict):
            info.files_parsed = parse_result.get('files_parsed', 0)
            info.files_seen = parse_result.get('files_seen', info.files_parsed)
            info.symbol_count = len(parse_result.get('symbols', []))
            info.edge_count = len(parse_result.get('edges', []))
            info.chunk_count = len(parse_result.get('chunks', []))
            info.parse_failures = len(parse_result.get('failures', []))

        info.availability = "AVAILABLE"
    except Exception as exc:
        info.error = str(exc)[:500]
    return info


def _extract_dependency_info(parse_result) -> DependencyInfo:
    """Extract dependency information from parsing result."""
    info = DependencyInfo(availability="NOT_AVAILABLE")
    try:
        # Extract from manifest_data or dependencies attribute
        manifest = getattr(parse_result, 'manifest_data', None)
        if manifest is None and isinstance(parse_result, dict):
            manifest = parse_result.get('manifest_data')

        if manifest:
            deps = manifest.get('dependencies', {}) if isinstance(manifest, dict) else {}
            info.direct_count = len(deps)
            info.total_count = info.direct_count
            info.manifest_files = manifest.get('manifest_files', []) if isinstance(manifest, dict) else []
            info.ecosystems = manifest.get('ecosystems', {}) if isinstance(manifest, dict) else {}

        # Also try dedicated dependencies attribute
        deps_attr = getattr(parse_result, 'dependencies', None)
        if deps_attr and isinstance(deps_attr, list):
            info.total_count = max(info.total_count, len(deps_attr))

        info.availability = "AVAILABLE" if info.total_count > 0 else "NO_DEPENDENCIES"
    except Exception as exc:
        info.error = str(exc)[:500]
    return info


def _run_security_analysis(local_path: str, admitted_files: List[str]) -> SecurityInfo:
    """Run static security analysis using the existing Semgrep integration."""
    info = SecurityInfo()
    try:
        from packages.security.semgrep_security import SecurityScanner, SecurityAnalysisStatus

        scanner = SecurityScanner()
        info.scanner_available = scanner.is_available()

        if not scanner.is_available():
            info.status = SecurityAnalysisStatus.NOT_AVAILABLE.value
            info.error = "semgrep CLI not available on PATH"
            return info

        result = scanner.scan_files(local_path, admitted_files)
        info.status = result.get("status", SecurityAnalysisStatus.ERROR.value)
        findings = result.get("findings", [])
        info.findings_count = len(findings)

        for f in findings:
            sev = f.severity if hasattr(f, 'severity') else f.get('severity', 'UNKNOWN')
            rule = f.rule_id if hasattr(f, 'rule_id') else f.get('rule_id', 'unknown')
            if sev in ('ERROR', 'CRITICAL'):
                info.error_count += 1
            elif sev == 'WARNING':
                info.warning_count += 1
            else:
                info.info_count += 1
            info.findings_by_rule[rule] = info.findings_by_rule.get(rule, 0) + 1

    except Exception as exc:
        info.status = "error"
        info.error = str(exc)[:500]
    return info