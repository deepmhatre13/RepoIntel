"""Semgrep security intelligence for RepoIntel (Stage 10).

Integrates Semgrep findings as structured repository evidence. All findings
are deterministic static-analysis results -- never fabricated. If the
``semgrep`` CLI is unavailable, the analyzer reports NOT_AVAILABLE and
returns no findings; it never claims the repository is secure.

Security boundary:
  - Semgrep runs ONLY against filtered, admitted repository files.
  - Repository code is never executed; Semgrep performs static matching only.
  - Controlled execution: shell invocation with timeouts against admitted paths.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from packages.shared.secret_redaction import redact_secrets


class SecurityAnalysisStatus(str, Enum):
    AVAILABLE = "available"
    NOT_AVAILABLE = "not_available"
    NO_FINDINGS = "no_findings"
    ERROR = "error"


@dataclass
class SecurityFinding:
    """A single structured security finding."""

    rule_id: str
    severity: str  # ERROR / WARNING / INFO / UNKNOWN
    message: str
    file_path: str
    line_start: int
    line_end: int
    finding_type: str = "STATIC"
    evidence: Optional[str] = None
    confidence: str = "MEDIUM"
    source_endpoint: str = "semgrep"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "finding_type": self.finding_type,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source_endpoint": self.source_endpoint,
        }
class SecurityScanner:
    """Safe Semgrep wrapper producing structured findings."""

    def __init__(self, semgrep_binary: str = "semgrep", timeout_seconds: int = 60):
        self.semgrep_binary = semgrep_binary
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def is_available() -> bool:
        """Check if the semgrep CLI is present on PATH."""
        try:
            result = subprocess.run(
                ["semgrep", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def scan_files(
        self,
        repo_path: str,
        files: List[str],
        config: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Semgrep over admitted files. Returns {status, findings, raw_output, error}."""
        if not files:
            return {"status": SecurityAnalysisStatus.NO_FINDINGS.value,
                    "findings": [], "raw_output": None, "error": None}

        # Build absolute, admitted paths (reject path traversal)
        abs_paths = []
        for rel in files:
            path = str(rel) if rel else ""
            abs_path = os.path.abspath(os.path.join(repo_path, path))
            if not abs_path.startswith(os.path.abspath(repo_path)):
                continue
            if not os.path.isfile(abs_path):
                continue
            abs_paths.append(abs_path)

        if not abs_paths:
            return {"status": SecurityAnalysisStatus.NO_FINDINGS.value,
                    "findings": [], "raw_output": None, "error": None}

        cmd = [self.semgrep_binary, "scan", "--json", "--no-git-ignore"]
        if config:
            cmd += ["--config", config]
        cmd += abs_paths

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=self.timeout_seconds)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            return {"status": SecurityAnalysisStatus.ERROR.value,
                    "findings": [], "raw_output": None, "error": str(exc)}

        if result.returncode not in (0, 1):
            return {"status": SecurityAnalysisStatus.ERROR.value,
                    "findings": [], "raw_output": None,
                    "error": result.stderr[-2000:]}

        try:
            parsed = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            return {"status": SecurityAnalysisStatus.ERROR.value,
                    "findings": [], "raw_output": None,
                    "error": f"Could not parse Semgrep output: {exc}"}

        findings = self._parse_semgrep_json(parsed)
        for f in findings:
            if f.evidence:
                f.evidence = redact_secrets(f.evidence).text
            if f.message:
                f.message = redact_secrets(f.message).text

        return {"status": SecurityAnalysisStatus.AVAILABLE.value,
                "findings": findings, "raw_output": parsed, "error": None}

    @staticmethod
    def _parse_semgrep_json(data: dict) -> List[SecurityFinding]:
        """Convert raw Semgrep JSON into structured SecurityFinding objects."""
        findings: List[SecurityFinding] = []
        results = data.get("results", [])
        for res in results:
            rule_id = res.get("check_id", "unknown")
            severity = res.get("extra", {}).get("severity", "UNKNOWN")
            message = res.get("extra", {}).get("message", rule_id)
            path = res.get("path", "")
            start_line = res.get("start", {}).get("line", 0)
            end_line = res.get("end", {}).get("line", start_line) or start_line
            evidence = res.get("extra", {}).get("lines", "") or ""
            findings.append(SecurityFinding(
                rule_id=rule_id,
                severity=severity.upper(),
                message=message[:2000],
                file_path=path,
                line_start=start_line,
                line_end=end_line,
                evidence=evidence,
                confidence="HIGH" if severity in ("ERROR", "WARNING") else "MEDIUM",
                source_endpoint="semgrep",
            ))
        return findings

    def scan_for_secrets(self, files: List[tuple]) -> List[SecurityFinding]:
        """Basic deterministic secret scan over admitted file contents."""
        findings: List[SecurityFinding] = []
        from packages.shared.secret_redaction import scan_for_secrets

        for item in files:
            if not item or len(item) < 2:
                continue
            path, content = item[0], item[1]
            if not path or not content:
                continue
            found = scan_for_secrets(content)
            for cat, count in (found or {}).items():
                if count > 0:
                    findings.append(SecurityFinding(
                        rule_id=f"secret:{cat.lower()}",
                        severity="ERROR",
                        message=f"Possible {cat} detected",
                        file_path=path,
                        line_start=0,
                        line_end=0,
                        finding_type="SECRET",
                        evidence="[REDACTED]",
                        confidence="HIGH",
                        source_endpoint="secret_scan",
                    ))
        return findings
# ---------------------------------------------------------------------------
# Integration: persistence + orchestration helpers
# ---------------------------------------------------------------------------

async def persist_security_findings(
    session,
    repo_id: str,
    findings: List[SecurityFinding],
) -> Dict[str, int]:
    """Persist structured security findings for a repository (re-scan semantics)."""
    from packages.knowledge.models import SecurityFindingModel as _SFM
    from sqlalchemy import delete as _sa_delete

    await session.execute(_sa_delete(_SFM).where(_SFM.repo_id == repo_id))
    for f in findings:
        session.add(_SFM(
            repo_id=repo_id,
            rule_id=f.rule_id,
            severity=f.severity,
            message=f.message,
            file_path=f.file_path,
            line_start=f.line_start,
            line_end=f.line_end,
            finding_type=f.finding_type,
            evidence=f.evidence,
            confidence=f.confidence,
            status="AVAILABLE",
            source_endpoint=f.source_endpoint,
        ))
    await session.commit()
    return {"persisted": len(findings)}


async def analyze_repository_security(
    repo_path: str,
    repo_id: str,
    files: List[str],
    config: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full security analysis for a repository.

    Returns a dict with status, findings (as dicts), and an error field.
    Never raises; never fabricates results.
    """
    scanner = SecurityScanner()

    if not scanner.is_available():
        return {
            "status": SecurityAnalysisStatus.NOT_AVAILABLE.value,
            "findings": [],
            "findings_count": 0,
            "error": "semgrep CLI not available on PATH",
        }

    result = scanner.scan_files(repo_path, files, config=config)
    findings = result["findings"]
    return {
        "status": result["status"],
        "findings": [f.to_dict() for f in findings],
        "findings_count": len(findings),
        "error": result.get("error"),
        "raw_output": result.get("raw_output"),
    }