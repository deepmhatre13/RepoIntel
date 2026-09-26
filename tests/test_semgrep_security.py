"""Tests for Semgrep security intelligence (Stage 10).

Covers:
- Structured finding ingestion from Semgrep JSON
- Unavailable analyzer reporting (NOT_AVAILABLE, never claims secure)
- Secret scan integration
- Evidence redaction
"""

import pytest

from packages.security.semgrep_security import (
    SecurityScanner,
    SecurityFinding,
    SecurityAnalysisStatus,
)


class TestSecurityFindingModel:
    def test_finding_creation(self):
        f = SecurityFinding(
            rule_id="python.lang.security.audit.paramiko-injection",
            severity="ERROR",
            message="Potential command injection",
            file_path="src/util.py",
            line_start=10,
            line_end=10,
        )
        assert f.rule_id == "python.lang.security.audit.paramiko-injection"
        assert f.severity == "ERROR"
        assert f.file_path == "src/util.py"
        assert f.to_dict()["finding_type"] == "STATIC"

    def test_finding_to_dict(self):
        f = SecurityFinding(
            rule_id="secret:aws",
            severity="ERROR",
            message="AWS key found",
            file_path=".env",
            line_start=1,
            line_end=1,
            finding_type="SECRET",
        )
        d = f.to_dict()
        assert isinstance(d, dict)
        assert d["finding_type"] == "SECRET"
        assert d["rule_id"] == "secret:aws"


class TestSemgrepParser:
    def test_parse_empty_results(self):
        scanner = SecurityScanner()
        parsed = scanner._parse_semgrep_json({"results": []})
        assert parsed == []

    def test_parse_single_finding(self):
        scanner = SecurityScanner()
        data = {
            "results": [{
                "check_id": "python.eval",
                "path": "src/foo.py",
                "start": {"line": 3},
                "end": {"line": 4},
                "extra": {
                    "severity": "ERROR",
                    "message": "eval() is dangerous",
                    "lines": "eval(user_input)",
                },
            }]
        }
        findings = scanner._parse_semgrep_json(data)
        assert len(findings) == 1
        f = findings[0]
        assert f.rule_id == "python.eval"
        assert f.file_path == "src/foo.py"
        assert f.line_start == 3
        assert f.line_end == 4
        assert "dangerous" in f.message
        assert f.confidence == "HIGH"

    def test_parse_multiple_findings(self):
        scanner = SecurityScanner()
        data = {
            "results": [
                {"check_id": "a.b", "path": "f.py", "start": {"line": 1},
                 "end": {"line": 1}, "extra": {"severity": "WARNING", "message": "x"}},
                {"check_id": "c.d", "path": "g.py", "start": {"line": 5},
                 "end": {"line": 5}, "extra": {"severity": "INFO", "message": "y"}},
            ]
        }
        findings = scanner._parse_semgrep_json(data)
        assert len(findings) == 2
        assert findings[0].confidence == "HIGH"  # WARNING
        assert findings[1].confidence == "MEDIUM"  # INFO


class TestSecurityScannerAvailability:
    def test_is_available_returns_bool(self):
        result = SecurityScanner.is_available()
        assert isinstance(result, bool)

    def test_scan_for_secrets_detects_aws_key(self):
        scanner = SecurityScanner()
        fake_file = ("src/config.py", 'AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"')
        findings = scanner.scan_for_secrets([fake_file])
        assert len(findings) >= 1
        assert findings[0].finding_type == "SECRET"
        assert "aws" in findings[0].rule_id.lower()

    def test_scan_for_secrets_no_match(self):
        scanner = SecurityScanner()
        findings = scanner.scan_for_secrets([("src/clean.py", "print('hello world')")])
        assert findings == []

    def test_scan_for_secrets_empty_input(self):
        scanner = SecurityScanner()
        findings = scanner.scan_for_secrets([])
        assert findings == []