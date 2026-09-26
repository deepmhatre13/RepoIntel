"""Targeted tests for Phase G (ConfidenceBreakdown), Phase I (Cross-Product),
Phase J (Temporal Drift), Phase L (Safe Path Resolution), and Phase M (API Models).
"""

import pytest
from packages.knowledge.relevance import (
    RelevanceDecision, RelevanceLevel, confidence_breakdown, confidence_from_relevance
)
from apps.api import _resolve_repo_root, TemporalDriftResponse, SymbolsResponse


def test_confidence_breakdown_high():
    decision = RelevanceDecision(
        level=RelevanceLevel.HIGH_RELEVANCE,
        reasoning="Direct symbol match",
        direct_symbols=["Calculator.add"],
    )
    bd = confidence_breakdown(decision, [], evidence_count=1, llm_available=False)
    assert bd.label == "HIGH"
    assert bd.retrieval_confidence == 1.0
    assert bd.direct_match is True
    assert "Direct symbol match" in bd.reasoning
    d = bd.to_dict()
    assert d["label"] == "HIGH"
    assert d["retrieval_confidence"] == 1.0


def test_confidence_breakdown_medium():
    decision = RelevanceDecision(
        level=RelevanceLevel.MEDIUM_RELEVANCE,
        reasoning="Medium relevance",
        evidence_terms={"calculator", "add"},
        query_terms=["calculator", "add"],
    )
    bd = confidence_breakdown(decision, [], evidence_count=2, llm_available=True)
    assert bd.label == "MEDIUM"
    assert bd.retrieval_confidence == 0.6
    assert bd.evidence_coverage == 1.0


def test_confidence_breakdown_low():
    decision = RelevanceDecision(
        level=RelevanceLevel.NO_RELEVANT_EVIDENCE,
        reasoning="No evidence found",
    )
    bd = confidence_breakdown(decision, [], evidence_count=0, llm_available=True)
    assert bd.label == "LOW"
    assert bd.retrieval_confidence == 0.0


@pytest.mark.asyncio
async def test_safe_resolve_repo_root_returns_none_for_missing():
    class DummySession:
        async def execute(self, stmt):
            class DummyResult:
                def scalar_one_or_none(self):
                    return None
            return DummyResult()

    res = await _resolve_repo_root(DummySession(), "nonexistent-id")
    assert res is None


def test_api_response_models():
    resp = TemporalDriftResponse(repository_id="repo1", drift={"availability": "INSUFFICIENT_HISTORY"})
    assert resp.repository_id == "repo1"
    assert resp.drift["availability"] == "INSUFFICIENT_HISTORY"

    sym_resp = SymbolsResponse(repository_id="repo1", symbols=[{"name": "foo"}])
    assert sym_resp.symbols[0]["name"] == "foo"


# ---------------------------------------------------------------------------
# Phase I -- Cross-Product Intelligence
# ---------------------------------------------------------------------------

def test_source_intelligence_dataclasses():
    """Source intelligence dataclasses have correct availability defaults."""
    from packages.project_health.source_intelligence import (
        SourceIntelligence, DependencyInfo, SecurityInfo, CodeHealthInfo,
    )
    si = SourceIntelligence()
    assert si.availability == "NOT_REQUESTED"
    assert si.dependencies.availability == "NOT_AVAILABLE"
    assert si.security.status == "NOT_AVAILABLE"
    assert si.code_health.availability == "NOT_AVAILABLE"


def test_dependency_features_with_source():
    """dependency_features returns AVAILABLE when source intelligence present."""
    from packages.project_health.features import ProjectHealthFeatures
    from packages.project_health.source_intelligence import (
        SourceIntelligence, DependencyInfo,
    )

    class FakeData:
        status = {}
        commits = []
        releases = []
        pull_requests = []
        issues = []
        contributors = []

    dep_info = DependencyInfo(
        total_count=10, direct_count=5,
        manifest_files=["requirements.txt"],
        ecosystems={"pip": 10},
        availability="AVAILABLE",
    )
    si = SourceIntelligence(availability="AVAILABLE", dependencies=dep_info)
    feats = ProjectHealthFeatures(FakeData(), source_intelligence=si)
    result = feats.dependency_features()
    assert result["availability"] == "AVAILABLE"
    assert result["total_dependencies"] == 10
    assert result["direct_dependencies"] == 5


def test_dependency_features_without_source():
    """dependency_features returns NOT_AVAILABLE when no source intelligence."""
    from packages.project_health.features import ProjectHealthFeatures

    class FakeData:
        status = {}
        commits = []
        releases = []
        pull_requests = []
        issues = []
        contributors = []

    feats = ProjectHealthFeatures(FakeData())
    result = feats.dependency_features()
    assert result["availability"] == "NOT_AVAILABLE"


def test_security_features_with_findings():
    """security_features returns AVAILABLE with findings when source present."""
    from packages.project_health.features import ProjectHealthFeatures
    from packages.project_health.source_intelligence import (
        SourceIntelligence, SecurityInfo,
    )

    class FakeData:
        status = {}
        commits = []
        releases = []
        pull_requests = []
        issues = []
        contributors = []

    sec_info = SecurityInfo(
        status="available",
        findings_count=3, error_count=1, warning_count=2,
        findings_by_rule={"py/security/check": 3},
    )
    si = SourceIntelligence(availability="AVAILABLE", security=sec_info)
    feats = ProjectHealthFeatures(FakeData(), source_intelligence=si)
    result = feats.security_features()
    assert result["availability"] == "AVAILABLE"
    assert result["findings_count"] == 3
    assert result["error_count"] == 1


def test_security_features_scanner_unavailable():
    """security_features reports NOT_AVAILABLE when scanner is missing."""
    from packages.project_health.features import ProjectHealthFeatures
    from packages.project_health.source_intelligence import (
        SourceIntelligence, SecurityInfo,
    )

    class FakeData:
        status = {}
        commits = []
        releases = []
        pull_requests = []
        issues = []
        contributors = []

    sec_info = SecurityInfo(status="not_available", scanner_available=False)
    si = SourceIntelligence(availability="AVAILABLE", security=sec_info)
    feats = ProjectHealthFeatures(FakeData(), source_intelligence=si)
    result = feats.security_features()
    assert result["availability"] == "NOT_AVAILABLE"
    assert result["scanner_available"] is False


def test_code_health_features_with_source():
    """code_health_features returns AVAILABLE when source analysis present."""
    from packages.project_health.features import ProjectHealthFeatures
    from packages.project_health.source_intelligence import (
        SourceIntelligence, CodeHealthInfo,
    )

    class FakeData:
        status = {}
        commits = []
        releases = []
        pull_requests = []
        issues = []
        contributors = []

    ch_info = CodeHealthInfo(
        files_parsed=50, files_seen=55, symbol_count=200,
        edge_count=300, chunk_count=150,
        languages={"python": 50},
        availability="AVAILABLE",
    )
    si = SourceIntelligence(availability="AVAILABLE", code_health=ch_info)
    feats = ProjectHealthFeatures(FakeData(), source_intelligence=si)
    result = feats.code_health_features()
    assert result["availability"] == "AVAILABLE"
    assert result["files_parsed"] == 50
    assert result["symbol_count"] == 200


def test_all_features_includes_source_dimensions():
    """all_features includes dependency/security/code_health from source."""
    from packages.project_health.features import ProjectHealthFeatures
    from packages.project_health.source_intelligence import (
        SourceIntelligence, DependencyInfo, SecurityInfo, CodeHealthInfo,
    )

    class FakeData:
        status = {}
        commits = []
        releases = []
        pull_requests = []
        issues = []
        contributors = []

    si = SourceIntelligence(
        availability="AVAILABLE",
        dependencies=DependencyInfo(total_count=5, availability="AVAILABLE"),
        security=SecurityInfo(status="no_findings"),
        code_health=CodeHealthInfo(files_parsed=10, availability="AVAILABLE"),
    )
    feats = ProjectHealthFeatures(FakeData(), source_intelligence=si)
    all_f = feats.all_features()
    assert "dependencies" in all_f
    assert "security" in all_f
    assert "code_health" in all_f
    assert all_f["dependencies"]["availability"] == "AVAILABLE"
    assert all_f["security"]["availability"] == "AVAILABLE"
    assert all_f["code_health"]["availability"] == "AVAILABLE"
