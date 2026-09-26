"""Evidence audit tests: claims are measured against the real index."""

import pytest

from evals.evidence_audit import EvidenceAuditResult, audit_evidence


class _FakeFacade:
    """Minimal facade double exposing only get_file_metadata."""

    def __init__(self, known_files):
        self.known_files = set(known_files)

    async def get_file_metadata(self, repo_id, file_path):
        return {"file_path": file_path} if file_path in self.known_files else None


@pytest.mark.asyncio
async def test_supported_and_unsupported_claims_measured():
    facade = _FakeFacade({"src/calculator.py"})
    sections = [
        {
            "section_title": "Overview",
            "content": (
                "The Calculator class lives in src/calculator.py. "   # cited + valid
                "It exposes add and multiply operations. "           # no citation
            ),
            "citations": [{"file_path": "src/calculator.py"}],
        },
        {
            "section_title": "Install",
            "content": "Installation uses pyproject.toml metadata.",
            "citations": [{"file_path": "missing/file.py"}],          # invalid citation
        },
    ]



    res = await audit_evidence(facade, "repo1", sections)

    assert isinstance(res, EvidenceAuditResult)
    assert res.total_claims == 3
    # Section citations cover both sentences of section 1; section 2 cites
    # only non-existent files, so its single claim stays unsupported.
    assert res.supported_claims == 2
    assert res.unsupported_claims == 1
    assert res.evidence_coverage == 1.0
    # Only src/calculator.py resolves against the index.
    assert "missing/file.py" in res.invalid_citations
    assert res.citation_validity == pytest.approx(1 / 3)
    assert 0.0 <= res.support_ratio <= 1.0
    # Confidence is derived from measured support, not asserted.
    assert res.confidence == round(
        0.5 * res.support_ratio + 0.3 * res.citation_validity
        + 0.2 * res.evidence_coverage, 4)


@pytest.mark.asyncio
async def test_no_citations_yields_zero_coverage_not_fake_success():
    facade = _FakeFacade(set())
    res = await audit_evidence(facade, "repo1", [
        {"section_title": "S", "content": "An unevidenced claim about behaviour.", "citations": []},
    ])
    assert res.total_claims == 1
    assert res.support_ratio == 0.0
    assert res.evidence_coverage == 0.0
    assert res.confidence == 0.0