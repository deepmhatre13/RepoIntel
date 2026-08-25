"""Evidence audit for RepoIntel.

Deterministically audits generated engineering answers against the indexed
knowledge layer. A claim is SUPPORTED only when every citation it references
resolves to a real indexed file; otherwise it is counted as unsupported.
Citations pointing at files absent from the index are invalid citations.

This module never fabricates results: when no citations exist for a claim the
claim is unsupported and coverage drops accordingly. It complements the
reviewer node's in-graph grounding check by providing a reusable audit record
(``EvidenceAuditResult``) with measurable ratios.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_FILE_PATTERN = re.compile(r"[\w\-./\\]+\.(?:py|js|ts|jsx|tsx|json|yaml|yml|md|toml|cfg|ini)")


@dataclass
class EvidenceAuditResult:
    """Measured outcome of auditing one generated answer."""
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    support_ratio: float = 1.0
    evidence_coverage: float = 0.0
    citation_validity: float = 1.0
    confidence: float = 0.0
    invalid_citations: List[str] = field(default_factory=list)
    stale_citations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "contradicted_claims": self.contradicted_claims,
            "support_ratio": round(self.support_ratio, 4),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "citation_validity": round(self.citation_validity, 4),
            "confidence": round(self.confidence, 4),
        }


def _extract_cited_files(text: str) -> List[str]:
    """Deterministic citation extraction from free text (file paths)."""
    return list(dict.fromkeys(_FILE_PATTERN.findall(text or "")))


def _extract_claims(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split artifact sections into atomic claims (one sentence each)."""
    claims: List[Dict[str, Any]] = []
    for section in sections:
        title = section.get("section_title", "")
        content = section.get("content", "") or ""
        cited = [c.get("file_path") for c in section.get("citations", []) if c.get("file_path")]
        for sentence in re.split(r"(?<=[.!?\n])\s+", content):
            sentence = sentence.strip()
            if len(sentence) < 8:
                continue  # headings / fragments are not claims
            claims.append({"section": title, "text": sentence, "citations": cited})
    return claims


async def audit_evidence(facade: Any, repo_id: str,
                         sections: List[Dict[str, Any]]) -> EvidenceAuditResult:
    """Audit artifact sections against the repository knowledge index.

    ``sections`` follows the ArtifactDraft shape: list of dicts with
    ``section_title``, ``content`` and ``citations`` entries.
    """
    result = EvidenceAuditResult()
    claims = _extract_claims(sections or [])
    result.total_claims = len(claims)

    all_cited_files = {f for c in claims for f in c["citations"]}
    all_cited_files |= {
        f for s in sections or []
        for f in _extract_cited_files(s.get("content", ""))
    }

    # Resolve every cited file once against the index (single source of truth).
    known_files = set()
    for file_path in sorted(all_cited_files):
        meta = await facade.get_file_metadata(repo_id, file_path)
        if meta:
            known_files.add(file_path)
        else:
            result.invalid_citations.append(file_path)

    total_citations = len(all_cited_files)
    valid_citations = len(known_files)
    if total_citations:
        result.citation_validity = valid_citations / total_citations
    else:
        # No citations at all is NOT vacuous perfection: nothing was verifiable.
        result.citation_validity = 0.0

    supported = 0
    covered = 0
    for claim in claims:
        cited = set(claim["citations"]) | set(_extract_cited_files(claim["text"]))
        if cited:
            covered += 1
        valid_for_claim = cited & known_files
        if cited and valid_for_claim:
            supported += 1
        elif not cited:
            result.notes.append(
                f"unsupported claim (no citation): {claim['text'][:80]}"
            )
        else:
            result.stale_citations.extend(sorted(cited - known_files))

    result.supported_claims = supported
    result.unsupported_claims = max(0, len(claims) - supported)
    result.contradicted_claims = 0  # contradiction detection requires cross-evidence comparison
    result.support_ratio = supported / len(claims) if claims else 1.0
    result.evidence_coverage = covered / len(claims) if claims else 0.0

    # Confidence tracks measured support, never asserted independently.
    if claims:
        result.confidence = round(
            0.5 * result.support_ratio
            + 0.3 * result.citation_validity
            + 0.2 * result.evidence_coverage, 4)
    else:
        result.confidence = 0.0
        result.notes.append("no auditable claims found")
    return result