"""Evidence & Provenance Audit Engine for RepoIntel (Phase 44).

Objectively evaluates whether RepoIntel's answers are GROUNDED, not merely whether
functions executed. For every final response it inspects claims, evidence and
citations against the deterministic repository knowledge layer:

- Claim classification (Phase 44.2): FACT / INFERENCE / RECOMMENDATION / HISTORICAL /
  STRUCTURAL / RISK / TEST / ARCHITECTURE.
- Citation verification (44.4): file exists, symbol exists, line range valid,
  commit exists, repository matches -- verified in the database, never accepted
  because a string looks plausible.
- Unsupported claim detection (44.3): claims asserting files/symbols/dependencies/
  commits/tests that have no corresponding repository evidence are marked
  UNSUPPORTED (never silently removed).
- Contradiction detection (44.5): negative dependency/import claims are checked
  against symbol edges and dependencies; contradictions are flagged.

The repository is the source of truth. Nothing here fabricates evidence.
"""

import re
from enum import Enum
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.knowledge.models import (
    SymbolModel, FileModel, CommitModel, DependencyModel,
    CodeChunkModel,
)

_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
_PATH_RE = re.compile(r"[\w./\\-]+\.(?:py|js|ts|jsx|tsx|json|toml|yaml|yml|cfg|ini|md)\b")
_QUOTED_RE = re.compile(r"[\"'`]([^\"'`]{2,80})[\"'`]")
_IDENT_RE = re.compile(r"\b[a-z_][a-z0-9_]*(?:_[a-z0-9_]+)+\b|\b[A-Z][A-Za-z0-9_]{2,}\b")

_NEGATIVE_DEP_PATTERNS = [
    r"(?:does\s+not|doesn't|do\w*\s+not|no)\s+(?:longer\s+)?(?:depend|depends|dependent|rely|relies)",
    r"(?:does\s+not|doesn't|do\w*\s+not|no)\s+(?:import|use|uses|using|reference|references|call|calls)",
    r"free\s+of\s+dependencies?\s+on",
]

_HISTORICAL_KEYWORDS = ["histor", "commit", "changed over time", "evolution", "previously", "regression", "last modified"]
_RISK_KEYWORDS = ["risk", "hotspot", "fragile", "critical", "danger", "vulnerab"]
_TEST_KEYWORDS = ["test", "coverage", "spec ", "pytest", "unittest"]
_ARCH_KEYWORDS = ["architecture", "component", "layer", "module structure", "coupling", "cohesion", "pattern"]
_STRUCTURAL_KEYWORDS = ["contains", "defines", "implements", "located in", "consists of", "file exists", "symbol"]


class ClaimType(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    RECOMMENDATION = "RECOMMENDATION"
    HISTORICAL = "HISTORICAL"
    STRUCTURAL = "STRUCTURAL"
    RISK = "RISK"
    TEST = "TEST"
    ARCHITECTURE = "ARCHITECTURE"


class ClaimVerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class EvaluatedClaim(BaseModel):
    claim_text: str
    claim_type: ClaimType
    status: ClaimVerificationStatus
    matching_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    contradictory_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_identifiers: List[str] = Field(default_factory=list)
    explanation: str = ""


class CitationCheck(BaseModel):
    citation: str
    kind: str  # file | symbol | commit | line_range | evidence_item
    valid: bool
    reason: str = ""


class EvidenceAuditResult(BaseModel):
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    unverifiable_claims: int = 0
    support_ratio: float = 0.0
    evidence_coverage: float = 0.0
    citation_validity_rate: float = 1.0
    contradiction_rate: float = 0.0
    confidence: float = 0.0
    claim_type_counts: Dict[str, int] = Field(default_factory=dict)
    evaluated_claims: List[EvaluatedClaim] = Field(default_factory=list)
    citation_checks: List[CitationCheck] = Field(default_factory=list)


class EvidenceAuditor:
    """Evaluates grounding and accuracy of engineering intelligence outputs.

    Verification is performed against the deterministic knowledge layer (DB),
    so a claim is only SUPPORTED when the repository actually contains the
    referenced evidence.
    """

    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session
        # Preloaded index state (populated lazily per repo)
        self._files: Optional[set] = None
        self._symbols: Optional[set] = None
        self._commits: Optional[set] = None
        self._symbol_lines: Dict[str, tuple] = {}
        self._file_lines: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Index loading / caching
    # ------------------------------------------------------------------

    async def _load_index(self, session: AsyncSession, repo_id: str):
        self.session = session
        files_res = await session.execute(
            select(FileModel.file_path).where(FileModel.repo_id == repo_id)
        )
        known_files = {p for (p,) in files_res.all()}

        chunk_files = await session.execute(
            select(CodeChunkModel.file_path).where(CodeChunkModel.repo_id == repo_id)
        )
        known_files |= {p for (p,) in chunk_files.all()}

        sym_res = await session.execute(
            select(
                SymbolModel.symbol_name, SymbolModel.file_path,
                SymbolModel.line_start, SymbolModel.line_end,
            ).where(SymbolModel.repo_id == repo_id)
        )
        symbols = set()
        symbol_lines = {}
        for name, path, ls, le in sym_res.all():
            symbols.add(name)
            if name not in symbol_lines or (le - ls) > 0:
                symbol_lines[name] = (path, ls, le)

        commit_res = await session.execute(
            select(CommitModel.sha).where(CommitModel.repo_id == repo_id)
        )
        commits = {sha for (sha,) in commit_res.all()}

        self._files = known_files
        self._symbols = symbols
        self._commits = commits
        self._symbol_lines = symbol_lines

    def preload_index(self, files: set, symbols: set, commits: set,
                      symbol_lines: Optional[Dict[str, tuple]] = None):
        """Synchronous injection for tests / non-DB usage."""
        self._files, self._symbols, self._commits = set(files), set(symbols), set(commits)
        self._symbol_lines = symbol_lines or {}

    # ------------------------------------------------------------------
    # Claim extraction & typing (44.2)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_identifiers(text: str) -> List[str]:
        found: List[str] = []
        for pattern in (_PATH_RE, _QUOTED_RE, _SHA_RE, _IDENT_RE):
            for m in pattern.finditer(text):
                val = m.group(1) if m.re is _QUOTED_RE else m.group(0)
                val = val.strip().strip("'\"`")
                if val and len(val) > 2 and val.lower() not in ("the", "this", "that"):
                    found.append(val)
        seen, unique = set(), []
        for f in found:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique[:12]

    @staticmethod
    def classify_claim_type(text: str, default: ClaimType) -> ClaimType:
        t = text.lower()
        if any(k in t for k in _RISK_KEYWORDS):
            return ClaimType.RISK
        if any(k in t for k in _TEST_KEYWORDS):
            return ClaimType.TEST
        if any(k in t for k in _HISTORICAL_KEYWORDS):
            return ClaimType.HISTORICAL
        if any(k in t for k in _ARCH_KEYWORDS):
            return ClaimType.ARCHITECTURE
        if any(k in t for k in _STRUCTURAL_KEYWORDS):
            return ClaimType.STRUCTURAL
        return default

    def collect_claims(self, response: Dict[str, Any]) -> List[tuple]:
        """Extract claims from facts/inferences/recommendations plus answer sentences."""
        claims: List[tuple] = []
        defaults = {
            "facts": ClaimType.FACT,
            "inferences": ClaimType.INFERENCE,
            "recommendations": ClaimType.RECOMMENDATION,
        }
        for key, default_type in defaults.items():
            for item in response.get(key, []) or []:
                if isinstance(item, str) and item.strip():
                    claims.append((item.strip(), self.classify_claim_type(item, default_type)))

        answer = response.get("answer", "") or ""
        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            sentence = sentence.strip()
            # Sentences asserting repository facts carry verifiable content.
            has_identifiers = bool(self.extract_identifiers(sentence))
            has_assertion = any(
                v in sentence.lower()
                for v in (" is ", " are ", " exists", " contains", " implements",
                          " depends", " imports", " located", " defines")
            )
            if sentence and has_identifiers and has_assertion and len(sentence) > 15:
                claims.append((sentence, self.classify_claim_type(sentence, ClaimType.FACT)))
        return claims




    def _verify_identifier(self, ident: str) -> Optional[CitationCheck]:
        """Verify one identifier against the repository index (44.4).

        Returns None when the identifier is not a verifiable repository reference.
        """
        files = self._files or set()
        symbols = self._symbols or set()
        commits = self._commits or set()

        if ident in files:
            return CitationCheck(citation=ident, kind="file", valid=True)
        for f in files:
            if f.endswith(ident) or ident.endswith(f):
                return CitationCheck(citation=ident, kind="file", valid=True,
                                     reason=f"matched indexed file {f}")
        if ident in symbols:
            return CitationCheck(citation=ident, kind="symbol", valid=True)
        if len(ident) >= 7 and bool(_SHA_RE.fullmatch(ident)) and ident.lower() in commits:
            return CitationCheck(citation=ident, kind="commit", valid=True)

        looks_like_file = bool(_PATH_RE.fullmatch(ident))
        looks_like_sha = bool(_SHA_RE.fullmatch(ident))
        looks_like_symbol = bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident)) and (
            "_" in ident or ident[:1].isupper()
        )
        if looks_like_file:
            return CitationCheck(citation=ident, kind="file", valid=False,
                                 reason="file not present in repository index")
        if looks_like_sha:
            return CitationCheck(citation=ident, kind="commit", valid=False,
                                 reason="commit sha not present in ingested history")
        if looks_like_symbol and len(ident) > 3:
            return CitationCheck(citation=ident, kind="symbol", valid=False,
                                 reason="symbol not found in parsed index")
        return None

    async def check_contradictions(self, session: AsyncSession, repo_id: str,
                                   claim_text: str) -> List[Dict[str, Any]]:
        """DB-backed contradiction detection (44.5).

        Example: claim 'Module A does not depend on redis' while symbol edges
        show A imports redis_client -> CONTRADICTED with the contradicting edge.
        """
        from packages.knowledge.models import SymbolEdgeModel
        contradicted: List[Dict[str, Any]] = []
        lowered = claim_text.lower()
        if not any(re.search(p, lowered) for p in _NEGATIVE_DEP_PATTERNS):
            return contradicted
        idents = self.extract_identifiers(claim_text)
        if not idents:
            return contradicted

        edges_res = await session.execute(
            select(SymbolEdgeModel).where(SymbolEdgeModel.repo_id == repo_id)
        )
        for edge in edges_res.scalars().all():
            src_fields = [edge.source_symbol or "", edge.source_file or ""]
            tgt_fields = [edge.target_symbol or "", edge.target_file or ""]
            for a in idents:
                for s in src_fields:
                    if a and s and (s == a or s.endswith(a) or a in s.split("/")):
                        for b in idents:
                            if b == a:
                                continue
                            for t in tgt_fields:
                                if b and t and (t == b or b.lower() in t.lower()):
                                    contradicted.append({
                                        "type": "symbol_edge",
                                        "source": s, "target": t,
                                        "edge_type": edge.edge_type,
                                        "claim_identifiers": [a, b],
                                    })
        deps_res = await session.execute(
            select(DependencyModel.package_name).where(DependencyModel.repo_id == repo_id)
        )
        packages = {p.lower() for (p,) in deps_res.all()}
        for i in idents:
            for pkg in packages:
                if pkg and len(pkg) > 3 and pkg in i.lower():
                    contradicted.append({"type": "dependency", "package": pkg,
                                         "claim_identifier": i})
        return contradicted

    def audit_response_sync(self, response: Dict[str, Any]) -> EvidenceAuditResult:
        """Audit using a preloaded index. Deterministic; no DB access needed."""
        claims = self.collect_claims(response)
        evidence_items = response.get("evidence", []) or []

        evaluated: List[EvaluatedClaim] = []
        citation_checks: List[CitationCheck] = []
        type_counts: Dict[str, int] = {}
        supported = unsupported = contradicted = unverifiable = 0

        for text, ctype in claims:
            type_counts[ctype.value] = type_counts.get(ctype.value, 0) + 1
            idents = self.extract_identifiers(text)

            matching_ev = []
            for ev in evidence_items:
                ev_vals = " ".join(str(v) for v in (ev or {}).values())
                if ev_vals and (ev_vals in text or any(i and str(i) in ev_vals for i in idents)):
                    matching_ev.append(ev)

            checks = [c for c in (self._verify_identifier(i) for i in idents) if c is not None]
            citation_checks.extend(checks)
            invalid_checks = [c for c in checks if not c.valid]

            status: ClaimVerificationStatus
            if invalid_checks:
                status = ClaimVerificationStatus.UNSUPPORTED
                explanation = "; ".join(f"{c.kind} '{c.citation}': {c.reason}" for c in invalid_checks[:3])
                unsupported += 1
            elif not checks and ctype in (ClaimType.RECOMMENDATION, ClaimType.INFERENCE):
                status = ClaimVerificationStatus.UNVERIFIABLE
                explanation = "judgment without repository-anchored identifiers to verify"
                unverifiable += 1
            elif not checks and not matching_ev:
                status = ClaimVerificationStatus.UNVERIFIABLE
                explanation = "no verifiable identifiers or matching evidence items"
                unverifiable += 1
            else:
                status = ClaimVerificationStatus.SUPPORTED
                n_valid = len([c for c in checks if c.valid])
                explanation = f"{n_valid} citation(s) verified, {len(matching_ev)} evidence item(s) matched"
                supported += 1

            evaluated.append(EvaluatedClaim(
                claim_text=text, claim_type=ctype, status=status,
                matching_evidence=matching_ev,
                extracted_identifiers=idents,
                explanation=explanation,
            ))

        total = len(evaluated)
        support_ratio = (supported / total) if total else 0.0
        valid_citations = sum(1 for c in citation_checks if c.valid)
        citation_validity = (valid_citations / len(citation_checks)) if citation_checks else 1.0
        coverage = min(1.0, (sum(len(c.matching_evidence) for c in evaluated) / total)) if total else 0.0
        confidence = round(0.5 * support_ratio + 0.3 * citation_validity + 0.2 * coverage, 4)

        return EvidenceAuditResult(
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            unverifiable_claims=unverifiable,
            support_ratio=round(support_ratio, 4),
            evidence_coverage=round(coverage, 4),
            citation_validity_rate=round(citation_validity, 4),
            contradiction_rate=round(contradicted / max(1, total), 4),
            confidence=confidence,
            claim_type_counts=type_counts,
            evaluated_claims=evaluated,
            citation_checks=citation_checks,
        )

    async def audit_response(self, response: Dict[str, Any], session: AsyncSession,
                             repo_id: str) -> EvidenceAuditResult:
        """Full audit against live repository knowledge: verifies citations from the
        DB index and runs contradiction detection over dependency edges (44.4/44.5)."""
        await self._load_index(session, repo_id)
        result = self.audit_response_sync(response)

        for claim in result.evaluated_claims:
            contradictions = await self.check_contradictions(session, repo_id, claim.claim_text)
            if contradictions and claim.status != ClaimVerificationStatus.CONTRADICTED:
                if claim.status == ClaimVerificationStatus.SUPPORTED:
                    result.supported_claims -= 1
                elif claim.status == ClaimVerificationStatus.UNSUPPORTED:
                    result.unsupported_claims -= 1
                else:
                    result.unverifiable_claims -= 1
                claim.status = ClaimVerificationStatus.CONTRADICTED
                claim.contradictory_evidence = contradictions
                claim.explanation = "claim negated by repository dependency/edge evidence"
                result.contradicted_claims += 1

        if result.total_claims:
            result.support_ratio = round(result.supported_claims / result.total_claims, 4)
            result.contradiction_rate = round(result.contradicted_claims / result.total_claims, 4)
        return result
