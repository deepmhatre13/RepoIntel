import hashlib
import json
from typing import List
from packages.knowledge.repository_facade import RepositoryFacade
from packages.knowledge.analysis_models import QAAnswer
from packages.agentic.llm_client import LLMClient

QA_ANSWER_SYSTEM = """
You are RepoIntel, an expert repository analyst AI.
Answer the user query based ONLY on the repository evidence provided.
If evidence is insufficient, set confidence to "LOW" and state clearly:
"The repository does not provide enough evidence to determine this."

Do NOT hallucinate or invent code, symbols, files, or commit history.
"""

from packages.knowledge.relevance import (
    evaluate_relevance,
    RelevanceLevel,
    confidence_breakdown,
    _query_files,
    _query_symbols,
)


def _evidence_id(item: dict) -> str:
    """Stable evidence identifier: repo + file + lines + symbol + content hash.

    Enables claim-to-evidence traceability across citations and audits (Phase G).
    """
    raw = "|".join([
        str(item.get("repo_id") or ""),
        str(item.get("file_path") or ""),
        str(item.get("line_start") or 0),
        str(item.get("line_end") or 0),
        str(item.get("symbol_name") or ""),
        str(item.get("chunk_id") or ""),
    ])
    return "ev-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def answer_repository_query(
    repo_id: str,
    query: str,
    facade: RepositoryFacade,
    llm_client: LLMClient,
    group_repo_ids: List[str] = None
    ) -> QAAnswer:
    # 1. Retrieve candidate evidence.
    if group_repo_ids:
        retrieved = facade.multi_repository_hybrid_search(group_repo_ids, query, top_k=5)
    else:
        retrieved = facade.hybrid_search(repo_id, query, top_k=5)

    evidence_items = []
    related_files = set()
    related_symbols = set()
    for item in retrieved:
        d = item.to_dict()
        d["evidence_id"] = _evidence_id(d)
        evidence_items.append(d)
        related_files.add(item.file_path)
        if item.symbol_name:
            related_symbols.add(item.symbol_name)

    # 2. Gather direct structured knowledge (symbols/files the question names) so
    #    the deterministic path can synthesize an answer instead of dumping chunks.
    structured_symbols = []
    direct_files = []
    query_files = _query_files(query)
    for f in query_files:
        syms = await facade.get_symbols_for_file(repo_id, f)
        if syms:
            direct_files.append(f)
            structured_symbols.extend(syms)
    for s in _query_symbols(query):
        syms = await facade.get_symbols_by_name(repo_id, s)
        structured_symbols.extend(syms)
    # De-duplicate structured symbols.
    seen = set()
    unique_struct = []
    for s in structured_symbols:
        key = (s.get("file_path"), s.get("symbol_name"))
        if key not in seen:
            seen.add(key)
            unique_struct.append(s)
    structured_symbols = unique_struct

    # 3. Run the deterministic relevance gate over the actually retrieved evidence.
    decision = evaluate_relevance(query, retrieved)
    relevance_support = decision.level

    # 4. If the repository provides no relevant evidence, refuse to invent an
    #    answer rather than reasoning from unrelated top-k documents.
    if relevance_support is RelevanceLevel.NO_RELEVANT_EVIDENCE and not direct_files:
        answer_text = (
            "No evidence was found that this repository supports the topic of "
            "the question. The indexed repository does not contain relevant "
            "files, symbols, or content for this question, so no answer can be "
            "grounded in repository evidence."
        )
        return QAAnswer(
            query=query,
            answer=answer_text,
            confidence="LOW",
            evidence=[],
            related_files=[],          # do not surface unrelated evidence
            related_symbols=[],
            related_commits=[],
            confidence_breakdown=confidence_breakdown(
                decision, retrieved, 0, llm_client.is_available(),
                direct_file_lookup=bool(direct_files),
            ).to_dict(),
            audit=None,
            degraded=not llm_client.is_available(),
        )

    # 5. Compute confidence from the relevance decision + evidence strength.
    degraded = not llm_client.is_available()
    # 5a. Measure real evidence freshness from ingested Git history (Phase G).
    try:
        freshness_result = await facade.get_evidence_freshness(
            repo_id, sorted({e["file_path"] for e in evidence_items if e.get("file_path")})
        )
    except Exception:
        freshness_result = None  # freshness stays NOT_MEASURED, never faked
    from packages.knowledge.relevance import confidence_from_relevance
    bd = confidence_breakdown(
        decision,
        retrieved,
        evidence_count=len(evidence_items),
        llm_available=not degraded,
        direct_file_lookup=bool(direct_files),
        freshness_result=freshness_result,
    )
    confidence = bd.label

    # 6. Synthesize the answer.
    if degraded:
        # Deterministic, evidence-grounded structured answer in DEGRADED MODE.
        lines = [
            "DEGRADED MODE: LLM/Gemini is unavailable (no GOOGLE_API_KEY); this "
            "answer was assembled only from extracted repository evidence and no "
            "generative reasoning was performed."
        ]
        if direct_files:
            lines.append("")
            lines.append(f"Target file(s): {', '.join(direct_files)}")
        if structured_symbols:
            lines.append("")
            lines.append("Symbols:")
            for s in structured_symbols:
                loc = f"{s.get('file_path')}::{s.get('symbol_name')}"
                sig = s.get("signature") or ""
                lines.append(f"- {loc}" + (f"  ({sig})" if sig else ""))
                doc = (s.get("docstring") or "").strip()
                if doc:
                    first = doc.splitlines()[0].strip()
                    lines.append(f"    docstring: {first}")
        elif evidence_items:
            lines.append("")
            lines.append("Supporting evidence:")
            for e in evidence_items:
                loc = e.get("file_path") or ""
                sym = e.get("symbol_name") or ""
                lines.append(f"- {loc}" + (f"::{sym}" if sym else ""))
        if evidence_items:
            lines.append("")
            lines.append(f"Evidence count: {len(evidence_items)} (relevance: {relevance_support.value})")
        answer_text = "\n".join(lines)
    else:
        context_str = json.dumps(evidence_items, indent=2)
        answer_text = llm_client.complete(
            f"User Query: {query}\n\nRepository Evidence:\n{context_str}",
            system=QA_ANSWER_SYSTEM,
        )

    # 7. Deterministic evidence audit of the final answer (Phase G).
    #    Verifies that claims in the answer resolve to real indexed repository
    #    facts and reports the measured support ratio. Audit failure never
    #    fabricates a passing result -- audit stays None.
    audit_payload = None
    try:
        from packages.evals.evidence_audit import EvidenceAuditor
        auditor = EvidenceAuditor()
        try:
            await auditor._load_index(facade.session, repo_id)
        except Exception:
            # DB index unavailable: audit only claim/evidence correspondence.
            auditor.preload_index(
                files={e.get("file_path") for e in evidence_items if e.get("file_path")},
                symbols={e.get("symbol_name") for e in evidence_items if e.get("symbol_name")},
                commits=set(),
            )
        audit_result = auditor.audit_response_sync({
            "answer": answer_text,
            "evidence": evidence_items,
        })
        audit_payload = {
            "total_claims": audit_result.total_claims,
            "supported_claims": audit_result.supported_claims,
            "unsupported_claims": audit_result.unsupported_claims,
            "unverifiable_claims": audit_result.unverifiable_claims,
            "support_ratio": audit_result.support_ratio,
            "evidence_coverage": audit_result.evidence_coverage,
            "citation_validity_rate": audit_result.citation_validity_rate,
            "confidence": audit_result.confidence,
            "claims": [
                {
                    "claim": c.claim_text[:200],
                    "type": c.claim_type.value,
                    "status": c.status.value,
                    "explanation": c.explanation,
                    "evidence_ids": [
                        e.get("evidence_id") for e in c.matching_evidence
                        if isinstance(e, dict) and e.get("evidence_id")
                    ],
                }
                for c in audit_result.evaluated_claims[:10]
            ],
        }
    except Exception:
        audit_payload = None

    return QAAnswer(
        query=query,
        answer=answer_text,
        confidence=bd.label,
        evidence=evidence_items,
        related_files=list(related_files),
        related_symbols=list(related_symbols),
        related_commits=[],
        confidence_breakdown=bd.to_dict(),
        audit=audit_payload,
        degraded=degraded,
    )
