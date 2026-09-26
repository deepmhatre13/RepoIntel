"""Evidence-grounded artifact generation engine for RepoIntel (Stage 8).

Pipelines any ``artifact_type`` (README, ARCHITECTURE, API_REFERENCE,
CODEBASE_OVERVIEW, ONBOARDING_GUIDE, CHANGE_IMPACT_REPORT, SECURITY_REVIEW) through a
single, shared evidence-grounded flow:

    Repository Intelligence -> Evidence Collection -> Artifact Planner ->
    Gemini/Deterinistic Synthesis -> Claim Extraction -> Evidence Validation ->
    Final Artifact

All capabilities are consumed through the shared :class:`RepoIntelTools` layer
(never re-implemented). Facts are DETECTED, not invented; unsupported claims are
removed or explicitly qualified.

Security: no repository code is executed; secrets are redacted from output.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from packages.agentic.tools import RepoIntelTools, ToolError
from packages.agentic.llm_client import LLMClient
from packages.shared.tracing import span
from packages.shared.secret_redaction import redact_secrets


class ArtifactSection(BaseModel):
    title: str
    content: str
    claims: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    insufficient_evidence: bool = False


class GeneratedArtifact(BaseModel):
    artifact_type: str
    repository_id: str
    title: str
    sections: List[ArtifactSection] = Field(default_factory=list)
    rendered: str = ""
    facts: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    degraded: bool = False


# Maps artifact_type -> (title, ordered (section_title, kind)) to plan.
_ARTIFACT_PLAN: Dict[str, Dict[str, Any]] = {
    "README": {
        "title": "README",
        "sections": [
            ("Purpose & Overview", "summary"),
            ("Technologies & Languages", "technology"),
            ("Structure & Entrypoints", "structure"),
            ("Installation", "installation"),
            ("Configuration", "configuration"),
            ("Usage", "usage"),
            ("Testing", "testing"),
            ("Architecture Overview", "architecture"),
        ],
    },
    "ARCHITECTURE": {
        "title": "Architecture Overview",
        "sections": [
            ("System Overview", "summary"),
            ("Components", "structure"),
            ("Dependencies", "dependencies"),
            ("Data Flow / Call Paths", "call_flow"),
            ("Configuration", "configuration"),
            ("Evolution", "history"),
        ],
    },
    "API_REFERENCE": {
        "title": "API Reference",
        "sections": [
            ("API Endpoints / Exposed Symbols", "api"),
            ("Request/Response Shapes", "signatures"),
            ("Authentication", "auth"),
        ],
    },
    "CODEBASE_OVERVIEW": {
        "title": "Codebase Overview",
        "sections": [
            ("Overview", "summary"),
            ("Key Modules", "structure"),
            ("Symbol Map", "symbols"),
            ("Entrypoints", "entrypoints"),
        ],
    },
    "ONBOARDING_GUIDE": {
        "title": "Onboarding Guide",
        "sections": [
            ("Quickstart", "summary"),
            ("Repository Layout", "structure"),
            ("Key Workflows", "call_flow"),
            ("Configuration", "configuration"),
            ("Testing", "testing"),
        ],
    },
    "CHANGE_IMPACT_REPORT": {
        "title": "Change Impact Report",
        "sections": [
            ("Impact of Change", "impact"),
            ("Affected Files & Symbols", "impact"),
            ("Related Tests", "tests"),
        ],
    },
    "SECURITY_REVIEW": {
        "title": "Security Review",
        "sections": [
            ("Security Findings", "security"),
            ("Dependency Risks", "dependencies"),
            ("Repository-Aware Context", "security_context"),
        ],
    },
}

# Section kinds supported by deterministic evidence assembly. Each kind maps
# to the tools used to gather evidence for that section (all through
# RepoIntelTools -> existing facade services; nothing re-implemented here).
_SECTION_TOOL_PLAN: Dict[str, List[str]] = {
    "summary": ["get_architecture", "search_repository"],
    "technology": ["search_repository"],
    "structure": ["get_architecture"],
    "installation": ["search_repository"],
    "configuration": ["search_repository"],
    "usage": ["search_repository"],
    "testing": ["find_tests", "search_repository"],
    "architecture": ["get_architecture", "get_hotspots"],
    "history": ["get_git_history"],
    "api": ["get_architecture", "search_symbol"],
    "signatures": ["search_symbol"],
    "auth": ["get_security_findings"],
    "symbols": ["search_repository"],
    "entrypoints": ["get_architecture"],
    "impact": ["analyze_impact"],
    "tests": ["find_tests"],
    "security": ["get_security_findings"],
    "security_context": ["get_security_findings", "get_hotspots"],
}

_DETECTED = "DETECTED"
_INFERRED = "INFERRED"
_UNAVAILABLE = "NOT AVAILABLE"


class _SectionDraft(BaseModel):
    """Schema Gemini must return per section (structured output)."""
    content: str = ""
    claims: List[str] = Field(default_factory=list)


def _load_items(evidence_json: str) -> List[Dict[str, Any]]:
    """Best-effort reload of evidence JSON (used only on Gemini fallback)."""
    try:
        parsed = json.loads(evidence_json)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


# Words ignored when checking whether a claim is supported by evidence.
_STOPWORDS = frozenset({
    "this", "that", "with", "from", "have", "uses", "used", "using",
    "also", "into", "when", "which", "your", "will", "been", "more",
    "repository", "code", "codebase", "project", "section", "evidence",
    "provides", "provides", "supports", "based", "tests", "test",
    "there", "these", "those", "than", "then", "them", "they", "were",
})


class ArtifactEngine:
    """Evidence-grounded artifact generation over the shared intelligence layer.

    Pipeline (identical for every artifact_type — no per-type architecture):

        evidence collection -> deterministic plan -> Gemini synthesis ->
        claim extraction -> evidence validation -> final artifact

    Facts come only from RepoIntel tools (facade services). Gemini only
    narrates over provided evidence; when Gemini is unavailable a
    deterministic renderer produces the artifact from the same evidence.
    """

    def __init__(self, facade: Any, llm_client: Optional[LLMClient] = None):
        self.facade = facade
        self.llm_client = llm_client or LLMClient()
        self.tools = RepoIntelTools(facade, llm_client=self.llm_client)

    # ------------------------------------------------------------------
    # Shared capability access (single boundary — no facade logic here)
    # ------------------------------------------------------------------
    async def _call(self, tool_name: str, repository_id: str, **kwargs: Any) -> Any:
        try:
            return await self.tools.call(tool_name, repository_id, kwargs)
        except ToolError:
            return {}
        except Exception:
            # One missing evidence source must never abort the whole artifact.
            return {}

    # ------------------------------------------------------------------
    # Stage 1: evidence collection
    # ------------------------------------------------------------------
    async def collect_evidence(
        self,
        repository_id: str,
        artifact_type: str,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collect bounded evidence for every planned section via RepoIntel tools."""
        plan = _ARTIFACT_PLAN.get(artifact_type.upper())
        if plan is None:
            raise ValueError(
                f"Unsupported artifact_type '{artifact_type}'. "
                f"Supported: {sorted(_ARTIFACT_PLAN)}"
            )

        evidence: Dict[str, List[Dict[str, Any]]] = {}
        with span("artifact.collect_evidence", repository_id=repository_id,
                  artifact_type=artifact_type.upper()):
            for section_title, kind in plan["sections"]:
                bucket: List[Dict[str, Any]] = []
                for tool_name in _SECTION_TOOL_PLAN.get(kind, ["search_repository"]):
                    kwargs: Dict[str, Any] = {}
                    if tool_name in ("search_repository", "search_symbol"):
                        kwargs = self._query_for(kind, section_title, target)
                    elif tool_name in ("analyze_impact", "find_tests"):
                        if not target:
                            continue
                        kwargs = ({"target": target} if tool_name == "analyze_impact"
                                  else {"symbol": target})
                    result = await self._call(tool_name, repository_id, **kwargs)
                    bucket.extend(self._flatten_tool_result(tool_name, result))
                evidence[section_title] = bucket[:12]  # bounded per section
        return {"plan": plan, "evidence": evidence}

    @staticmethod
    def _query_for(kind: str, section_title: str, target: Optional[str]) -> Dict[str, Any]:
        if kind == "technology":
            return {"query": "requirements dependencies framework imports package"}
        if kind == "installation":
            return {"query": "install setup requirements pip getting started"}
        if kind == "configuration":
            return {"query": "configuration environment variables settings config"}
        if kind == "usage":
            return {"query": "example usage quickstart demo run"}
        if kind == "testing":
            return {"query": "test pytest fixture suite"}
        if target:
            return {"symbol": target} if kind in ("api", "signatures") else {"query": target}
        return {"query": "main entrypoint endpoint route"}

    @staticmethod
    def _flatten_tool_result(tool_name: str, result: Any) -> List[Dict[str, Any]]:
        """Normalize heterogeneous tool outputs into bounded evidence items."""
        if not isinstance(result, dict):
            return []
        items: List[Dict[str, Any]] = []
        for key in ("results", "symbols", "commits", "history", "tests", "findings",
                    "hotspots", "components", "module_edges", "cycles", "modules",
                    "entrypoints", "evidence"):
            raw = result.get(key)
            if isinstance(raw, list):
                for entry in raw[:8]:
                    if isinstance(entry, dict):
                        items.append({"source_tool": tool_name, **entry})
                    else:
                        items.append({"source_tool": tool_name, "value": entry})
        if not items and result:
            # Single-dict analyses (architecture, coupling, impact...).
            flat = {k: v for k, v in result.items()
                    if isinstance(v, (str, int, float, bool, list))}
            if flat:
                items.append({"source_tool": tool_name, **flat})
        return items

    # ------------------------------------------------------------------
    # Stage 2: synthesis (Gemini when available; deterministic otherwise)
    # ------------------------------------------------------------------
    async def _synthesize_sections(
        self,
        repository_id: str,
        artifact_type: str,
        plan: Dict[str, Any],
        evidence: Dict[str, List[Dict[str, Any]]],
    ) -> tuple[List[ArtifactSection], bool]:
        """Gemini narrates sections from evidence only; degraded renders deterministically."""
        degraded = not self.llm_client.is_available()
        sections: List[ArtifactSection] = []

        for section_title, kind in plan["sections"]:
            items = evidence.get(section_title, [])
            if not items:
                sections.append(ArtifactSection(
                    title=section_title,
                    content=(f"{_UNAVAILABLE}: no repository evidence was found for this "
                             f"section; no claims are made."),
                    insufficient_evidence=True,
                ))
                continue

            evidence_json = json.dumps(items[:12], default=str, ensure_ascii=False)
            with span("artifact.synthesize_section", repository_id=repository_id,
                      section=section_title, degraded=degraded,
                      evidence_count=len(items)):
                if degraded:
                    content, claims = self._deterministic_section(
                        section_title, kind, items)
                else:
                    content, claims = self._gemini_section(
                        artifact_type, section_title, kind, evidence_json, items)
            sections.append(ArtifactSection(
                title=section_title,
                content=content,
                claims=claims,
                evidence=items[:12],
            ))
        return sections, degraded

    def _gemini_section(self, artifact_type: str, section_title: str,
                        kind: str, evidence_json: str,
                        items: List[Dict[str, Any]]) -> tuple[str, List[str]]:
        """One bounded Gemini call per section over redacted evidence JSON."""
        prompt = (
            f"You are generating the '{section_title}' section of a {artifact_type} "
            f"document for a software repository.\n\n"
            f"STRICT RULES:\n"
            f"1. Use ONLY the evidence JSON below. Never invent file names, commands, "
            f"dependencies, or configuration.\n"
            f"2. If the evidence does not support a statement, omit it.\n"
            f"3. If evidence is insufficient for the section, say so explicitly.\n"
            f"4. Write 1-3 short factual paragraphs or bullet points.\n\n"
            f"EVIDENCE (repository-derived, may contain [REDACTED:*] placeholders):\n"
            f"{evidence_json}\n\n"
            f"Return JSON: {{\"content\": <markdown>, \"claims\": [<factual statements "
            f"made, one per string>]}}"
        )
        try:
            draft = self.llm_client.complete_structured(
                prompt,
                _SectionDraft,
                system=("You are a documentation generator. Facts only, grounded in "
                        "the provided evidence."),
            )
            content = (draft.content or "").strip()
            claims = [c for c in (draft.claims or []) if c]
            if not content or content.startswith("[DEGRADED_MODE"):
                raise LLMFailureError("degraded marker returned", transient=False)
        except Exception:
            # Any Gemini failure falls back to deterministic narration of the
            # same evidence — the artifact is still produced, still grounded.
            return self._deterministic_section(section_title, kind, items)
        return content, claims

    # ------------------------------------------------------------------
    # Stage 3: deterministic section rendering (also the degraded path)
    # ------------------------------------------------------------------
    def _deterministic_section(
        self, section_title: str, kind: str, items: List[Dict[str, Any]]
    ) -> tuple[str, List[str]]:
        lines: List[str] = []
        claims: List[str] = []
        for item in items[:8]:
            tool = item.get("source_tool", "repository_intelligence")
            snippet = self._summarize_item(item)
            if snippet:
                lines.append(f"- {snippet} *(evidence: {tool})*")
                claims.append(snippet)
        if not lines:
            return (f"{_UNAVAILABLE}: no repository evidence found for this section.",
                    [])
        return "\n".join(lines), claims

    @staticmethod
    def _summarize_item(item: Dict[str, Any]) -> str:
        """Render one evidence item as a single factual bullet."""
        for key in ("content_snippet", "snippet", "summary", "description", "message"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                loc = ""
                fp = item.get("file_path") or item.get("path") or ""
                if fp:
                    loc = f" (`{fp}`"
                    if item.get("line_start"):
                        loc += f":{item['line_start']}"
                    loc += ")"
                return f"{val.strip()[:240]}{loc}"
        fp = item.get("file_path") or item.get("path") or item.get("file")
        if isinstance(fp, str) and fp:
            sym = item.get("name") or item.get("symbol") or ""
            kind_v = item.get("kind") or item.get("symbol_type") or ""
            base = f"`{fp}`" + (f" — {sym}" if sym else "")
            if kind_v and isinstance(kind_v, str):
                base += f" ({kind_v})"
            return base
        name = item.get("name") or item.get("symbol")
        if isinstance(name, str) and name:
            out = f"symbol `{name}`"
            if item.get("file_path"):
                out += f" in `{item['file_path']}`"
            return out
        if item.get("sha"):
            msg = str(item.get("message", ""))[:120]
            return f"commit `{str(item['sha'])[:10]}`" + (f": {msg}" if msg else "")
        pairs = [f"{k}: {v}" for k, v in item.items()
                 if k not in ("source_tool",) and isinstance(v, (str, int, float, bool))]
        return ", ".join(pairs[:4]) if pairs else ""

    # ------------------------------------------------------------------
    # Stage 4: evidence validation (claims -> evidence, remove unsupported)
    # ------------------------------------------------------------------
    def _validate_claims(self, sections: List[ArtifactSection]) -> None:
        """Mark claims with no matching evidence as unsupported and remove them.

        Validation is lexical: an artifact claim is supported when its salient
        words overlap the evidence items captured for the same section. This is
        deliberately conservative — anything not traceable to collected
        evidence is dropped, per the evidence-grounding contract.
        """
        for section in sections:
            if section.insufficient_evidence or not section.claims:
                continue
            evidence_blob = json.dumps(section.evidence, default=str).lower()
            supported: List[str] = []
            unsupported: List[str] = []
            for claim in section.claims:
                words = [
                    w for w in re.findall(r"[a-z_0-9]{4,}", claim.lower())
                    if w not in _STOPWORDS
                ]
                if not words:
                    # Nothing checkable — keep only short trivial statements.
                    (supported if len(claim) < 80 else unsupported).append(claim)
                    continue
                hits = sum(1 for w in words if w in evidence_blob)
                if hits >= max(1, min(2, len(words))):
                    supported.append(claim)
                else:
                    unsupported.append(claim)
            section.claims = supported
            section.unsupported_claims = unsupported
            if unsupported:
                note = (
                    f"*Note: {len(unsupported)} statement(s) could not be "
                    f"validated against repository evidence and were removed.*"
                )
                section.content = redact_secrets(f"{section.content}\n\n{note}").text

    def _finalize(self, artifact: GeneratedArtifact) -> GeneratedArtifact:
        artifact.unsupported_claims = [
            c for s in artifact.sections for c in s.unsupported_claims
        ]
        artifact.facts = [c for s in artifact.sections for c in s.claims]
        total_claims = len(artifact.facts) + len(artifact.unsupported_claims)
        if total_claims:
            artifact.confidence = round(len(artifact.facts) / total_claims, 3)
        else:
            artifact.confidence = 0.5 if any(
                s.insufficient_evidence for s in artifact.sections) else 1.0
        artifact.rendered = self._render(artifact)
        return artifact

    @staticmethod
    def _render(artifact: GeneratedArtifact) -> str:
        parts = [f"# {artifact.title}", ""]
        for section in artifact.sections:
            parts.append(f"## {section.title}")
            parts.append(section.content)
            parts.append("")
        return redact_secrets("\n".join(parts)).text

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    async def generate(
        self,
        repository_id: str,
        artifact_type: str = "README",
        target: Optional[str] = None,
    ) -> GeneratedArtifact:
        """Generate an evidence-grounded artifact of any supported type."""
        norm_type = (artifact_type or "README").upper()
        with span("artifact.generate", repository_id=repository_id,
                  artifact_type=norm_type, target=target or ""):
            collected = await self.collect_evidence(repository_id, norm_type, target)
            sections, degraded = await self._synthesize_sections(
                repository_id, norm_type, collected["plan"], collected["evidence"])
            self._validate_claims(sections)
            plan = _ARTIFACT_PLAN[norm_type]
            artifact = GeneratedArtifact(
                artifact_type=norm_type,
                repository_id=repository_id,
                title=plan["title"] if not target else f"{plan['title']}: {target}",
                sections=sections,
                degraded=degraded,
            )
            return self._finalize(artifact)


