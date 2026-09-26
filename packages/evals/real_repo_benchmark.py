"""Real-Repository Engineering Benchmark & Production Readiness for RepoIntel (Phase 45).

MEASUREMENT, NOT ASSERTION:
- Every metric in this module is computed from actual executions.
- Ground truth comes from an independent oracle that scans RAW repository files
  directly (never through RepoIntel's own index), so retrieval quality is
  measured against repository reality rather than self-referential output.
- Baselines are really executed: Naive LLM (no retrieval), Simple Vector RAG
  (top-k vector hits only), RepoIntel-without-graph (hybrid lexical ranking),
  and Full RepoIntel (category capability + graph/history/audit).
- The ProductionReadinessReport is assembled ONLY from outcomes passed in as
  measured evidence; categories with no measurement are UNVERIFIED.

No numbers in this file are fabricated. If a run cannot be performed, the
report says so honestly.
"""

import os
import re
import json
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field



class RepoSizeTier(str, Enum):
    SMALL = "SMALL"    # < 50 indexed source files (measured)
    MEDIUM = "MEDIUM"  # 50 - 500
    LARGE = "LARGE"    # > 500


def classify_tier(file_count: int) -> RepoSizeTier:
    if file_count < 50:
        return RepoSizeTier.SMALL
    if file_count <= 500:
        return RepoSizeTier.MEDIUM
    return RepoSizeTier.LARGE


class BaselineType(str, Enum):
    NAIVE_LLM = "NAIVE_LLM"
    SIMPLE_VECTOR_RAG = "SIMPLE_VECTOR_RAG"
    REPOINTEL_SANS_GRAPH = "REPOINTEL_SANS_GRAPH"
    FULL_REPOINTEL = "FULL_REPOINTEL"


# ---------------------------------------------------------------------------
# Independent oracle: expected evidence computed from RAW files on disk.
# This is deliberately independent of RepoIntel's ingestion/index path.
# ---------------------------------------------------------------------------

class OracleGroundTruth:
    def __init__(self, repo_path: str, filtered_files: List[str]):
        self.repo_path = repo_path
        self.file_contents: Dict[str, str] = {}
        for rel in filtered_files:
            fp = os.path.join(repo_path, rel)
            if os.path.isfile(fp):
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        self.file_contents[rel] = f.read()
                except Exception:
                    pass

    def expected_files(self, terms: List[str], k: int = 10) -> List[str]:
        """Rank raw files by oracle-term density; returns top-k as ground truth."""
        if not terms:
            return []
        scored: List[Tuple[float, str]] = []
        lowered_terms = [t.lower() for t in terms]
        for path, content in self.file_contents.items():
            blob = content.lower()
            score = sum(blob.count(t) for t in lowered_terms)
            name_bonus = sum(1 for t in lowered_terms if t in path.lower())
            total = score + name_bonus * 10
            if total > 0:
                scored.append((float(total), path))
        scored.sort(reverse=True)
        return [p for _s, p in scored[:k]]

    def expected_symbols(self, terms: List[str], k: int = 10) -> List[str]:
        """Symbol-like definitions mentioning oracle terms in raw files."""
        if not terms:
            return []
        found: List[Tuple[int, str]] = []
        pattern = re.compile(
            r"(?:def|class|function|func|const)\s+([A-Za-z_][\w]*)", re.MULTILINE
        )
        for path, content in self.file_contents.items():
            for m in pattern.finditer(content):
                name = m.group(1)
                nl = name.lower()
                if any(t.lower() in nl for t in terms):
                    found.append((content.count(name), f"{name} ({path})"))
        found.sort(reverse=True)
        seen, unique = set(), []
        for _cnt, item in found:
            key = item.split(" ")[0]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique[:k]

    def test_files_for(self, terms: List[str]) -> List[str]:
        return [
            p for p in self.expected_files(terms, k=50)
            if "test" in p.lower() or p.startswith("tests/") or "/test_" in p.replace("\\\\", "/")
        ]


class BenchmarkResultMetrics(BaseModel):
    baseline: BaselineType
    query: str
    category: str
    recall_at_k: float
    precision_at_k: float
    supported_claim_ratio: float
    contradiction_rate: float
    citation_validity_rate: float
    latency_seconds: float
    tokens_used: int
    estimated_cost_usd: float


class ProductionReadinessCategory(str, Enum):
    INGESTION = "INGESTION"
    PARSING = "PARSING"
    STORAGE = "STORAGE"
    SEARCH = "SEARCH"
    GRAPH = "GRAPH"
    HISTORY = "HISTORY"
    ARCHITECTURE = "ARCHITECTURE"
    TEST_INTELLIGENCE = "TEST_INTELLIGENCE"
    IMPACT = "IMPACT"
    RISK = "RISK"
    INVESTIGATION = "INVESTIGATION"
    DECISION = "DECISION"
    REPORTING = "REPORTING"
    COPILOT = "COPILOT"
    RESILIENCE = "RESILIENCE"
    PERFORMANCE = "PERFORMANCE"
    EVIDENCE_GROUNDING = "EVIDENCE_GROUNDING"
    OBSERVABILITY = "OBSERVABILITY"


class ReadinessStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNVERIFIED = "UNVERIFIED"
    FAILED = "FAILED"


class ProductionReadinessReport(BaseModel):
    timestamp: str
    overall_status: ReadinessStatus
    verified_categories: int
    category_evaluations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    known_limitations: List[str] = Field(default_factory=list)
    unmeasured_categories: List[str] = Field(default_factory=list)


def build_production_readiness_report(
    measured_evidence: Dict[str, Dict[str, Any]],
) -> ProductionReadinessReport:
    """Assemble the report ONLY from measured evidence (no invented scores).

    measured_evidence maps category name -> {
        "status": "VERIFIED"|"PARTIAL"|"UNVERIFIED"|"FAILED",
        "evidence": <what was actually executed and observed>,
        "limitations": [...],
        "measured_metrics": {...}
    }
    Categories absent from the dict are reported UNVERIFIED.
    """
    evaluations: Dict[str, Dict[str, Any]] = {}
    for cat in ProductionReadinessCategory:
        provided = measured_evidence.get(cat.value)
        if provided is None:
            evaluations[cat.value] = {
                "status": ReadinessStatus.UNVERIFIED.value,
                "evidence": "No measurement was produced for this category.",
                "limitations": ["Not exercised by this benchmark run."],
                "measured_metrics": {},
            }
        else:
            evaluations[cat.value] = provided

    status_counts: Dict[str, int] = {}
    for ev in evaluations.values():
        status_counts[ev["status"]] = status_counts.get(ev["status"], 0) + 1

    if status_counts.get(ReadinessStatus.FAILED.value):
        overall = ReadinessStatus.FAILED
    elif status_counts.get(ReadinessStatus.UNVERIFIED.value) and not status_counts.get(
        ReadinessStatus.PARTIAL.value
    ):
        overall = ReadinessStatus.PARTIAL if status_counts.get(ReadinessStatus.VERIFIED.value) else ReadinessStatus.UNVERIFIED
    elif status_counts.get(ReadinessStatus.PARTIAL.value) or status_counts.get(
        ReadinessStatus.UNVERIFIED.value
    ):
        overall = ReadinessStatus.PARTIAL
    else:
        overall = ReadinessStatus.VERIFIED

    limitations: List[str] = []
    for cat, ev in evaluations.items():
        for lim in ev.get("limitations", []) or []:
            limitations.append(f"{cat}: {lim}")

    return ProductionReadinessReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        overall_status=overall,
        verified_categories=status_counts.get(ReadinessStatus.VERIFIED.value, 0),
        category_evaluations=evaluations,
        known_limitations=limitations,
        unmeasured_categories=[
            c.value for c in ProductionReadinessCategory
            if c.value not in measured_evidence
        ],
    )


class RealRepoBenchmarkRunner:
    """Runs measured engineering-question benchmarks on real repositories.

    All retrieval metrics are computed against OracleGroundTruth derived from raw
    repository files. All grounding metrics come from EvidenceAuditor verified
    against the indexed knowledge layer.
    """

    def __init__(self, session_factory=None):
        self.session_factory = session_factory

    def load_questions(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        default = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "evals", "datasets", "benchmark_questions.json",
        )
        with open(path or default, "r", encoding="utf-8") as f:
            return json.load(f)["questions"]

    # ------------------------------------------------------------------
    # Baseline executions (all real)
    # ------------------------------------------------------------------

    async def _baseline_naive_llm(self, repo_id: str, query: str) -> Dict[str, Any]:
        """LLM with NO repository evidence (measured; deterministic fallback when no key)."""
        from packages.agentic.llm_client import LLMClient
        llm = LLMClient()
        t0 = time.time()
        answer = llm.complete(
            f"Answer this engineering question about a codebase you cannot see:\n{query}"
        )
        latency = time.time() - t0
        return {
            "answer": answer,
            "facts": [], "inferences": [], "recommendations": [],
            "evidence": [],
            "related_files": [], "related_symbols": [],
            "latency": latency,
            "tokens": llm.total_input_tokens + llm.total_output_tokens,
            "cost_usd": llm.total_cost_usd,
            "llm_available": llm.is_available(),
        }

    async def _baseline_simple_rag(self, repo_id: str, query: str, vector_store) -> Dict[str, Any]:
        """Top-k vector hits only -- no rerank, no graph, no history."""
        t0 = time.time()
        hits = vector_store.vector_search(repo_id, query, top_k=10)
        latency = time.time() - t0
        return {
            "answer": f"Retrieved {len(hits)} chunks by similarity.",
            "facts": [f"Chunk in {h.file_path} lines {h.line_start}-{h.line_end}" for h in hits[:5]],
            "inferences": [], "recommendations": [],
            "evidence": [
                {"file_path": h.file_path, "symbol_name": h.symbol_name,
                 "line_start": h.line_start, "line_end": h.line_end,
                 "retrieval_source": h.retrieval_source}
                for h in hits
            ],
            "related_files": [h.file_path for h in hits],
            "related_symbols": [h.symbol_name for h in hits if h.symbol_name],
            "latency": latency, "tokens": 0, "cost_usd": 0.0,
            "llm_available": False,
        }

    async def _baseline_no_graph(self, repo_id: str, query: str, facade) -> Dict[str, Any]:
        """Hybrid lexical search + rerank but WITHOUT graph/history enrichment."""
        t0 = time.time()
        hits = facade.hybrid_search(repo_id, query, top_k=10)
        latency = time.time() - t0
        return {
            "answer": f"Hybrid retrieval returned {len(hits)} ranked items.",
            "facts": [f"Match in {h.file_path}" for h in hits[:5]],
            "inferences": [], "recommendations": [],
            "evidence": [
                {"file_path": h.file_path, "symbol_name": h.symbol_name,
                 "line_start": h.line_start, "line_end": h.line_end,
                 "retrieval_source": h.retrieval_source}
                for h in hits
            ],
            "related_files": [h.file_path for h in hits],
            "related_symbols": [h.symbol_name for h in hits if h.symbol_name],
            "latency": latency, "tokens": 0, "cost_usd": 0.0,
            "llm_available": False,
        }

    async def _full_repointel_search(self, repo_id: str, question: Dict[str, Any], facade) -> Dict[str, Any]:
        """SEARCH / ARCHITECTURE / DEPENDENCY categories via deterministic capabilities."""
        category = question["category"]
        query = question["question"]
        t0 = time.time()
        try:
            if category == "SEARCH":
                res = await facade.search_code(repo_id, query)
                payload = {
                    "answer": f"Search located {len(res.get('results', []))} evidence locations.",
                    "facts": [
                        f"{r.get('file_path')}:{r.get('line_start')}-{r.get('line_end')} ({r.get('symbol_name')})"
                        for r in res.get("results", [])[:8]
                    ],
                    "evidence": res.get("results", []),
                    "related_files": [r.get("file_path") for r in res.get("results", [])],
                    "related_symbols": [
                        r.get("symbol_name") for r in res.get("results", []) if r.get("symbol_name")
                    ],
                }
            elif category == "ARCHITECTURE":
                rep = await facade.generate_engineering_report(repo_id, "ENGINEERING_OVERVIEW")
                payload = {
                    "answer": rep.get("title", "Architecture overview"),
                    "facts": rep.get("facts", []),
                    "evidence": rep.get("evidence", []),
                }
            elif category == "DEPENDENCY":
                deps_res = await facade.get_dependency_manifest(repo_id)
                payload = {
                    "answer": f"Dependency manifest lists {len(deps_res)} packages.",
                    "facts": [f"{d['package_name']} ({d['ecosystem']})" for d in deps_res[:8]],
                    "evidence": [{"package_name": d["package_name"]} for d in deps_res[:10]],
                }
            else:
                raise ValueError(f"category {category} handled elsewhere")
        except Exception as exc:
            payload = {
                "answer": f"capability error: {type(exc).__name__}: {exc}",
                "facts": [], "evidence": [], "_error": f"{type(exc).__name__}: {exc}",
            }
        payload.setdefault("inferences", [])
        payload.setdefault("recommendations", [])
        payload.setdefault("related_files", [])
        payload.setdefault("related_symbols", [])
        payload["latency"] = time.time() - t0
        payload.setdefault("tokens", 0)
        payload.setdefault("cost_usd", 0.0)
        payload["llm_available"] = False
        return payload

    async def _full_repointel_history(self, repo_id: str, question: Dict[str, Any], facade) -> Dict[str, Any]:
        """HISTORY / TESTS / IMPACT / RISK categories via graph + git capabilities."""
        category = question["category"]
        target = _pick_target(question)
        t0 = time.time()
        try:
            if category == "HISTORY":
                history = await facade.get_symbol_history(repo_id, target, limit=20)
                payload = {
                    "answer": f"Symbol '{target}' has {len(history)} recorded changes.",
                    "facts": [
                        f"{(h.get('commit_sha') or '')[:10]} by {h.get('author')} "
                        f"({h.get('change_type')}) lines {h.get('line_start')}-{h.get('line_end')}"
                        for h in history[:8]
                    ],
                    "evidence": [
                        {"commit_sha": h.get("commit_sha"), "file_path": h.get("file_path")}
                        for h in history[:10]
                    ],
                    "related_commits": [h.get("commit_sha") for h in history],
                    "related_files": sorted({h.get("file_path") for h in history if h.get("file_path")}),
                    "related_symbols": [target] if history else [],
                }
            elif category == "TESTS":
                tests = await facade.get_tests_for_symbol(repo_id, target)
                payload = {
                    "answer": f"Found {len(tests)} tests related to '{target}'.",
                    "facts": [f"{t.get('test_name')} in {t.get('test_file')}" for t in tests[:8]],
                    "evidence": [
                        {"test_file": t.get("test_file"), "test_name": t.get("test_name")}
                        for t in tests[:10]
                    ],
                    "related_files": sorted({t.get("test_file") for t in tests if t.get("test_file")}),
                    "related_symbols": [target] if tests else [],
                }
            elif category == "IMPACT":
                impact = await facade.analyze_impact(repo_id, target)
                payload = {
                    "answer": (
                        f"Impact analysis for '{target}': "
                        f"{impact.get('total_dependents', 0)} direct dependents."
                    ),
                    "facts": [f"Affected files: {impact.get('affected_files', [])}"],
                    "evidence": [impact],
                    "related_files": impact.get("affected_files", []),
                    "related_symbols": impact.get("affected_symbols", []),
                }
            elif category == "RISK":
                hotspots = await facade.get_hotspots(repo_id)
                payload = {
                    "answer": f"Ranked {len(hotspots)} risk hotspots.",
                    "facts": [
                        f"{h.get('file_or_symbol')}: score {h.get('hotspot_score')}"
                        for h in hotspots[:8]
                    ],
                    "evidence": list(hotspots[:10]),
                    "related_files": [h.get("file_or_symbol") for h in hotspots[:5]],
                }
            else:
                raise ValueError(f"category {category} handled elsewhere")
        except Exception as exc:
            payload = {
                "answer": f"capability error: {type(exc).__name__}: {exc}",
                "facts": [], "evidence": [], "_error": f"{type(exc).__name__}: {exc}",
            }
        payload.setdefault("inferences", [])
        payload.setdefault("recommendations", [])
        payload.setdefault("related_symbols", [])
        payload["latency"] = time.time() - t0
        payload.setdefault("tokens", 0)
        payload.setdefault("cost_usd", 0.0)
        payload["llm_available"] = False
        return payload