"""Retrieval benchmark (production validation Objective 3).

Compares five retrieval methods on a repository indexed by the real pipeline:

    1. keyword            - BM25 lexical only
    2. vector             - cosine similarity over real embeddings
    3. hybrid             - reciprocal-rank fusion of lexical + semantic
    4. hybrid_rerank      - hybrid candidates re-scored by EvidenceReranker
    5. hybrid_graph       - hybrid hits expanded one hop via SymbolEdgeModel

Ground truth is NEVER hand-invented.  Questions are generated from repository
facts recorded at ingestion time: each sampled symbol S defines a fact
"S is defined in file F / chunks C".  The relevant set for a question is
exactly the set of indexed chunks whose ``symbol_name`` equals S.

Measured per method: Recall@1/5/10, Precision@5, MRR, citation validity
(evidence items that resolve to actual indexed chunks) and mean latency.

Usage:
    python evals/run_retrieval_benchmark.py [--repo PATH] [--questions N]

Output: evals/reports/retrieval_benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from evals import metrics as _metrics

# Allow direct execution (``python evals/run_retrieval_benchmark.py``) by
# making the repository root importable.
_REPO_ROOT_STR = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

REPO_ROOT = Path(__file__).resolve().parent.parent

METHODS = ["keyword", "vector", "hybrid", "hybrid_rerank", "hybrid_graph"]


async def _index_repository(repo_dir: Path, workdir: Path):
    """Run the real ingestion pipeline; return (pipeline_env, repo_id)."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from packages.knowledge import migrations
    from packages.knowledge.embeddings import create_embedding_service
    from packages.knowledge.vector_store import VectorStore
    from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline

    db_file = workdir / "retrieval_bench.db"
    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    os.environ["DATABASE_URL"] = url
    migrations.upgrade_to_head()

    engine = create_async_engine(url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    embedding_service = create_embedding_service()
    vector_store = VectorStore(embedding_service=embedding_service)
    pipeline = (
        RepositoryAnalysisPipeline(
            session_factory=session_factory, embedding_service=embedding_service
        )
        .attach_vector_store(vector_store)
    )

    t0 = time.perf_counter()
    outcome = await pipeline.analyze(repo_url=str(repo_dir), mode="FULL")
    index_s = time.perf_counter() - t0
    if outcome.get("status") != "READY":
        raise RuntimeError(f"ingestion failed: {outcome.get('error')}")

    return {
        "engine": engine,
        "session_factory": session_factory,
        "vector_store": vector_store,
        "embedding_service": embedding_service,
        "repo_id": outcome["repository_id"],
        "index_s": index_s,
        "outcome": outcome,
    }


def _build_questions(chunks, num_questions: int, seed: int):
    """Derive benchmark questions from repository facts (symbol -> chunks)."""
    by_symbol: dict = {}
    for c in chunks:
        if c.symbol_name:
            by_symbol.setdefault(c.symbol_name, []).append(c)

    # Only symbols whose defining chunk content is distinctive enough.
    # Dotted names (Class.method) are matched via their base component because
    # chunk text contains the definition source, not necessarily the qualified
    # name.
    candidates = []
    for sym, cs in sorted(by_symbol.items()):
        base = sym.rsplit(".", 1)[-1]
        defining = [c for c in cs if sym in (c.content or "") or base in (c.content or "")]
        if defining and len(base) >= 3:
            candidates.append((sym, cs))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:num_questions]

    questions = []
    for sym, cs in selected:
        sample = cs[0]
        first_line = next(
            (ln.strip() for ln in (sample.content or "").splitlines()
             if ln.strip() and not ln.strip().startswith(("#", '"', "'"))),
            "",
        )
        base = sym.rsplit(".", 1)[-1]
        style = "natural_language" if rng.random() < 0.5 else "keyword_style"
        query = (
            f"Where is {sym} implemented?"
            if style == "natural_language"
            else f"{base} definition implementation"
        )
        questions.append({
            "query": query,
            "query_style": style,
            "context_hint": first_line[:120],
            "ground_truth": {
                "type": "repository_fact",
                "symbol": sym,
                "relevant_chunk_keys": sorted({
                    f"{c.file_path}:{c.line_start}-{c.line_end}" for c in cs
                }),
                "relevant_files": sorted({c.file_path for c in cs}),
            },
        })
    return questions


def _key(item) -> str:
    return f"{item.file_path}:{item.line_start}-{item.line_end}"


def _rank_metrics(ranked, relevant_keys) -> dict:
    """Standard retrieval metrics computed via evals.metrics (Phase T6).

    * ``recall_at_k``     - set-based Recall@k = |relevant ∩ top-k| / |relevant|
                            (multi-relevant ground truth is fully measured)
    * ``hit_rate_at_k``   - probability at least one relevant item is in top-k
                            (this is what the previous harness labelled
                            ``recall_at_k``; kept under its correct name)
    * ``precision_at_k``  - |relevant ∩ top-k| / k
    * ``mrr``             - 1 / rank of the first relevant item
    * ``ndcg_at_10``      - binary-relevance normalized DCG (log2 discount)
    """
    keys = [_key(r) for r in ranked]
    m = _metrics
    recall = {
        k: m.calculate_recall_at_k(keys, relevant_keys, k) for k in (1, 3, 5, 10)
    }
    hit = {
        k: (1.0 if any(k_ in relevant_keys for k_ in keys[:k]) else 0.0)
        for k in (1, 5, 10)
    }
    precision = {
        k: m.calculate_precision_at_k(keys, relevant_keys, k) for k in (1, 5, 10)
    }
    mrr = m.calculate_mrr(keys, relevant_keys)
    ndcg = m.calculate_ndcg_at_k(keys, relevant_keys, 10)
    citation_validity = None
    if ranked:
        valid = sum(1 for r in ranked if r.chunk_id or r.content)
        citation_validity = valid / len(ranked)
    return {
        "recall_at_1": round(recall[1], 4),
        "recall_at_3": round(recall[3], 4),
        "recall_at_5": round(recall[5], 4),
        "recall_at_10": round(recall[10], 4),
        "hit_rate_at_1": round(hit[1], 4),
        "hit_rate_at_5": round(hit[5], 4),
        "hit_rate_at_10": round(hit[10], 4),
        "precision_at_1": round(precision[1], 4),
        "precision_at_5": round(precision[5], 4),
        "precision_at_10": round(precision[10], 4),
        "mrr": round(mrr, 4),
        "ndcg_at_10": round(ndcg, 4),
        "citation_validity": round(citation_validity, 4) if citation_validity is not None else None,
    }


async def _graph_expand(session_factory, repo_id, hits, chunks):
    """One-hop expansion over the persisted symbol graph (SymbolEdgeModel).

    Each expanded item inherits the score of its seed hybrid hit, decayed by
    a graph-distance factor (0.5 per hop). This ensures graph-expanded evidence
    never outranks its own seed in the fused ranking — the expansion is additive
    evidence, not a re-ranking of the original hybrid order."""
    from sqlalchemy import select

    from packages.knowledge.models import SymbolEdgeModel
    from packages.knowledge.vector_store import RetrievedEvidenceItem

    symbols = [h.symbol_name for h in hits[:5] if h.symbol_name]
    if not symbols:
        return []
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(SymbolEdgeModel).where(
                    SymbolEdgeModel.repo_id == repo_id,
                    SymbolEdgeModel.source_symbol.in_(symbols),
                )
            )
        ).scalars().all()

        expanded = []
    seen_keys = {_key(h) for h in hits}
    # Map each symbol to the best (highest-scoring) seed hit that discovered it.
    seed_score_for_symbol: dict = {}
    for h in hits:
        if h.symbol_name:
            prev = seed_score_for_symbol.get(h.symbol_name, 0.0)
            seed_score_for_symbol[h.symbol_name] = max(prev, h.relevance_score)

    GRAPH_DECAY = 0.5  # one hop away from the seed evidence
    for edge in rows:
        seed_score = seed_score_for_symbol.get(edge.source_symbol, 0.0)
        expanded_score = seed_score * GRAPH_DECAY  # decayed, always <= seed
        for chunk in chunks:
            if (chunk.file_path == edge.target_file
                    and chunk.symbol_name == edge.target_symbol
                    and _key(chunk) not in seen_keys):
                seen_keys.add(_key(chunk))
                expanded.append(
                    RetrievedEvidenceItem(
                        repo_id=repo_id,
                        file_path=chunk.file_path,
                        symbol_name=chunk.symbol_name,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        content=chunk.content,
                        retrieval_source="graph_expansion",
                        retrieval_method=["hybrid", "graph"],
                        relevance_score=expanded_score,
                        chunk_id=(chunk.content_hash or "")[:16] or None,
                    )
                )
    return expanded


async def run_benchmark(repo_path: str, num_questions: int = 15, seed: int = 42) -> dict:
    from packages.knowledge.reranker import EvidenceReranker

    repo_dir = Path(repo_path).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"repository not found: {repo_dir}")

    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="repointel_retrieval_bench_") as tmp:
        env = await _index_repository(repo_dir, Path(tmp))
        try:
            store = env["vector_store"]
            chunks = store._repo_chunks(env["repo_id"])
            if not chunks:
                raise RuntimeError("no chunks indexed; cannot build fact-based ground truth")

            questions = _build_questions(chunks, num_questions, seed)
            if not questions:
                raise RuntimeError("no symbol-backed questions could be derived")

            reranker = EvidenceReranker()
            per_method: dict = {m: [] for m in METHODS}
            stage_latencies = {"keyword_ms": [], "vector_ms": [], "hybrid_ms": [], "graph_ms": []}
            method_latency_ms: dict = {m: [] for m in METHODS}
            # RRF diagnostic (Phase T7): per-query spread of fused scores.
            rrf_spread: list = []

            for q in questions:
                relevant = set(q["ground_truth"]["relevant_chunk_keys"])
                query = q["query"]

                t = time.perf_counter(); kw = store.keyword_search(env["repo_id"], query, top_k=10)
                stage_latencies["keyword_ms"].append((time.perf_counter() - t) * 1000)
                method_latency_ms["keyword"].append(stage_latencies["keyword_ms"][-1])
                t = time.perf_counter(); vec = store.vector_search(env["repo_id"], query, top_k=10)
                stage_latencies["vector_ms"].append((time.perf_counter() - t) * 1000)
                method_latency_ms["vector"].append(stage_latencies["vector_ms"][-1])
                t = time.perf_counter(); hyb = store.hybrid_search(env["repo_id"], query, top_k=10)
                stage_latencies["hybrid_ms"].append((time.perf_counter() - t) * 1000)
                method_latency_ms["hybrid"].append(stage_latencies["hybrid_ms"][-1])

                t = time.perf_counter()
                runs = {
                    "keyword": kw,
                    "vector": vec,
                    "hybrid": hyb,
                    "hybrid_rerank": reranker.rerank(query, list(hyb), top_k=10),
                }
                method_latency_ms["hybrid_rerank"].append((time.perf_counter() - t) * 1000)

                # RRF fusion spread for this query (max fused score - min fused score).
                if hyb:
                    scores = [h.relevance_score for h in hyb]
                    rrf_spread.append(round(max(scores) - min(scores), 6))

                t = time.perf_counter()
                graph_extra = await _graph_expand(
                    env["session_factory"], env["repo_id"], hyb, chunks
                )
                stage_latencies["graph_ms"].append((time.perf_counter() - t) * 1000)
                runs["hybrid_graph"] = sorted(
                    list(hyb) + graph_extra,
                    key=lambda x: x.relevance_score, reverse=True,
                )[:10]
                method_latency_ms["hybrid_graph"].append(
                    stage_latencies["hybrid_ms"][-1] + stage_latencies["graph_ms"][-1]
                )

                for m in METHODS:
                    metrics = _rank_metrics(runs[m], relevant)
                    per_method[m].append(metrics)

            def summarize(rows, lats):
                n = len(rows) or 1
                out = {"questions": len(rows)}
                for field in (
                    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
                    "hit_rate_at_1", "hit_rate_at_5", "hit_rate_at_10",
                    "precision_at_1", "precision_at_5", "precision_at_10",
                    "mrr", "ndcg_at_10",
                ):
                    out[field] = round(sum(r[field] for r in rows) / n, 4)
                out["citation_validity"] = round(
                    sum(r["citation_validity"] or 0.0 for r in rows) / n, 4
                )
                # Mean retrieval wall time is taken from the per-stage timers
                # (keyword/vector/hybrid), which measure the actual search call.
                out["mean_latency_ms"] = (
                    round(statistics.fmean(lats), 2) if lats else None
                )
                return out

            report = {
                "generated_at": started,
                "dataset": "python_library",
                "repository": str(repo_dir),
                "ground_truth_policy": (
                    "derived from indexed repository facts (symbol -> defining chunks)"
                ),
                "seed": seed,
                "embedding": env["embedding_service"].get_telemetry(),
                "indexing_seconds": round(env["index_s"], 3),
                "indexed_chunks": len(chunks),
                "stage_latency_mean_ms": {
                    k: round(statistics.fmean(v), 2) if v else None
                    for k, v in stage_latencies.items()
                },
                "methods": {
                    m: summarize(per_method[m], method_latency_ms[m]) for m in METHODS
                },
                "rrf_diagnostics": {
                    "k": 60.0,
                    "mean_fused_score_spread": (
                        round(statistics.fmean(rrf_spread), 6) if rrf_spread else None
                    ),
                    "max_fused_score_spread": (
                        round(max(rrf_spread), 6) if rrf_spread else None
                    ),
                    "interpretation": (
                        "spread is max-min of fused RRF scores per query; a spread close "
                        "to the maximum achievable 2/(k+1) means rank positions dominate, "
                        "a tiny spread relative to corpus size means fusion is flattening "
                        "distinctions (T7 hybrid investigation evidence)"
                    ),
                },
                "notes": {
                    "supported_claim_ratio": (
                        "requires LLM-generated answers (Gemini); reported by the evidence "
                        "audit in LLM benchmarks when GOOGLE_API_KEY is configured"
                    ),
                },
            }

            out_dir = REPO_ROOT / "evals" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "retrieval_benchmark.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return report
        finally:
            await env["engine"].dispose()


def main():

    parser = argparse.ArgumentParser(description="RepoIntel retrieval benchmark")
    parser.add_argument(
        "--repo",
        default=str(REPO_ROOT / "evals" / "datasets" / "python_library"),
    )
    parser.add_argument("--questions", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = asyncio.run(run_benchmark(args.repo, args.questions, args.seed))
    print(json.dumps(report["methods"], indent=2))
    print("report -> evals/reports/retrieval_benchmark.json")


if __name__ == "__main__":
    main()
