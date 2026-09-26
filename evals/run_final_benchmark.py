"""Final release benchmark (production validation Objective 27).

Runs one repository through the full lifecycle against a fresh migrated
database and records only measured values:

    repository, commit, files, symbols, chunks, graph_nodes, graph_edges,
    commits, indexing_seconds, incremental_seconds, query_latency,
    embedding_seconds, llm_calls, token_usage, retrieval_metrics,
    evidence_metrics, errors.

Unmeasurable metrics are ``null``. When GOOGLE_API_KEY is configured the LLM
usage numbers come from the real Gemini calls made through LLMClient;
otherwise they are null with llm_available=false (degraded mode is explicit).

Usage:
    python evals/run_final_benchmark.py [--repo PATH]

Output: evals/reports/final_benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_REPO = Path(_REPO_ROOT) / "evals" / "datasets" / "python_library"


async def _count_all(session_factory, model, repo_id) -> int:
    from sqlalchemy import func, select

    async with session_factory() as session:
        return (
            await session.execute(
                select(func.count()).select_from(model).where(model.repo_id == repo_id)
            )
        ).scalar_one()


async def run_final_benchmark(repo_path: Path) -> dict:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from packages.agentic.llm_client import LLMClient
    from packages.knowledge import migrations
    from packages.knowledge.embeddings import create_embedding_service
    from packages.knowledge.models import (
        CodeChunkModel, CommitModel, DependencyModel, FileModel,
        SymbolEdgeModel, SymbolModel,
    )
    from packages.knowledge.vector_store import VectorStore
    from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline

    errors: list = []
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo_path),
        "fabrication_policy": "unmeasurable -> null",
    }

    with tempfile.TemporaryDirectory(prefix="repointel_final_bench_") as tmp:
        workdir = Path(tmp)
        db_file = workdir / "final.db"
        url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
        os.environ["DATABASE_URL"] = url
        t_mig0 = time.perf_counter()
        migrations.upgrade_to_head()
        migration_s = time.perf_counter() - t_mig0
        engine = create_async_engine(url)
        session_factory = async_sessionmaker(engine, class_=AsyncSession,
                                             expire_on_commit=False)

        llm = LLMClient()
        embedding_service = create_embedding_service()

        # --- initial FULL index ------------------------------------------
        vector_store = VectorStore(embedding_service=embedding_service)
        pipeline = RepositoryAnalysisPipeline(
            session_factory=session_factory, embedding_service=embedding_service
        ).attach_vector_store(vector_store)

        t_idx0 = time.perf_counter()
        full = await pipeline.analyze(repo_url=str(repo_path), mode="FULL")
        indexing_seconds = time.perf_counter() - t_idx0
        if full.get("status") != "READY":
            errors.append(f"FULL analyze failed: {full.get('error')}")
        repo_id = full.get("repository_id")

        report["commit"] = (full.get("repository") or {}).get("head_commit")
        report["branch"] = (full.get("repository") or {}).get("branch")
        report["indexing_seconds"] = round(indexing_seconds, 3)
        report["migration_seconds"] = round(migration_s, 3)

        stage_durations = full.get("stage_durations") or {}
        report["stage_durations"] = {
            name: meta.get("duration_seconds") for name, meta in stage_durations.items()
        }
        embedding_stage = stage_durations.get("embedding") or {}
        report["embedding_seconds"] = embedding_stage.get("duration_seconds")

        # --- entity counts -------------------------------------------------
        if repo_id:
            report["files"] = await _count_all(session_factory, FileModel, repo_id)
            report["symbols"] = await _count_all(session_factory, SymbolModel, repo_id)
            report["chunks"] = await _count_all(session_factory, CodeChunkModel, repo_id)
            report["graph_nodes"] = report["symbols"]
            report["graph_edges"] = await _count_all(session_factory, SymbolEdgeModel, repo_id)
            report["commits"] = await _count_all(session_factory, CommitModel, repo_id)
            report["dependencies"] = await _count_all(session_factory, DependencyModel, repo_id)
        else:
            for key in ("files", "symbols", "chunks", "graph_nodes",
                        "graph_edges", "commits", "dependencies"):
                report[key] = None

        # --- incremental update -------------------------------------------
        target_file = None
        for p in sorted(repo_path.rglob("*.py")):
            if "__pycache__" not in p.parts:
                target_file = p
                break
        incremental_seconds = None
        if target_file is not None and full.get("status") == "READY":
            original = target_file.read_text(encoding="utf-8")
            try:
                target_file.write_text(
                    original + "\n\ndef _benchmark_probe_function():\n    return 2718\n",
                    encoding="utf-8",
                )
                t_inc0 = time.perf_counter()
                inc = await pipeline.analyze(repo_url=str(repo_path), mode="INCREMENTAL")
                incremental_seconds = time.perf_counter() - t_inc0
                report["incremental_status"] = inc.get("status")
                inc_report = inc.get("incremental_report") or {}
                report["incremental_files_modified"] = len(inc_report.get("files_modified") or [])
            finally:
                target_file.write_text(original, encoding="utf-8")
        report["incremental_seconds"] = round(incremental_seconds, 3) if incremental_seconds else None

        # --- query latency (hybrid retrieval) ------------------------------
        latencies = []
        if vector_store._repo_chunks(repo_id):
            probes = [
                "where is the main entry point implemented",
                "database connection handling",
                "configuration loading and environment variables",
                "error handling and logging utilities",
            ]
            for q in probes:
                try:
                    t_q0 = time.perf_counter()
                    vector_store.hybrid_search(repo_id, q, top_k=5)
                    latencies.append((time.perf_counter() - t_q0) * 1000)
                except Exception as exc:
                    errors.append(f"query failed: {type(exc).__name__}: {exc}")
        report["query_latency_mean_ms"] = (
            round(statistics.fmean(latencies), 2) if latencies else None
        )
        report["query_latency_samples"] = len(latencies)

        # --- retrieval metrics (measured by the dedicated benchmark) -------
        rb_path = Path(_REPO_ROOT) / "evals" / "reports" / "retrieval_benchmark.json"
        if rb_path.exists():
            try:
                rb = json.loads(rb_path.read_text(encoding="utf-8"))
                report["retrieval_metrics"] = {
                    "source": "evals/reports/retrieval_benchmark.json",
                    "generated_at": rb.get("generated_at"),
                    "methods": rb.get("methods"),
                    "ground_truth_policy": rb.get("ground_truth_policy"),
                }
            except Exception as exc:
                report["retrieval_metrics"] = {"error": str(exc)}
        else:
            report["retrieval_metrics"] = None

        # --- LLM usage (real Gemini only when configured) -------------------
        usage = llm.get_usage_summary()
        report["llm_calls"] = usage["calls"] if llm.is_available() else None
        report["token_usage"] = (
            {"input_tokens": usage["input_tokens"],
             "output_tokens": usage["output_tokens"],
             "total_tokens": usage["total_tokens"],
             "estimated_cost_usd": usage["estimated_cost_usd"]}
            if llm.is_available() else None
        )
        report["llm_available"] = llm.is_available()
        report["llm_mode"] = "GEMINI_LIVE" if llm.is_available() else "DEGRADED_DETERMINISTIC"
        report["secrets_redacted_by_llm_client"] = usage["secrets_redacted"]

        report["errors"] = errors or []
        await engine.dispose()

    out_dir = Path(_REPO_ROOT) / "evals" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "stage_durations"}, indent=2))
    print("report -> evals/reports/final_benchmark.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="RepoIntel final release benchmark")
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    args = parser.parse_args()
    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        raise SystemExit(f"repository not found: {repo_path}")
    asyncio.run(run_final_benchmark(repo_path))


if __name__ == "__main__":
    main()
