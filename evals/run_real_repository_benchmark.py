"""Real-repository ingestion benchmark (production validation Objective 5).

Clones public repositories defined in ``evals/datasets/real_repositories.json``
(never committed into RepoIntel), ingests each through the real pipeline, and
records measured metrics only:

    clone_seconds, files, source_files, loc, symbols, chunks, dependencies,
    graph_nodes, graph_edges, commits, archetype, head_commit, branch,
    stage/total durations, memory_rss_mb (when psutil is available).

Any metric that cannot be measured is reported as ``null`` -- never guessed.
Repositories whose clone fails (e.g. no network access) are recorded as
BLOCKED with the exact reason and the command required.

Usage:
    python evals/run_real_repository_benchmark.py [--only id1,id2] [--all]
        [--list]

Output: evals/reports/real_repository_benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DATASETS = Path(_REPO_ROOT) / "evals" / "datasets" / "real_repositories.json"
CACHE_ROOT = Path(tempfile.gettempdir()) / "repointel_bench_cache"
SOURCE_EXT = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rb", ".go", ".rs", ".java", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".php",
}


def load_config() -> dict:
    return json.loads(DATASETS.read_text(encoding="utf-8"))


def count_loc(repo_dir: Path) -> int:
    """Count lines in source files (measured, best-effort)."""
    loc = 0
    for p in repo_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SOURCE_EXT:
            continue
        parts = set(p.parts)
        if parts & {".git", "node_modules", "__pycache__", ".venv", "venv",
                    "dist", "build"}:
            continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                loc += sum(1 for _ in fh)
        except OSError:
            continue
    return loc


def count_source_files(repo_dir: Path) -> int:
    n = 0
    for p in repo_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SOURCE_EXT:
            continue
        parts = set(p.parts)
        if parts & {".git", "node_modules", "__pycache__", ".venv", "venv",
                    "dist", "build"}:
            continue
        n += 1
    return n


def ensure_clone(entry: dict) -> tuple:
    """Clone (or reuse a cached clone); return (path, clone_seconds, reused)."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    target = CACHE_ROOT / entry["id"]
    depth = entry.get("clone_depth_default")
    if (target / ".git").exists():
        return target, 0.0, True
    args = ["clone"]
    if depth:
        args += ["--depth", str(depth)]
    args += [entry["url"], str(target)]
    t0 = time.perf_counter()
    proc = subprocess.run(["git"] + args, capture_output=True, text=True,
                          timeout=1800)
    seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"git clone failed ({proc.returncode}): {proc.stderr.strip()[:300]}"
        )
    return target, round(seconds, 2), False


async def measure_repo(entry: dict, workdir: Path) -> dict:
    """Clone + ingest one repository; collect only measurable facts."""
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from packages.knowledge import migrations
    from packages.knowledge.embeddings import create_embedding_service
    from packages.knowledge.models import (
        CodeChunkModel, CommitModel, DependencyModel, FileModel,
        SymbolEdgeModel, SymbolModel,
    )
    from packages.knowledge.vector_store import VectorStore
    from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline

    result: dict = {
        "id": entry["id"],
        "category": entry["category"],
        "category_label": entry["category_label"],
        "url": entry["url"],
        "language": entry.get("language"),
        "license": entry.get("license"),
    }
    try:
        repo_dir, clone_s, reused = ensure_clone(entry)
    except Exception as exc:  # network/auth/etc.
        result.update({
            "status": "BLOCKED",
            "blocked_reason": f"{type(exc).__name__}: {exc}",
            "command_required": f"git clone {entry['url']}",
        })
        return result

    result["clone_seconds"] = clone_s or None
    result["clone_reused"] = reused

    db_file = workdir / f"{entry['id']}.db"
    url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    os.environ["DATABASE_URL"] = url
    migrations.upgrade_to_head()
    engine = create_async_engine(url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession,
                                         expire_on_commit=False)

    process = None
    try:
        import psutil  # optional dependency

        process = psutil.Process()
    except Exception:
        pass

    embedding_service = create_embedding_service()
    vector_store = VectorStore(embedding_service=embedding_service)
    pipeline = RepositoryAnalysisPipeline(
        session_factory=session_factory, embedding_service=embedding_service
    ).attach_vector_store(vector_store)

    t0 = time.perf_counter()
    outcome = await pipeline.analyze(repo_url=str(repo_dir), mode="FULL")
    total_s = time.perf_counter() - t0

    memory_mb = None
    if process is not None:
        try:
            memory_mb = round(process.memory_info().rss / (1024 * 1024), 1)
        except Exception:
            memory_mb = None

    result["status"] = outcome.get("status")
    result["error"] = outcome.get("error")
    repo_meta = outcome.get("repository") or {}
    result["head_commit"] = repo_meta.get("head_commit")
    result["branch"] = repo_meta.get("branch")
    result["archetype"] = repo_meta.get("archetype") or None
    result["total_analysis_seconds"] = round(total_s, 3)
    result["memory_rss_mb_end"] = memory_mb
    stage_durations = outcome.get("stage_durations") or {}
    result["stage_durations"] = {
        name: meta.get("duration_seconds") for name, meta in stage_durations.items()
    }

    try:
        result["source_files_on_disk"] = count_source_files(repo_dir)
        result["loc_on_disk"] = count_loc(repo_dir)
    except Exception as exc:
        result["source_files_on_disk"] = None
        result["loc_on_disk"] = None
        result["loc_error"] = f"{type(exc).__name__}: {exc}"

    async with session_factory() as session:
        repo_id = outcome.get("repository_id")

        async def _count(model):
            return (
                await session.execute(
                    select(func.count()).select_from(model).where(model.repo_id == repo_id)
                )
            ).scalar_one()

        try:
            result["files_persisted"] = await _count(FileModel)
            result["symbols"] = await _count(SymbolModel)
            result["chunks"] = await _count(CodeChunkModel)
            result["dependencies"] = await _count(DependencyModel)
            result["graph_nodes"] = await _count(SymbolModel)
            result["graph_edges"] = await _count(SymbolEdgeModel)
            result["commits"] = await _count(CommitModel)
        except Exception as exc:
            result["count_error"] = f"{type(exc).__name__}: {exc}"

    result["embedding_telemetry"] = embedding_service.get_telemetry()
    await engine.dispose()
    return result


async def run_selected(ids: list, workdir: Path) -> list:
    config = load_config()
    entries = [e for e in config["repositories"] if not ids or e["id"] in ids]
    results = []
    for entry in entries:
        print(f"[real-repo-bench] measuring {entry['id']} ({entry['url']}) ...")
        t0 = time.perf_counter()
        try:
            r = await measure_repo(entry, workdir)
        except Exception as exc:  # never let one repo kill the whole run
            r = {
                "id": entry["id"], "url": entry["url"],
                "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            }
        r["wall_seconds"] = round(time.perf_counter() - t0, 2)
        print(
            f"[real-repo-bench] {entry['id']}: status={r.get('status')} "
            f"files={r.get('files_persisted')} symbols={r.get('symbols')} "
            f"chunks={r.get('chunks')} wall={r['wall_seconds']}s"
        )
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(description="Real repository ingestion benchmark")
    parser.add_argument("--only", default="", help="comma-separated repo ids to measure")
    parser.add_argument("--list", action="store_true", help="list configured repositories and exit")
    args = parser.parse_args()

    config = load_config()
    if args.list:
        for e in config["repositories"]:
            print(f"{e['category']}. {e['id']:24s} {e['url']} [{e.get('license')}]")
        return

    ids = [i.strip() for i in args.only.split(",") if i.strip()]
    with tempfile.TemporaryDirectory(prefix="repointel_realrepo_bench_") as tmp:
        results = asyncio.run(run_selected(ids, Path(tmp)))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "fabrication": "forbidden; unmeasurable metrics are null",
            "clone_cache": str(CACHE_ROOT),
            "repositories_never_committed": True,
        },
        "dataset": str(DATASETS),
        "configured_repositories": [
            {"id": e["id"], "url": e["url"], "category": e["category"]}
            for e in config["repositories"]
        ],
        "results": results,
    }
    out_dir = Path(_REPO_ROOT) / "evals" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "real_repository_benchmark.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {out_file}")


if __name__ == "__main__":
    main()
