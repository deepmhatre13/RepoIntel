import os
import sys
import json
import traceback
import asyncio

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline
from packages.knowledge.db import init_db, AsyncSessionLocal
from packages.knowledge.models import CodeChunkModel, RepositoryModel, PipelineRunModel, SymbolModel, FileModel
from sqlalchemy import select, func

async def run_validation():
    try:
        print("Initializing DB...")
        await init_db()

        repo_url = "https://github.com/deepmhatre13/Aiverse"
        print(f"Starting analysis for {repo_url}...")

        pipeline = RepositoryAnalysisPipeline()
        result = await pipeline.analyze(
            repo_url=repo_url,
            mode="FULL",
            force=True,
            generate_artifact=False
        )

        print("\n================== ANALYSIS OUTCOME ==================")
        print(f"run_id:             {result.get('run_id')}")
        print(f"repository_id:      {result.get('repository_id')}")
        print(f"status:             {result.get('status')}")
        print(f"error:              {result.get('error')}")
        print(f"duration_seconds:   {result.get('duration_seconds')}")
        print(f"artifact_counts:    {json.dumps(result.get('artifact_counts'), indent=2)}")

        repo_id = result.get('repository_id')
        async with AsyncSessionLocal() as session:
            file_res = await session.execute(select(func.count()).select_from(FileModel).where(FileModel.repo_id == repo_id))
            files_cnt = file_res.scalar() or 0

            sym_res = await session.execute(select(func.count()).select_from(SymbolModel).where(SymbolModel.repo_id == repo_id))
            sym_cnt = sym_res.scalar() or 0

            chunk_res = await session.execute(select(func.count()).select_from(CodeChunkModel).where(CodeChunkModel.repo_id == repo_id))
            chunk_cnt = chunk_res.scalar() or 0

            req_chunks_res = await session.execute(
                select(CodeChunkModel)
                .where(CodeChunkModel.repo_id == repo_id)
                .where(CodeChunkModel.file_path.like("%requirements.txt%"))
            )
            req_chunks = req_chunks_res.scalars().all()
            print(f"\nFound {len(req_chunks)} requirements.txt chunks in DB.")
            for idx, chk in enumerate(req_chunks, 1):
                has_nul = "\x00" in chk.content
                has_django = "Django==5.0.6" in chk.content or "Django" in chk.content
                print(f"  Chunk #{idx} [{chk.file_path}]: NUL present={has_nul}, Django present={has_django}, length={len(chk.content)}")
                assert not has_nul, f"CRITICAL FAILURE: NUL character found in persisted DB chunk for {chk.file_path}!"

            all_chunks_res = await session.execute(
                select(CodeChunkModel.file_path, CodeChunkModel.content)
                .where(CodeChunkModel.repo_id == repo_id)
            )
            all_rows = all_chunks_res.all()
            nul_violations = [fp for fp, cnt in all_rows if "\x00" in cnt]
            print(f"\nSanity check on all {len(all_rows)} DB chunks for NUL characters: {len(nul_violations)} violations found.")
            assert len(nul_violations) == 0, f"NUL characters found in DB chunks: {nul_violations}"

        print(f"\nSUMMARY METRICS:")
        print(f"  Total DB Files:   {files_cnt}")
        print(f"  Total DB Symbols: {sym_cnt}")
        print(f"  Total DB Chunks:  {chunk_cnt}")

        return result
    except Exception as exc:
        print("EXCEPTION ENCOUNTERED IN VALIDATION SCRIPT:")
        traceback.print_exc()
        return {"status": "FAILED", "error": str(exc)}

if __name__ == "__main__":
    res = asyncio.run(run_validation())
    if res.get("status") in ("READY", "PARTIAL"):
        print("\nAIVERSE INGESTION VALIDATION SUCCESSFUL!")
    else:
        print(f"\nAIVERSE INGESTION FAILED: {res.get('error')}")
        sys.exit(1)
