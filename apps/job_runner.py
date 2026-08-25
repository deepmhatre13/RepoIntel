import uuid
from typing import Dict, Any, Optional

from packages.knowledge.db import init_db, AsyncSessionLocal
from packages.knowledge.vector_store import VectorStore
from packages.knowledge.embeddings import EmbeddingService
from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline
from packages.runtime.recovery import ExecutionResilienceManager


class JobRunner:
    """Job architecture (Phase 15) now driven by the production pipeline
    orchestrator (Phases 40-43): lifecycle tracking, incremental indexing,
    resource budgets, cancellation and resilient retries."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.resilience = ExecutionResilienceManager()

    def create_job(self, repo_url: str) -> str:
        run_id = str(uuid.uuid4())
        self.jobs[run_id] = {
            "run_id": run_id,
            "repo_url": repo_url,
            "status": "PENDING",
            "result": None,
            "error": None,
            "stage": None,
            "progress": 0.0,
        }
        return run_id

    async def execute_job(
        self,
        run_id: str,
        mode: str = "FULL",
        force: bool = False,
        generate_artifact: bool = True,
    ):
        if run_id not in self.jobs:
            return
        job = self.jobs[run_id]
        job["status"] = "RUNNING"
        try:
            await init_db()
            embedding_service = EmbeddingService()
            vector_store = VectorStore(embedding_service)

            pipeline = RepositoryAnalysisPipeline(
                resilience=self.resilience,
                session_factory=AsyncSessionLocal,
            ).attach_vector_store(vector_store)

            outcome = await pipeline.analyze(
                repo_url=job["repo_url"],
                run_id=run_id,
                mode=mode,
                force=force,
                generate_artifact=generate_artifact,
            )
            job["status"] = outcome["status"]
            job["result"] = outcome
            job["error"] = outcome.get("error")
            job["stage"] = None
            job["progress"] = 100.0
        except Exception as e:  # defensive: jobs must always reach a terminal status
            job["status"] = "FAILED"
            job["error"] = f"{type(e).__name__}: {e}"

    # ---------------- Cancellation (Phase 43.5) ----------------

    def cancel_job(self, run_id: str, reason: str = "") -> bool:
        if run_id not in self.jobs:
            return False
        self.resilience.cancel_job(run_id, reason=reason)
        return True

    def get_job_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(run_id)
        if job is None:
            return None
        return {
            "run_id": run_id,
            "status": job["status"],
            "error": job["error"],
            "stage": job.get("stage"),
            "progress": job.get("progress"),
        }


job_runner = JobRunner()