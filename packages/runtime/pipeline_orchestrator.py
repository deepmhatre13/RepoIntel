"""Repository Analysis Pipeline Orchestrator (Phases 40-43).

Single production path for repository analysis used by the CLI, the job runner,
the API and the benchmark. Composes, around the existing knowledge layer:

    Repository Lifecycle (Phase 40)      -> packages.runtime.lifecycle
    Incremental Indexing (Phase 41)      -> packages.runtime.incremental_indexer
    Resource Budgets (Phase 42)          -> packages.runtime.resource_manager
    Failure Recovery (Phase 43)          -> packages.runtime.recovery

It reuses Phases 1-39 implementations unchanged (ingestion, parsing,
persistence, git history, test intelligence, LangGraph artifact generation).
RepositoryFacade is NOT modified; this layer operates beside it.

Security boundary (analysis-only): repositories are cloned/filtered/parsed.
No repository code is ever executed. setup.py / package scripts / CI configs /
Makefiles are treated as data only.
"""

import os
import json
import time
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from sqlalchemy import select

from packages.shared import IngestionManifest
from packages.shared.telemetry import telemetry
from packages.ingestion.cloning import clone_repository, cleanup_repository
from packages.ingestion.filtering import filter_repository
from packages.ingestion.metadata import extract_metadata, detect_archetype
from packages.ingestion.git_history import parse_git_commits, persist_git_commits
from packages.parsing.pipeline import run_parsing_pipeline
from packages.knowledge.models import (
    RepositoryModel, PipelineRunModel, TestCaseModel, TestToCodeRelationModel,
)
from packages.knowledge.db import AsyncSessionLocal
from packages.knowledge.persistence import (
    persist_parsing_results, count_repo_artifacts,
)
from packages.runtime.lifecycle import (
    RepositoryLifecycleManager, RepositoryLifecycleState, LifecycleStage,
    FailureDetail, PipelineRunRecord,
)
from packages.runtime.resource_manager import (
    ResourceManager, ResourceBudgetConfig,
)
from packages.runtime.recovery import (
    ExecutionResilienceManager, CancellationToken, CancellationRequested,
)
from packages.runtime.incremental_indexer import (
    IncrementalIndexer, IncrementalRunReport,
)

logger = logging.getLogger(__name__)


class RepositoryAnalysisPipeline:
    def __init__(
        self,
        resource_config: Optional[ResourceBudgetConfig] = None,
        resilience: Optional[ExecutionResilienceManager] = None,
        lifecycle_manager: Optional[RepositoryLifecycleManager] = None,
        session_factory=None,
        embedding_service=None,
    ):
        self.resources = ResourceManager(resource_config or ResourceBudgetConfig())
        self.resilience = resilience or ExecutionResilienceManager()
        self.lifecycle = lifecycle_manager or RepositoryLifecycleManager()
        self.incremental = IncrementalIndexer()
        self.session_factory = session_factory or AsyncSessionLocal
        self.embedding_service = embedding_service

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    async def _resolve_and_ingest(
        self, run_id: str, repo_url: str, cancel_token: CancellationToken,
    ):
        """CLONING + INGESTION stages. Returns (local_path, is_temp_clone, head, branch, manifest, admitted_paths, skipped)."""
        # ---- CLONE / LOAD ----
        if os.path.isdir(repo_url):
            local_path = repo_url
            is_temp_clone = False
        else:
            is_temp_clone = True
            local_path = await asyncio.wait_for(
                asyncio.to_thread(clone_repository, repo_url),
                timeout=self.resources.timeout_for_stage("clone"),
            )
        cancel_token.check()

        # ---- INGEST (filter + metadata + resource admission) ----
        started = self.lifecycle.start_stage(run_id, LifecycleStage.INGESTION)
        await self._checkpoint(run_id)
        filtered_files = filter_repository(local_path)
        languages, manifest_files, entrypoints, ci_configs = extract_metadata(local_path, filtered_files)
        archetype = detect_archetype(local_path, filtered_files, manifest_files)

        files_with_sizes: Dict[str, int] = {}
        total_bytes = 0
        for rel in filtered_files:
            fp = os.path.join(local_path, rel)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                files_with_sizes[rel] = size
                total_bytes += size

        admitted, skipped, budget_warnings = self.resources.admit_files(files_with_sizes)
        admitted_paths = sorted(admitted.keys())
        for w in budget_warnings:
            self.lifecycle.add_warning(run_id, w, stage=LifecycleStage.INGESTION)

        manifest = IngestionManifest(
            repo_url=repo_url,
            local_path=local_path,
            size_mb=total_bytes / (1024 * 1024),
            file_count=len(admitted_paths),
            languages=languages,
            manifest_files=manifest_files,
            entrypoints=entrypoints,
            ci_configs=ci_configs,
            archetype=archetype,
        )
        head_commit, branch = self.incremental.git_head(local_path)
        self.lifecycle.complete_stage(
            run_id, LifecycleStage.INGESTION, started,
            counts={"files_seen": len(filtered_files), "files_admitted": len(admitted_paths)},
        )
        await self._checkpoint(run_id)
        self.lifecycle.record_counts(run_id, files_seen=len(filtered_files))
        await self._checkpoint(run_id)
        telemetry.record_stage_duration("ingestion", time.monotonic() - started)
        return local_path, is_temp_clone, head_commit, branch, manifest, admitted_paths, skipped

    async def analyze(
        self,
        repo_url: str,
        repo_id: Optional[str] = None,
        mode: str = "FULL",            # FULL | INCREMENTAL
        force: bool = False,           # rebuild indexes even if unchanged
        generate_artifact: bool = False,
        artifact_type: str = "README",
        max_commits: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full lifecycle. Per-file parse failures do not raise; they are
        recorded and the repository is marked PARTIAL. Raises only for hard failures."""
        run_id = run_id or str(uuid.uuid4())
        repo_id = repo_id or str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
        cancel_token = self.resilience.register_run(run_id)
        self.resources.start_deadline()
        t0 = time.monotonic()
        local_path: Optional[str] = None
        is_temp_clone = False
        incremental_report: Optional[IncrementalRunReport] = None

        # Persist the CREATED run immediately (Phase 40.3): a run that has been
        # accepted must be observable in the database even if the worker dies
        # before ingestion begins.  In-memory lifecycle state is a cache of the
        # durable ``pipeline_runs`` row, never the other way around.
        run = self.lifecycle.create_run(run_id=run_id, repo_id=repo_id, mode=mode)
        await self._persist_run_record(run)

        try:
            local_path, is_temp_clone, head_commit, branch, manifest, admitted_paths, skipped = (
                await self._resolve_and_ingest(run_id, repo_url, cancel_token)
            )
            cancel_token.check()
            self.resources.check_deadline("ingestion")

            reprocess_files, deleted_files, incremental_report = await self._plan_scope(
                run_id, repo_id, local_path, admitted_paths, mode, force
            )

            # ---------------- PARSING ----------------
            started = self.lifecycle.start_stage(run_id, LifecycleStage.PARSING)
            await self._checkpoint(run_id)
            parse_res = await asyncio.wait_for(
                asyncio.to_thread(run_parsing_pipeline, repo_id, local_path, reprocess_files),
                timeout=self.resources.timeout_for_stage("parse"),
            )
            parse_failures = [
                FailureDetail(
                    stage=LifecycleStage.PARSING.value,
                    file=f.get("file"),
                    error_type=f.get("error_type", "Unknown"),
                    message=f.get("message", ""),
                    recoverable=bool(f.get("recoverable", True)),
                )
                for f in getattr(parse_res, "failures", [])
            ]
            self.lifecycle.complete_stage(
                run_id, LifecycleStage.PARSING, started,
                counts={
                    "files_attempted": len(reprocess_files),
                    "files_parsed": parse_res.files_parsed,
                    "files_failed": len(parse_failures),
                },
                errors=parse_failures,
            )
            await self._checkpoint(run_id)
            self.lifecycle.record_counts(
                run_id,
                files_parsed=parse_res.files_parsed,
                files_failed=len(parse_failures),
                symbols_extracted=len(parse_res.symbols),
                chunks_created=len(parse_res.chunks),
            )
            await self._checkpoint(run_id)

            # -------- INDEXING: KNOWLEDGE / EMBEDDING / GRAPH / GIT --------
            started_index = self.lifecycle.start_stage(run_id, LifecycleStage.KNOWLEDGE)
            await self._checkpoint(run_id)
            incremental_files = (
                None if incremental_report is None
                else sorted(set(reprocess_files) | set(deleted_files))
            )

            async def _knowledge_progress(processed: int, total: int) -> None:
                if total:
                    run.progress_percentage = round(55.0 + (20.0 * processed / total), 1)
                self.lifecycle.record_counts(run_id, chunks_created=len(parse_res.chunks))
                await self._checkpoint(run_id)

            async with self.session_factory() as session:
                await persist_parsing_results(
                    session, manifest, parse_res,
                    incremental_files=incremental_files,
                    head_commit=head_commit or None,
                    branch=branch or None,
                    embedding_service=getattr(self, "embedding_service", None),
                    embedding_model=(
                        self.embedding_service.model_name
                        if getattr(self, "embedding_service", None) is not None
                        else None
                    ),
                    embedding_provider=(
                        getattr(self.embedding_service, "provider_name", None)
                        if getattr(self, "embedding_service", None) is not None
                        else None
                    ),
                    progress_callback=_knowledge_progress,
                )

                started_git = self.lifecycle.start_stage(run_id, LifecycleStage.GIT)
                await self._checkpoint(run_id)
                commits = []
                if os.path.isdir(os.path.join(local_path, ".git")):
                    commits = await asyncio.to_thread(
                        parse_git_commits, local_path,
                        max_commits or self.resources.config.max_history_commits,
                    )
                    if commits:
                        await persist_git_commits(
                            session, repo_id, commits, parse_res.symbols,
                            repo_path=local_path,
                            max_history_commits=self.resources.config.max_history_commits,
                        )
                self.lifecycle.complete_stage(
                    run_id, LifecycleStage.GIT, started_git,
                    counts={"commits_ingested": len(commits)},
                )
                await self._checkpoint(run_id)

                # TESTS (full refresh per run -> idempotent)
                from packages.parsing.test_parser import (
                    discover_tests_in_repo, map_tests_to_production_symbols,
                )
                test_scope = incremental_files if incremental_files is not None else admitted_paths
                test_cases = discover_tests_in_repo(local_path, test_scope)
                test_mappings = map_tests_to_production_symbols(test_cases, parse_res.symbols)
                if incremental_files is None:
                    await session.execute(
                        TestToCodeRelationModel.__table__.delete().where(
                            TestToCodeRelationModel.repo_id == repo_id)
                    )
                    await session.execute(
                        TestCaseModel.__table__.delete().where(TestCaseModel.repo_id == repo_id)
                    )
                for tc in test_cases:
                    session.add(TestCaseModel(repo_id=repo_id, file_path=tc.file_path, test_name=tc.test_name))
                for tm in test_mappings:
                    session.add(TestToCodeRelationModel(
                        repo_id=repo_id, test_name=tm.test_name, test_file=tm.test_file,
                        target_symbol=tm.target_symbol, target_file=tm.target_file,
                        relation_type=tm.relation_type, confidence=tm.confidence,
                    ))

                curr_fp = self.incremental.compute_fingerprint(
                    local_path, admitted_paths, repo_id=repo_id
                )
                from packages.knowledge.persistence import persist_file_fingerprints as _pff
                await _pff(session, repo_id, {
                    p: {"content_hash": f.content_hash, "size_bytes": f.size_bytes}
                    for p, f in curr_fp.files.items()
                })

                repo_row = (
                    await session.execute(select(RepositoryModel).where(RepositoryModel.id == repo_id))
                ).scalar_one_or_none()
                if repo_row is None:
                    repo_row = RepositoryModel(id=repo_id, repo_url=repo_url)
                    session.add(repo_row)
                repo_row.head_commit = head_commit or None
                repo_row.branch = branch or ""
                repo_row.fingerprint_hash = curr_fp.compute_overall_hash()
                repo_row.last_run_id = run_id

                counts_after = await count_repo_artifacts(session, repo_id)
                await session.commit()

            # Vector store: invalidate changed paths, then re-upsert fresh chunks
            if incremental_files:
                removed = self._vector_store_invalidate(repo_id, incremental_files)
                if incremental_report is not None:
                    incremental_report.chunks_invalidated += removed
            self._vector_store_upsert(parse_res)

            self.lifecycle.complete_stage(
                run_id, LifecycleStage.KNOWLEDGE, started_index,
                counts={
                    "symbols_persisted": len(parse_res.symbols),
                    "chunks_persisted": len(parse_res.chunks),
                    "edges_persisted": len(parse_res.edges),
                },
            )
            await self._checkpoint(run_id)
            telemetry.record_stage_duration("indexing", time.monotonic() - started_index)
            cancel_token.check()
            self.resources.check_deadline("indexing")

            # ---------------- ANALYZING ----------------
            started = self.lifecycle.start_stage(run_id, LifecycleStage.ANALYSIS)
            await self._checkpoint(run_id)
            artifact_generated = False
            artifact_warning = None
            if generate_artifact:
                artifact_generated, artifact_warning = await self._generate_artifact(
                    repo_id, repo_url, manifest, artifact_type, run_id
                )
            structure_changed = (
                incremental_report is None or len(incremental_report.files_modified) > 0
                or len(incremental_report.files_added) > 0 or len(incremental_report.files_deleted) > 0
            )
            self.lifecycle.complete_stage(
                run_id, LifecycleStage.ANALYSIS, started,
                counts={"artifact_generated": int(artifact_generated)},
                warnings=[artifact_warning] if artifact_warning else None,
            )
            await self._checkpoint(run_id)

        except CancellationRequested as exc:
            run.lifecycle_state = RepositoryLifecycleState.FAILED
            run.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            await self._persist_run_record(run)
            return self._build_outcome(
                run_id, repo_id, run, None, incremental_report, counts_after=None,
                duration=time.monotonic() - t0, cancelled=True,
                error=f"cancelled: {exc}", local_path=local_path, is_temp_clone=is_temp_clone,
            )
        except Exception as exc:
            rec = self.resilience.record_failure(exc, stage="pipeline", attempts=1)
            run.errors.append(FailureDetail(
                stage="pipeline", error_type=type(exc).__name__,
                message=str(exc)[:500], recoverable=rec.is_retryable,
            ))
            run.lifecycle_state = RepositoryLifecycleState.FAILED
            run.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            await self._persist_run_record(run)
            return self._build_outcome(
                run_id, repo_id, run, None, incremental_report, counts_after=None,
                duration=time.monotonic() - t0, cancelled=False,
                error=f"{type(exc).__name__}: {exc}",
                local_path=local_path, is_temp_clone=is_temp_clone,
                failure_category=rec.category.value,
            )

        # ---------------- READY / PARTIAL ----------------
        final_run = self.lifecycle.finalize_run(run_id)
        await self._persist_run_record(final_run)
        duration = time.monotonic() - t0
        if incremental_report is not None:
            incremental_report.duration_seconds = round(duration, 3)
            incremental_report.architecture_recomputed = structure_changed
        if is_temp_clone and local_path:
            cleanup_repository(local_path)
        return self._build_outcome(
            final_run.run_id, repo_id, final_run,
            {"head_commit": head_commit, "branch": branch,
             "archetype": getattr(manifest, "archetype", None) and str(getattr(manifest, "archetype")),
             "artifacts": counts_after},
            incremental_report, counts_after=counts_after, duration=duration,
            cancelled=False, error=None, local_path=None, is_temp_clone=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _plan_scope(
        self, run_id: str, repo_id: str, local_path: str,
        admitted_paths: List[str], mode: str, force: bool,
    ):
        """Decide which files to reprocess (Phase 41). Returns (reprocess, deleted, report|None)."""
        if mode != "INCREMENTAL" or force:
            return list(admitted_paths), [], None

        async with self.session_factory() as session:
            prev_fp = await self.incremental.load_previous_fingerprint(session, repo_id)
        if prev_fp is None:
            return list(admitted_paths), [], None

        curr_fp = self.incremental.compute_fingerprint(local_path, admitted_paths, repo_id=repo_id)
        diffs = self.incremental.compute_diff(prev_fp, curr_fp)
        summary = self.incremental.diff_summary(diffs)
        changed = summary["ADDED"] + summary["MODIFIED"] + summary["RENAMED"]
        deleted_files = summary["DELETED"]

        direct: List[str] = []
        indirect: List[str] = []
        symbols_changed: List[str] = []
        async with self.session_factory() as session:
            direct, indirect = await self.incremental.expand_affected_files(
                session, repo_id, changed + deleted_files
            )
            symbols_changed = await self.incremental.changed_symbols_for_files(
                session, repo_id, changed + deleted_files
            )

        reprocess = sorted(set(changed))
        report = IncrementalRunReport(
            previous_commit=prev_fp.commit_sha,
            current_commit=curr_fp.commit_sha,
            previous_fingerprint=prev_fp.compute_overall_hash(),
            current_fingerprint=curr_fp.compute_overall_hash(),
            files_added=summary["ADDED"],
            files_modified=summary["MODIFIED"],
            files_deleted=deleted_files,
            files_unchanged_count=len(summary["UNCHANGED"]),
            directly_affected=direct,
            indirectly_affected=indirect,
            symbols_changed=symbols_changed,
        )
        telemetry.record_event(
            "incremental_scope",
            added=len(summary["ADDED"]), modified=len(summary["MODIFIED"]),
            deleted=len(deleted_files), reprocess=len(reprocess),
            unchanged=len(summary["UNCHANGED"]),
        )
        return reprocess, deleted_files, report

    def _vector_store_invalidate(self, repo_id: str, paths: List[str]) -> int:
        """Invalidate vector-store chunks for paths using the process-local store if reachable."""
        store = getattr(self, "_last_vector_store", None)
        if store is None:
            return 0
        removed = store.invalidate_paths(repo_id, paths)
        telemetry.record_event("cache_invalidation", store="vector", removed=removed)
        return removed

    def _vector_store_upsert(self, parse_res):
        store = getattr(self, "_last_vector_store", None)
        if store is not None and getattr(parse_res, "chunks", []):
            store.upsert_chunks(parse_res.chunks)

    def attach_vector_store(self, vector_store) -> "RepositoryAnalysisPipeline":
        """Register the process-local VectorStore so incremental runs keep it consistent."""
        self._last_vector_store = vector_store
        return self

    async def _generate_artifact(self, repo_id: str, repo_url: str,
                                 manifest, artifact_type: str, run_id: str):
        """LangGraph artifact generation (Phase 5-9 pipeline reuse).

        Builds the canonical Planner -> Retriever -> Synthesis -> Reviewer ->
        bounded-reflection graph, grounds it on the *same in-memory vector
        store that was populated during indexing* (the repository facade is the
        knowledge boundary; the agent never touches raw files or the DB
        directly), then durably persists the resulting ``AgentRunModel`` and
        ``ArtifactModel`` rows so the README endpoint can serve the artifact.

        Returns ``(generated: bool, warning: Optional[str])``. LLM
        unavailability degrades gracefully -- it never fails the pipeline.
        """
        try:
            from packages.knowledge.vector_store import VectorStore as _VS
            from packages.knowledge.repository_facade import RepositoryFacade
            from packages.knowledge.models import AgentRunModel, ArtifactModel
            from packages.agentic import LLMClient, RepoIntelState, build_main_graph

            llm_client = LLMClient()
            if not llm_client.is_available():
                return False, "artifact skipped: Gemini unavailable (no GOOGLE_API_KEY); deterministic indexing completed"

            # Reuse the store that already holds this run's indexed chunks.
            vs = getattr(self, "_last_vector_store", None)
            if vs is None:
                from packages.knowledge.embeddings import create_embedding_service
                vs = _VS(create_embedding_service())

            async with self.session_factory() as session:
                facade = RepositoryFacade(session, vs)
                app = build_main_graph(llm_client, facade)
                initial_state = RepoIntelState(
                    run_id=run_id,
                    thread_id=run_id,
                    repo_id=repo_id,
                    repo_url=repo_url,
                    archetype=(
                        manifest.archetype.value
                        if hasattr(manifest.archetype, "value")
                        else str(manifest.archetype)
                    ),
                    artifact_type=artifact_type,
                )
                graph_config = {
                    "configurable": {
                        "thread_id": run_id,
                        "checkpoint_ns": "artifact_generation",
                        "checkpoint_id": run_id,
                    }
                }
                final_state_dict = await asyncio.wait_for(
                    app.ainvoke(
                        initial_state.model_dump(),
                        config=graph_config,
                    ),
                    timeout=max(self.resources.timeout_for_stage("llm") * 4, 60),
                )
                final_state = RepoIntelState.model_validate(final_state_dict)

                # Reassemble the grounded README from the graph's artifact draft.
                artifact_content = ""
                total_citations = 0
                draft = getattr(final_state, "artifact_draft", None)
                if draft is not None:
                    lines = []
                    for sec in draft.sections:
                        lines.append(f"## {sec.section_title}\n\n{sec.content}\n")
                        total_citations += len(sec.citations or [])
                    artifact_content = "\n".join(lines)

                verdict = (
                    final_state.review_report.verdict.value
                    if final_state.review_report is not None
                    else "N/A"
                )
                run_errors = list(getattr(final_state, "errors", []) or [])

                # Durable agent-run record + artifact (join-compatible with the
                # /artifacts and /repositories/{id}/artifacts endpoints which
                # join ArtifactModel.run_id -> AgentRunModel.run_id).
                await session.merge(AgentRunModel(
                    run_id=run_id,
                    repo_id=repo_id,
                    status=(
                        "COMPLETED" if verdict != "NEEDS_HUMAN_REVIEW" else "NEEDS_HUMAN_REVIEW"
                    ),
                    result_json=json.dumps({
                        "repo_id": repo_id,
                        "repo_url": repo_url,
                        "artifact_type": artifact_type,
                        "review_verdict": verdict,
                        "reflection_iterations": getattr(
                            final_state, "reflection_iteration_count", 0
                        ),
                        "citations": total_citations,
                        "errors": run_errors,
                    }),
                ))
                if artifact_content:
                    session.add(ArtifactModel(
                        run_id=run_id,
                        artifact_type=artifact_type,
                        content=artifact_content,
                        version=1,
                    ))
                await session.commit()

            telemetry.increment("llm_artifacts_generated")
            if artifact_content:
                return True, None
            return True, (
                f"artifact generated but empty for run {run_id}; "
                f"review_verdict={verdict} errors={run_errors}"
            )
        except Exception as exc:
            category, _retryable = self.resilience.classify_error(exc)
            self.resilience.record_failure(
                exc, stage="artifact_generation", attempts=1
            )
            return False, f"artifact generation degraded ({category.value}): {exc}"

    async def _checkpoint(self, run_id: str) -> None:
        """Durably flush the in-memory lifecycle record to ``pipeline_runs``.

        DURABILITY INVARIANT (Phase 40.3): every lifecycle state transition is
        persisted to the database immediately, before the pipeline proceeds to
        the next stage.  A run that has entered a stage must have its
        ``status``/``current_stage`` columns reflect that stage in PostgreSQL
        even if the worker is killed the instant after the transition -- this
        is what makes the run observable from the API and recoverable after a
        crash.  Persistence failures are recorded on the run (never raised) so
        observability cannot break the analysis itself; the subsequent
        checkpoint for the next transition will retry on fresh evidence.
        """
        run = self.lifecycle.runs.get(run_id)
        if run is None:
            return
        await self._persist_run_record(run)

    async def _checkpoint(self, run_id: str) -> None:
        """Durably flush the in-memory lifecycle record to ``pipeline_runs``.

        DURABILITY INVARIANT (Phase 40.3): every lifecycle state transition is
        persisted to the database immediately, before the pipeline proceeds to
        the next stage.  A run that has entered a stage must have its
        ``status``/``current_stage`` columns reflect that stage in PostgreSQL
        even if the worker is killed the instant after the transition -- this
        is what makes the run observable from the API and recoverable after a
        crash.  Persistence failures are recorded on the run (never raised) so
        observability cannot break the analysis itself; the next checkpoint
        retries on fresh evidence.
        """
        run = self.lifecycle.runs.get(run_id)
        if run is None:
            return
        await self._persist_run_record(run)

    async def _persist_run_record(self, run: PipelineRunRecord):
        """Persist the run record through the existing job/run architecture (Phase 40.2)."""
        try:
            async with self.session_factory() as session:
                row = (
                    await session.execute(
                        select(PipelineRunModel).where(PipelineRunModel.run_id == run.run_id)
                    )
                ).scalar_one_or_none()
                if row is None:
                    row = PipelineRunModel(run_id=run.run_id)
                    session.add(row)
                row.repository_id = run.repository_id
                row.mode = run.mode
                row.current_stage = run.current_stage.value if run.current_stage else None
                row.status = run.lifecycle_state.value
                row.progress = run.progress_percentage
                row.files_seen = run.files_seen
                row.files_parsed = run.files_parsed
                row.files_failed = run.files_failed
                row.symbols_extracted = run.symbols_extracted
                row.chunks_created = run.chunks_created
                row.graph_edges_created = run.graph_edges_created
                row.errors_json = json.dumps([e.model_dump() for e in run.errors])
                row.warnings_json = json.dumps(run.warnings)
                row.report_json = json.dumps(run.model_dump(), default=str)
                if run.completed_at:
                    try:
                        row.completed_at = datetime.fromisoformat(run.completed_at.replace("Z", "+00:00"))
                    except ValueError:
                        row.completed_at = datetime.now(timezone.utc)
                    # keep repository status in sync with lifecycle outcome
                    repo_row = (
                        await session.execute(
                            select(RepositoryModel).where(RepositoryModel.id == run.repository_id)
                        )
                    ).scalar_one_or_none()
                    if repo_row is not None:
                        repo_row.status = run.lifecycle_state.value
                        repo_row.last_run_id = run.run_id
                await session.commit()
        except Exception as exc:  # observability must never break the pipeline
            # A silent warning here would leave the run stuck in its previous
            # durable state (e.g. CREATED) even though the pipeline finished,
            # which the UI/API would report as "still in progress". Record the
            # persistence failure on the run itself and log it loudly, then
            # retry once through a fresh session so a transient DB hiccup does
            # not swallow the lifecycle transition.
            logger.error(
                "Failed to persist run record %s (state=%s): %s",
                run.run_id, run.lifecycle_state.value, exc,
            )
            run.errors.append(FailureDetail(
                stage="persistence",
                error_type=type(exc).__name__,
                message=f"run record persistence failed: {exc}"[:500],
                recoverable=True,
            ))
            try:
                async with self.session_factory() as session:
                    row = (
                        await session.execute(
                            select(PipelineRunModel).where(
                                PipelineRunModel.run_id == run.run_id
                            )
                        )
                    ).scalar_one_or_none()
                    if row is None:
                        row = PipelineRunModel(run_id=run.run_id)
                        session.add(row)
                    row.repository_id = run.repository_id
                    row.status = run.lifecycle_state.value
                    row.current_stage = (
                        run.current_stage.value if run.current_stage else None
                    )
                    row.progress = run.progress_percentage
                    row.errors_json = json.dumps([e.model_dump() for e in run.errors])
                    await session.commit()
            except Exception as retry_exc:
                logger.error(
                    "Run record %s could not be persisted on retry either: %s",
                    run.run_id, retry_exc,
                )

    def _build_outcome(
        self,
        run_id: str,
        repo_id: str,
        run: PipelineRunRecord,
        repo_meta: Optional[Dict[str, Any]],
        incremental_report: Optional[IncrementalRunReport],
        counts_after: Optional[Dict[str, int]],
        duration: float,
        cancelled: bool,
        error: Optional[str],
        local_path: Optional[str],
        is_temp_clone: bool,
        failure_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        if is_temp_clone and local_path and os.path.isdir(local_path):
            cleanup_repository(local_path)
        status = "CANCELLED" if cancelled else run.lifecycle_state.value
        return {
            "run_id": run_id,
            "repository_id": repo_id,
            "status": status,
            "error": error,
            "failure_category": failure_category,
            "duration_seconds": round(duration, 3),
            "lifecycle": run.model_dump(),
            "incremental_report": incremental_report.model_dump() if incremental_report else None,
            "repository": repo_meta or {},
            "artifact_counts": counts_after or {},
            "stage_durations": {
                name: {"status": sp.status, "duration_seconds": sp.duration_seconds,
                       "counts": sp.counts, "errors": [e.model_dump() for e in sp.errors]}
                for name, sp in run.stages.items()
            },
        }
