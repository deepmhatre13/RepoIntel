import os
import sys
import json
import time
import asyncio
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON as RichJSON

from packages.ingestion import run_ingestion_pipeline, cleanup_repository
from packages.parsing import run_parsing_pipeline
from packages.knowledge import (
    init_db, AsyncSessionLocal, EmbeddingService, VectorStore,
    persist_parsing_results, RepositoryFacade
)
from packages.agentic import LLMClient, RepoIntelState, build_main_graph
from packages.runtime.pipeline_orchestrator import RepositoryAnalysisPipeline

console = Console()


def _git_head(repo_path: str):
    """Return (head_commit, branch) for a git checkout; ('', '') otherwise."""
    try:
        from packages.runtime.incremental_indexer import IncrementalIndexer

        return IncrementalIndexer().git_head(repo_path)
    except Exception:
        return "", ""


async def _persist_artifact(session, run_id: str, artifact_type: str, content: str):
    """Persist the generated artifact so it is retrievable and auditable."""
    if not content:
        return None
    from packages.knowledge.models import ArtifactModel

    artifact = ArtifactModel(run_id=run_id, artifact_type=artifact_type, content=content, version=1)
    session.add(artifact)
    await session.flush()
    return artifact.id


async def analyze_repo(repo_url: str, output_dir: str = "./output", artifact_type: str = "README"):
    started_at = time.monotonic()
    console.print(Panel(f"[bold blue]RepoIntel Analysis Initiated[/bold blue]\nTarget: {repo_url}\nArtifact Type: {artifact_type}"))
    
    await init_db()
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service)
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))

    console.print("[yellow]Stage 1/8: Ingestion...[/yellow]")
    manifest, filtered_files = run_ingestion_pipeline(repo_url)
    
    from packages.ingestion.git_history import parse_git_commits, persist_git_commits
    commits = parse_git_commits(manifest.local_path, max_depth=50)

    console.print("[yellow]Stage 2/8: Parsing (AST, Symbols, Dependencies & Chunks)...[/yellow]")
    parsing_res = run_parsing_pipeline(repo_id, manifest.local_path, filtered_files)

    from packages.parsing.test_parser import discover_tests_in_repo, map_tests_to_production_symbols
    test_cases = discover_tests_in_repo(manifest.local_path, filtered_files)
    test_mappings = map_tests_to_production_symbols(test_cases, parsing_res.symbols)

    console.print("[yellow]Stage 3/8: Persistence (Database & Vector Store)...[/yellow]")
    vector_store.upsert_chunks(parsing_res.chunks)
    async with AsyncSessionLocal() as session:
        await persist_parsing_results(session, manifest, parsing_res)
        if commits:
            console.print("[yellow]Stage 4/8: Git History Indexing...[/yellow]")
            await persist_git_commits(session, repo_id, commits, parsing_res.symbols, repo_path=manifest.local_path)
        from packages.knowledge.models import TestCaseModel, TestToCodeRelationModel
        for tc in test_cases:
            session.add(TestCaseModel(repo_id=repo_id, file_path=tc.file_path, test_name=tc.test_name))
        for tm in test_mappings:
            session.add(TestToCodeRelationModel(
                repo_id=repo_id, test_name=tm.test_name, test_file=tm.test_file,
                target_symbol=tm.target_symbol, target_file=tm.target_file,
                relation_type=tm.relation_type, confidence=tm.confidence
            ))
        await session.commit()

        facade = RepositoryFacade(session, vector_store)
        console.print(f"[yellow]Stage 5/8: Agentic Analysis ({artifact_type})...[/yellow]")
        llm_client = LLMClient()
        run_id = str(uuid.uuid4())
        app = build_main_graph(llm_client, facade)
        initial_state = RepoIntelState(
            run_id=run_id,
            thread_id=str(uuid.uuid4()),
            repo_id=repo_id,
            repo_url=repo_url,
            archetype=manifest.archetype.value if hasattr(manifest.archetype, 'value') else str(manifest.archetype),
            artifact_type=artifact_type
        )
        final_state_dict = await app.ainvoke(initial_state.model_dump())
        final_state = RepoIntelState.model_validate(final_state_dict)

        console.print("[yellow]Stage 6/8: Artifact Generation...[/yellow]")
        os.makedirs(output_dir, exist_ok=True)
        ext = ".md" if not artifact_type.endswith(".md") else ""
        artifact_path = os.path.join(output_dir, f"{artifact_type.lower()}{ext}")
        artifact_content = ""
        total_citations = 0
        if final_state.artifact_draft:
            lines = []
            for sec in final_state.artifact_draft.sections:
                lines.append(f"## {sec.section_title}\n\n{sec.content}\n")
                total_citations += len(sec.citations)
            artifact_content = "\n".join(lines)
            with open(artifact_path, "w", encoding="utf-8") as f:
                f.write(artifact_content)
        artifact_id = await _persist_artifact(session, run_id, artifact_type, artifact_content)

        # Persist the agentic run record so lifecycle status is queryable.
        from packages.knowledge.models import AgentRunModel
        review_verdict = (
            final_state.review_report.verdict.value if final_state.review_report else "N/A"
        )
        await session.merge(AgentRunModel(
            run_id=run_id,
            repo_id=repo_id,
            status="COMPLETED" if review_verdict != "NEEDS_HUMAN_REVIEW" else "NEEDS_HUMAN_REVIEW",
            result_json=json.dumps({
                "repo_id": repo_id,
                "repo_url": repo_url,
                "artifact_type": artifact_type,
                "review_verdict": review_verdict,
            }),
        ))

        # Persist lifecycle outcome on the repository row so `status <repo>`
        # reflects reality (Phase 40.1: no pretend success).
        from packages.knowledge.models import RepositoryModel
        from sqlalchemy import select as _select

        run_errors = list(getattr(final_state, "errors", []) or [])
        repo_row = (
            await session.execute(_select(RepositoryModel).where(RepositoryModel.id == repo_id))
        ).scalar_one_or_none()
        if repo_row is not None:
            if review_verdict == "NEEDS_HUMAN_REVIEW":
                repo_row.status = "FAILED"
            elif run_errors:
                repo_row.status = "PARTIAL"
            else:
                repo_row.status = "READY"
            repo_row.last_run_id = run_id
            if getattr(manifest, "local_path", None):
                head_commit, branch = _git_head(manifest.local_path)
                if head_commit:
                    repo_row.head_commit = head_commit
                if branch:
                    repo_row.branch = branch
        await session.commit()
        cleanup_repository(manifest.local_path)

    warnings = list(getattr(final_state, "warnings", []) or [])
    errors = list(getattr(final_state, "errors", []) or [])

    table = Table(title="RepoIntel Execution Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Repository ID", repo_id)
    table.add_row("Run ID", run_id)
    table.add_row("Status", "READY" if not errors else "PARTIAL")
    table.add_row("Repository URL", repo_url)
    table.add_row("Archetype", final_state.archetype)
    table.add_row("Artifact ID", str(artifact_id) if artifact_id else "N/A")
    table.add_row("Artifact Type", artifact_type)
    table.add_row("Generated Artifact", artifact_path)
    table.add_row("Files Processed", str(manifest.file_count))
    table.add_row("Symbols Extracted", str(len(parsing_res.symbols)))
    table.add_row("Chunks Created", str(len(parsing_res.chunks)))
    table.add_row("Graph Relationships", str(len(parsing_res.edges)))
    table.add_row("Dependencies", str(len(parsing_res.dependencies)))
    table.add_row("Git Commits Ingested", str(len(commits)))
    table.add_row("Discovered Tests", str(len(test_cases)))
    table.add_row("Test-Code Mappings", str(len(test_mappings)))
    table.add_row("Reflection Iterations", str(final_state.reflection_iteration_count))
    table.add_row("Review Verdict", review_verdict)
    table.add_row("Citations Count", str(total_citations))
    table.add_row("Duration (s)", f"{round(time.monotonic() - started_at, 2)}")
    table.add_row("Warnings", str(len(warnings)))
    table.add_row("Errors", str(len(errors)))
    console.print(table)

    if warnings:
        warn_table = Table(title="Warnings (never silently swallowed)")
        warn_table.add_column("#")
        warn_table.add_column("Message")
        for i, w in enumerate(warnings[:20], start=1):
            warn_table.add_row(str(i), str(w)[:100])
        console.print(warn_table)
    if errors:
        err_table = Table(title="Errors")
        err_table.add_column("#")
        err_table.add_column("Message")
        for i, e in enumerate(errors[:20], start=1):
            err_table.add_row(str(i), str(e)[:100])
        console.print(err_table)

    console.print("[bold green]Analysis Complete Successfully![/bold green]")


async def _load_vector_store(repo_id: str) -> "tuple[EmbeddingService, VectorStore, AsyncSessionLocal]":
    """Build a vector store backed by real embeddings + persisted chunks.

    Loads the persisted code chunks (and their embeddings) for *repo_id* from the
    database into an in-process ``VectorStore`` so query-time semantic retrieval
    works without re-ingesting the repository.
    """
    from packages.knowledge.persistence import load_code_chunks, load_embeddings, code_chunks_to_vector_store

    embedding_service = EmbeddingService()
    vs = VectorStore(embedding_service)
    try:
        async with AsyncSessionLocal() as session:
            rows = await load_code_chunks(session, repo_id)
            if rows:
                vs.upsert_chunks(code_chunks_to_vector_store(rows))
                cached = await load_embeddings(session, repo_id)
                embedding_service.warm_cache(cached)
    except Exception as exc:  # pragma: no cover
        console.print(f"[yellow]Note: could not load persisted chunks: {exc}[/yellow]")
    return embedding_service, vs


async def cli_query(repo_url: str, query: str):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    _embedding_service, vs = await _load_vector_store(repo_id)
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        llm = LLMClient()
        from packages.agentic.agents.qa_analyst import answer_repository_query
        res = await answer_repository_query(repo_id, query, facade, llm)
        console.print(Panel(f"[bold cyan]Answer:[/bold cyan]\n{res.answer}\n\n[bold yellow]Confidence:[/bold yellow] {res.confidence}"))
        if res.related_files:
            console.print(f"\n[bold]Related Files:[/bold]")
            for f in res.related_files:
                console.print(f"  - {f}")
        if res.related_symbols:
            console.print(f"\n[bold]Related Symbols:[/bold]")
            for s in res.related_symbols:
                console.print(f"  - {s}")
        if res.related_commits:
            console.print(f"\n[bold]Related Commits:[/bold]")
            for c in res.related_commits:
                console.print(f"  - {c}")


async def cli_history(repo_url: str, symbol_name: str, limit: int = 20):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    vs = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        history = await facade.get_symbol_history(repo_id, symbol_name, limit=limit)
        freq = await facade.get_symbol_frequency(repo_id, symbol_name)
        coc = await facade.get_co_change_relationships(repo_id, symbol_name)
        
        console.print(Panel(f"[bold magenta]Symbol History: {symbol_name}[/bold magenta]"))
        if not history:
            console.print("[yellow]No historical data found for this symbol.[/yellow]")
            return
        
        table = Table(title=f"Change History for {symbol_name}")
        table.add_column("Commit", style="cyan")
        table.add_column("Date", style="green")
        table.add_column("Author", style="yellow")
        table.add_column("Change Type", style="magenta")
        table.add_column("Lines", style="blue")
        table.add_column("Summary")
        for h in history:
            table.add_row(
                h.get("commit_sha", "")[:8],
                h.get("timestamp", "")[:10] if h.get("timestamp") else "N/A",
                h.get("author", ""),
                h.get("change_type", ""),
                f"{h.get('line_start', '?')}-{h.get('line_end', '?')}",
                h.get("change_summary", h.get("confidence", ""))
            )
        console.print(table)
        
        if freq:
            console.print(f"\n[bold]Frequency Metrics:[/bold]")
            table2 = Table(title="Change Frequency")
            for k, v in freq.items():
                table2.add_row(k, str(v))
            if not table2.rows:
                table2.add_row("metric", "value")
                for k, v in freq.items():
                    table2.add_row(k, str(v))
            console.print(table2)
        
        if coc:
            console.print(f"\n[bold]Co-Change Relationships:[/bold]")
            for r in coc[:10]:
                console.print(f"  [cyan]{r['symbol_a']}[/cyan] <-> [cyan]{r['symbol_b']}[/cyan] "
                              f"(count: {r.get('co_change_count', 0)}, ratio: {r.get('co_change_ratio', 'N/A')})")


async def cli_diff(repo_url: str, commit_sha: str):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    vs = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        analysis = await facade.get_commit_analysis(repo_id, commit_sha)
        if not analysis:
            console.print("[yellow]No commit analysis found. The repository may not have been ingested with git history.[/yellow]")
            return
        
        console.print(Panel(f"[bold magenta]Commit Analysis: {commit_sha[:8]}[/bold magenta]\n{analysis.get('summary', '')}"))
        
        changed_files = analysis.get("changed_files", [])
        table = Table(title="Changed Files")
        table.add_column("File", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("+", style="green")
        table.add_column("-", style="red")
        for f in changed_files:
            table.add_row(f.get("file_path", ""), f.get("change_type", ""), str(f.get("additions", 0)), str(f.get("deletions", 0)))
        console.print(table)
        
        changed_symbols = analysis.get("changed_symbols", [])
        if changed_symbols:
            table2 = Table(title="Changed Symbols")
            table2.add_column("Symbol", style="cyan")
            table2.add_column("Change Type", style="yellow")
            for s in changed_symbols:
                table2.add_row(s.get("symbol_name", ""), s.get("change_category", "BODY_CHANGE"))
            console.print(table2)
        
        api_changes = analysis.get("api_changes", [])
        if api_changes:
            console.print(f"\n[bold]API Changes ({len(api_changes)}):[/bold]")
            for ac in api_changes:
                console.print(f"  [cyan]{ac.get('api_symbol', '')}[/cyan] - {ac.get('change_type', '')}: {ac.get('before_signature', '')} -> {ac.get('after_signature', '')}")
        
        dep_changes = analysis.get("dependency_changes", [])
        if dep_changes:
            console.print(f"\n[bold]Dependency Changes ({len(dep_changes)}):[/bold]")
            for dc in dep_changes:
                console.print(f"  [cyan]{dc.get('package_name', '')}[/cyan] ({dc.get('ecosystem', '')}) - {dc.get('change_type', '')}: {dc.get('old_version', '')} -> {dc.get('new_version', '')}")


async def cli_architecture(repo_url: str):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    vs = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        snapshot = await facade.create_architecture_snapshot(repo_id, repo_id)
        if not snapshot:
            console.print("[yellow]No architecture data found. Run analyze first.[/yellow]")
            return
        
        console.print(Panel(f"[bold magenta]Architecture Snapshot[/bold magenta]"))
        modules = snapshot.get("modules", [])
        table = Table(title="Components")
        table.add_column("Component", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Symbol Count", style="green")
        table.add_column("File Count", style="blue")
        for m in snapshot.get("major_components", []):
            meta = snapshot.get("component_metadata", {}).get(m["name"], {})
            table.add_row(m["name"], meta.get("type", "Unknown"), str(m.get("symbol_count", 0)), str(m.get("file_count", 0)))
        console.print(table)
        
        edges = snapshot.get("module_edges", [])
        if edges:
            console.print(f"\n[bold]Component Edges ({len(edges)}):[/bold]")
            for e in edges[:20]:
                console.print(f"  [cyan]{e.get('source_component', '')}[/cyan] --{e.get('edge_type', '')}--> [cyan]{e.get('target_component', '')}[/cyan]")
        
        entry = snapshot.get("entrypoints", [])
        if entry:
            console.print(f"\n[bold]Entrypoints:[/bold]")
            for e in entry:
                console.print(f"  - {e}")


async def cli_hotspots(repo_url: str, limit: int = 20):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    vs = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        hotspots = await facade.get_hotspots(repo_id)
        
        console.print(Panel(f"[bold magenta]Code Hotspots[/bold magenta]"))
        table = Table(title="Top Hotspots")
        table.add_column("#", style="cyan")
        table.add_column("Symbol/File", style="yellow")
        table.add_column("Score", style="green")
        table.add_column("Module", style="blue")
        table.add_column("Commits", style="magenta")
        table.add_column("Authors", style="red")
        table.add_column("Reasons")
        for i, h in enumerate(hotspots[:limit], 1):
            table.add_row(str(i), h.get("file_or_symbol", ""), str(h.get("hotspot_score", 0)),
                          h.get("affected_module", "N/A"), str(h.get("commit_count", 0)),
                          str(h.get("authors_count", 0)), ", ".join(h.get("reasons", [])))
        console.print(table)


async def cli_coupling(repo_url: str, file_path: str = ""):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    vs = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        
        table = Table(title="Module Coupling Matrix")
        table.add_column("Module", style="cyan")
        table.add_column("Fan-In", style="green")
        table.add_column("Fan-Out", style="yellow")
        table.add_column("Total", style="magenta")
        
        coupling_data = await facade.get_coupling_metrics(repo_id, file_path) if file_path else await facade.get_module_coupling(repo_id)
        
        if file_path:
            table.add_row(coupling_data.get("file_path", file_path), str(coupling_data.get("fan_in", 0)),
                          str(coupling_data.get("fan_out", 0)), str(coupling_data.get("fan_in", 0) + coupling_data.get("fan_out", 0)))
        else:
            modules = coupling_data.get("modules", {})
            for mod, data in sorted(modules.items(), key=lambda x: x[1].get("total", 0), reverse=True):
                table.add_row(mod, str(data.get("fan_in", 0)), str(data.get("fan_out", 0)), str(data.get("total", 0)))
        
        console.print(table)
        
        high_coupled = coupling_data.get("highly_coupled_modules", [])
        if high_coupled:
            console.print(f"\n[bold red]Highly Coupled Modules:[/bold red]")
            for m in high_coupled:
                console.print(f"  [cyan]{m['module']}[/cyan] - fan_in: {m['fan_in']}, fan_out: {m['fan_out']}")
        
        cycles = await facade.detect_cycles(repo_id)
        if cycles:
            console.print(f"\n[bold red]Circular Dependencies Detected ({len(cycles)}):[/bold red]")
            for c in cycles:
                console.print(f"  {' -> '.join(c.get('cycle', []))}")
        else:
            console.print(f"\n[green]No circular dependencies detected.[/green]")


async def cli_tests(repo_url: str, symbol: str):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    vs = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        tests = await facade.get_tests_for_symbol(repo_id, symbol)
        gaps = await facade.get_test_gap_analysis(repo_id, symbol)
        
        console.print(Panel(f"[bold magenta]Tests for: {symbol}[/bold magenta]"))
        if not tests:
            console.print("[yellow]No related tests found.[/yellow]")
        else:
            table = Table(title="Related Tests")
            table.add_column("Test", style="cyan")
            table.add_column("File", style="yellow")
            table.add_column("Relation Type", style="green")
            table.add_column("Confidence", style="magenta")
            for t in tests:
                table.add_row(t.get("test_name", ""), t.get("test_file", ""), t.get("relation_type", ""), t.get("confidence", ""))
            console.print(table)
        
        if gaps:
            console.print(f"\n[bold red]Test Gap: {gaps.get('target', symbol)}[/bold red]")
            console.print(f"  {gaps.get('message', 'No relevant tests identified.')}")
            console.print(f"  Confidence: {gaps.get('confidence', 'UNKNOWN')}")


async def cli_impact(repo_url: str, target: str):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    vs = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        impact = await facade.analyze_impact(repo_id, target)
        
        console.print(Panel(f"[bold magenta]Impact Analysis: {target}[/bold magenta]"))
        table = Table(title="Impact Report")
        table.add_column("Aspect", style="cyan")
        table.add_column("Details", style="yellow")
        table.add_row("Target", impact.get("target", ""))
        table.add_row("Type", impact.get("target_type", ""))
        table.add_row("Risk Level", impact.get("risk_level", ""))
        table.add_row("Total Dependents", str(impact.get("total_dependents", 0)))
        table.add_row("Affected Symbols", str(len(impact.get("affected_symbols", []))))
        table.add_row("Affected Files", str(len(impact.get("affected_files", []))))
        table.add_row("Configuration Impact", str(len(impact.get("configuration_impact", []))))
        table.add_row("Documentation Impact", str(len(impact.get("documentation_impact", []))))
        table.add_row("Test Impact", str(len(impact.get("test_impact", []))))
        console.print(table)
        
        if impact.get("affected_symbols"):
            console.print(f"\n[bold]Affected Symbols:[/bold]")
            for s in impact["affected_symbols"][:20]:
                console.print(f"  - {s}")
        
        if impact.get("test_impact"):
            console.print(f"\n[bold]Related Tests:[/bold]")
            for t in impact["test_impact"][:20]:
                console.print(f"  - {t}")
        
        if impact.get("risk_reasons"):
            console.print(f"\n[bold]Risk Reasons:[/bold]")
            for r in impact["risk_reasons"]:
                console.print(f"  - {r}")


async def cli_change_plan(repo_url: str, requirement: str, target: str = ""):
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    vs = VectorStore()
    async with AsyncSessionLocal() as session:
        facade = RepositoryFacade(session, vs)
        llm = LLMClient()
        from packages.agentic.agents.impact_planner import plan_code_change
        from packages.knowledge.analysis_models import ChangePlan
        
        plan = await plan_code_change(repo_id, requirement, target or "Modify component", facade, llm)
        
        console.print(Panel(f"[bold magenta]Change Plan: {requirement}[/bold magenta]"))
        if plan.requirement_analysis:
            console.print(f"\n[bold]Requirement Analysis:[/bold]")
            ra = plan.requirement_analysis
            console.print(f"  Objective: {ra.objective}")
            if ra.target_entities:
                console.print(f"  Entities: {', '.join(ra.target_entities)}")
            if ra.likely_affected_domains:
                console.print(f"  Affected Domains: {', '.join(ra.likely_affected_domains)}")
            if ra.uncertainties:
                console.print(f"  Uncertainties: {', '.join(ra.uncertainties)}")
        
        table = Table(title="Change Plan Steps")
        table.add_column("#", style="cyan")
        table.add_column("Component", style="yellow")
        table.add_column("Action", style="green")
        table.add_column("Risk", style="magenta")
        table.add_column("Prerequisites", style="blue")
        for step in plan.steps:
            table.add_row(str(step.step_number), step.component, step.action, step.risk_level, ", ".join(str(p) for p in step.prerequisite_steps))
        console.print(table)
        
        if plan.recommended_tests_to_run:
            console.print(f"\n[bold]Recommended Tests to Run:[/bold]")
            for t in plan.recommended_tests_to_run:
                console.print(f"  - {t}")
        
        if plan.rollout_considerations:
            console.print(f"\n[bold]Rollout Considerations:[/bold]")
            for r in plan.rollout_considerations:
                console.print(f"  - {r}")


# ---------------------------------------------------------------------------
# Database administration commands (Step 3/4: db-current / db-upgrade /
# db-downgrade / db-check)
# ---------------------------------------------------------------------------

def cli_db_current():
    """Print the current database migration revision."""
    from packages.knowledge import migrations

    def _run():
        head = migrations.head_revision()
        cur = migrations.current_revision()
        return head, cur

    head, cur = _run()
    if cur is None:
        console.print("[yellow]Migration state:[/] no revisions applied yet")
    else:
        console.print(f"[cyan]Migration state:[/] {cur}")
    console.print(f"[cyan]Latest available:[/] {head}")
    if cur != head:
        console.print("[red]Database schema is not current.[/red]")
        console.print("Action: run `python -m repointel.cli db-upgrade`")
        return False
    return True


def cli_db_upgrade():
    """Run all pending database migrations to bring the schema current."""
    from packages.knowledge.db import init_db

    console.print("[yellow]Applying pending database migrations...[/yellow]")
    asyncio.run(init_db())
    from packages.knowledge import migrations

    head = migrations.head_revision()
    cur = migrations.current_revision()
    console.print(f"[green]Migration complete: current revision = {cur} (latest {head})[/green]")
    return cur == head


def cli_db_downgrade():
    """Revert the most recently applied database migration."""
    from packages.knowledge import migrations

    prev = migrations.downgrade_one()
    console.print(f"[yellow]Downgraded database to revision:[/yellow] {prev or 'base'}")
    return True


def cli_db_check():
    """Validate the database schema against the ORM and report health."""
    from packages.knowledge.database_validator import validate_database

    result = validate_database()
    for line in result.summary_lines():
        console.print(line)
    if not result.ok:
        console.print("[red]Database is not fully valid.[/red]")
        for p in result.problems():
            console.print(f"  [red]-[/red] {p}")
        console.print("[yellow]Action: run `python -m repointel.cli db-upgrade`[/yellow]")
        return False
    console.print("[green]Database validation PASSED.[/green]")
    return True


def cli_eval():
    """Run the evaluation suite and report retrieval metrics."""
    from evals.run_evals import run_evaluation_suite

    run_evaluation_suite()
    return True

async def cli_status(repo_url: str):
    """Show repository lifecycle status and artifact counts (Phase 40.1)."""
    await init_db()
    repo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_url))
    from sqlalchemy import select as _select
    from packages.knowledge.models import RepositoryModel, PipelineRunModel
    from packages.knowledge.persistence import count_repo_artifacts
    async with AsyncSessionLocal() as session:
        repo = (
            await session.execute(_select(RepositoryModel).where(RepositoryModel.id == repo_id))
        ).scalar_one_or_none()
        if repo is None:
            console.print(f"[red]Repository not indexed:[/] {repo_url}")
            return
        counts = await count_repo_artifacts(session, repo_id)
        runs = (
            await session.execute(
                _select(PipelineRunModel)
                .where(PipelineRunModel.repository_id == repo_id)
                .order_by(PipelineRunModel.started_at.desc())
                .limit(5)
            )
        ).scalars().all()

    table = Table(title=f"Repository Status: {repo_id[:12]}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Lifecycle Status", repo.status)
    table.add_row("Branch", repo.branch or "N/A")
    table.add_row("HEAD Commit", (repo.head_commit or "N/A")[:12])
    table.add_row("Fingerprint", (repo.fingerprint_hash or "N/A")[:16])
    table.add_row("Archetype", repo.archetype)
    for key, value in counts.items():
        table.add_row(f"Indexed {key}", str(value))
    console.print(table)

    if runs:
        run_table = Table(title="Recent Pipeline Runs")
        run_table.add_column("Run")
        run_table.add_column("Mode")
        run_table.add_column("Status")
        run_table.add_column("Progress")
        run_table.add_column("Files Parsed")
        for r in runs:
            run_table.add_row(r.run_id[:8], r.mode, r.status, f"{r.progress:.0f}%", str(r.files_parsed))
        console.print(run_table)


async def cli_update(repo_url: str):
    """Incremental re-analysis: only changed content is reprocessed (Phase 41.2)."""
    return await _run_pipeline_cli(repo_url, mode="INCREMENTAL", force=False)


async def cli_reindex(repo_url: str, force: bool = True):
    """Forced full rebuild of indexes (Phase 41.7 reindex --force)."""
    return await _run_pipeline_cli(repo_url, mode="FULL", force=force)


async def _run_pipeline_cli(repo_url: str, mode: str, force: bool):
    console.print(Panel(
        f"[bold blue]RepoIntel {mode} {'Rebuild' if force else 'Update'}[/bold blue]\nTarget: {repo_url}"
    ))
    await init_db()
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service)

    pipeline = RepositoryAnalysisPipeline(
        session_factory=AsyncSessionLocal,
        embedding_service=embedding_service,
    ).attach_vector_store(vector_store)
    outcome = await pipeline.analyze(repo_url=repo_url, mode=mode, force=force)

    lifecycle = outcome.get("lifecycle", {})
    stage_durations = outcome.get("stage_durations", {})
    table = Table(title=f"Pipeline Outcome: {outcome['status']}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Run ID", outcome["run_id"][:12])
    table.add_row("Status", outcome["status"])
    table.add_row("Duration", f"{outcome['duration_seconds']}s")
    table.add_row("Files Seen / Parsed / Failed",
                  f"{lifecycle.get('files_seen', 0)} / {lifecycle.get('files_parsed', 0)} / {lifecycle.get('files_failed', 0)}")
    table.add_row("Symbols Extracted", str(lifecycle.get('symbols_extracted', 0)))
    table.add_row("Chunks Created", str(lifecycle.get('chunks_created', 0)))
    inc = outcome.get("incremental_report")
    if inc:
        table.add_row("Files Added/Modified/Deleted",
                      f"{len(inc['files_added'])} / {len(inc['files_modified'])} / {len(inc['files_deleted'])}")
        table.add_row("Unchanged Files", str(inc.get("files_unchanged_count", 0)))
        table.add_row("Directly/Indirectly Affected",
                      f"{len(inc.get('directly_affected', []))} / {len(inc.get('indirectly_affected', []))}")
        table.add_row("Chunks Invalidated", str(inc.get("chunks_invalidated", 0)))
    for stage_name, sp in stage_durations.items():
        if sp.get("duration_seconds"):
            table.add_row(f"Stage {stage_name}", f"{sp['duration_seconds']}s ({sp['status']})")
    if outcome.get("error"):
        table.add_row("Error", str(outcome["error"])[:80])
    console.print(table)

    failures = [e for e in lifecycle.get("errors", [])]
    if failures:
        fail_table = Table(title="Recorded Failures (never silently swallowed)")
        fail_table.add_column("Stage")
        fail_table.add_column("File")
        fail_table.add_column("Type")
        fail_table.add_column("Message")
        fail_table.add_column("Recoverable")
        for f in failures[:20]:
            fail_table.add_row(str(f.get("stage")), str(f.get("file")), str(f.get("error_type")),
                               str(f.get("message"))[:60], str(f.get("recoverable")))
        console.print(fail_table)
    return outcome


def _print_usage():
    """Print CLI usage to stdout."""
    console.print("Usage: python -m repointel.cli <command> [ARGS] [--verbose]")
    console.print("Commands:")
    console.print("  analyze <url> [artifact_type]   - Full analysis and documentation generation")
    console.print("  query <url> <question>          - Query repository with a natural language question")
    console.print("  history <url> <symbol> [limit]  - Query symbol history")
    console.print("  diff <url> <commit_sha>         - Analyze git commit diff")
    console.print("  architecture <url>              - Show architecture snapshot")
    console.print("  hotspots <url> [limit]          - Show code hotspots")
    console.print("  coupling <url> [file/module]    - Show coupling metrics")
    console.print("  tests <url> <symbol>            - Show tests for a symbol")
    console.print("  impact <url> <target>           - Impact analysis")
    console.print("  change-plan <url> <requirement> - Generate change plan")
    console.print("  update <url>                    - Incremental re-analysis of changed content")
    console.print("  reindex <url> [--force]         - Forced full index rebuild")
    console.print("  status <url>                    - Repository lifecycle status")
    console.print("  db-current                      - Show current database migration revision")
    console.print("  db-upgrade                      - Apply pending database migrations")
    console.print("  db-downgrade                    - Revert the latest database migration")
    console.print("  db-check                        - Validate the database schema")
    console.print("  eval                            - Run the retrieval evaluation suite")
    console.print("  --verbose                       - Surface full tracebacks for unexpected failures")


def _dispatch(cmd: str, args: list, verbose: bool) -> int:
    """Dispatch a CLI command and translate expected failures into concise,
    structured errors with a non-zero exit code. Unexpected exceptions are only
    exposed as full tracebacks when ``verbose`` is enabled."""
    try:
        if cmd == "analyze":
            artifact_type = args[2] if len(args) > 2 else "README"
            asyncio.run(analyze_repo(args[1], artifact_type=artifact_type))
        elif cmd == "query":
            q = args[2] if len(args) > 2 else "What does this codebase do?"
            asyncio.run(cli_query(args[1], q))
        elif cmd == "history":
            symbol = args[2] if len(args) > 2 else ""
            limit = int(args[3]) if len(args) > 3 else 20
            if not symbol:
                console.print("[red]Usage:[/] python -m repointel.cli history <url> <symbol> [limit]")
                return 1
            asyncio.run(cli_history(args[1], symbol, limit))
        elif cmd == "diff":
            sha = args[2] if len(args) > 2 else ""
            if not sha:
                console.print("[red]Usage:[/] python -m repointel.cli diff <url> <commit_sha>")
                return 1
            asyncio.run(cli_diff(args[1], sha))
        elif cmd == "architecture":
            asyncio.run(cli_architecture(args[1]))
        elif cmd == "hotspots":
            limit = int(args[2]) if len(args) > 2 else 20
            asyncio.run(cli_hotspots(args[1], limit))
        elif cmd == "coupling":
            file_path = args[2] if len(args) > 2 else ""
            asyncio.run(cli_coupling(args[1], file_path))
        elif cmd == "tests":
            symbol = args[2] if len(args) > 2 else ""
            if not symbol:
                console.print("[red]Usage:[/red] python -m repointel.cli tests <url> <symbol>")
                return 1
            asyncio.run(cli_tests(args[1], symbol))
        elif cmd == "impact":
            target = args[2] if len(args) > 2 else ""
            if not target:
                console.print("[red]Usage:[/red] python -m repointel.cli impact <url> <target>")
                return 1
            asyncio.run(cli_impact(args[1], target))
        elif cmd == "change-plan":
            requirement = args[2] if len(args) > 2 else ""
            target = args[3] if len(args) > 3 else ""
            if not requirement:
                console.print("[red]Usage:[/] python -m repointel.cli change-plan <url> <requirement> [target]")
                return 1
            asyncio.run(cli_change_plan(args[1], requirement, target))
        elif cmd == "update":
            asyncio.run(cli_update(args[1]))
        elif cmd == "reindex":
            force = "--force" in sys.argv
            asyncio.run(cli_reindex(args[1], force=force))
        elif cmd == "status":
            asyncio.run(cli_status(args[1]))
        elif cmd == "db-current":
            return 0 if cli_db_current() else 1
        elif cmd == "db-upgrade":
            return 0 if cli_db_upgrade() else 1
        elif cmd == "db-downgrade":
            return 0 if cli_db_downgrade() else 1
        elif cmd == "db-check":
            return 0 if cli_db_check() else 1
        elif cmd == "eval":
            return 0 if cli_eval() else 1
        else:
            console.print(f"[red]Unknown command:[/red] {cmd}")
            _print_usage()
            return 1
        return 0
    except SystemExit as exc:
        # Propagate explicit exit codes from command handlers.
        code = exc.code
        return 0 if code in (None, 0) else int(code)
    except (asyncio.CancelledError, KeyboardInterrupt):
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 2
    except Exception as exc:
        # Expected operational failures surface as concise structured errors;
        # the underlying cause stays in logs and never contains secrets.
        console.print("[red]Command FAILED[/red]")
        console.print(f"  [cyan]Stage:[/cyan] {cmd}")
        console.print(f"  [cyan]Error:[/cyan] {type(exc).__name__}: {exc}")
        console.print(
            "  [cyan]Action:[/cyan] verify the repository path/URL is reachable and the "
            "database schema is current (`python -m repointel.cli db-check`)."
        )
        if verbose:
            import traceback

            console.print("[red]--- verbose traceback ---[/red]")
            traceback.print_exc()
        return 1



def main():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    verbose = "--verbose" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--verbose"]
    if not argv:
        _print_usage()
        sys.exit(1)

    cmd = argv[0]
    # Database administration commands operate without a repository target.
    if cmd in ("db-current", "db-upgrade", "db-downgrade", "db-check", "eval"):
        raise SystemExit(_dispatch(cmd, argv, verbose))

    if len(argv) < 2:
        _print_usage()
        sys.exit(1)

    raise SystemExit(_dispatch(cmd, argv, verbose))


if __name__ == "__main__":
    main()
