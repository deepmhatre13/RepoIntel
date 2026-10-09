import json
import uuid
from typing import Dict, Any
from packages.agentic.state.schema import RepoIntelState, Plan
from packages.agentic.llm_client import LLMClient

PLANNER_SYSTEM_PROMPT = """
You are an expert software architect and technical documentation planner.
Your goal is to inspect a software repository's ingestion manifest and create a targeted, structured retrieval plan to generate high-quality repository documentation.

Supported artifact types:
- README: Overview, Installation, Usage, Configuration, Testing.
- ARCHITECTURE: System Overview, Components, Data Flow, Dependencies, Runtime Flow, Configuration.
- API_DOCUMENTATION: Endpoints, Request Models, Response Models, Authentication, Errors, Examples.
- DEVELOPER_GUIDE: Setup, Development Workflow, Code Standards, Debugging, Testing.
- CONFIGURATION_GUIDE: Environment Variables, Config Files, Database Settings, Third-Party Credentials.
- DEPENDENCY_REPORT: Runtime Dependencies, Dev Dependencies, Frameworks, Vulnerabilities/Ecosystems.
- CODEBASE_MAP: Directory Structure, Key Modules, Symbol Map, Entrypoints.
- ONBOARDING_GUIDE: Quickstart, Repository Layout, Key Workflows, Best Practices.
- CHANGE_SUMMARY: Recent Commit Summaries, Modified Symbols, Diff Impact, Version Changes.

Do NOT generate the document text yourself.
Generate a structured JSON Plan with a document_outline and individual evidence tasks.
Derive the outline from the requested artifact type and repository archetype
(web service, CLI, library, or monorepo). Keep its order reader-friendly and
make the task target_section values align with the outline.
Each task must specify:
1. task_id: Unique string ID
2. objective: Clear goal of what evidence to find
3. retrieval_query: Optimized search query for code/docs search
4. evidence_type: One of code, manifest, symbol_graph, file_metadata, tests, git_history
5. target_section: Section title in final artifact
6. priority: Integer priority (1 is highest)
"""

_EVIDENCE_TYPES = {
    "code", "manifest", "symbol_graph", "file_metadata", "tests", "git_history",
    "architecture", "design_rationale",
}


def _normalize_evidence_type(task) -> None:
    value = task.evidence_type.strip().lower().replace("-", "_")
    if value in _EVIDENCE_TYPES:
        task.evidence_type = value
    elif any(term in value for term in ("depend", "package", "manifest")):
        task.evidence_type = "manifest"
    elif any(term in value for term in ("graph", "symbol", "call", "import")):
        task.evidence_type = "symbol_graph"
    elif any(term in value for term in ("test", "coverage")):
        task.evidence_type = "tests"
    elif any(term in value for term in ("history", "commit", "change")):
        task.evidence_type = "git_history"
    elif any(term in value for term in ("file", "metadata", "path")):
        task.evidence_type = "file_metadata"
    else:
        task.evidence_type = "code"

def planner_node(
    state: RepoIntelState,
    llm_client: LLMClient,
    repository_memory: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    memory_context = json.dumps(repository_memory or {}, indent=2)
    prompt = f"""
    Repository URL: {state.repo_url}
    Archetype: {state.archetype}
    Target Artifact: {state.artifact_type}
    Reflection Iteration: {state.reflection_iteration_count}
    
    Cross-run repository memory (prior archetype, recurring review issues, and human corrections):
    {memory_context}

    Review Feedback from previous runs (if any):
    {state.review_report.model_dump_json() if state.review_report else "None"}
    
    Please produce a structured Plan for retrieving evidence.
    """
    
    plan = llm_client.complete_structured(
        prompt=prompt,
        schema=Plan,
        system=PLANNER_SYSTEM_PROMPT
    )
    plan.plan_id = str(uuid.uuid4())
    plan.repo_id = state.repo_id
    plan.artifact_type = state.artifact_type
    for task in plan.tasks:
        _normalize_evidence_type(task)
    outline = list(dict.fromkeys(
        section.strip() for section in plan.document_outline if section.strip()
    ))
    for task in plan.tasks:
        if task.target_section not in outline:
            outline.append(task.target_section)
    plan.document_outline = outline or list(dict.fromkeys(
        task.target_section for task in plan.tasks
    ))

    plan_history = list(state.plan_history)
    if state.plan:
        plan_history.append(state.plan)

    return {
        "plan": plan,
        "plan_history": plan_history
    }
