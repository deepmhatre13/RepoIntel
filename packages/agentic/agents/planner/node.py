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
Generate a structured JSON Plan consisting of individual tasks.
Each task must specify:
1. task_id: Unique string ID
2. objective: Clear goal of what evidence to find
3. retrieval_query: Optimized search query for code/docs search
4. evidence_type: Type of evidence needed
5. target_section: Section title in final artifact
6. priority: Integer priority (1 is highest)
"""

def planner_node(state: RepoIntelState, llm_client: LLMClient) -> Dict[str, Any]:
    prompt = f"""
    Repository URL: {state.repo_url}
    Archetype: {state.archetype}
    Target Artifact: {state.artifact_type}
    Reflection Iteration: {state.reflection_iteration_count}
    
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

    plan_history = list(state.plan_history)
    if state.plan:
        plan_history.append(state.plan)

    return {
        "plan": plan,
        "plan_history": plan_history
    }
