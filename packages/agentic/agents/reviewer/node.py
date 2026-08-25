import json
from typing import Dict, Any, List
from packages.agentic.state.schema import (
    RepoIntelState, ReviewReport, ReviewVerdict, ReviewIssue, IssueType
)
from packages.agentic.llm_client import LLMClient
from packages.knowledge.repository_facade import RepositoryFacade

REVIEWER_SYSTEM_PROMPT = """
You are an expert documentation reviewer and technical auditor.
Your job is to evaluate an Artifact Draft for Groundedness, Completeness, Structural Quality, and Citation Coverage.

Groundedness Verification Results:
{grounding_verification}

Output a structured ReviewReport:
- verdict: "APPROVE", "REVISE_SYNTHESIS", "REVISE_PLAN", or "NEEDS_HUMAN_REVIEW"
- issues: List of ReviewIssue (issue_type: WORDING, COVERAGE_GAP, GROUNDING_FAILURE; severity: LOW, MEDIUM, HIGH)
- groundedness_score: Float between 0.0 and 1.0
- coverage_score: Float between 0.0 and 1.0

Rules for Verdict:
- If grounding verification failed or hallucination detected -> REVISE_SYNTHESIS
- If major missing sections (high severity COVERAGE_GAP) -> REVISE_PLAN
- If scores are high (>= 0.85) and no high-severity issues -> APPROVE
"""

async def reviewer_node(state: RepoIntelState, facade: RepositoryFacade, llm_client: LLMClient) -> Dict[str, Any]:
    if not state.artifact_draft:
        return {"errors": state.errors + ["Reviewer executed without draft"]}

    # 1. Deterministic citation verification
    verification_results = []
    has_grounding_failure = False

    for sec in state.artifact_draft.sections:
        for citation in sec.citations:
            if citation.file_path == "manifests/dependencies.json":
                continue
            meta = await facade.get_file_metadata(state.repo_id, citation.file_path)
            if not meta:
                verification_results.append(f"FAILED: File '{citation.file_path}' cited in section '{sec.section_title}' does not exist in repository.")
                has_grounding_failure = True
            else:
                verification_results.append(f"PASSED: Citation '{citation.file_path}:{citation.line_start}-{citation.line_end}' verified.")

    grounding_str = "\n".join(verification_results) if verification_results else "No external file citations to verify."

    # 2. LLM evaluation
    draft_data = state.artifact_draft.model_dump_json()
    prompt = f"""
    Draft Content:
    {draft_data}
    
    Reflection Iteration: {state.reflection_iteration_count}/{state.max_reflection_iterations}
    
    Audit this draft and provide a ReviewReport.
    """

    system_prompt = REVIEWER_SYSTEM_PROMPT.format(grounding_verification=grounding_str)

    report = llm_client.complete_structured(
        prompt=prompt,
        schema=ReviewReport,
        system=system_prompt
    )

    # Force grounding failure if deterministic check failed
    if has_grounding_failure and report.verdict == ReviewVerdict.APPROVE:
        report.verdict = ReviewVerdict.REVISE_SYNTHESIS
        report.issues.append(ReviewIssue(
            issue_type=IssueType.GROUNDING_FAILURE,
            description="Deterministic check detected non-existent cited files.",
            severity="HIGH"
        ))

    # Check iteration budget
    new_iteration = state.reflection_iteration_count + 1
    if new_iteration > state.max_reflection_iterations and report.verdict != ReviewVerdict.APPROVE:
        report.verdict = ReviewVerdict.NEEDS_HUMAN_REVIEW

    review_history = list(state.review_history)
    if state.review_report:
        review_history.append(state.review_report)

    return {
        "review_report": report,
        "review_history": review_history,
        "reflection_iteration_count": new_iteration,
        "route_decision": report.verdict.value
    }
