import pytest
from pydantic import ValidationError

from packages.agentic.agents.planner.node import planner_node
from packages.agentic.llm_client import LLMClient
from packages.agentic.prompt_safety import enclose_repository_evidence
from packages.agentic.state.schema import (
    Plan,
    PlanTask,
    RepoIntelState,
    ReviewIssue,
    ReviewReport,
    ReviewVerdict,
)


def make_state(artifact_type="README", review_report=None):
    return RepoIntelState(
        run_id="planner-run",
        thread_id="planner-thread",
        repo_id="planner-repo",
        repo_url="https://example.com/repo.git",
        archetype="SERVICE",
        artifact_type=artifact_type,
        review_report=review_report,
    )


def test_degraded_plan_matches_requested_artifact_type():
    result = planner_node(make_state("ARCHITECTURE"), LLMClient(api_key=""))
    plan = result["plan"]

    assert plan.artifact_type == "ARCHITECTURE"
    assert plan.document_outline == [task.target_section for task in plan.tasks]
    assert {task.target_section for task in plan.tasks} >= {
        "System Overview", "Components", "Data Flow", "Runtime Flow"
    }
    assert "Installation" not in {task.target_section for task in plan.tasks}
    assert {task.evidence_type for task in plan.tasks} <= {
        "code", "manifest", "symbol_graph", "file_metadata", "tests", "git_history"
    }


def test_degraded_plan_adds_high_severity_review_gap():
    feedback = ReviewReport(
        verdict=ReviewVerdict.REVISE_PLAN,
        issues=[ReviewIssue(
            issue_type="COVERAGE_GAP",
            description="Explain how runtime security is configured.",
            target_section="Runtime Security",
            severity="HIGH",
        )],
    )

    plan = planner_node(
        make_state("ARCHITECTURE", feedback), LLMClient(api_key="")
    )["plan"]

    targeted_task = next(
        task for task in plan.tasks if task.target_section == "Runtime Security"
    )
    assert "runtime security" in targeted_task.retrieval_query.lower()
    assert targeted_task.evidence_type == "code"


def test_live_plan_unknown_evidence_type_is_normalized():
    class PlanClient:
        def complete_structured(self, prompt, schema, system=None):
            assert schema is Plan
            return Plan(
                plan_id="remote-plan",
                repo_id="remote-repo",
                tasks=[PlanTask(
                    task_id="task-1",
                    objective="Inspect dependencies",
                    retrieval_query="package dependencies",
                    evidence_type="dependency_manifest_v2",
                    target_section="Dependencies",
                )],
            )

    result = planner_node(make_state(), PlanClient())

    assert result["plan"].tasks[0].evidence_type == "manifest"
    assert result["plan"].document_outline == ["Dependencies"]


@pytest.mark.parametrize(
    ("archetype", "expected_sections"),
    [
        ("CLI", ["Commands", "Usage"]),
        ("LIBRARY", ["API Reference", "Examples"]),
        ("MONOREPO", ["Repository Structure", "Packages"]),
        ("SERVICE", ["API", "Configuration"]),
    ],
)
def test_readme_outline_matches_repository_archetype(archetype, expected_sections):
    state = make_state("README")
    state.archetype = archetype

    plan = planner_node(state, LLMClient(api_key=""))["plan"]

    assert plan.document_outline == [task.target_section for task in plan.tasks]
    assert set(expected_sections) <= set(plan.document_outline)


def test_llm_mode_rejects_unsupported_values():
    state_data = make_state().model_dump()
    state_data["llm_mode"] = "SILENTLY_FAKE_LIVE"
    with pytest.raises(ValidationError):
        RepoIntelState.model_validate(state_data)


def test_degraded_retriever_selects_manifest_from_task_evidence_type():
    client = LLMClient(api_key="")
    task_context = enclose_repository_evidence(
        '{"target_section":"Runtime Dependencies",'
        '"retrieval_query":"runtime package list",'
        '"evidence_type":"manifest"}'
    )
    prompt = (
        f"Task context:\n{task_context}\n"
        "Current Round: 1/4\nCitations Gathered So Far: 0"
    )

    decision = client._generate_tool_call_decision(prompt)

    assert decision.tool_name == "get_dependency_manifest"


def test_degraded_retriever_finishes_at_citation_budget():
    client = LLMClient(api_key="")
    task_context = enclose_repository_evidence(
        '{"target_section":"Overview","retrieval_query":"entry points",'
        '"evidence_type":"code"}'
    )
    prompt = (
        f"Task context:\n{task_context}\n"
        "Current Round: 2/4\nCitations Gathered So Far: 3"
    )

    decision = client._generate_tool_call_decision(prompt)

    assert decision.tool_name == "finish_task"