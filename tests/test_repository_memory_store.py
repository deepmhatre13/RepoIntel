import pytest
from langgraph.store.memory import InMemoryStore

from packages.agentic.agents.planner.node import planner_node
from packages.agentic.state.schema import (
    HumanFeedback,
    Plan,
    PlanTask,
    RepoIntelState,
    ReviewIssue,
)
from packages.knowledge.persistent_memory import (
    PersistentMemoryStore,
    open_repository_memory_store,
)


@pytest.mark.asyncio
async def test_repository_memory_aggregates_cross_run_context_by_repository():
    memory = PersistentMemoryStore(InMemoryStore())
    issue = ReviewIssue(
        issue_type="COVERAGE_GAP",
        description="Document the Docker Compose setup.",
        target_section="Installation",
    )

    await memory.save_archetype("repo-1", "SERVICE")
    await memory.save_review_issues("repo-1", "run-1", [issue])
    await memory.save_review_issues("repo-1", "run-2", [issue])
    await memory.save_review_issues("repo-2", "run-3", [issue])
    await memory.save_human_correction(
        "repo-1",
        "run-2",
        HumanFeedback(
            approved=False,
            decision="REQUEST_REVISION",
            comments="Include the compose startup command.",
        ),
    )

    context = await memory.load_repository_context("repo-1")
    other_repo_context = await memory.load_repository_context("repo-2")

    assert context["archetype"] == "SERVICE"
    assert context["recurring_review_issues"] == [
        {"description": "Document the Docker Compose setup.", "runs": 2}
    ]
    assert context["human_review_corrections"][0]["comments"] == (
        "Include the compose startup command."
    )
    assert other_repo_context["recurring_review_issues"] == []


@pytest.mark.asyncio
async def test_sqlite_configuration_uses_local_langgraph_store():
    async with open_repository_memory_store("sqlite+aiosqlite:///:memory:") as store:
        memory = PersistentMemoryStore(store)
        await memory.save_archetype("repo-local", "CLI")

        context = await memory.load_repository_context("repo-local")

    assert context["archetype"] == "CLI"


def test_planner_receives_cross_run_repository_context():
    class CapturingClient:
        prompt = ""

        def complete_structured(self, prompt, schema, system=None):
            self.prompt = prompt
            return Plan(
                plan_id="plan-1",
                repo_id="repo-1",
                artifact_type="README",
                tasks=[PlanTask(
                    task_id="task-1",
                    objective="Find startup instructions",
                    retrieval_query="startup instructions",
                    evidence_type="code",
                    target_section="Usage",
                )],
            )

    client = CapturingClient()
    planner_node(
        RepoIntelState(
            run_id="run-1",
            thread_id="thread-1",
            repo_id="repo-1",
            repo_url="https://example.com/repo",
            archetype="SERVICE",
        ),
        client,
        {"recurring_review_issues": [{"description": "Add Docker setup."}]},
    )

    assert "Add Docker setup." in client.prompt