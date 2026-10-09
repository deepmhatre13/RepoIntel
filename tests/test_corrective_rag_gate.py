import pytest

from packages.agentic.agents.retriever.node import ToolCallDecision, retriever_node
from packages.agentic.state.schema import EvidenceCitation, Plan, PlanTask, RepoIntelState, TaskEvidence
from packages.knowledge.corrective_rag import (
    EvidenceSufficiency,
    assess_evidence_sufficiency,
)
from packages.knowledge.query_transform import QueryVariants
from packages.knowledge.vector_store import RetrievedEvidenceItem


def make_task_evidence(score):
    return TaskEvidence(
        task_id="task-1",
        query="startup command",
        citations=[EvidenceCitation(
            file_path="README.md",
            line_start=1,
            line_end=2,
            content="Run the service with the documented command.",
            relevance_score=score,
        )],
    )


def test_evidence_sufficiency_classifies_relevance_scores():
    task = PlanTask(
        task_id="task-1",
        objective="Find startup instructions",
        retrieval_query="startup command",
        evidence_type="code",
        target_section="Usage",
    )

    assert assess_evidence_sufficiency(make_task_evidence(0.8), task) == EvidenceSufficiency.CORRECT
    assert assess_evidence_sufficiency(make_task_evidence(0.25), task) == EvidenceSufficiency.AMBIGUOUS
    assert assess_evidence_sufficiency(make_task_evidence(0.02), task) == EvidenceSufficiency.INCORRECT


@pytest.mark.asyncio
async def test_incorrect_evidence_triggers_one_wider_retrieval_before_return():
    events = []

    class Client:
        def __init__(self):
            self.decision_calls = 0

        def complete_structured(self, prompt, schema, system=None):
            if schema is QueryVariants:
                events.append("reformulate")
                return QueryVariants(queries=["startup command", "run service", "application entrypoint"])
            assert schema is ToolCallDecision
            self.decision_calls += 1
            if self.decision_calls == 1:
                return ToolCallDecision(
                    tool_name="hybrid_retrieve",
                    query_or_param="startup command",
                    reasoning="Find how to launch the service",
                )
            return ToolCallDecision(
                tool_name="finish_task",
                query_or_param="",
                reasoning="Initial search completed",
            )

    class Facade:
        def hybrid_search(self, repo_id, query, top_k):
            events.append(("initial", top_k))
            return [RetrievedEvidenceItem(
                repo_id=repo_id,
                file_path="README.md",
                symbol_name=None,
                line_start=1,
                line_end=2,
                content="A low-confidence startup mention.",
                retrieval_source="hybrid",
                relevance_score=0.02,
                chunk_id="weak",
            )]

        async def multi_query_search(self, repo_id, queries, top_k, ranking_query):
            events.append(("corrective", top_k, tuple(queries)))
            return [RetrievedEvidenceItem(
                repo_id=repo_id,
                file_path="README.md",
                symbol_name=None,
                line_start=10,
                line_end=14,
                content="Run the application with python -m repointel.cli.",
                retrieval_source="hybrid",
                relevance_score=0.82,
                chunk_id="strong",
            )]

    task = PlanTask(
        task_id="task-1",
        objective="Find startup instructions",
        retrieval_query="startup command",
        evidence_type="code",
        target_section="Usage",
    )
    state = RepoIntelState(
        run_id="run-1",
        thread_id="thread-1",
        repo_id="repo-1",
        repo_url="https://example.com/repo",
        archetype="CLI",
        plan=Plan(plan_id="plan-1", repo_id="repo-1", tasks=[task]),
    )

    result = await retriever_node(state, Facade(), Client())

    assert events[0] == ("initial", 5)
    assert events[1] == "reformulate"
    assert events[2][0:2] == ("corrective", 10)
    evidence = result["evidence_store"]["task-1"]
    assert len(evidence.citations) == 2
    assert evidence.citations[1].relevance_score == 0.82
    assert "Corrective-RAG final evidence assessment: CORRECT" in evidence.reasoning