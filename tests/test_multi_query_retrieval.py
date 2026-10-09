import pytest

from packages.agentic.agents.planner.node import _normalize_evidence_type
from packages.agentic.agents.retriever.node import ToolCallDecision, retriever_node
from packages.agentic.state.schema import Plan, PlanTask, RepoIntelState
from packages.knowledge.query_transform import (
    HypotheticalDocument,
    QueryVariants,
    generate_hypothetical_document,
    is_broad_query,
    transform_query,
)
from packages.knowledge.repository_facade import RepositoryFacade
from packages.knowledge.vector_store import RetrievedEvidenceItem


def test_broad_query_heuristic():
    assert not is_broad_query("database connection")
    assert is_broad_query("database and cache behavior")
    assert is_broad_query("Explain the overall startup, routing, dependencies, configuration, and storage behavior")


def test_query_transform_returns_distinct_reformulations():
    class Client:
        def complete_structured(self, prompt, schema, system=None):
            assert schema is QueryVariants
            return QueryVariants(queries=["db config", "cache setup", "db config"])

    assert transform_query("database and cache setup", Client()) == [
        "db config",
        "cache setup",
    ]


def test_planner_preserves_hyde_evidence_types():
    for evidence_type in ("ARCHITECTURE", "DESIGN_RATIONALE"):
        task = PlanTask(
            task_id="task-1",
            objective="Explain the system design",
            retrieval_query="system design",
            evidence_type=evidence_type,
            target_section="Architecture",
        )

        _normalize_evidence_type(task)

        assert task.evidence_type == evidence_type.casefold()


def test_hypothetical_document_is_generated_as_structured_output():
    class Client:
        def complete_structured(self, prompt, schema, system=None):
            assert schema is HypotheticalDocument
            assert "not repository evidence" in prompt
            return HypotheticalDocument(
                document="A service may coordinate storage and request handling."
            )

    assert generate_hypothetical_document("Describe system architecture", Client()) == (
        "A service may coordinate storage and request handling."
    )


@pytest.mark.asyncio
async def test_multi_query_search_merges_chunks_before_one_rerank():
    shared = RetrievedEvidenceItem(
        repo_id="repo-1",
        file_path="src/db.py",
        symbol_name="connect",
        line_start=1,
        line_end=4,
        content="def connect(): pass",
        retrieval_source="hybrid",
        relevance_score=0.02,
        chunk_id="shared",
    )
    unique_db = RetrievedEvidenceItem(
        repo_id="repo-1",
        file_path="src/db.py",
        symbol_name="configure_db",
        line_start=6,
        line_end=9,
        content="def configure_db(): pass",
        retrieval_source="hybrid",
        relevance_score=0.015,
        chunk_id="db-only",
    )
    unique_cache = RetrievedEvidenceItem(
        repo_id="repo-1",
        file_path="src/cache.py",
        symbol_name="configure_cache",
        line_start=1,
        line_end=4,
        content="def configure_cache(): pass",
        retrieval_source="hybrid",
        relevance_score=0.015,
        chunk_id="cache-only",
    )

    class VectorStore:
        def __init__(self):
            self.queries = []

        def hybrid_search(self, repo_id, query, top_k):
            self.queries.append(query)
            if query == "database settings":
                return [shared, unique_db]
            return [shared, unique_cache]

    class Reranker:
        def __init__(self):
            self.calls = []

        def rerank(self, query, items, top_k):
            self.calls.append((query, items, top_k))
            return items

    class Assembler:
        def assemble(self, items):
            return items

    vector_store = VectorStore()
    reranker = Reranker()
    facade = RepositoryFacade(None, vector_store)
    facade.reranker = reranker
    facade.context_assembler = Assembler()

    results = await facade.multi_query_search(
        "repo-1",
        ["database settings", "cache configuration"],
        top_k=5,
        ranking_query="database and cache settings",
    )

    assert set(vector_store.queries) == {"database settings", "cache configuration"}
    assert len(reranker.calls) == 1
    assert reranker.calls[0][0] == "database and cache settings"
    assert {item.chunk_id for item in reranker.calls[0][1]} == {
        "shared", "db-only", "cache-only"
    }
    assert next(item for item in results if item.chunk_id == "shared").relevance_score == pytest.approx(0.04)


def test_hyde_search_uses_hypothetical_dense_query_and_original_lexical_query():
    item = RetrievedEvidenceItem(
        repo_id="repo-1",
        file_path="src/app.py",
        symbol_name="create_app",
        line_start=1,
        line_end=4,
        content="def create_app(): pass",
        retrieval_source="hybrid",
        relevance_score=0.5,
    )

    class VectorStore:
        received = None

        def hybrid_search(self, repo_id, query, top_k, *, vector_query, keyword_query):
            self.received = (repo_id, query, top_k, vector_query, keyword_query)
            return [item]

    class Reranker:
        received = None

        def rerank(self, query, candidates, top_k):
            self.received = (query, candidates, top_k)
            return candidates

    class Assembler:
        def assemble(self, candidates):
            return candidates

    vector_store = VectorStore()
    reranker = Reranker()
    facade = RepositoryFacade(None, vector_store)
    facade.reranker = reranker
    facade.context_assembler = Assembler()

    facade.hyde_search(
        "repo-1",
        "What is the architecture?",
        "A service separates API, persistence, and background processing.",
        top_k=5,
    )

    assert vector_store.received == (
        "repo-1",
        "What is the architecture?",
        10,
        "A service separates API, persistence, and background processing.",
        "What is the architecture?",
    )
    assert reranker.received[0] == "What is the architecture?"


@pytest.mark.asyncio
async def test_retriever_uses_multi_query_search_for_broad_decision():
    transformed = ["database configuration", "cache configuration", "startup configuration"]

    class Client:
        def __init__(self):
            self.decision_calls = 0

        def complete_structured(self, prompt, schema, system=None):
            if schema is QueryVariants:
                return QueryVariants(queries=transformed)
            assert schema is ToolCallDecision
            self.decision_calls += 1
            if self.decision_calls == 1:
                return ToolCallDecision(
                    tool_name="hybrid_retrieve",
                    query_or_param="How do database and cache configuration work overall?",
                    reasoning="Broad compound request",
                )
            return ToolCallDecision(
                tool_name="finish_task",
                query_or_param="",
                reasoning="Enough evidence",
            )

    class Facade:
        def __init__(self):
            self.queries = None

        async def multi_query_search(self, repo_id, queries, top_k, ranking_query):
            self.queries = queries
            return [RetrievedEvidenceItem(
                repo_id=repo_id,
                file_path="src/config.py",
                symbol_name=None,
                line_start=1,
                line_end=2,
                content="DATABASE_URL is configured here",
                retrieval_source="hybrid",
                relevance_score=0.5,
                chunk_id="config-chunk",
            )]

        def hybrid_search(self, *args, **kwargs):
            raise AssertionError("broad query should use multi_query_search")

    task = PlanTask(
        task_id="task-1",
        objective="Find database and cache configuration",
        retrieval_query="database and cache configuration",
        evidence_type="code",
        target_section="Configuration",
    )
    state = RepoIntelState(
        run_id="run-1",
        thread_id="thread-1",
        repo_id="repo-1",
        repo_url="https://example.com/repo",
        archetype="SERVICE",
        plan=Plan(plan_id="plan-1", repo_id="repo-1", tasks=[task]),
    )
    facade = Facade()

    result = await retriever_node(state, facade, Client())

    assert facade.queries == transformed
    assert len(result["evidence_store"]["task-1"].citations) == 1


@pytest.mark.asyncio
async def test_architecture_retrieval_uses_hyde_search():
    class Client:
        def __init__(self):
            self.decision_calls = 0

        def complete_structured(self, prompt, schema, system=None):
            if schema is HypotheticalDocument:
                return HypotheticalDocument(document="A layered service delegates persistence to a repository.")
            assert schema is ToolCallDecision
            self.decision_calls += 1
            if self.decision_calls == 1:
                return ToolCallDecision(
                    tool_name="hybrid_retrieve",
                    query_or_param="What is the system architecture?",
                    reasoning="Architecture task",
                )
            return ToolCallDecision(
                tool_name="finish_task",
                query_or_param="",
                reasoning="Evidence gathered",
            )

    class Facade:
        received = None

        def hyde_search(self, repo_id, query, hypothetical_document, top_k):
            self.received = (repo_id, query, hypothetical_document, top_k)
            return [RetrievedEvidenceItem(
                repo_id=repo_id,
                file_path="src/main.py",
                symbol_name="create_app",
                line_start=1,
                line_end=3,
                content="The application factory assembles the service components.",
                retrieval_source="hybrid",
                relevance_score=0.8,
                chunk_id="architecture-1",
            )]

        def hybrid_search(self, *args, **kwargs):
            raise AssertionError("architecture task should use HyDE")

    task = PlanTask(
        task_id="task-architecture",
        objective="Describe the system architecture",
        retrieval_query="system architecture",
        evidence_type="architecture",
        target_section="Architecture",
    )
    state = RepoIntelState(
        run_id="run-1",
        thread_id="thread-1",
        repo_id="repo-1",
        repo_url="https://example.com/repo",
        archetype="SERVICE",
        plan=Plan(plan_id="plan-1", repo_id="repo-1", tasks=[task]),
    )
    facade = Facade()

    await retriever_node(state, facade, Client())

    assert facade.received == (
        "repo-1",
        "What is the system architecture?",
        "A layered service delegates persistence to a repository.",
        5,
    )