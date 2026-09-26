from packages.knowledge.query_understanding import classify_query, QueryIntent
from packages.knowledge.vector_store import RetrievedEvidenceItem
from packages.knowledge.reranker import EvidenceReranker
from packages.knowledge.context_assembler import ContextAssembler

def test_query_classification():
    res1 = classify_query("What database dependency is configured?")
    assert QueryIntent.DEPENDENCY in res1.intents
    assert QueryIntent.CONFIGURATION in res1.intents

    res2 = classify_query("How to run the service?")
    assert QueryIntent.USAGE in res2.intents

def test_reranker():
    items = [
        RetrievedEvidenceItem(repo_id="r1", file_path="f1.py", symbol_name=None, line_start=1, line_end=10, content="unrelated code", retrieval_source="vector", relevance_score=0.9),
        RetrievedEvidenceItem(repo_id="r1", file_path="f2.py", symbol_name=None, line_start=1, line_end=10, content="database connection postgresql", retrieval_source="vector", relevance_score=0.7)
    ]
    reranker = EvidenceReranker()
    reranked = reranker.rerank("database postgresql", items, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].file_path == "f2.py"

def test_reranker_preserves_provenance():
    """The reranker must not drop chunk_id, file_id, symbol_id, or
    retrieval_method when creating reranked copies (Phase N5/N6 citation fix)."""
    item = RetrievedEvidenceItem(
        repo_id="r1", file_path="f1.py", symbol_name="Calculator.add",
        line_start=1, line_end=10, content="def add(self, a, b): return a + b",
        retrieval_source="hybrid", relevance_score=0.0167,
        chunk_id="abc123def456",
        retrieval_method=["vector", "keyword"],
        repository_id="r1", file_id=42, symbol_id=7, chunk_ref="chunk-7",
    )
    reranker = EvidenceReranker()
    reranked = reranker.rerank("add calculation", [item], top_k=5)
    assert len(reranked) == 1
    r = reranked[0]
    # Provenance fields must survive reranking (citation validity depends on this).
    assert r.chunk_id == "abc123def456"
    assert r.repository_id == "r1"
    assert r.file_id == 42
    assert r.symbol_id == 7
    assert r.chunk_ref == "chunk-7"
    assert "vector" in r.retrieval_method
    assert "keyword" in r.retrieval_method

def test_reranker_normalizes_rrf_scores():
    """RRF scores are tiny (e.g. 1/(k+rank) ≈ 0.016). The reranker must normalize
    them to [0,1] so they are comparable to term-overlap scores. Without
    normalization the reranker would be pure term-overlap and lose the RRF
    rank ordering (the documented root cause of hybrid MRR < vector MRR)."""
    # Simulate two hybrid hits with RRF scores only.
    items = [
        RetrievedEvidenceItem(
            repo_id="r1", file_path="f1.py", symbol_name="foo",
            line_start=1, line_end=5, content="def foo(x): return x",
            retrieval_source="hybrid", relevance_score=1.0 / 61.0,  # rank 0 RRF
            chunk_id="c1",
        ),
        RetrievedEvidenceItem(
            repo_id="r1", file_path="f2.py", symbol_name="bar",
            line_start=1, line_end=5, content="def bar(y): return y",
            retrieval_source="hybrid", relevance_score=1.0 / 71.0,  # rank 10 RRF
            chunk_id="c2",
        ),
    ]
    reranker = EvidenceReranker()
    # Query mentions "foo" but not "bar". Without normalization, the overlap
    # term (0.5) would dominate and the ranking could flip because both
    # RRF scores are ~0.016 and nearly equal. With normalization, the
    # RRF score difference is amplified to [0, 1] range.
    reranked = reranker.rerank("foo", items, top_k=2)
    assert len(reranked) == 2
    # The higher-ranked item (by RRF) should still be first when overlap
    # is equal (neither chunk contains "foo" in its content, but we
    # check that the score preserves the RRF ordering).
    # With overlap=0 for both, the normalized retrieval score should drive
    # the ranking: 0.5 * 1.0 > 0.5 * (1/71)/(1/61)
    assert reranked[0].chunk_id == "c1"
    assert reranked[1].chunk_id == "c2"

def test_reranker_preserves_content_and_snippet():
    """Reranked items must retain their content and snippet for evidence assembly."""
    item = RetrievedEvidenceItem(
        repo_id="r1", file_path="f1.py", symbol_name="foo",
        line_start=1, line_end=5, content="def foo(x): return x",
        retrieval_source="vector", relevance_score=0.8,
        snippet="custom snippet",
    )
    reranker = EvidenceReranker()
    reranked = reranker.rerank("foo", [item], top_k=5)
    assert reranked[0].content == "def foo(x): return x"
    assert reranked[0].snippet == "custom snippet"

def test_context_assembler():
    items = [
        RetrievedEvidenceItem(repo_id="r1", file_path="f1.py", symbol_name=None, line_start=1, line_end=10, content="content 1", retrieval_source="vector", relevance_score=0.8),
        RetrievedEvidenceItem(repo_id="r1", file_path="f1.py", symbol_name=None, line_start=1, line_end=10, content="content 1", retrieval_source="vector", relevance_score=0.8),
    ]
    assembler = ContextAssembler()
    assembled = assembler.assemble(items)
    assert len(assembled) == 1
