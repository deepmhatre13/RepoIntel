import pytest
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

def test_context_assembler():
    items = [
        RetrievedEvidenceItem(repo_id="r1", file_path="f1.py", symbol_name=None, line_start=1, line_end=10, content="content 1", retrieval_source="vector", relevance_score=0.8),
        RetrievedEvidenceItem(repo_id="r1", file_path="f1.py", symbol_name=None, line_start=1, line_end=10, content="content 1", retrieval_source="vector", relevance_score=0.8),
    ]
    assembler = ContextAssembler()
    assembled = assembler.assemble(items)
    assert len(assembled) == 1
