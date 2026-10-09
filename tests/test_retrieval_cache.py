from packages.knowledge.repository_facade import RepositoryFacade
from packages.knowledge.vector_store import RetrievedEvidenceItem, VectorStore
from packages.parsing.chunking import CodeChunk


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.ttls[key] = ttl


class FakeVectorStore:
    def __init__(self):
        self.version = "content-v1"
        self.search_calls = 0

    def repository_content_hash(self, repo_id):
        return self.version

    def hybrid_search(self, repo_id, query, top_k, **kwargs):
        self.search_calls += 1
        return [RetrievedEvidenceItem(
            repo_id=repo_id,
            file_path="src/config.py",
            symbol_name="load_config",
            line_start=1,
            line_end=3,
            content=f"configuration for {query}",
            retrieval_source="hybrid",
            relevance_score=0.8,
            chunk_id=f"chunk-{self.version}",
        )]


class IdentityReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


class IdentityAssembler:
    def assemble(self, candidates):
        return candidates


def test_hybrid_cache_normalizes_queries_and_invalidates_on_content_change(monkeypatch):
    monkeypatch.delenv("REPOINTEL_RETRIEVAL_CACHE_TTL", raising=False)
    vector_store = FakeVectorStore()
    facade = RepositoryFacade(None, vector_store)
    redis_cache = FakeRedis()
    facade._retrieval_cache = redis_cache
    facade._retrieval_cache_initialized = True
    facade.reranker = IdentityReranker()
    facade.context_assembler = IdentityAssembler()

    first = facade.hybrid_search("repo-1", "Database   configuration", top_k=5)
    normalized_hit = facade.hybrid_search("repo-1", " database configuration ", top_k=5)

    assert vector_store.search_calls == 1
    assert normalized_hit[0].chunk_id == first[0].chunk_id
    assert next(iter(redis_cache.ttls.values())) == 3600

    vector_store.version = "content-v2"
    refreshed = facade.hybrid_search("repo-1", "database configuration", top_k=5)

    assert vector_store.search_calls == 2
    assert refreshed[0].chunk_id == "chunk-content-v2"
    assert len(redis_cache.values) == 2


def test_vector_store_content_hash_changes_after_reindexing():
    vector_store = VectorStore()
    original = CodeChunk(
        repo_id="repo-1",
        file_path="src/app.py",
        symbol_name="run",
        line_start=1,
        line_end=2,
        content="def run(): return 1",
        content_hash="hash-v1",
        language="Python",
    )
    vector_store.upsert_chunks([original])
    first_hash = vector_store.repository_content_hash("repo-1")

    vector_store.invalidate_paths("repo-1", ["src/app.py"])
    updated = CodeChunk(
        repo_id="repo-1",
        file_path="src/app.py",
        symbol_name="run",
        line_start=1,
        line_end=2,
        content="def run(): return 2",
        content_hash="hash-v2",
        language="Python",
    )
    vector_store.upsert_chunks([updated])

    assert vector_store.repository_content_hash("repo-1") != first_hash