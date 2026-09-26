"""Regression: incremental update must not force embedding-model load (Phase 11).

The Phase 11 manual finding showed an unchanged-repository incremental update
spending ~28s in KNOWLEDGE almost entirely on embedding-model initialization.
Root cause: ``persist_parsing_results`` read ``embedding_service.dimension``
during the model-mismatch check, and ``EmbeddingService.dimension`` lazily loads
the model. ``_resolve_current_embedding_dimension`` now reuses the dimension
recorded on stored same-model rows instead of loading the model.
"""
import pytest


class _DimensionTrackingProvider:
    """Provider whose ``dimension`` property records every access."""

    def __init__(self, dim: int = 4):
        self._dim = dim
        self.dimension_accesses = 0
        self.embed_calls = 0

    @property
    def model_name(self):
        return "tracked-model"

    @property
    def provider_name(self):
        return "tracked"

    @property
    def dimension(self):
        self.dimension_accesses += 1
        return self._dim

    def embed_texts(self, texts):
        self.embed_calls += 1
        return [[float(len(t))] * self._dim for t in texts]

    def embed_query(self, q):
        return self.embed_texts([q])[0]


@pytest.fixture
def chunk_service():
    """Return (embedding_service, provider) wired to the tracking provider."""
    from packages.knowledge.embeddings import EmbeddingService

    provider = _DimensionTrackingProvider()
    return EmbeddingService(model_name="tracked-model", provider=provider), provider


@pytest.mark.anyio
async def test_unchanged_incremental_update_does_not_touch_dimension(
    tmp_path, chunk_service
):
    """A no-op incremental update must not access .dimension (no model load)."""
    pytest.importorskip("sqlalchemy.ext.asyncio")
    # Import aiosqlite so the async engine can open tmp databases.
    pytest.importorskip("aiosqlite")

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from packages.knowledge.models import Base
    from packages.knowledge import persistence
    from packages.shared import IngestionManifest
    from packages.parsing.pipeline import ParsingPipelineResult

    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 't.db')}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    service, provider = chunk_service

    manifest = IngestionManifest(
        repo_url="p11", local_path=str(tmp_path), size_mb=0.0, file_count=1,
        languages={"python": 1}, manifest_files=[], entrypoints=[], ci_configs=[],
    )
    from packages.parsing.python_ast import SymbolNode
    from packages.parsing.chunking import CodeChunk

    def _mk_chunk(i):
        return CodeChunk(
            repo_id="p11", file_path="m.py", symbol_name=None,
            line_start=i * 10, line_end=i * 10 + 5,
            content=f"def f{i}():\n    pass\n",
            content_hash=f"hash{i}",
            language="python",
        )

    parse_res = ParsingPipelineResult(
        repo_id="p11",
        symbols=[
            SymbolNode(file_path="m.py", symbol_name=f"f{i}",
                       symbol_type="function", line_start=i * 10, line_end=i * 10 + 5,
                       signature=f"def f{i}()")
            for i in range(2)
        ],
        edges=[],
        chunks=[_mk_chunk(0), _mk_chunk(1)],
    )

    async with SessionLocal() as session:
        # First persist: fresh index -> no stored rows -> dimension IS accessed
        await persistence.persist_parsing_results(
            session, manifest, parse_res,
            embedding_service=service,
            embedding_model="tracked-model",
            embedding_provider="tracked",
        )
        await session.commit()

    fresh_accesses = provider.dimension_accesses
    fresh_embed_calls = provider.embed_calls
    # Fresh index: no stored rows agree, so the real dimension is consulted.
    assert fresh_accesses >= 1

    # Second persist with the SAME content: stored rows exist and agree, so the
    # mismatch check must reuse the stored dimension, not load the model.
    async with SessionLocal() as session:
        await persistence.persist_parsing_results(
            session, manifest, parse_res,
            embedding_service=service,
            embedding_model="tracked-model",
            embedding_provider="tracked",
        )
        await session.commit()

    assert provider.dimension_accesses == fresh_accesses, (
        "persist_parsing_results accessed .dimension on an unchanged re-index; "
        "this forces an embedding-model load and slows incremental updates"
    )
    assert provider.embed_calls == fresh_embed_calls, (
        "persist_parsing_results regenerated embeddings for unchanged chunks"
    )

    # All chunks have embeddings with the same provenance and no duplicates.
    from sqlalchemy import select, func
    from packages.knowledge.models import CodeChunkModel

    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(CodeChunkModel).where(CodeChunkModel.repo_id == "p11")
            )
        ).scalars().all()
        assert len(rows) == 2, f"expected 2 chunks, got {len(rows)}"
        for r in rows:
            assert r.embedding is not None
            assert r.embedding_dimension == 4
            assert r.embedding_model == "tracked-model"
            assert r.embedding_provider == "tracked"
        dup_count = (
            await session.execute(
                select(func.count()).select_from(CodeChunkModel)
                .where(CodeChunkModel.repo_id == "p11")
                .group_by(
                    CodeChunkModel.file_path, CodeChunkModel.line_start,
                    CodeChunkModel.line_end, CodeChunkModel.content_hash,
                )
            )
        ).scalars().all()
        assert all(c == 1 for c in dup_count), "duplicate chunk rows detected"

    await engine.dispose()
