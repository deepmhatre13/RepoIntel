import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Dict, Any, Iterable, Optional
from sqlalchemy import delete, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from packages.knowledge.models import (
    RepositoryModel, FileModel, SymbolModel, SymbolEdgeModel,
    DependencyModel, CodeChunkModel, FileFingerprintModel,
    TestCaseModel, TestToCodeRelationModel, CommitModel
)
from packages.shared import IngestionManifest
from packages.shared.telemetry import telemetry
from packages.parsing.pipeline import ParsingPipelineResult

logger = logging.getLogger(__name__)


def _validate_embedding(
    vector: Optional[list],
    expected_dimension: Optional[int],
    *,
    repo_id: Optional[str] = None,
    chunk_id: Optional[int] = None,
    source: str = "unknown",
) -> None:
    if vector is None:
        return
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimension}, got {len(vector)}"
        )
    for index, value in enumerate(vector):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            sample = list(vector[:5]) if hasattr(vector, "__getitem__") else []
            logger.error(
                "Invalid embedding component: repo_id=%s chunk_id=%s source=%s "
                "value_type=%s length=%s first_element_types=%s "
                "first_values=%s failing_index=%s failing_value=%r",
                repo_id,
                chunk_id,
                source,
                type(vector).__name__,
                len(vector),
                [type(item).__name__ for item in sample],
                [repr(item)[:48] for item in sample],
                index,
                value,
            )
            raise
        if not finite:
            raise ValueError("Embedding contains a non-finite value")


async def _ensure_repository_row(
    session: AsyncSession,
    repo_id: str,
    manifest: IngestionManifest,
    head_commit: Optional[str] = None,
    branch: Optional[str] = None,
) -> RepositoryModel:
    """Create or update the repository row (idempotent via merge)."""
    existing = (
        await session.execute(select(RepositoryModel).where(RepositoryModel.id == repo_id))
    ).scalar_one_or_none()
    manifest_json = json.dumps({
        "manifest_files": manifest.manifest_files,
        "entrypoints": manifest.entrypoints,
        "ci_configs": manifest.ci_configs,
        "languages": manifest.languages,
    })
    archetype = manifest.archetype.value if hasattr(manifest.archetype, "value") else str(manifest.archetype)
    if existing:
        repo = existing
        repo.archetype = archetype
        repo.size_mb = manifest.size_mb
        repo.file_count = manifest.file_count
        repo.manifest_data = manifest_json
    else:
        repo = RepositoryModel(
            id=repo_id,
            repo_url=manifest.repo_url,
            archetype=archetype,
            size_mb=manifest.size_mb,
            file_count=manifest.file_count,
            manifest_data=manifest_json,
        )
    if head_commit:
        repo.head_commit = head_commit
    if branch is not None:
        repo.branch = branch
    await session.merge(repo)
    return repo


# ---------------------------------------------------------------------------
# Scoped invalidation (Phase 41.5 cache invalidation)
# ---------------------------------------------------------------------------

async def invalidate_file_rows(session: AsyncSession, repo_id: str, file_paths: Iterable[str]) -> Dict[str, int]:
    """Delete all persisted knowledge rows scoped to specific file paths.

    Returns counts of deleted rows per table so invalidation is observable.
    """
    paths = list(file_paths)
    counts = {"symbols": 0, "symbol_edges": 0, "chunks": 0, "files": 0}
    if not paths:
        return counts

    res = await session.execute(
        delete(SymbolModel).where(SymbolModel.repo_id == repo_id, SymbolModel.file_path.in_(paths))
    )
    counts["symbols"] += res.rowcount or 0

    res = await session.execute(
        delete(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id, SymbolEdgeModel.source_file.in_(paths)
        )
    )
    counts["symbol_edges"] += res.rowcount or 0

    # Edges may also reference the files as targets
    res = await session.execute(
        delete(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id, SymbolEdgeModel.target_file.in_(paths)
        )
    )
    counts["symbol_edges"] += res.rowcount or 0

    res = await session.execute(
        delete(CodeChunkModel).where(CodeChunkModel.repo_id == repo_id, CodeChunkModel.file_path.in_(paths))
    )
    counts["chunks"] += res.rowcount or 0

    res = await session.execute(
        delete(FileModel).where(FileModel.repo_id == repo_id, FileModel.file_path.in_(paths))
    )
    counts["files"] += res.rowcount or 0

    # Test mappings referencing invalidated production files are stale too.
    await session.execute(
        delete(TestToCodeRelationModel).where(
            TestToCodeRelationModel.repo_id == repo_id,
            TestToCodeRelationModel.target_file.in_(paths),
        )
    )
    await session.execute(
        delete(TestCaseModel).where(TestCaseModel.repo_id == repo_id, TestCaseModel.file_path.in_(paths))
    )
    await session.flush()
    return counts


async def persist_file_fingerprints(session: AsyncSession, repo_id: str, fingerprints: Dict[str, Dict[str, Any]]):
    """Replace the stored per-file fingerprint set for a repository."""
    await session.execute(delete(FileFingerprintModel).where(FileFingerprintModel.repo_id == repo_id))
    for path, fp in fingerprints.items():
        session.add(FileFingerprintModel(
            repo_id=repo_id,
            file_path=path,
            content_hash=fp["content_hash"],
            size_bytes=fp.get("size_bytes", 0),
        ))
    await session.flush()


async def load_file_fingerprints(session: AsyncSession, repo_id: str) -> Dict[str, FileFingerprintModel]:
    res = await session.execute(select(FileFingerprintModel).where(FileFingerprintModel.repo_id == repo_id))
    return {fp.file_path: fp for fp in res.scalars().all()}

# ---------------------------------------------------------------------------
# Single-file result persistence (used by full + incremental paths)
# ---------------------------------------------------------------------------

async def upsert_file_result(
    session: AsyncSession,
    repo_id: str,
    file_path: str,
    language: str,
    content_hash: str,
    size_bytes: int,
    symbols: list,
    edges: list,
    chunks: list,
    embedding_service=None,
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
) -> int:
    """Persist one parsed file's artifacts, replacing previous rows for the file.

    Idempotent by construction: re-persisting identical content yields identical rows.
    When ``embedding_service`` is provided, each chunk's content is embedded and
    the vector (plus model + provider + dimension + created_at provenance) is
    Returns number of chunks persisted.
    """
    for chunk in chunks:
        if "\x00" in chunk.content:
            raise ValueError(
                f"Database safety invariant violation: NUL byte ('\\x00') in chunk content for file '{chunk.file_path}'"
            )

    existing_embeddings: Dict[tuple, tuple[int, list]] = {}
    if embedding_service is not None and embedding_model:
        existing_rows = await session.execute(
            select(CodeChunkModel).where(
                CodeChunkModel.repo_id == repo_id,
                CodeChunkModel.file_path == file_path,
                CodeChunkModel.content_hash.in_([c.content_hash for c in chunks]),
                CodeChunkModel.embedding_model == embedding_model,
                CodeChunkModel.embedding_provider == embedding_provider,
                CodeChunkModel.embedding.isnot(None),
            )
        )
        existing_embeddings = {}
        for row in existing_rows.scalars():
            raw_embedding = row.embedding
            existing_embeddings[(row.content_hash, row.line_start, row.line_end)] = (
                row.id,
                list(raw_embedding),
            )

    await invalidate_file_rows(session, repo_id, [file_path])

    session.add(FileModel(
        repo_id=repo_id, file_path=file_path, language=language,
        content_hash=content_hash, size_bytes=size_bytes,
    ))
    for sym in symbols:
        session.add(SymbolModel(
            repo_id=repo_id, file_path=sym.file_path, symbol_name=sym.symbol_name,
            symbol_type=sym.symbol_type, line_start=sym.line_start, line_end=sym.line_end,
            signature=sym.signature, docstring=sym.docstring,
        ))
    for edge in edges:
        session.add(SymbolEdgeModel(
            repo_id=repo_id, source_symbol=edge.source_symbol, source_file=edge.source_file,
            edge_type=edge.edge_type, target_symbol=edge.target_symbol, target_file=edge.target_file,
        ))

    # Pre-compute embeddings for the file's chunks (batched).
    # Stage 3: when the provider is unavailable, mark semantic retrieval as
    # unavailable rather than falling back to fake vectors.
    missing_chunks = [
        c for c in chunks
        if (c.content_hash, c.line_start, c.line_end) not in existing_embeddings
    ]
    if embedding_service is not None and embedding_model and missing_chunks:
        try:
            generated = embedding_service.embed_texts([c.content for c in missing_chunks])
            generated_by_key = {
                (chunk.content_hash, chunk.line_start, chunk.line_end): vector
                for chunk, vector in zip(missing_chunks, generated)
            }
        except Exception as exc:
            # Degraded mode: persist chunks without vectors rather than failing
            # the whole ingestion run, but never swallow the reason silently.
            logger.warning(
                "Embedding unavailable for %s (model=%s): %s: %s",
                file_path, embedding_model, type(exc).__name__, exc,
            )
            generated_by_key = {}
    else:
        generated_by_key = {}

    _now = datetime.now(timezone.utc)
    for i, chunk in enumerate(chunks):
        if "\x00" in chunk.content:
            raise ValueError(
                f"Database safety invariant violation: NUL byte ('\\x00') in chunk content for file '{chunk.file_path}'"
            )
        key = (chunk.content_hash, chunk.line_start, chunk.line_end)
        reused = existing_embeddings.get(key)
        if reused is not None:
            chunk_id, emb = reused
            embedding_source = "existing_database_row"
        else:
            chunk_id = None
            emb = generated_by_key.get(key)
            embedding_source = "newly_generated" if emb is not None else "missing"
        _validate_embedding(
            emb,
            embedding_dimension,
            repo_id=repo_id,
            chunk_id=chunk_id,
            source=embedding_source,
        )
        session.add(CodeChunkModel(
            repo_id=repo_id, file_path=chunk.file_path, symbol_name=chunk.symbol_name,
            line_start=chunk.line_start, line_end=chunk.line_end, content=chunk.content,
            content_hash=chunk.content_hash, language=chunk.language,
            embedding=emb,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_dimension=len(emb) if emb else None,
            embedding_created_at=_now,
        ))
    await session.flush()
    return len(chunks)


async def replace_dependencies(session: AsyncSession, repo_id: str, dependencies: list) -> int:
    """Replace the dependency manifest rows for a repository (idempotent)."""
    await session.execute(delete(DependencyModel).where(DependencyModel.repo_id == repo_id))
    for dep in dependencies:
        session.add(DependencyModel(
            repo_id=repo_id, package_name=dep.package_name, ecosystem=dep.ecosystem,
            version_spec=dep.version_spec, dependency_type=dep.dependency_type,
        ))
    await session.flush()
    return len(dependencies)


async def count_repo_artifacts(session: AsyncSession, repo_id: str) -> Dict[str, int]:
    """Artifact counts used for idempotency verification."""
    async def _count(model):
        res = await session.execute(select(func.count()).select_from(model).where(model.repo_id == repo_id))
        return int(res.scalar() or 0)

    return {
        "files": await _count(FileModel),
        "symbols": await _count(SymbolModel),
        "symbol_edges": await _count(SymbolEdgeModel),
        "dependencies": await _count(DependencyModel),
        "chunks": await _count(CodeChunkModel),
        "commits": await _count(CommitModel),
    }


async def _resolve_current_embedding_dimension(
    session,
    repo_id: str,
    embedding_service,
    current_model: Optional[str],
    current_provider: Optional[str],
) -> Optional[int]:
    """Return the current embedding dimension without forcing a model load.

    ``EmbeddingService.dimension`` lazily loads the embedding model, which is an
    expensive (multi-second) operation. Stored chunk rows produced by the *same*
    model+provider carry the dimension that model actually produced, so when the
    stored rows agree we can reuse that value for the mismatch check instead of
    loading the model. The model is only loaded when no same-model rows exist
    (fresh index - embeddings will be computed anyway) or when stored rows
    disagree (genuine inconsistency that must be resolved against the real
    dimension so incompatible vectors are invalidated).
    """
    from sqlalchemy import select

    if embedding_service is None:
        return None
    dims = (
        (
            await session.execute(
                select(CodeChunkModel.embedding_dimension)
                .where(CodeChunkModel.repo_id == repo_id)
                .where(CodeChunkModel.embedding.isnot(None))
                .where(CodeChunkModel.embedding_model == current_model)
                .where(CodeChunkModel.embedding_provider == current_provider)
                .distinct()
            )
        )
        .scalars()
        .all()
    )
    dims = {d for d in dims if d is not None}
    if len(dims) == 1:
        return dims.pop()
    return embedding_service.dimension


async def persist_parsing_results(
    session: AsyncSession,
    manifest: IngestionManifest,
    parsing_res: ParsingPipelineResult,
    incremental_files: Optional[List[str]] = None,
    head_commit: Optional[str] = None,
    branch: Optional[str] = None,
    embedding_service=None,
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    persist_batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
):
    """Idempotently persist parsing results.

    - Full mode (default): replaces all previously indexed artifacts of the repo,
      guaranteeing re-analysis does not duplicate files/symbols/chunks/edges/deps.
    - Incremental mode (incremental_files given): only the listed files are replaced;
      untouched files keep their existing rows.
    - ``embedding_service`` (optional): if provided, each persisted chunk's content
      is embedded and stored for persistent semantic retrieval.
    """
    repo_id = parsing_res.repo_id
    await _ensure_repository_row(session, repo_id, manifest, head_commit=head_commit, branch=branch)

    # Stage 3: if the embedding model / provider changed since the last index,
    # invalidate stale embeddings so no incompatible vectors are ever compared.
    if embedding_service is not None and embedding_model:
        provider_name = (
            getattr(embedding_service, "provider_name", None)
            or embedding_provider
        )
        current_dimension = await _resolve_current_embedding_dimension(
            session, repo_id, embedding_service, embedding_model, provider_name
        )
        await invalidate_embeddings_for_model_mismatch(
            session, repo_id,
            current_model=embedding_model,
            current_provider=provider_name,
            current_dimension=current_dimension,
        )

    files_filter = set(incremental_files) if incremental_files is not None else None
    resolved_embedding_dimension = None
    if embedding_service is not None and embedding_model:
        resolved_embedding_dimension = await _resolve_current_embedding_dimension(
            session, repo_id, embedding_service, embedding_model, embedding_provider
        )

    symbols_by_file: Dict[str, list] = {}
    edges_by_file: Dict[str, list] = {}
    for sym in parsing_res.symbols:
        symbols_by_file.setdefault(sym.file_path, []).append(sym)
    for edge in parsing_res.edges:
        edges_by_file.setdefault(edge.source_file, []).append(edge)

    chunks_by_file: Dict[str, list] = {}
    seen_chunk_keys = set()
    for chunk in parsing_res.chunks:
        key = (chunk.file_path, chunk.line_start, chunk.line_end, chunk.content_hash)
        if key in seen_chunk_keys:
            continue
        seen_chunk_keys.add(key)
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    file_hashes = getattr(parsing_res, "parsed_file_hashes", {})
    all_paths = set(symbols_by_file) | set(edges_by_file) | set(chunks_by_file) | set(file_hashes)
    if files_filter is not None:
        stale_paths = set(files_filter) - set(all_paths)
        if stale_paths:
            await invalidate_file_rows(session, repo_id, sorted(stale_paths))
    else:
        stored_paths = set(
            (
                await session.execute(
                    select(FileModel.file_path).where(FileModel.repo_id == repo_id)
                )
            ).scalars().all()
        )
        stale_paths = stored_paths - all_paths
        if stale_paths:
            await invalidate_file_rows(session, repo_id, sorted(stale_paths))

    batch_size = persist_batch_size or int(os.getenv("KNOWLEDGE_PERSIST_BATCH_SIZE", "25"))
    if batch_size < 1:
        raise ValueError("persist_batch_size must be at least 1")
    processed = 0
    for rel_path in sorted(all_paths):
        if files_filter is not None and rel_path not in files_filter:
            continue
        h, sz = file_hashes.get(rel_path, ("", 0))
        file_chunks = chunks_by_file.get(rel_path, [])
        language = file_chunks[0].language if file_chunks else "Unknown"
        await upsert_file_result(
            session, repo_id, rel_path,
            language=language,
            content_hash=h,
            size_bytes=sz,
            symbols=symbols_by_file.get(rel_path, []),
            edges=edges_by_file.get(rel_path, []),
            chunks=file_chunks,
            embedding_service=embedding_service,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            embedding_dimension=resolved_embedding_dimension,
        )
        processed += 1
        if processed % batch_size == 0:
            await session.commit()
            if progress_callback is not None:
                await progress_callback(processed, len(all_paths))

    if files_filter is None:
        await replace_dependencies(session, repo_id, parsing_res.dependencies)

    await session.commit()
    if progress_callback is not None and processed % batch_size:
        await progress_callback(processed, len(all_paths))


# ---------------------------------------------------------------------------
# Embedding versioning / model-mismatch invalidation (Stage 3)
# ---------------------------------------------------------------------------

async def invalidate_embeddings_for_model_mismatch(
    session: AsyncSession,
    repo_id: str,
    current_model: str,
    current_provider: Optional[str] = None,
    current_dimension: Optional[int] = None,
) -> Dict[str, int]:
    """Invalidate stored embeddings whose model/provider/dimension no longer match.

    Determines the set of chunks whose persisted embedding provenance differs from
    the current active model+provider+dimension. Those rows are left in place with
    ``embedding=None`` (their vectors are NOT safely comparable), so semantic
    retrieval will simply skip them and the next persist pass re-embeds them.

    Returns a count of invalidated rows so the operation is observable.
    """
    rows = await session.execute(
        select(CodeChunkModel).where(CodeChunkModel.repo_id == repo_id)
    )
    invalidated = 0
    for row in rows.scalars().all():
        mismatch = False
        if row.embedding is not None:
            if current_model and row.embedding_model != current_model:
                mismatch = True
            elif current_provider and row.embedding_provider != current_provider:
                mismatch = True
            elif current_dimension and row.embedding_dimension != current_dimension:
                mismatch = True
        if mismatch:
            row.embedding = None
            row.embedding_model = None
            row.embedding_provider = None
            row.embedding_dimension = None
            row.embedding_created_at = None
            invalidated += 1
    if invalidated:
        telemetry.increment("embedding_invalidated_model_mismatch", invalidated)
    return {"invalidated": invalidated}


# ---------------------------------------------------------------------------
# Persistent embedding / chunk loading (query-time reconstruction)
# ---------------------------------------------------------------------------

async def load_code_chunks(session: AsyncSession, repo_id: str) -> List[CodeChunkModel]:
    """Load all persisted code chunks for a repository, in DB order."""
    res = await session.execute(
        select(CodeChunkModel).where(CodeChunkModel.repo_id == repo_id)
    )
    return list(res.scalars().all())


async def load_embeddings(session: AsyncSession, repo_id: str) -> Dict[str, list]:
    """Load persisted embeddings keyed by chunk content-hash.

    Returned dict maps ``content_hash -> embedding`` for chunks that have a
    non-null vector.  Used to warm the in-process embedding cache and avoid
    recomputing embeddings that are already persisted.
    """
    res = await session.execute(
        select(CodeChunkModel.content_hash, CodeChunkModel.embedding)
        .where(CodeChunkModel.repo_id == repo_id)
    )
    out: Dict[str, list] = {}
    for content_hash, embedding in res.all():
        if content_hash and embedding is not None:
            out[content_hash] = list(embedding)
    return out


def code_chunks_to_vector_store(chunk_rows: List[CodeChunkModel]) -> List[Any]:
    """Convert persisted CodeChunkModel rows back into :class:`CodeChunk` objects.

    Returns a flat list suitable for feeding into ``VectorStore.upsert_chunks``.
    """
    from packages.parsing.chunking import CodeChunk

    out: List[CodeChunk] = []
    for row in chunk_rows:
        out.append(CodeChunk(
            repo_id=row.repo_id,
            file_path=row.file_path,
            symbol_name=row.symbol_name,
            line_start=row.line_start,
            line_end=row.line_end,
            content=row.content,
            content_hash=row.content_hash,
            language=row.language,
        ))
    return out