"""Incremental Indexer & Change-Aware Reprocessor for RepoIntel (Phase 41).

Computes deterministic repository fingerprints from Git metadata plus per-file
content hashes, diffs two repository states into ADDED / MODIFIED / DELETED /
RENAMED / UNCHANGED categories, expands changes across the symbol dependency
graph (directly + indirectly affected files), and executes scoped cache
invalidation so only necessary work is redone.

Fingerprints persist through the knowledge layer (FileFingerprintModel), so
change detection works across processes -- not just within one run.
"""

import os
import re
import hashlib
import subprocess
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.knowledge.models import SymbolEdgeModel, RepositoryModel
from packages.shared.telemetry import telemetry


def _git(repo_path: str, *args: str) -> str:
    res = subprocess.run(["git", *args], cwd=repo_path, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()[:200]}")
    return res.stdout.strip()


class ChangeCategory(str, Enum):
    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"
    RENAMED = "RENAMED"
    UNCHANGED = "UNCHANGED"


class FileStateFingerprint(BaseModel):
    file_path: str
    content_hash: str
    size_bytes: int


class RepositoryFingerprint(BaseModel):
    repo_id: str = ""
    commit_sha: str = ""
    branch: str = ""
    files: Dict[str, FileStateFingerprint] = Field(default_factory=dict)

    def compute_overall_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.commit_sha.encode("utf-8"))
        h.update(self.branch.encode("utf-8"))
        for path in sorted(self.files.keys()):
            h.update(f"{path}:{self.files[path].content_hash}".encode("utf-8"))
        return h.hexdigest()


class IncrementalRunReport(BaseModel):
    previous_commit: str = ""
    current_commit: str = ""
    previous_fingerprint: str = ""
    current_fingerprint: str = ""
    files_added: List[str] = Field(default_factory=list)
    files_modified: List[str] = Field(default_factory=list)
    files_deleted: List[str] = Field(default_factory=list)
    files_renamed: Dict[str, str] = Field(default_factory=dict)  # old -> new
    files_unchanged_count: int = 0
    directly_affected: List[str] = Field(default_factory=list)
    indirectly_affected: List[str] = Field(default_factory=list)
    symbols_changed: List[str] = Field(default_factory=list)
    chunks_invalidated: int = 0
    embeddings_updated: int = 0
    graph_edges_updated: int = 0
    tests_reanalyzed: bool = False
    architecture_recomputed: bool = False
    risk_recomputed: bool = False
    duration_seconds: float = 0.0


class IncrementalIndexer:
    """Computes repository state, diffs states, and performs dependency-aware invalidation."""

    def __init__(self, max_dependency_hops: int = 3, max_affected_files: int = 2000):
        self.max_dependency_hops = max_dependency_hops
        self.max_affected_files = max_affected_files

    # ---------------- Fingerprinting (41.1) ----------------

    def git_head(self, repo_path: str) -> Tuple[str, str]:
        """Returns (commit_sha, branch); ('', '') for non-git directories."""
        if not os.path.isdir(os.path.join(repo_path, ".git")):
            return "", ""
        try:
            sha = _git(repo_path, "rev-parse", "HEAD")
            branch = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
            return sha, branch
        except RuntimeError:
            return "", ""

    def compute_fingerprint(
        self,
        repo_path: str,
        filtered_files: List[str],
        repo_id: str = "",
        file_hashes: Optional[Dict[str, str]] = None,
    ) -> RepositoryFingerprint:
        """Fingerprint from Git HEAD + per-file content hashes (deterministic).

        Prefer Git metadata for identity; only hash the files we would index.
        `file_hashes` may be supplied by the caller (e.g. already computed during
        parsing) to avoid re-reading every file.
        """
        commit_sha, branch = self.git_head(repo_path)
        files: Dict[str, FileStateFingerprint] = {}
        for rel in filtered_files:
            if file_hashes and rel in file_hashes:
                chash = file_hashes[rel]
                size = 0
            else:
                fp = os.path.join(repo_path, rel)
                if not os.path.isfile(fp):
                    continue
                h = hashlib.sha256()
                size = 0
                with open(fp, "rb") as f:
                    while True:
                        block = f.read(65536)
                        if not block:
                            break
                        h.update(block)
                        size += len(block)
                chash = h.hexdigest()
            files[rel] = FileStateFingerprint(file_path=rel, content_hash=chash, size_bytes=size)
        return RepositoryFingerprint(repo_id=repo_id, commit_sha=commit_sha, branch=branch, files=files)

    # ---------------- Diffing (41.3) ----------------

    def compute_diff(
        self,
        prev_fp: RepositoryFingerprint,
        curr_fp: RepositoryFingerprint,
    ) -> Dict[str, ChangeCategory]:
        diffs: Dict[str, ChangeCategory] = {}
        prev_files = prev_fp.files
        curr_files = curr_fp.files

        prev_by_hash: Dict[str, List[str]] = {}
        for path, item in prev_files.items():
            prev_by_hash.setdefault(item.content_hash, []).append(path)

        claimed_renames: Set[str] = set()

        for path, curr_item in curr_files.items():
            if path not in prev_files:
                candidates = [p for p in prev_by_hash.get(curr_item.content_hash, []) if p not in curr_files]
                if len(candidates) == 1:
                    claimed_renames.add(candidates[0])
                    diffs[path] = ChangeCategory.RENAMED
                else:
                    diffs[path] = ChangeCategory.ADDED
            elif prev_files[path].content_hash != curr_item.content_hash:
                diffs[path] = ChangeCategory.MODIFIED
            else:
                diffs[path] = ChangeCategory.UNCHANGED

        for path in prev_files:
            if path not in curr_files:
                diffs[path] = ChangeCategory.RENAMED if path in claimed_renames else ChangeCategory.DELETED

        return diffs

    @staticmethod
    def diff_summary(diffs: Dict[str, ChangeCategory]) -> Dict[str, List[str]]:
        summary: Dict[str, List[str]] = {c.value: [] for c in ChangeCategory}
        for path, cat in sorted(diffs.items()):
            summary[cat.value].append(path)
        return summary

    # ---------------- Dependency-aware affected set (41.4) ----------------

    async def expand_affected_files(
        self,
        session: AsyncSession,
        repo_id: str,
        changed_files: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Return (directly_affected, indirectly_affected) files using symbol edges.

        Directly affected: files that import/call into any changed file.
        Indirectly affected: transitive importers up to max_dependency_hops.
        """
        changed = set(changed_files)

        def _stem(path: str) -> str:
            stem = re.sub(r"\.(py|js|ts|jsx|tsx)$", "", path.replace("\\", "/").rsplit("/", 1)[-1])
            return stem

        async def importers_of(paths: Set[str]) -> Dict[str, Set[str]]:
            res = await session.execute(
                select(SymbolEdgeModel).where(SymbolEdgeModel.repo_id == repo_id)
            )
            edges = res.scalars().all()
            target_stems = {_stem(p) for p in paths if _stem(p)}
            by_source: Dict[str, Set[str]] = {}
            for e in edges:
                tgt = e.target_file or ""
                src = e.source_file or ""
                if not src or src in paths:
                    continue
                hit = tgt in paths
                if not hit and tgt and _stem(tgt) in target_stems:
                    # Module-level dependency (e.g. "import a.b" resolves by module name).
                    hit = True
                if hit:
                    by_source.setdefault(src, set()).add(tgt)
            return by_source

        direct_map = await importers_of(changed)
        directly_affected = set(direct_map.keys()) - changed

        frontier = set(directly_affected)
        indirectly_affected: Set[str] = set()
        for hop in range(max(0, self.max_dependency_hops - 1)):
            if not frontier or len(indirectly_affected) > self.max_affected_files:
                break
            next_map = await importers_of(frontier)
            frontier = set(next_map.keys()) - directly_affected - changed - indirectly_affected
            indirectly_affected |= frontier
            telemetry.record_event("dependency_expansion_hop", hop=hop + 1, found=len(frontier))

        return (
            sorted(directly_affected)[: self.max_affected_files],
            sorted(indirectly_affected)[: self.max_affected_files],
        )

    async def changed_symbols_for_files(self, session: AsyncSession, repo_id: str,
                                        file_paths: List[str]) -> List[str]:
        if not file_paths:
            return []
        res = await session.execute(
            select(SymbolEdgeModel.target_symbol).where(
                SymbolEdgeModel.repo_id == repo_id,
                SymbolEdgeModel.source_file.in_(file_paths),
            )
        )
        return sorted({s for (s,) in res.all() if s})

    # ---------------- Fingerprint persistence ----------------

    async def load_previous_fingerprint(self, session: AsyncSession, repo_id: str) -> Optional[RepositoryFingerprint]:
        from packages.knowledge.persistence import load_file_fingerprints
        stored = await load_file_fingerprints(session, repo_id)
        if not stored:
            return None
        repo_row = (
            await session.execute(select(RepositoryModel).where(RepositoryModel.id == repo_id))
        ).scalar_one_or_none()
        return RepositoryFingerprint(
            repo_id=repo_id,
            commit_sha=(repo_row.head_commit or "") if repo_row else "",
            branch=(repo_row.branch or "") if repo_row else "",
            files={
                path: FileStateFingerprint(file_path=path, content_hash=m.content_hash, size_bytes=m.size_bytes)
                for path, m in stored.items()
            },
        )
