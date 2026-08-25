"""History capability mixin for RepositoryFacade.

Owns the historical / AST-aware diff / change-tracking intelligence surfaced by
RepositoryFacade: symbol history, co-change relationships, symbol changes,
structured commit diffs, changed symbols, API changes, dependency changes, and
git commit listings. Implementation detail, not exposed to agents.
"""

import json
from typing import List, Dict, Any, Optional

from sqlalchemy import select

from packages.knowledge.models import (
    SymbolHistoryModel,
    CoChangeRelationModel,
    SymbolChangeModel,
    SymbolEdgeModel,
    CommitAnalysisModel,
    APIChangeModel,
    DependencyChangeModel,
    CommitModel,
    FileChangeModel,
)


class HistoryCapabilities:
    """Historical / AST diff / change intelligence for RepositoryFacade."""

    async def get_symbol_history(
        self, repo_id: str, symbol: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return commit history for a symbol, ordered by timestamp desc."""
        stmt = (
            select(SymbolHistoryModel)
            .where(SymbolHistoryModel.repo_id == repo_id, SymbolHistoryModel.symbol_name == symbol)
            .order_by(SymbolHistoryModel.timestamp.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        return [
            {
                "symbol_name": r.symbol_name,
                "file_path": r.file_path,
                "commit_sha": r.commit_sha,
                "author": r.author,
                "change_type": r.change_type,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "content_hash": r.content_hash,
                "change_summary": r.change_summary,
                "confidence": r.evidence,
            }
            for r in rows
        ]

    async def resolve_symbol_history(
        self, repo_id: str, symbol: str, module: str = ""
    ) -> List[Dict[str, Any]]:
        """Resolve symbol history supporting exact symbol, file+symbol, symbol+module."""
        history = await self.get_symbol_history(repo_id, symbol, limit=50)
        if history:
            for h in history:
                h["resolution_confidence"] = "HIGH_EXACT"
            return history

        if module:
            broad = await self.get_symbol_history(repo_id, symbol, limit=100)
            matched = [h for h in broad if module.lower() in (h.get("file_path") or "").lower()]
            if matched:
                for h in matched:
                    h["resolution_confidence"] = "MEDIUM_MODULE"
                return matched

        return [
            {
                "symbol_name": symbol,
                "file_path": None,
                "commit_sha": None,
                "author": None,
                "change_type": None,
                "timestamp": None,
                "resolution_confidence": "LOW_INSUFFICIENT_EVIDENCE",
                "message": "Insufficient repository evidence to resolve historical identity.",
            }
        ]

    async def get_symbol_frequency(self, repo_id: str, symbol: str) -> Dict[str, Any]:
        """Compute deterministic change frequency metrics for a symbol."""
        stmt = select(SymbolHistoryModel).where(
            SymbolHistoryModel.repo_id == repo_id, SymbolHistoryModel.symbol_name == symbol
        )
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        if not rows:
            return {}

        authors = set(r.author for r in rows)
        timestamps = sorted(r.timestamp for r in rows if r.timestamp)
        commit_count = len(rows)
        first_seen = timestamps[0].isoformat() if timestamps else None
        last_modified = timestamps[-1].isoformat() if timestamps else None

        avg_interval = None
        if len(timestamps) >= 2:
            total_delta = (timestamps[-1] - timestamps[0]).total_seconds()
            avg_interval = round(total_delta / (len(timestamps) - 1), 2)

        return {
            "symbol_name": symbol,
            "commit_count": commit_count,
            "authors_count": len(authors),
            "authors": sorted(authors),
            "first_seen": first_seen,
            "last_modified": last_modified,
            "average_change_interval_seconds": avg_interval,
        }
    async def get_co_change_relationships(
        self, repo_id: str, symbol: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return co-change relationships for a symbol."""
        stmt = (
            select(CoChangeRelationModel)
            .where(
                CoChangeRelationModel.repo_id == repo_id,
                (CoChangeRelationModel.symbol_a == symbol) | (CoChangeRelationModel.symbol_b == symbol),
            )
            .order_by(CoChangeRelationModel.co_change_count.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        return [
            {
                "symbol_a": r.symbol_a,
                "symbol_b": r.symbol_b,
                "co_change_count": r.co_change_count,
                "co_change_ratio": r.co_change_ratio,
                "supporting_commits": r.supporting_commits,
            }
            for r in rows
        ]

    async def get_symbol_changes(
        self, repo_id: str, symbol: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return symbol change records."""
        stmt = (
            select(SymbolChangeModel)
            .where(SymbolChangeModel.repo_id == repo_id, SymbolChangeModel.symbol_name == symbol)
            .order_by(SymbolChangeModel.id.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        return [
            {
                "commit_sha": r.commit_sha,
                "file_path": r.file_path,
                "symbol_name": r.symbol_name,
                "change_category": r.change_category,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "before_signature": r.before_signature,
                "after_signature": r.after_signature,
                "confidence": r.confidence,
                "evidence": r.evidence,
            }
            for r in rows
        ]

    async def get_dependents(
        self, repo_id: str, target: str, max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Return direct dependents (incoming edges) of a symbol/file."""
        stmt = select(SymbolEdgeModel).where(
            SymbolEdgeModel.repo_id == repo_id, SymbolEdgeModel.target_symbol == target
        ).limit(max_results)
        res = await self.session.execute(stmt)
        edges = res.scalars().all()
        return [
            {
                "source_symbol": e.source_symbol,
                "source_file": e.source_file,
                "edge_type": e.edge_type,
            }
            for e in edges
        ]
    async def get_commit_diff(self, repo_id: str, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Get structured diff analysis for a commit."""
        stmt = select(CommitAnalysisModel).where(
            CommitAnalysisModel.repo_id == repo_id, CommitAnalysisModel.commit_sha == commit_sha
        )
        res = await self.session.execute(stmt)
        ca = res.scalar_one_or_none()
        if not ca:
            return None
        return {
            "commit_sha": ca.commit_sha,
            "changed_files": json.loads(ca.changed_files_json),
            "changed_symbols": json.loads(ca.changed_symbols_json),
            "dependency_changes": json.loads(ca.dependency_changes_json),
            "api_changes": json.loads(ca.api_changes_json),
            "configuration_changes": json.loads(ca.configuration_changes_json),
            "summary": ca.summary,
            "risk_signals": json.loads(ca.risk_signals_json),
        }

    async def get_commit_analysis(self, repo_id: str, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Alias for get_commit_diff."""
        return await self.get_commit_diff(repo_id, commit_sha)

    async def get_changed_symbols(
        self, repo_id: str, file_path: str = "", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return symbols changed across commits."""
        stmt = select(SymbolHistoryModel).where(SymbolHistoryModel.repo_id == repo_id)
        if file_path:
            stmt = stmt.where(SymbolHistoryModel.file_path == file_path)
        stmt = stmt.order_by(SymbolHistoryModel.timestamp.desc()).limit(limit)
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        return [
            {
                "symbol_name": r.symbol_name,
                "file_path": r.file_path,
                "commit_sha": r.commit_sha,
                "author": r.author,
                "change_type": r.change_type,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ]

    async def get_api_changes(self, repo_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return API changes."""
        stmt = (
            select(APIChangeModel)
            .where(APIChangeModel.repo_id == repo_id)
            .order_by(APIChangeModel.id.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        return [
            {
                "api_symbol": r.api_symbol,
                "before_signature": r.before_signature,
                "after_signature": r.after_signature,
                "commit_sha": r.commit_sha,
                "change_type": r.change_type,
                "confidence": r.confidence,
                "evidence": r.evidence,
            }
            for r in rows
        ]

    async def get_dependency_changes(self, repo_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return dependency changes."""
        stmt = (
            select(DependencyChangeModel)
            .where(DependencyChangeModel.repo_id == repo_id)
            .order_by(DependencyChangeModel.id.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        return [
            {
                "package_name": r.package_name,
                "ecosystem": r.ecosystem,
                "change_type": r.change_type,
                "old_version": r.old_version,
                "new_version": r.new_version,
                "commit_sha": r.commit_sha,
                "confidence": r.confidence,
                "evidence": r.evidence,
            }
            for r in rows
        ]

    async def get_git_commits(self, repo_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent git commits for a repository."""
        stmt = (
            select(CommitModel)
            .where(CommitModel.repo_id == repo_id)
            .order_by(CommitModel.timestamp.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        commits = []
        for r in rows:
            commits.append(
                {
                    "sha": r.sha,
                    "author": r.author,
                    "message": r.message,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "changed_files": await self._get_commit_files(repo_id, r.sha),
                }
            )
        return commits

    async def _get_commit_files(self, repo_id: str, sha: str) -> List[str]:
        """Helper to get changed file paths for a commit."""
        stmt = select(FileChangeModel.file_path).where(
            FileChangeModel.repo_id == repo_id, FileChangeModel.commit_sha == sha
        )
        res = await self.session.execute(stmt)
        return [row[0] for row in res.all()]