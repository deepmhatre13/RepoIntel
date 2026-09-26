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

    async def get_evidence_freshness(
        self, repo_id: str, file_paths: List[str], now: Any = None
    ) -> Dict[str, Dict[str, Any]]:
        """Measure real evidence freshness per file from ingested Git history.

        Returns ``{file_path: {"last_commit_utc": iso_or_None, "age_days": float|None,
        "freshness": float|None, "status": "MEASURED"|"NO_HISTORY"}}``.

        Freshness is a monotone decay of commit age (measured, not invented):
        ``exp(-age_days / 180)`` — evidence committed today scores ~1.0, evidence
        six months old ~0.37, a year old ~0.14. ``None``/MISSING is returned when
        no ingested history exists for a file — it is never reported as a
        perfect score, so stale-evidence handling is measurable (Phase G).
        """
        from datetime import datetime, timezone
        import math

        paths = [p for p in (file_paths or []) if p]
        if not paths:
            return {}
        unique = list(dict.fromkeys(paths))

        rows = (await self.session.execute(
            select(FileChangeModel.file_path, CommitModel.timestamp)
            .join(
                CommitModel,
                (FileChangeModel.commit_sha == CommitModel.sha)
                & (CommitModel.repo_id == repo_id),
            )
            .where(
                FileChangeModel.repo_id == repo_id,
                FileChangeModel.file_path.in_(unique),
            )
        )).all()

        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        latest: Dict[str, datetime] = {}
        for path, ts in rows:
            if ts is None:
                continue
            ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if path not in latest or ts > latest[path]:
                latest[path] = ts

        result: Dict[str, Dict[str, Any]] = {}
        for path in unique:
            ts = latest.get(path)
            if ts is None:
                result[path] = {
                    "last_commit_utc": None,
                    "age_days": None,
                    "freshness": None,
                    "status": "NO_HISTORY",
                }
                continue
            age_days = max(0.0, (now_dt - ts).total_seconds() / 86400.0)
            result[path] = {
                "last_commit_utc": ts.isoformat(),
                "age_days": round(age_days, 2),
                "freshness": round(math.exp(-age_days / 180.0), 4),
                "status": "MEASURED",
            }
        return result

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

    async def get_temporal_drift(self, repo_id: str) -> Dict[str, Any]:
        """Analyze temporal repository drift using Git history and snapshot data.

        Returns INSUFFICIENT_HISTORY when history is too thin (< 2 commits).
        Otherwise computes symbol churn, API surface drift, dependency drift, and architecture drift.
        """
        commits = await self.get_git_commits(repo_id, limit=100)
        if len(commits) < 2:
            return {
                "availability": "INSUFFICIENT_HISTORY",
                "reason": f"Only {len(commits)} commit(s) in repository history; at least 2 are required for drift analysis.",
                "commits_count": len(commits),
                "symbol_churn": None,
                "api_drift": None,
                "dependency_drift": None,
                "architecture_drift": None,
            }

        symbol_history = await self.get_changed_symbols(repo_id, limit=200)
        churn_by_commit: Dict[str, int] = {}
        for sh in symbol_history:
            sha = sh.get("commit_sha") or "unknown"
            churn_by_commit[sha] = churn_by_commit.get(sha, 0) + 1

        api_changes = await self.get_api_changes(repo_id, limit=50)
        dep_changes = await self.get_dependency_changes(repo_id, limit=50)

        from packages.knowledge.models import ArchitectureSnapshotModel
        stmt = (
            select(ArchitectureSnapshotModel)
            .where(ArchitectureSnapshotModel.repo_id == repo_id)
            .order_by(ArchitectureSnapshotModel.created_at.desc())
            .limit(2)
        )
        res = await self.session.execute(stmt)
        snapshots = res.scalars().all()
        arch_drift = None
        if len(snapshots) >= 2:
            s_new, s_old = snapshots[0], snapshots[1]
            arch_drift = {
                "compared_commits": [s_old.commit_sha, s_new.commit_sha],
                "layer_change": s_new.layer_count != s_old.layer_count,
                "component_count_delta": s_new.component_count - s_old.component_count,
            }

        return {
            "availability": "AVAILABLE",
            "commits_analyzed": len(commits),
            "symbol_churn": {
                "total_symbol_modifications": len(symbol_history),
                "commits_with_symbol_changes": len(churn_by_commit),
                "peak_churn_commit": max(churn_by_commit.items(), key=lambda x: x[1])[0] if churn_by_commit else None,
            },
            "api_drift": {
                "total_api_changes": len(api_changes),
                "recent_changes": api_changes[:10],
            },
            "dependency_drift": {
                "total_dependency_changes": len(dep_changes),
                "recent_changes": dep_changes[:10],
            },
            "architecture_drift": arch_drift,
        }