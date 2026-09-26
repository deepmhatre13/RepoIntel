"""CRUD helpers for Project Health persistence (ORM-based, async).

Encapsulates the normalized observation model so callers work with plain
dicts / Pydantic objects and never touch ORM query plumbing. The health
service uses these helpers to upsert repository identity, append historical
observations, and read back metric history for temporal analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from packages.knowledge.db import AsyncSessionLocal
from packages.project_health.orm import (
    ProjectHealthObservationModel,
    ProjectHealthRepositoryModel,
)


def persistence_session():
    """Async context manager over a Project-Health-aware SQLAlchemy session.

    Reuses the shared RepoIntel engine so the same database serves both
    Repository Intelligence and Project Health persistence.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _open():
        async with AsyncSessionLocal() as session:
            yield session

    return _open()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _norm_ts(ts: datetime) -> datetime:
    """Canonicalize a timestamp to naive UTC for stable key comparisons.

    SQLite loads datetimes as naive even when aware values were stored, so
    dedup keys must be compared in one canonical form (naive UTC).
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


async def upsert_repository(
    session: AsyncSession, *, health_key: str, owner: str, name: str,
    full_name: str, url: str, metadata_json: str = "{}",
) -> None:
    """Insert or update the stable repository identity row (idempotent)."""
    existing = (
        await session.execute(
            select(ProjectHealthRepositoryModel).where(
                ProjectHealthRepositoryModel.id == health_key
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.metadata_json = metadata_json
        existing.url = url
        existing.full_name = full_name
        existing.updated_at = _utcnow()
    else:
        session.add(
            ProjectHealthRepositoryModel(
                id=health_key,
                owner=owner,
                name=name,
                full_name=full_name,
                url=url,
                metadata_json=metadata_json,
            )
        )
    await session.commit()


async def touch_analysis(session: AsyncSession, health_key: str) -> None:
    """Bump the analysis count / last-analysis timestamp on the repository row."""
    await session.execute(
        update(ProjectHealthRepositoryModel)
        .where(ProjectHealthRepositoryModel.id == health_key)
        .values(
            analysis_count=ProjectHealthRepositoryModel.analysis_count + 1,
            last_analysis_at=_utcnow(),
            updated_at=_utcnow(),
        )
    )
    await session.commit()
async def add_observations(
    session: AsyncSession,
    health_key: str,
    observations: List[Dict[str, Any]],
) -> int:
    """Append a batch of normalized historical observations (deduplicated).

    Rows are keyed on ``(health_key, timestamp, metric)``; a repeated
    observation for the same timestamp+metric is skipped so incremental
    monitoring refreshes do not duplicate history. Returns count written.
    """
    if not observations:
        return 0
    # Batch-level dedup within this payload: later rows for an identical
    # (health_key, timestamp, metric) key overwrite earlier ones so a single
    # batch can never violate the unique constraint internally.
    unique: Dict[tuple, Dict[str, Any]] = {}
    for obs in observations:
        key = (health_key, _norm_ts(obs["timestamp"]), obs["metric"])
        unique[key] = obs

    # Rows already persisted (e.g. repeated monitoring refreshes) are skipped.
    keys = list(unique.keys())
    existing = set()
    chunk = 100
    for i in range(0, len(keys), chunk):
        part = keys[i:i + chunk]
        rows = (
            await session.execute(
                select(
                    ProjectHealthObservationModel.health_key,
                    ProjectHealthObservationModel.timestamp,
                    ProjectHealthObservationModel.metric,
                ).where(
                    tuple_(
                        ProjectHealthObservationModel.health_key,
                        ProjectHealthObservationModel.timestamp,
                        ProjectHealthObservationModel.metric,
                    ).in_(part)
                )
            )
        ).all()
        for hk, db_ts, metric in rows:
            existing.add((hk, _norm_ts(db_ts), metric))

    pending = [obs for key, obs in unique.items() if key not in existing]
    if not pending:
        return 0
    session.add_all(
        ProjectHealthObservationModel(
            health_key=health_key,
            timestamp=obs["timestamp"],
            metric=obs["metric"],
            value=obs.get("value"),
            source=obs.get("source", "github"),
            confidence=obs.get("confidence", 1.0),
            observation_window=obs.get("observation_window"),
            availability=obs.get("availability", "NOT_AVAILABLE"),
        )
        for obs in pending
    )
    await session.commit()
    return len(pending)


async def get_metric_history(
    session: AsyncSession,
    health_key: str,
    metric: str,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Chronological observations for one metric as plain dicts."""
    rows = (
        await session.execute(
            select(ProjectHealthObservationModel)
            .where(
                ProjectHealthObservationModel.health_key == health_key,
                ProjectHealthObservationModel.metric == metric,
            )
            .order_by(ProjectHealthObservationModel.timestamp.asc())
            .limit(limit)
        )
    ).scalars().all()
    out = []
    for r in rows:
        out.append(
            {
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "metric": r.metric,
                "value": r.value,
                "source": r.source,
                "confidence": r.confidence,
                "observation_window": r.observation_window,
                "availability": r.availability,
            }
        )
    return out


async def get_all_metric_history(
    session: AsyncSession, health_key: str, limit: int = 20000
) -> Dict[str, Any]:
    """All persisted observations for a repository, grouped by metric name."""
    rows = (
        await session.execute(
            select(ProjectHealthObservationModel)
            .where(ProjectHealthObservationModel.health_key == health_key)
            .order_by(ProjectHealthObservationModel.timestamp.asc())
            .limit(limit)
        )
    ).scalars().all()
    by_metric: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_metric.setdefault(r.metric, []).append(
            {
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "metric": r.metric,
                "value": r.value,
                "source": r.source,
                "confidence": r.confidence,
                "observation_window": r.observation_window,
                "availability": r.availability,
            }
        )
    return {"metrics": by_metric, "metric_count": len(by_metric)}


async def get_repository_row(
    session: AsyncSession, health_key: str
) -> Optional[ProjectHealthRepositoryModel]:
    return (
        await session.execute(
            select(ProjectHealthRepositoryModel).where(
                ProjectHealthRepositoryModel.id == health_key
            )
        )
    ).scalar_one_or_none()


async def save_analysis(session: AsyncSession, health_key: str,
                        analysis_json: str) -> bool:
    """Store the latest health analysis snapshot for a project (monitoring hook)."""
    row = await get_repository_row(session, health_key)
    if row is None:
        return False
    row.latest_analysis_json = analysis_json
    await session.commit()
    return True


async def load_analysis(session: AsyncSession, health_key: str) -> Optional[str]:
    """Retrieve the latest persisted analysis snapshot for a project."""
    row = await get_repository_row(session, health_key)
    return row.latest_analysis_json if row is not None else None