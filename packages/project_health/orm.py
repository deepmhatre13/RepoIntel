"""SQLAlchemy ORM models for Project Health Intelligence (Stage 4).

Declared on the shared RepoIntel ``Base`` so they participate in the same
Alembic migration target (``alembic/versions/0007_project_health.py``).

Normalized observation model: one table holds the historical time series
(``timestamp, metric, value, source, confidence, observation window``) for a
project — not a separate table per metric. A lightweight repository table
keys each series by a stable ``health_key`` (``owner/name``) so a future
monitoring scheduler can append new observations incrementally.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from packages.knowledge.models import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProjectHealthRepositoryModel(Base):
    """Stable identity + latest snapshot for a GitHub project under health watch."""

    __tablename__ = "project_health_repositories"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # = health_key
    owner: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(128))
    full_name: Mapped[str] = mapped_column(String(256))
    url: Mapped[str] = mapped_column(String(512))
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    latest_analysis_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    analysis_count: Mapped[int] = mapped_column(Integer, default=0)
    last_analysis_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class ProjectHealthObservationModel(Base):
    """A single normalized historical health observation."""

    __tablename__ = "project_health_observations"
    __table_args__ = (
        UniqueConstraint(
            "health_key", "timestamp", "metric", name="uq_health_obs_key_ts_metric"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    health_key: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    metric: Mapped[str] = mapped_column(String(128))
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="github")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    observation_window: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    availability: Mapped[str] = mapped_column(String(32), default="NOT_AVAILABLE")