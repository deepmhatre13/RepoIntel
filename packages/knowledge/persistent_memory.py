from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from packages.knowledge.models import Base

class CheckpointModel(Base):
    __tablename__ = "checkpoints"

    thread_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    state_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class PersistentMemoryStore:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _namespace(repo_id: str, category: str) -> tuple[str, ...]:
        return ("repointel", repo_id, "memory", category)

    async def load_repository_context(self, repo_id: str) -> Dict[str, Any]:
        archetype = await self.store.aget(self._namespace(repo_id, "profile"), "archetype")
        review_runs = await self.store.asearch(
            self._namespace(repo_id, "review_issues"), limit=100
        )
        corrections = await self.store.asearch(
            self._namespace(repo_id, "human_corrections"), limit=20
        )

        issue_counts = Counter(
            issue.get("description", "").strip()
            for item in review_runs
            for issue in item.value.get("issues", [])
            if issue.get("description", "").strip()
        )
        recurring_issues = [
            {"description": description, "runs": count}
            for description, count in issue_counts.most_common()
            if count > 1
        ]

        return {
            "archetype": archetype.value.get("archetype") if archetype else None,
            "recurring_review_issues": recurring_issues,
            "human_review_corrections": [item.value for item in corrections],
        }

    async def save_archetype(self, repo_id: str, archetype: str) -> None:
        await self.store.aput(
            self._namespace(repo_id, "profile"),
            "archetype",
            {"archetype": archetype},
        )

    async def save_review_issues(self, repo_id: str, run_id: str, issues) -> None:
        await self.store.aput(
            self._namespace(repo_id, "review_issues"),
            run_id,
            {"issues": [issue.model_dump(mode="json") for issue in issues]},
        )

    async def save_human_correction(self, repo_id: str, run_id: str, feedback) -> None:
        await self.store.aput(
            self._namespace(repo_id, "human_corrections"),
            run_id,
            feedback.model_dump(mode="json"),
        )


@asynccontextmanager
async def open_repository_memory_store(database_url: Optional[str] = None):
    """Open the LangGraph Store using the application's configured database."""
    if database_url is None:
        from packages.shared.config import settings

        database_url = settings.database_url

    if database_url.startswith(("postgresql://", "postgres://", "postgresql+asyncpg://")):
        from langgraph.store.postgres import AsyncPostgresStore

        connection_string = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        async with AsyncPostgresStore.from_conn_string(connection_string) as store:
            await store.setup()
            yield store
        return

    from langgraph.store.memory import InMemoryStore

    yield InMemoryStore()
