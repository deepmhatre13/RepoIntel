import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import String, Text, DateTime, select
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession
from packages.knowledge.models import Base, AgentRunModel, ArtifactModel, ReviewFeedbackModel

class CheckpointModel(Base):
    __tablename__ = "checkpoints"

    thread_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    state_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class PersistentMemoryStore:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_run(self, run_id: str, repo_id: str, thread_id: str, status: str = "RUNNING"):
        run = AgentRunModel(run_id=run_id, repo_id=repo_id, status=status)
        await self.session.merge(run)
        await self.session.commit()

    async def save_artifact_version(self, run_id: str, artifact_type: str, content: str) -> int:
        art = ArtifactModel(run_id=run_id, artifact_type=artifact_type, content=content)
        self.session.add(art)
        await self.session.commit()
        await self.session.refresh(art)
        return art.id

    async def save_review_feedback(self, run_id: str, verdict: str, issues_json: str):
        rev = ReviewFeedbackModel(run_id=run_id, verdict=verdict, issues_json=issues_json)
        self.session.add(rev)
        await self.session.commit()

    async def save_checkpoint(self, thread_id: str, run_id: str, state_dict: Dict[str, Any]):
        cp = CheckpointModel(thread_id=thread_id, run_id=run_id, state_json=json.dumps(state_dict))
        await self.session.merge(cp)
        await self.session.commit()

    async def load_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        stmt = select(CheckpointModel).where(CheckpointModel.thread_id == thread_id)
        res = await self.session.execute(stmt)
        cp = res.scalar_one_or_none()
        if cp:
            return json.loads(cp.state_json)
        return None
