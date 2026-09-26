"""Database-backed async LangGraph saver. Schema is owned by Alembic, not nodes.

Thread IDs are supplied by the authorized runtime; this storage component does
not authorize callers. PostgreSQL and SQLite are the supported dialects.
"""
import base64
import json
from datetime import datetime, timezone

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, WRITES_IDX_MAP
from sqlalchemy import Column, DateTime, LargeBinary, MetaData, String, Table, delete, select

checkpoint_metadata = MetaData()
checkpoints_tbl = Table(
    "langgraph_checkpoints", checkpoint_metadata,
    Column("thread_id", String(255), primary_key=True),
    Column("checkpoint_ns", String(255), primary_key=True, server_default=""),
    Column("checkpoint_id", String(64), primary_key=True),
    Column("parent_checkpoint_id", String(64)),
    Column("checkpoint", LargeBinary, nullable=False),
    Column("metadata", LargeBinary),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
)
writes_tbl = Table(
    "langgraph_checkpoint_writes", checkpoint_metadata,
    Column("thread_id", String(255), primary_key=True),
    Column("checkpoint_ns", String(255), primary_key=True, server_default=""),
    Column("checkpoint_id", String(64), primary_key=True),
    Column("task_id", String(255), primary_key=True),
    Column("idx", String(16), primary_key=True),
    Column("channel", String(255), nullable=False),
    Column("type", String(100)),
    Column("blob", LargeBinary),
)


class PostgresCheckpointer(BaseCheckpointSaver):
    """Async saver; call application migrations before executing a graph.

    Checkpoints contain full channel values, not deltas. Pending writes can
    arrive before the corresponding checkpoint put, so they intentionally have
    no immediate FK to checkpoints. Thread deletion removes both atomically.
    """

    def __init__(self, session_factory=None, *, serde=None):
        super().__init__(serde=serde)
        self._session_factory = session_factory

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        from packages.knowledge.db import AsyncSessionLocal
        return AsyncSessionLocal()

    def _dump(self, value):
        kind, blob = self.serde.dumps_typed(value)
        return json.dumps({"repointel_serde": 1, "type": kind,
                           "blob": base64.b64encode(blob).decode("ascii")}).encode("utf-8")

    def _load(self, blob):
        payload = json.loads(bytes(blob))
        if isinstance(payload, dict) and payload.get("repointel_serde") == 1:
            return self.serde.loads_typed((payload["type"], base64.b64decode(payload["blob"], validate=True)))
        # Older saver payloads were JsonPlus JSON. Do not discard unreadable rows.
        return self.serde.loads_typed(("json", bytes(blob)))

    @staticmethod
    def _insert(session, table):
        dialect = session.get_bind().dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        else:
            raise ValueError(f"Unsupported checkpoint dialect: {dialect}")
        return insert(table)

    @staticmethod
    def _config(thread_id, namespace, checkpoint_id):
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": namespace,
                                  "checkpoint_id": checkpoint_id}}

    async def _tuple(self, session, row):
        config = self._config(row.thread_id, row.checkpoint_ns, row.checkpoint_id)
        rows = (await session.execute(select(writes_tbl).where(
            writes_tbl.c.thread_id == row.thread_id,
            writes_tbl.c.checkpoint_ns == row.checkpoint_ns,
            writes_tbl.c.checkpoint_id == row.checkpoint_id,
        ))).all()
        writes = [(wr.task_id, wr.channel, self.serde.loads_typed((wr.type, bytes(wr.blob))))
                  for wr in sorted(rows, key=lambda wr: (wr.task_id, int(wr.idx)))]
        return CheckpointTuple(
            config=config, checkpoint=self._load(row.checkpoint),
            metadata=self._load(row.metadata) if row.metadata else {},
            parent_config=self._config(row.thread_id, row.checkpoint_ns, row.parent_checkpoint_id)
            if row.parent_checkpoint_id else None, pending_writes=writes,
        )

    async def aget_tuple(self, config):
        cfg = config["configurable"]
        stmt = select(checkpoints_tbl).where(
            checkpoints_tbl.c.thread_id == cfg["thread_id"],
            checkpoints_tbl.c.checkpoint_ns == cfg.get("checkpoint_ns", ""),
        )
        if cfg.get("checkpoint_id"):
            stmt = stmt.where(checkpoints_tbl.c.checkpoint_id == cfg["checkpoint_id"])
        async with self._session() as session:
            row = (await session.execute(stmt.order_by(checkpoints_tbl.c.checkpoint_id.desc()).limit(1))).first()
            return await self._tuple(session, row) if row else None


    async def aput(self, config, checkpoint, metadata, new_versions):
        cfg = config["configurable"]
        thread_id, namespace = cfg["thread_id"], cfg.get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        async with self._session() as session:
            stmt = self._insert(session, checkpoints_tbl).values(
                thread_id=thread_id, checkpoint_ns=namespace, checkpoint_id=checkpoint_id,
                parent_checkpoint_id=cfg.get("checkpoint_id"),
                checkpoint=self._dump(checkpoint), metadata=self._dump(dict(metadata or {})),
            )
            await session.execute(stmt.on_conflict_do_update(
                index_elements=["thread_id", "checkpoint_ns", "checkpoint_id"],
                set_={"checkpoint": stmt.excluded.checkpoint, "metadata": stmt.excluded.metadata},
            ))
            await session.commit()
        return self._config(thread_id, namespace, checkpoint_id)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        cfg = config["configurable"]
        async with self._session() as session:
            for index, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, index)
                kind, blob = self.serde.dumps_typed(value)
                stmt = self._insert(session, writes_tbl).values(
                    thread_id=cfg["thread_id"], checkpoint_ns=cfg.get("checkpoint_ns", ""),
                    checkpoint_id=cfg["checkpoint_id"], task_id=task_id, idx=str(idx),
                    channel=channel, type=kind, blob=blob,
                )
                keys = ["thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"]
                # Successful task writes are immutable on redelivery. Special
                # channels (error, interrupt, resume) may be replaced.
                if idx < 0:
                    stmt = stmt.on_conflict_do_update(index_elements=keys, set_={
                        "channel": stmt.excluded.channel, "type": stmt.excluded.type,
                        "blob": stmt.excluded.blob,
                    })
                else:
                    stmt = stmt.on_conflict_do_nothing(index_elements=keys)
                await session.execute(stmt)
            await session.commit()

    async def alist(self, config=None, *, filter=None, before=None, limit=None):
        if limit is not None and limit <= 0:
            return
        stmt = select(checkpoints_tbl)
        if config is not None:
            cfg = config["configurable"]
            stmt = stmt.where(checkpoints_tbl.c.thread_id == cfg["thread_id"])
            if "checkpoint_ns" in cfg:
                stmt = stmt.where(checkpoints_tbl.c.checkpoint_ns == cfg["checkpoint_ns"])
            if cfg.get("checkpoint_id"):
                stmt = stmt.where(checkpoints_tbl.c.checkpoint_id == cfg["checkpoint_id"])
        if before is not None:
            stmt = stmt.where(checkpoints_tbl.c.checkpoint_id < before["configurable"]["checkpoint_id"])
        stmt = stmt.order_by(checkpoints_tbl.c.checkpoint_id.desc())
        if limit is not None and not filter:
            stmt = stmt.limit(limit)
        async with self._session() as session:
            rows = await session.stream(stmt)
            count = 0
            async for row in rows:
                item = await self._tuple(session, row)
                if filter and any(item.metadata.get(key) != value for key, value in filter.items()):
                    continue
                yield item
                count += 1
                if limit is not None and count >= limit:
                    break

    async def adelete_thread(self, thread_id):
        async with self._session() as session:
            await session.execute(delete(writes_tbl).where(writes_tbl.c.thread_id == thread_id))
            await session.execute(delete(checkpoints_tbl).where(checkpoints_tbl.c.thread_id == thread_id))
            await session.commit()


postgres_checkpointer = PostgresCheckpointer()


def resolve_checkpointer():
    """Resolve the checkpointer a graph run must checkpoint into.

    Durability is not optional in the worker path, so an unusable database must
    surface as an error instead of silently degrading to a process-local saver.
    In-memory checkpointing is available only when explicitly opted into
    (``REPOINTEL_ALLOW_INMEMORY_CHECKPOINTS=1``), which the tests and local CLI
    analysis use; it is never inferred from a failed database import.
    """
    import os

    from packages.shared.config import settings

    allow_in_memory = os.getenv("REPOINTEL_ALLOW_INMEMORY_CHECKPOINTS") == "1"
    if not allow_in_memory:
        try:
            allow_in_memory = str(settings.get("REPOINTEL_ALLOW_INMEMORY_CHECKPOINTS") or "").lower() in {"1", "true", "yes"}
        except Exception:  # configuration layer is optional for pure unit usage
            allow_in_memory = False

    if allow_in_memory:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    # Constructing the saver does not touch the database; the first checkpoint
    # write does. An unusable database therefore fails the run loudly rather
    # than being swallowed here.
    return postgres_checkpointer

