"""WebSocket event streaming for RepoIntel.

Provides real-time job event streaming to connected clients.
Events are typed, ordered, and correlated to job/run IDs.

Architecture:
    Celery task -> event bridge -> WebSocket -> client
    
Events are NOT fake progress. They represent real operational milestones:
- job accepted
- ingestion started/completed
- parsing started/completed
- indexing started/completed
- LangGraph run started
- planner started/completed
- retrieval started
- tool invoked
- evidence assembled
- generator completed
- reviewer completed
- reflection/replan triggered
- finalization
- completed/failed
- human review required
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Typed event types for job streaming."""
    
    # Job lifecycle
    JOB_ACCEPTED = "job_accepted"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    
    # Ingestion
    INGESTION_STARTED = "ingestion_started"
    INGESTION_COMPLETED = "ingestion_completed"
    INGESTION_FAILED = "ingestion_failed"
    
    # Parsing
    PARSING_STARTED = "parsing_started"
    PARSING_COMPLETED = "parsing_completed"
    PARSING_FAILED = "parsing_failed"
    
    # Indexing
    INDEXING_STARTED = "indexing_started"
    INDEXING_COMPLETED = "indexing_completed"
    INDEXING_FAILED = "indexing_failed"
    
    # LangGraph agent
    AGENT_RUN_STARTED = "agent_run_started"
    AGENT_RUN_COMPLETED = "agent_run_completed"
    AGENT_RUN_FAILED = "agent_run_failed"
    
    # Planner
    PLANNER_STARTED = "planner_started"
    PLANNER_COMPLETED = "planner_completed"
    
    # Retriever
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    TOOL_INVOKED = "tool_invoked"
    EVIDENCE_ASSEMBLED = "evidence_assembled"
    
    # Generator/Reviewer
    GENERATOR_STARTED = "generator_started"
    GENERATOR_COMPLETED = "generator_completed"
    REVIEWER_STARTED = "reviewer_started"
    REVIEWER_COMPLETED = "reviewer_completed"
    
    # Reflection
    REFLECTION_TRIGGERED = "reflection_triggered"
    REPLAN_TRIGGERED = "replan_triggered"
    
    # Finalization
    FINALIZATION_STARTED = "finalization_started"
    FINALIZATION_COMPLETED = "finalization_completed"
    
    # Human review
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    HUMAN_FEEDBACK_RECEIVED = "human_feedback_received"
    
    # Progress
    PROGRESS_UPDATE = "progress_update"
    
    # Error
    ERROR = "error"


@dataclass
class JobEvent:
    """A single job event."""
    event_type: str
    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: Dict = field(default_factory=dict)
    sequence: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "sequence": self.sequence,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def create_job_event(
    event_type: EventType,
    run_id: str,
    data: Optional[Dict] = None,
    sequence: int = 0,
) -> JobEvent:
    """Create a new job event."""
    return JobEvent(
        event_type=event_type.value,
        run_id=run_id,
        data=data or {},
        sequence=sequence,
    )


class EventStreamManager:
    """Manages WebSocket connections and event distribution.
    
    This is the bridge between Celery tasks and WebSocket clients.
    Events are distributed to all connected clients for a given run_id.
    """
    
    def __init__(self):
        # run_id -> set of connected WebSockets
        self._connections: Dict[str, Set[WebSocket]] = {}
        # run_id -> list of events (for replay on reconnect)
        self._event_history: Dict[str, List[JobEvent]] = {}
        # Maximum events to keep in history per run
        self._max_history = 1000
        # Sequence counter per run
        self._sequence_counters: Dict[str, int] = {}
    
    async def connect(
        self,
        websocket: WebSocket,
        run_id: str,
        last_seen_sequence: Optional[int] = None,
    ):
        """Register a new WebSocket connection for a run.

        If ``last_seen_sequence`` is provided (e.g. by a reconnecting client),
        only events with a sequence number strictly greater than that value
        are replayed. This guarantees no duplicates and no missing events on
        reconnect, as long as those events are still within the bounded
        history window.
        """
        await websocket.accept()

        if run_id not in self._connections:
            self._connections[run_id] = set()
        self._connections[run_id].add(websocket)

        # Replay events that the client has not yet seen.
        history = self._event_history.get(run_id, [])
        if last_seen_sequence is not None:
            # Reconnect: replay only events *after* the last-seen sequence.
            replayable = [e for e in history if e.sequence > last_seen_sequence]
        else:
            # Fresh connection: replay the most recent window of events.
            replayable = history[-50:]

        for event in replayable:
            try:
                await websocket.send_text(event.to_json())
            except Exception:
                break

        logger.info(
            "WebSocket connected for run %s (%d total, replay=%d)",
            run_id,
            len(self._connections[run_id]),
            len(replayable),
        )
    
    def disconnect(self, websocket: WebSocket, run_id: str):
        """Unregister a WebSocket connection."""
        if run_id in self._connections:
            self._connections[run_id].discard(websocket)
            if not self._connections[run_id]:
                del self._connections[run_id]
        logger.info("WebSocket disconnected for run %s", run_id)
    
    async def emit(self, event: JobEvent):
        """Emit an event to all connected clients for the run."""
        run_id = event.run_id
        
        # Assign sequence number
        if run_id not in self._sequence_counters:
            self._sequence_counters[run_id] = 0
        self._sequence_counters[run_id] += 1
        event.sequence = self._sequence_counters[run_id]
        
        # Store in history
        if run_id not in self._event_history:
            self._event_history[run_id] = []
        self._event_history[run_id].append(event)
        
        # Trim history if needed
        if len(self._event_history[run_id]) > self._max_history:
            self._event_history[run_id] = self._event_history[run_id][-self._max_history:]
        
        # Send to all connected clients
        if run_id not in self._connections:
            return
        
        disconnected = set()
        for ws in self._connections[run_id]:
            try:
                await ws.send_text(event.to_json())
            except Exception:
                disconnected.add(ws)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws, run_id)
    
    async def emit_event(
        self,
        event_type: EventType,
        run_id: str,
        data: Optional[Dict] = None,
    ):
        """Convenience method to create and emit an event."""
        event = create_job_event(event_type, run_id, data)
        await self.emit(event)
    
    def get_history(self, run_id: str) -> List[Dict]:
        """Get event history for a run."""
        return [e.to_dict() for e in self._event_history.get(run_id, [])]
    
    def cleanup_run(self, run_id: str):
        """Clean up resources for a completed run."""
        self._connections.pop(run_id, None)
        self._event_history.pop(run_id, None)
        self._sequence_counters.pop(run_id, None)


# Global singleton instance
event_stream_manager = EventStreamManager()