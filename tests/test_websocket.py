"""WebSocket event streaming tests for RepoIntel."""

import pytest
import json
from datetime import datetime, timezone

from packages.websocket.event_stream import (
    EventStreamManager,
    JobEvent,
    EventType,
    create_job_event,
)
from packages.runtime.pipeline_orchestrator import _stream_langgraph_events


class TestJobEvent:
    """Test JobEvent creation and serialization."""

    def test_create_event(self):
        event = JobEvent(
            event_type=EventType.JOB_ACCEPTED.value,
            run_id="run-123",
            data={"repo_url": "https://github.com/example/repo"},
        )
        assert event.event_type == "job_accepted"
        assert event.run_id == "run-123"
        assert event.data["repo_url"] == "https://github.com/example/repo"
        assert event.sequence == 0

    def test_event_to_dict(self):
        event = JobEvent(
            event_type=EventType.INGESTION_STARTED.value,
            run_id="run-456",
        )
        d = event.to_dict()
        assert d["event_type"] == "ingestion_started"
        assert d["run_id"] == "run-456"
        assert "timestamp" in d
        assert "data" in d
        assert "sequence" in d

    def test_event_to_json(self):
        event = JobEvent(
            event_type=EventType.JOB_COMPLETED.value,
            run_id="run-789",
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "job_completed"
        assert parsed["run_id"] == "run-789"

    def test_create_job_event_helper(self):
        event = create_job_event(
            EventType.PARSING_STARTED,
            "run-abc",
            {"file_count": 42},
        )
        assert event.event_type == "parsing_started"
        assert event.run_id == "run-abc"
        assert event.data["file_count"] == 42


class TestEventStreamManager:
    """Test EventStreamManager."""

    @pytest.fixture
    def manager(self):
        return EventStreamManager()

    def test_init(self, manager):
        assert manager._connections == {}
        assert manager._event_history == {}
        assert manager._max_history == 1000

    @pytest.mark.asyncio
    async def test_emit_without_connections(self, manager):
        """Emitting without connections should not raise."""
        event = create_job_event(EventType.JOB_ACCEPTED, "run-1")
        await manager.emit(event)
        # Event should be stored in history
        assert "run-1" in manager._event_history
        assert len(manager._event_history["run-1"]) == 1

    def test_sequence_assignment(self, manager):
        """Events should get sequential numbers."""
        async def _test():
            for i in range(5):
                event = create_job_event(EventType.PROGRESS_UPDATE, "run-1")
                await manager.emit(event)
            
            history = manager.get_history("run-1")
            sequences = [e["sequence"] for e in history]
            assert sequences == [1, 2, 3, 4, 5]

        import asyncio
        asyncio.run(_test())

    def test_history_limit(self, manager):
        """History should be trimmed to max_history."""
        async def _test():
            for i in range(1005):
                event = create_job_event(EventType.PROGRESS_UPDATE, "run-1")
                await manager.emit(event)
            
            history = manager.get_history("run-1")
            assert len(history) == 1000

        import asyncio
        asyncio.run(_test())

    def test_cleanup_run(self, manager):
        """Cleanup should remove all run data."""
        async def _test():
            event = create_job_event(EventType.JOB_ACCEPTED, "run-1")
            await manager.emit(event)
            assert "run-1" in manager._event_history
            
            manager.cleanup_run("run-1")
            assert "run-1" not in manager._event_history
            assert "run-1" not in manager._sequence_counters

        import asyncio
        asyncio.run(_test())

    def test_get_history(self, manager):
        """get_history should return event dicts."""
        async def _test():
            event = create_job_event(EventType.JOB_ACCEPTED, "run-1", {"key": "value"})
            await manager.emit(event)
            
            history = manager.get_history("run-1")
            assert len(history) == 1
            assert history[0]["event_type"] == "job_accepted"
            assert history[0]["data"]["key"] == "value"

        import asyncio
        asyncio.run(_test())


class TestEventTypes:
    """Test EventType enum."""

    def test_all_event_types_unique(self):
        values = [e.value for e in EventType]
        assert len(values) == len(set(values))

    def test_event_type_values(self):
        assert EventType.JOB_ACCEPTED.value == "job_accepted"
        assert EventType.INGESTION_STARTED.value == "ingestion_started"
        assert EventType.PARSING_COMPLETED.value == "parsing_completed"
        assert EventType.JOB_FAILED.value == "job_failed"


class TestLangGraphEventBridge:
    @pytest.mark.asyncio
    async def test_streams_graph_events_and_returns_root_output(self):
        class Chunk:
            content = "hello"

        class Graph:
            async def astream_events(self, initial_state, config, version):
                assert initial_state == {"value": 1}
                assert version == "v2"
                yield {
                    "event": "on_chain_start",
                    "name": "synthesize_section",
                    "run_id": "event-1",
                    "parent_ids": ["root"],
                    "metadata": {"langgraph_node": "Synthesis"},
                    "data": {"input": {"value": 1}},
                }
                yield {
                    "event": "on_chat_model_stream",
                    "name": "model",
                    "run_id": "event-2",
                    "parent_ids": ["root", "node"],
                    "metadata": {},
                    "data": {"chunk": Chunk()},
                }
                yield {
                    "event": "on_chain_end",
                    "name": "LangGraph",
                    "run_id": "root",
                    "parent_ids": [],
                    "metadata": {},
                    "data": {"output": {"value": 2}},
                }

        manager = EventStreamManager()
        result = await _stream_langgraph_events(
            Graph(), {"value": 1}, {"configurable": {}}, "run-1", manager
        )

        assert result == {"value": 2}
        history = manager.get_history("run-1")
        assert len(history) == 3
        assert history[0]["data"]["node"] == "Synthesis"
        assert history[1]["data"]["token"] == "hello"
