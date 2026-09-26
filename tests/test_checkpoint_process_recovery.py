"""Process-boundary checkpoint test; does not simulate a Celery worker crash."""
import os
from pathlib import Path
import subprocess
import sys
import time


# A separate interpreter ensures no in-memory graph or saver survives the kill.
_CHILD = r'''
import asyncio
import json
from pathlib import Path
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from packages.knowledge.checkpoint_store import PostgresCheckpointer

class State(TypedDict):
    count: int
    approved: bool

async def main():
    mode, url, marker_path, calls_path = sys.argv[1:]
    marker, calls = Path(marker_path), Path(calls_path)
    engine = create_async_engine(url)
    saver = PostgresCheckpointer(async_sessionmaker(engine))

    def prepare(state):
        with calls.open("a", encoding="utf-8") as output:
            output.write("prepare\n")
        return {"count": state["count"] + 1}

    def review(state):
        return {"approved": interrupt({"count": state["count"]})["approved"]}

    builder = StateGraph(State)
    builder.add_node("prepare", prepare)
    builder.add_node("review", review)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "killed-process-review"}}
    if mode == "start":
        await graph.ainvoke({"count": 0, "approved": False}, config, durability="sync")
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("review",)
        assert snapshot.tasks[0].interrupts
        marker.write_text("suspended", encoding="utf-8")
        # The parent kills this process. No normal engine/graph cleanup occurs.
        await asyncio.Event().wait()
    else:
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("review",)
        assert snapshot.tasks[0].interrupts
        result = await graph.ainvoke(Command(resume={"approved": True}), config, durability="sync")
        assert result == {"count": 1, "approved": True}
        assert (await graph.aget_state(config)).next == ()
        history = [item async for item in graph.aget_state_history(config)]
        assert len(history) >= 4
        marker.write_text(json.dumps(result), encoding="utf-8")
        await engine.dispose()

asyncio.run(main())
'''


def test_checkpoint_resume_after_process_kill(tmp_path, monkeypatch):
    """Kill at a persisted interrupt; restart and apply a native resume command."""
    import json
    from packages.knowledge import migrations

    root = Path(__file__).resolve().parents[1]
    url = f"sqlite+aiosqlite:///{(tmp_path / 'recovery.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    migrations.upgrade_to_head()
    marker = tmp_path / "state.json"
    calls = tmp_path / "calls.txt"
    command = [sys.executable, "-c", _CHILD]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with (tmp_path / "child.log").open("w+", encoding="utf-8") as log:
        process = subprocess.Popen(
            [*command, "start", url, str(marker), str(calls)],
            cwd=root, env=env, stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 45
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if not marker.exists():
                log.seek(0)
                raise AssertionError(f"Child did not reach interrupt (exit={process.poll()}): {log.read()}")
            assert process.poll() is None
            assert marker.read_text(encoding="utf-8") == "suspended"
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        assert process.returncode != 0
    completed = subprocess.run(
        [*command, "resume", url, str(marker), str(calls)], cwd=root, env=env,
        capture_output=True, text=True, timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(marker.read_text(encoding="utf-8")) == {"count": 1, "approved": True}
    assert calls.read_text(encoding="utf-8").splitlines() == ["prepare"]
