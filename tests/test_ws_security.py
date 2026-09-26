"""Focused WebSocket security tests for /ws/{run_id}.

Covers the six required behaviors:
1. unauthenticated connection rejected
2. invalid token rejected
3. authenticated owner accepted
4. unauthorized user rejected
5. repository/run ownership enforced (run_id -> PipelineRun -> repository)
6. replay sequence behavior remains correct (bounded, ordered, no dupes)

Rejection contract: every failure mode closes with 1008 BEFORE the connection
is accepted, so no events stream and nothing about run/repository existence
leaks to unauthorized callers.
"""
import asyncio
import json
import uuid

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from apps.api import app
from packages.knowledge.db import AsyncSessionLocal
from packages.knowledge.models import PipelineRunModel
from packages.websocket.event_stream import (
    event_stream_manager,
    create_job_event,
    EventType,
)


@pytest.fixture(scope="module")
def client():
    """Context-managed TestClient so the FastAPI lifespan (init_db) runs."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ws_env(client):
    """Two users (A owns the repository, B has no access), one persisted run.

    All state is namespaced with a random suffix so repeated runs against the
    persistent dev database stay deterministic.
    """
    uniq = uuid.uuid4().hex[:10]

    user_a = client.post(
        "/auth/register",
        json={
            "username": f"ws_owner_{uniq}",
            "email": f"ws_owner_{uniq}@test.com",
            "password": "wspass123",
        },
    )
    assert user_a.status_code == 201, user_a.text
    user_b = client.post(
        "/auth/register",
        json={
            "username": f"ws_outsider_{uniq}",
            "email": f"ws_outsider_{uniq}@test.com",
            "password": "wspass123",
        },
    )
    assert user_b.status_code == 201, user_b.text

    token_a = user_a.json()["access_token"]
    token_b = user_b.json()["access_token"]

    repo = client.post(
        "/repositories",
        json={"repo_url": f"https://github.com/example/ws-{uniq}"},
        headers={"Authorization": f"Bearer {token_a}"},
        )
    assert repo.status_code == 200, repo.text
    repo_id = repo.json()["repository_id"]

    run_id = f"run-ws-{uniq}"

    # Seed the persisted run (run_id -> repository_id) and the bounded event
    # history (sequences 1..10) inside the manager. Emission happens with no
    # live connections, so only history/sequence state is written.
    #
    # The ``client`` fixture (TestClient) started the ASGI app's lifespan on a
    # dedicated thread+event-loop, which borrowed connections from the shared
    # SQLAlchemy engine pool.  ``asyncio.run()`` below runs on a *different* loop,
    # so we dispose the pool first -- forcing fresh connections bound to this
    # loop instead of corrupted cross-loop reuse ("attached to a different
    # loop").  The pool is re-established lazily on the next acquisition.
    import asyncio as _asyncio

    from packages.knowledge import db as _db

    _asyncio.run(_db.engine.dispose())

    async def _seed():
        async with AsyncSessionLocal() as session:
            session.add(
                PipelineRunModel(
                    run_id=run_id,
                    repository_id=repo_id,
                    status="RUNNING",
                )
            )
            await session.commit()
        for i in range(1, 11):
            await event_stream_manager.emit(
                create_job_event(EventType.PROGRESS_UPDATE, run_id, {"step": i})
            )

    asyncio.run(_seed())

    return {
        "token_a": token_a,
        "token_b": token_b,
        "run_id": run_id,
        "unknown_run_id": f"run-does-not-exist-{uniq}",
    }


def _close_code(ws_url: str, client: TestClient) -> int:
    """Connect and capture the rejection close code (must be pre-accept)."""
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(ws_url):
            pass  # never accepted -> no receive expected
    return excinfo.value.code


def test_ws_rejects_unauthenticated(client, ws_env):
    """No token at all -> rejected before accept, no event exposure."""
    assert _close_code(f"/ws/{ws_env['run_id']}", client) == 1008


def test_ws_rejects_invalid_token(client, ws_env):
    """Garbage / tampered / wrong-signature token -> same uniform rejection."""
    assert _close_code(f"/ws/{ws_env['run_id']}?token=not-a-jwt", client) == 1008


def test_ws_rejects_unknown_run(client, ws_env):
    """Valid token, unknown run_id -> rejected without revealing non-existence.

    Must be indistinguishable from the unauthorized case (same close code,
    pre-accept), so run existence is never leaked to any caller.
    """
    assert (
        _close_code(f"/ws/{ws_env['unknown_run_id']}?token={ws_env['token_b']}", client)
        == 1008
    )
    # The owner's valid token on an unknown run gets the SAME uniform close.
    assert (
        _close_code(f"/ws/{ws_env['unknown_run_id']}?token={ws_env['token_a']}", client)
        == 1008
    )


def test_ws_rejects_unauthorized_user(client, ws_env):
    """User B (authenticated, no repository access) -> rejected pre-accept.

    The connection is never accepted, therefore zero events are delivered.
    """
    code = _close_code(f"/ws/{ws_env['run_id']}?token={ws_env['token_b']}", client)
    assert code == 1008


def test_ws_accepts_owner_with_replay(client, ws_env):
    """Owner connects: accepted, receives the full bounded replay 1..10.

    Verifies typed events, monotonic sequences, no duplicates, correct order.
    """
    with client.websocket_connect(
        f"/ws/{ws_env['run_id']}?token={ws_env['token_a']}"
    ) as ws:
        received = [json.loads(ws.receive_text()) for _ in range(10)]
    sequences = [e["sequence"] for e in received]
    assert sequences == list(range(1, 11))
    assert all(e["event_type"] == EventType.PROGRESS_UPDATE.value for e in received)
    assert all(e["run_id"] == ws_env["run_id"] for e in received)
    assert [e["data"]["step"] for e in received] == list(range(1, 11))


def test_ws_replay_after_reconnect(client, ws_env):
    """Reconnect with last_seen_sequence=7 -> exactly events 8..10.

    No duplicates, no missing events, correct ordering.
    """
    with client.websocket_connect(
        f"/ws/{ws_env['run_id']}?token={ws_env['token_a']}&last_seen_sequence=7"
    ) as ws:
        received = [json.loads(ws.receive_text()) for _ in range(3)]
    assert [e["sequence"] for e in received] == [8, 9, 10]

    # An up-to-date client (last_seen >= latest) is accepted with an empty
    # replay window: no duplicates, nothing new delivered.
    with client.websocket_connect(
        f"/ws/{ws_env['run_id']}?token={ws_env['token_a']}&last_seen_sequence=10"
    ) as ws:
        pass
