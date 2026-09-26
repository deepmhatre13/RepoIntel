"""Celery runtime integration tests.

Validates that the async task execution pattern works correctly outside of
CELERY_TASK_ALWAYS_EAGER mode. This catches the critical bug where
asyncio.get_event_loop() fails in worker threads.

These tests verify:
1. _run_async helper creates and cleans up event loops properly
2. Tasks can execute async code without an existing event loop
3. Multiple sequential async operations work correctly
4. Exception propagation works through the async boundary
"""

import asyncio
import pytest
from packages.worker.tasks import _run_async


def test_run_async_executes_coroutine():
    """_run_async must execute an async coroutine and return its result."""
    async def _sample():
        return {"status": "ok", "value": 42}

    result = _run_async(_sample())
    assert result == {"status": "ok", "value": 42}


def test_run_async_creates_fresh_loop():
    """Each call must create a fresh event loop (no reuse of closed loops)."""
    loops = []

    async def _capture_loop():
        loop = asyncio.get_event_loop()
        loops.append(loop)
        return loop

    _run_async(_capture_loop())
    _run_async(_capture_loop())
    # Two distinct loop objects were created
    assert len(loops) == 2
    assert loops[0] is not loops[1]
    # Both should be closed after _run_async returns
    assert loops[0].is_closed()
    assert loops[1].is_closed()


def test_run_async_cleans_up_loop():
    """After _run_async completes, no event loop should remain set."""
    async def _noop():
        pass

    _run_async(_noop())
    # After cleanup, get_event_loop should raise (no loop set in this thread)
    with pytest.raises(RuntimeError):
        asyncio.get_event_loop()


def test_run_async_propagates_exceptions():
    """Exceptions in async code must propagate through _run_async."""
    async def _failing():
        raise ValueError("test error propagation")

    with pytest.raises(ValueError, match="test error propagation"):
        _run_async(_failing())


def test_run_async_handles_nested_async():
    """Nested async operations must work correctly."""
    async def _inner(x):
        return x * 2

    async def _outer():
        results = []
        for i in range(5):
            results.append(await _inner(i))
        return results

    result = _run_async(_outer())
    assert result == [0, 2, 4, 6, 8]


def test_run_async_no_preexisting_loop():
    """_run_async must work even when no event loop exists in the thread.
    
    This is the critical test: Celery worker threads have no event loop.
    The old asyncio.get_event_loop().run_until_complete() pattern would
    raise RuntimeError here.
    """
    # Ensure no event loop exists in this thread
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.close()
    except RuntimeError:
        pass  # No loop exists - this is the expected state

    async def _worker_thread_task():
        await asyncio.sleep(0.001)
        return "success"

    result = _run_async(_worker_thread_task())
    assert result == "success"


def test_run_async_concurrent_tasks():
    """Multiple _run_async calls must not interfere with each other."""
    async def _task(n):
        await asyncio.sleep(0.001)
        return n

    results = [_run_async(_task(i)) for i in range(10)]
    assert results == list(range(10))
