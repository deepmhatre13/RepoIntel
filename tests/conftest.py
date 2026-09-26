"""Shared pytest fixtures for the RepoIntel test suite.

Hermeticity
-----------
By default the suite must be hermetic: tests exercise deterministic/degraded
paths and must never make real Gemini network calls, regardless of whether the
developer's shell or the project ``.env`` provides a ``GOOGLE_API_KEY``.

The autouse fixture enforces this by

1. clearing ambient credential variables from ``os.environ`` (so tests never
   depend on the developer's shell), and
2. pointing the configuration layer's ``_ENV_PATH`` at a location with no
   ``.env`` file and resetting the load-once flag (so the developer's real
   ``.env`` cannot leak credentials into tests).

Tests that verify the configuration mechanism itself (e.g.
``tests/test_configuration.py``) create their own temporary ``.env`` file and
re-point ``_ENV_PATH`` explicitly, which keeps them deterministic too.
Subprocess-based tests pass explicit environments and are unaffected.

Event-loop isolation
--------------------
Several test modules call ``asyncio.run()`` directly (e.g.
``test_repository_access_durability.py``, ``test_ws_security.py``).  Each
``asyncio.run()`` creates a *new* event loop, but the SQLAlchemy async engine
is a module-level singleton whose connection pool pins connections to the
loop that created them.  Reusing a connection across loops raises
``asyncpg.InterfaceError: cannot perform operation: another operation is in
progress``.  The ``_dispose_engine`` autouse fixture disposes the engine's
connection pool after every test so each ``asyncio.run()`` starts from a clean
pool, preventing cross-test connection corruption.
"""
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _hermetic_test_environment(monkeypatch):
    """Keep the test suite deterministic and network-free by default."""
    from packages.shared import config as app_config

    for var in ("GOOGLE_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    # A path that never contains a .env file -> degraded configuration.
    monkeypatch.setattr(
        app_config, "_ENV_PATH", Path(app_config.__file__).parent / "_no_dotenv_in_tests"
    )
    # Force the config singleton to re-resolve inside this scrubbed context.
    app_config.Settings._loaded = False
    yield
    # Next test re-resolves from its own scrubbed environment.
    app_config.Settings._loaded = False


@pytest.fixture(autouse=True)
def _dispose_engine():
    """Dispose the shared SQLAlchemy async engine after every test.

    The engine (``packages.knowledge.db.engine``) is a process-wide singleton
    whose connection pool associates pooled connections with the event loop that
    created them.  Tests that call ``asyncio.run()`` independently each spin up
    a fresh loop; without disposal, a connection borrowed by one loop can be
    returned to the pool and reused by another, producing
    ``asyncpg.InterfaceError: cannot perform operation: another operation is in
    progress``.  Disposing synchronously at teardown empties the pool so the
    next test's loop gets connections bound to its own loop.
    """
    yield
    # Run disposal in a fresh loop — the teardown runs after the test's loop
    # may already be closed, so we create one explicitly.
    import asyncio

    from packages.knowledge import db

    async def _dispose():
        await db.engine.dispose()

    try:
        asyncio.run(_dispose())
    except Exception:
        # Best-effort: if the engine was never initialized or the loop is
        # already gone, silently continue — the next test reinitializes it.
        pass