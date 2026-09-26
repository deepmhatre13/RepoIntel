"""Authoritative application configuration for RepoIntel.

Design
------
There is exactly **one** configuration mechanism in RepoIntel: this module's
``Settings`` singleton. It loads the project ``.env`` file **once** (lazily, on
first attribute access) via ``python-dotenv`` and caches the result in
``os.environ``. No other module in the codebase calls ``load_dotenv``.

Why a singleton accessed through ``os.environ``?
  * ``LLMClient`` and ``packages.knowledge.db`` already read ``os.getenv`` for
    ``GOOGLE_API_KEY`` / ``DATABASE_URL`` / ``GEMINI_MODEL``.
  * By materializing ``.env`` into ``os.environ`` at a single chokepoint we keep
    those existing call-sites unchanged (no import-time network calls, no
    Gemini invocation) while guaranteeing the file is loaded before they run.
  * ``load_dotenv(override=False)`` never overwrites an existing real
    environment variable, so explicit shell/CI configuration always wins —
    deterministic degraded mode is preserved when the key is genuinely absent.

The module exposes ``Settings`` for callers that want a structured snapshot,
but the canonical flow is::

    configuration (this module) -> os.environ -> LLMClient / db / embeddings

"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("RepoIntel.config")

# Walk up from this file to locate the project root (the directory that owns
# ``.env`` / ``pyproject.toml``). This makes RepoIntel relocatable: it works
# regardless of the current working directory at startup (CLI, API, tests).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # packages/shared/config.py -> repo root

_ENV_FILENAME = os.environ.get("REPOINTEL_ENV_FILE", ".env")
if Path(_ENV_FILENAME).is_absolute():
    # Explicit absolute path (e.g. REPOINTEL_ENV_FILE=D:\\cfg\\.env) — use as-is.
    _ENV_PATH = Path(_ENV_FILENAME)
else:
    _ENV_PATH = _PROJECT_ROOT / _ENV_FILENAME


class Settings:
    """Lightweight configuration snapshot backed by ``os.environ``.

    Loading is idempotent and thread-safe: the ``.env`` file is parsed at most
    once per process. Existing environment variables are never overwritten
    (dotenv ``override=False``), so CI / shell configuration always takes
    precedence.
    """

    _lock = threading.Lock()
    _loaded: bool = False

    # ---- public, canonical entry points ---------------------------------

    @classmethod
    def load_dotenv(cls) -> bool:
        """Load the project ``.env`` into ``os.environ`` exactly once.

        Returns ``True`` if the file existed and was parsed. Safe to call from
        application lifespan, CLI entrypoints, or tests — it is a no-op after
        the first successful call.
        """
        if cls._loaded:
            return False
        with cls._lock:
            if cls._loaded:
                return False
            loaded = cls._do_load()
            cls._loaded = True
            return loaded

    # ---- structured accessors (optional convenience) --------------------

    @classmethod
    def get(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Return ``os.environ[key]`` after ensuring ``load_dotenv`` ran."""
        cls.load_dotenv()
        return os.getenv(key, default)

    @property
    def google_api_key(self) -> Optional[str]:
        return self.get("GOOGLE_API_KEY")

    @property
    def gemini_model(self) -> str:
        # Standardized authoritative default: gemini-3.8-flash.
        # Override with GEMINI_MODEL in environment/.env if explicitly configured.
        return self.get("GEMINI_MODEL") or "gemini-3.8-flash"

    @property
    def database_url(self) -> str:
        return self.get("DATABASE_URL") or "sqlite+aiosqlite:///./repointel.db"

    @property
    def embedding_model(self) -> str:
        return self.get("EMBEDDING_MODEL") or "all-MiniLM-L6-v2"

    @property
    def embedding_provider(self) -> str:
        return self.get("EMBEDDING_PROVIDER") or "local"

    @property
    def log_level(self) -> str:
        return self.get("LOG_LEVEL") or "INFO"

    @property
    def llm_available(self) -> bool:
        """True when a non-empty ``GOOGLE_API_KEY`` is configured locally.

        This is a *configuration* check only — it must never make a network
        request.
        """
        return bool(self.google_api_key)

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _do_load() -> bool:
        try:
            from dotenv import load_dotenv  # python-dotenv, already a dependency
        except ImportError:
            logger.warning("python-dotenv not installed; skipping .env load")
            return False

        if not _ENV_PATH.exists():
            logger.debug(".env not found at %s; relying on process environment", _ENV_PATH)
            return False

        # override=False => never clobber a real environment variable. This
        # keeps deterministic degraded mode valid when the key is absent from
        # both .env and the process environment.
        return load_dotenv(dotenv_path=str(_ENV_PATH), override=False, interpolate=True)


# Module-level singleton. Importing this module does NOT load .env (the load
# is lazy), avoiding import-time side effects. Call ``settings.load_dotenv()``
# or ``settings.get(...)`` to trigger loading.
settings = Settings()


def ensure_configured() -> None:
    """Eagerly load ``.env`` into ``os.environ``.

    Intended to be called once, early in application startup (API lifespan or
    CLI entrypoint), before any component reads environment-based configuration.
    """
    settings.load_dotenv()
