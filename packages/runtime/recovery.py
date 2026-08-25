"""Failure Recovery & Resilient Execution for RepoIntel (Phase 43).

- Classifies failures into the ten Phase 43.1 categories with retryability.
- Retries ONLY transient categories, using exponential backoff + jitter (43.2).
- Cooperative cancellation tokens so long runs stop safely and persist CANCELLED
  status (43.5).
- Stage timeouts via asyncio.wait_for wrappers (43.6).
- Deterministic degraded mode helpers: when Gemini is unavailable, deterministic
  capabilities keep working and results are explicitly marked llm_available=False,
  never pretending LLM synthesis succeeded (43.7).

Idempotent artifacts are guaranteed by the persistence layer (40.4/43.4): a retried
job replaces rows instead of duplicating them.
"""

import asyncio
import logging
import random
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Awaitable, Tuple
from pydantic import BaseModel
from packages.shared.telemetry import telemetry
from packages.runtime.resource_manager import ResourceLimitExceeded

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"
    INVALID_INPUT = "INVALID_INPUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    LLM_FAILURE = "LLM_FAILURE"
    PARSE_FAILURE = "PARSE_FAILURE"
    DATABASE_FAILURE = "DATABASE_FAILURE"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    TIMEOUT = "TIMEOUT"
    EVIDENCE_FAILURE = "EVIDENCE_FAILURE"


# Categories worth retrying (43.2). Everything else fails fast.
RETRYABLE_CATEGORIES = {
    ErrorCategory.TRANSIENT,
    ErrorCategory.NETWORK_FAILURE,
    ErrorCategory.TIMEOUT,
    ErrorCategory.DATABASE_FAILURE,
    ErrorCategory.LLM_FAILURE,
}

_PERMANENT_MARKERS = (
    "syntaxerror", "syntax error", "invalid syntax", "not a git repository",
    "malformed", "unsupported file",
)
_TRANSIENT_MARKERS = (
    "temporarily", "try again", "connection reset", "connection refused",
    "broken pipe", "server disconnected", "temporarily unavailable",
)
_RATE_LIMIT_MARKERS = ("429", "quota", "rate limit", "resource_exhausted")
_DB_MARKERS = ("sqlalchemy", "database", "sqlite", "postgres", "deadlock", "locked")

class FailureRecord(BaseModel):
    category: ErrorCategory
    message: str
    stage: str
    is_retryable: bool
    attempt_count: int = 1


class CancellationRequested(Exception):
    """Raised inside protected sections when a cooperative cancel was requested."""


class CancellationToken:
    """Cooperative cancellation checked between stages/files (43.5)."""

    def __init__(self):
        self._cancelled = False
        self.reason: Optional[str] = None

    def cancel(self, reason: str = ""):
        self._cancelled = True
        self.reason = reason or self.reason

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check(self):
        if self._cancelled:
            raise CancellationRequested(self.reason or "cancellation requested")


class ExecutionResilienceManager:
    """Retry, classification, cancellation, timeout and degraded-mode utilities."""

    def __init__(self, max_attempts: int = 3, backoff_base_seconds: float = 0.25,
                 backoff_max_seconds: float = 8.0):
        self.max_attempts = max_attempts
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.cancel_tokens: Dict[str, CancellationToken] = {}
        self.failures: List[FailureRecord] = []

    # ---------------- Classification (43.1) ----------------

    def classify_error(self, exc: BaseException) -> Tuple[ErrorCategory, bool]:
        if isinstance(exc, CancellationRequested):
            return ErrorCategory.TRANSIENT, False
        if isinstance(exc, ResourceLimitExceeded):
            return ErrorCategory.RESOURCE_LIMIT, False
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return ErrorCategory.TIMEOUT, True
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            return ErrorCategory.PERMANENT, False

        from packages.agentic.llm_client import LLMFailureError, StructuredOutputValidationError
        if isinstance(exc, LLMFailureError):
            return ErrorCategory.LLM_FAILURE, bool(getattr(exc, "transient", True))
        if isinstance(exc, StructuredOutputValidationError):
            return ErrorCategory.LLM_FAILURE, True

        msg = str(exc).lower()
        type_name = type(exc).__name__.lower()

        if any(m in msg for m in _RATE_LIMIT_MARKERS):
            return ErrorCategory.LLM_FAILURE, True
        if "timeout" in msg or "timed out" in msg:
            return ErrorCategory.TIMEOUT, True
        if any(m in msg for m in _DB_MARKERS):
            return ErrorCategory.DATABASE_FAILURE, True
        if any(m in msg for m in _TRANSIENT_MARKERS) or any(
                m in msg for m in ("connection", "network", "socket", "getaddrinfo")):
            return ErrorCategory.NETWORK_FAILURE, True
        if "memory" in msg or "overflow" in msg:
            return ErrorCategory.RESOURCE_LIMIT, False
        if "syntaxerror" in type_name or "syntax error" in msg or "invalid syntax" in msg:
            return ErrorCategory.PARSE_FAILURE, False
        if any(m in msg for m in _PERMANENT_MARKERS):
            return ErrorCategory.INVALID_INPUT, False
        if "invalid" in msg or "not found" in msg or isinstance(exc, ValueError):
            return ErrorCategory.INVALID_INPUT, False
        if "parse" in msg or "parseerror" in type_name:
            return ErrorCategory.PARSE_FAILURE, False
        return ErrorCategory.TRANSIENT, True

    def record_failure(self, exc: BaseException, stage: str, attempts: int) -> FailureRecord:
        category, retryable = self.classify_error(exc)
        rec = FailureRecord(
            category=category, message=str(exc)[:500], stage=stage,
            is_retryable=retryable, attempt_count=attempts,
        )
        self.failures.append(rec)
        telemetry.record_event("failure", stage=stage, category=category.value, retryable=retryable)
        return rec

    # ---------------- Retry policy (43.2) ----------------

    def _backoff_delay(self, attempt: int) -> float:
        base = self.backoff_base_seconds * (2 ** (attempt - 1))
        jitter = random.uniform(0, base * 0.25)
        return min(base + jitter, self.backoff_max_seconds)

    async def execute_with_retry(self, stage: str, func: Callable, *args,
                                 cancel_token: Optional[CancellationToken] = None,
                                 max_attempts: Optional[int] = None, **kwargs) -> Any:
        """Run func with retry only for retryable categories; exponential backoff + jitter.

        Sync and async callables are both supported. Cancellation is checked before
        each attempt so cooperative cancels stop new work promptly.
        """
        attempts_allowed = max_attempts or self.max_attempts
        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt < attempts_allowed:
            if cancel_token is not None:
                cancel_token.check()
            attempt += 1
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                rec = self.record_failure(exc, stage, attempt)
                if not rec.is_retryable or attempt >= attempts_allowed:
                    logger.error("Stage '%s' failed permanently on attempt %s (%s): %s",
                                 stage, attempt, rec.category.value, exc)
                    raise
                delay = self._backoff_delay(attempt)
                telemetry.record_event("retry", stage=stage, attempt=attempt,
                                       category=rec.category.value, delay=round(delay, 3))
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    # ---------------- Cancellation registry (43.5) ----------------

    def register_run(self, run_id: str) -> CancellationToken:
        token = CancellationToken()
        self.cancel_tokens[run_id] = token
        return token

    def get_token(self, run_id: str) -> Optional[CancellationToken]:
        return self.cancel_tokens.get(run_id)

    def cancel_job(self, run_id: str, reason: str = "") -> bool:
        token = self.cancel_tokens.get(run_id)
        if token is None:
            token = CancellationToken()
            self.cancel_tokens[run_id] = token
        token.cancel(reason)
        telemetry.record_event("cancellation_requested", run_id=run_id, reason=reason)
        return True

    def is_cancelled(self, run_id: str) -> bool:
        token = self.cancel_tokens.get(run_id)
        return bool(token and token.cancelled)

    # ---------------- Stage timeouts (43.6) ----------------

    async def with_stage_timeout(self, coro: Awaitable[Any], stage: str, timeout_seconds: float) -> Any:
        """Enforce a stage-specific timeout; TimeoutError is classified as TIMEOUT (retryable)."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            telemetry.record_event("timeout", stage=stage, timeout_seconds=timeout_seconds)
            raise

    # ---------------- Degraded mode (43.7) ----------------

    async def execute_degraded_mode(self, repo_id: str, query: str, facade: Any,
                                    reason: str = "") -> Dict[str, Any]:
        """Deterministic capabilities continue when Gemini is unavailable.

        Never pretends LLM synthesis succeeded: the result is explicitly marked
        llm_available=False with the failure reason.
        """
        search_res: Dict[str, Any] = {}
        impact: Dict[str, Any] = {}
        health: Dict[str, Any] = {}
        errors: List[str] = []
        for name, coro in (
            ("search", facade.search_code(repo_id, query)),
            ("impact", facade.analyze_impact(repo_id, query)),
            ("health", facade.get_repository_health(repo_id)),
        ):
            try:
                res = await coro
                if name == "search":
                    search_res = res or {}
                elif name == "impact":
                    impact = res or {}
                else:
                    health = res or {}
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        related_symbols = search_res.get("related_symbols", []) if isinstance(search_res, dict) else []
        results = search_res.get("results", []) if isinstance(search_res, dict) else []

        facts = [f"Search matched {len(results)} evidence items in deterministic index."]
        if related_symbols:
            facts.append(f"Matched symbols: {related_symbols}")
        if isinstance(impact, dict):
            facts.append(f"Direct dependents count: {impact.get('total_dependents', 0)}")

        telemetry.record_event("degraded_mode", repo_id=repo_id, reason=reason)
        return {
            "mode": "DEGRADED_DETERMINISTIC",
            "llm_available": False,
            "llm_unavailable_reason": reason or "Gemini unavailable",
            "query": query,
            "answer": (
                "Gemini synthesis is currently unavailable. "
                "The following deterministic repository evidence was retrieved without LLM reasoning."
            ),
            "search_results": results,
            "impact_analysis": impact if isinstance(impact, dict) else {},
            "health_status": health.get("overall_status", "UNKNOWN") if isinstance(health, dict) else "UNKNOWN",
            "facts": facts,
            "inferences": [],
            "recommendations": [],
            "errors": errors,
            "confidence": 0.6,
        }
