"""Large Repository Performance & Resource Controls for RepoIntel (Phase 42).

Enforces measurable resource budgets (file count/size, total size, bounded parse
concurrency, embedding batching, graph traversal bounds, context token budget,
LLM call cap, execution deadline) and records performance telemetry.
Budgets load from environment variables so deployments can tune without code changes.
"""

import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterable
from pydantic import BaseModel, Field

from packages.shared.telemetry import telemetry


class ResourceLimitExceeded(Exception):
    """Category RESOURCE_LIMIT per Phase 43.1 -- not retryable."""

    def __init__(self, message: str, limit: str = ""):
        super().__init__(message)
        self.limit = limit


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class ResourceBudgetConfig(BaseModel):
    max_repository_files: int = Field(default_factory=lambda: _env_int("MAX_REPOSITORY_FILES", 20000))
    max_file_size_bytes: int = Field(default_factory=lambda: _env_int("MAX_FILE_SIZE", 2 * 1024 * 1024))
    max_total_repo_size_bytes: int = Field(default_factory=lambda: _env_int("MAX_TOTAL_REPOSITORY_SIZE", 500 * 1024 * 1024))
    max_parse_workers: int = Field(default_factory=lambda: _env_int("MAX_PARSE_WORKERS", 4))
    max_embedding_batch: int = Field(default_factory=lambda: _env_int("MAX_EMBEDDING_BATCH", 64))
    max_graph_nodes: int = Field(default_factory=lambda: _env_int("MAX_GRAPH_NODES", 100))
    max_graph_edges: int = Field(default_factory=lambda: _env_int("MAX_GRAPH_EDGES", 500))
    max_search_results: int = Field(default_factory=lambda: _env_int("MAX_SEARCH_RESULTS", 20))
    max_history_commits: int = Field(default_factory=lambda: _env_int("MAX_HISTORY_COMMITS", 50))
    max_context_tokens: int = Field(default_factory=lambda: _env_int("MAX_CONTEXT_TOKENS", 8000))
    max_llm_calls: int = Field(default_factory=lambda: _env_int("MAX_LLM_CALLS", 25))
    max_execution_seconds: float = Field(default_factory=lambda: _env_int("MAX_EXECUTION_SECONDS", 3600))
    # Stage-specific timeouts (Phase 43.6)
    clone_timeout_seconds: float = Field(default_factory=lambda: _env_int("CLONE_TIMEOUT_SECONDS", 180))
    parse_timeout_seconds: float = Field(default_factory=lambda: _env_int("PARSE_TIMEOUT_SECONDS", 600))
    embedding_timeout_seconds: float = Field(default_factory=lambda: _env_int("EMBEDDING_TIMEOUT_SECONDS", 120))
    graph_timeout_seconds: float = Field(default_factory=lambda: _env_int("GRAPH_TIMEOUT_SECONDS", 120))
    llm_timeout_seconds: float = Field(default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 90))
    investigation_timeout_seconds: float = Field(default_factory=lambda: _env_int("INVESTIGATION_TIMEOUT_SECONDS", 300))


class SkippedFileNotice(BaseModel):
    file_path: str
    reason: str
    size_bytes: int


class ResourceManager:
    """Monitors and enforces safety thresholds across the analysis pipeline."""

    def __init__(self, config: Optional[ResourceBudgetConfig] = None):
        self.config = config or ResourceBudgetConfig()
        self._deadline: Optional[float] = None

    # ------------- Execution deadline (42.1 MAX_EXECUTION_SECONDS) -------------

    def start_deadline(self):
        self._deadline = time.monotonic() + self.config.max_execution_seconds

    def check_deadline(self, stage: str = ""):
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise ResourceLimitExceeded(
                f"Execution exceeded MAX_EXECUTION_SECONDS="
                f"{self.config.max_execution_seconds}s during stage '{stage}'",
                limit="max_execution_seconds",
            )

    def timeout_for_stage(self, stage: str) -> float:
        mapping = {
            "clone": self.config.clone_timeout_seconds,
            "parse": self.config.parse_timeout_seconds,
            "embedding": self.config.embedding_timeout_seconds,
            "graph": self.config.graph_timeout_seconds,
            "llm": self.config.llm_timeout_seconds,
            "investigation": self.config.investigation_timeout_seconds,
        }
        return mapping.get(stage, self.config.max_execution_seconds)

    # ---------------- File admission (42.2 large file handling) ---------------

    def admit_files(self, files_with_sizes: Dict[str, int]) -> Tuple[Dict[str, int], List[SkippedFileNotice], List[str]]:
        """Admit files under budget; skipped/oversized files are recorded, never silent."""
        admitted: Dict[str, int] = {}
        skipped: List[SkippedFileNotice] = []
        warnings: List[str] = []

        if len(files_with_sizes) > self.config.max_repository_files:
            warnings.append(
                f"Repository has {len(files_with_sizes)} files; only "
                f"{self.config.max_repository_files} admitted (MAX_REPOSITORY_FILES)."
            )
            telemetry.record_event("resource_limit", limit="max_repository_files")

        total_bytes = 0
        for path in sorted(files_with_sizes.keys()):
            size = files_with_sizes[path]
            if len(admitted) >= self.config.max_repository_files:
                skipped.append(SkippedFileNotice(
                    file_path=path, reason="repository file-count budget exhausted",
                    size_bytes=size))
                continue
            if size > self.config.max_file_size_bytes:
                skipped.append(SkippedFileNotice(
                    file_path=path,
                    reason=f"file size {size} bytes exceeds MAX_FILE_SIZE {self.config.max_file_size_bytes}",
                    size_bytes=size))
                telemetry.record_event("skipped_file", file=path, size=size)
                continue
            if total_bytes + size > self.config.max_total_repo_size_bytes:
                skipped.append(SkippedFileNotice(
                    file_path=path, reason="total repository size budget exhausted",
                    size_bytes=size))
                continue
            admitted[path] = size
            total_bytes += size

        return admitted, skipped, warnings

    @staticmethod
    def hash_content(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ---------------- Bounded concurrency (42.3) ------------------------------

    def run_bounded(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> Tuple[List[Any], List[BaseException]]:
        """Map fn over items with at most MAX_PARSE_WORKERS threads; errors returned, not raised mid-batch."""
        results: List[Any] = []
        errors: List[BaseException] = []
        items = list(items)
        if not items:
            return results, errors
        workers = max(1, min(self.config.max_parse_workers, len(items)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(lambda item: _safe_call(fn, item), items):
                if isinstance(res, BaseException):
                    errors.append(res)
                else:
                    results.append(res)
        return results, errors

    # ---------------- Embedding batching (42.4) -------------------------------

    def batch_items(self, items: List[Any], batch_size: Optional[int] = None) -> List[List[Any]]:
        bsize = batch_size or self.config.max_embedding_batch
        telemetry.increment("embedding_batches")
        return [items[i:i + bsize] for i in range(0, len(items), bsize)]

    # ---------------- Context budget (42.6) ------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def truncate_context_items(self, items: List[str], max_tokens: Optional[int] = None) -> Tuple[List[str], bool]:
        """Keep highest-priority (caller-ordered) items within the token budget."""
        limit = max_tokens or self.config.max_context_tokens
        kept: List[str] = []
        used = 0
        truncated = False
        for item in items:
            t = self.estimate_tokens(item)
            if used + t > limit:
                truncated = True
                continue
            kept.append(item)
            used += t
        if truncated:
            telemetry.record_event("context_truncated", kept=len(kept), dropped=len(items) - len(kept))
        return kept, truncated


def _safe_call(fn: Callable[[Any], Any], item: Any):
    try:
        return fn(item)
    except Exception as exc:  # surfaced to caller, never swallowed
        return exc
