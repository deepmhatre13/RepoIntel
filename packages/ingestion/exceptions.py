"""Typed ingestion exceptions for RepoIntel.

Spec §5 mandates a deterministic ingestion layer with a specific exception set:
    CloneFailedError            -> git clone itself failed (retryable)
    RepoTooLargeError           -> size cap exceeded (non-retryable)
    InvalidRepositoryError      -> URL/scheme/host invalid or blocked by SSRF guard
    UnsupportedRepoStructureError -> clone succeeded but result unusable
    FilteringError              -> failure during .gitignore + denylist filtering
    MetadataExtractionError     -> failure during language/manifest/archetype extraction

All exceptions carry a ``retryable`` flag so the API error taxonomy and Celery
retry policy can map ingestion failures without inspecting message strings.
"""


class IngestionError(Exception):
    """Base class for all ingestion errors."""

    def __init__(self, message: str, retryable: bool = True):
        self.message = message
        self.retryable = retryable
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidRepositoryError(IngestionError):
    """The URL/scheme/host is invalid or blocked (SSRF guard)."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class RepoTooLargeError(IngestionError):
    """Repository exceeds the configurable size cap."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class CloneFailedError(IngestionError):
    """Git clone itself failed (network, bad ref, permissions, timeout)."""

    def __init__(self, message: str):
        super().__init__(message, retryable=True)


class UnsupportedRepoStructureError(IngestionError):
    """Cloning succeeded but the resulting structure is unusable."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class FilteringError(IngestionError):
    """Failure during .gitignore-aware + denylist filtering."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


class MetadataExtractionError(IngestionError):
    """Failure during language/manifest/archetype metadata extraction."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)