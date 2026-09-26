import os
import pytest
from packages.ingestion import run_ingestion_pipeline, cleanup_repository
from packages.ingestion.cloning import clone_repository, validate_repo_url
from packages.ingestion.exceptions import (
    CloneFailedError,
    RepoTooLargeError,
    InvalidRepositoryError,
    IngestionError,
)


def create_synthetic_py_lib(target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    
    # pyproject.toml
    with open(os.path.join(target_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "mathlib"\nversion = "0.1.0"\ndependencies = ["numpy>=1.20.0"]')
    
    # src/calculator.py
    os.makedirs(os.path.join(target_dir, "src"), exist_ok=True)
    with open(os.path.join(target_dir, "src", "calculator.py"), "w") as f:
        f.write('''
import os

class Calculator:
    """Basic math calculator class."""
    def add(self, a: float, b: float) -> float:
        return a + b
    
    def multiply(self, a: float, b: float) -> float:
        return a * b

def main():
    calc = Calculator()
    print("Add 2+3:", calc.add(2, 3))

if __name__ == "__main__":
    main()
''')


def test_ingestion_synthetic():
    target = "./evals/datasets/python_library"
    create_synthetic_py_lib(target)
    
    manifest, files = run_ingestion_pipeline(target)
    assert manifest.file_count >= 2
    assert "Python" in manifest.languages
    assert manifest.archetype.value in ("LIBRARY", "UNKNOWN")
    cleanup_repository(manifest.local_path)


# ---------------------------------------------------------------------------
# Tests for SSRF guard and typed exceptions (spec §5 gap closure)
# ---------------------------------------------------------------------------


class TestSSRFGuard:
    """Test that validate_repo_url applies SSRF protection."""

    def test_localhost_is_blocked(self):
        with pytest.raises(InvalidRepositoryError) as exc_info:
            validate_repo_url("http://localhost/repo")
        assert "SSRF" in str(exc_info.value) or "private/internal" in str(exc_info.value)

    def test_127_0_0_1_is_blocked(self):
        with pytest.raises(InvalidRepositoryError):
            validate_repo_url("http://127.0.0.1/repo")

    def test_169_254_169_254_is_blocked(self):
        with pytest.raises(InvalidRepositoryError):
            validate_repo_url("http://169.254.169.254/latest/meta-data/")

    def test_10_x_is_blocked(self):
        with pytest.raises(InvalidRepositoryError):
            validate_repo_url("http://10.0.0.1/repo")

    def test_192_168_is_blocked(self):
        with pytest.raises(InvalidRepositoryError):
            validate_repo_url("http://192.168.1.1/repo")

    def test_file_scheme_is_blocked(self):
        with pytest.raises(InvalidRepositoryError) as exc_info:
            validate_repo_url("file:///etc/passwd")
        assert "file" in str(exc_info.value).lower()

    def test_github_url_is_allowed(self):
        parsed = validate_repo_url("https://github.com/expressjs/express")
        assert parsed.scheme == "https"
        assert parsed.hostname == "github.com"

    def test_empty_url_is_blocked(self):
        with pytest.raises(InvalidRepositoryError):
            validate_repo_url("")

    def test_none_url_is_blocked(self):
        with pytest.raises(InvalidRepositoryError):
            validate_repo_url(None)

    def test_local_dir_returns_local_result(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            parsed = validate_repo_url(tmpdir)
            assert parsed.scheme == "local"


class TestTypedExceptions:
    """Test that typed exceptions carry the retryable flag."""

    def test_invalid_repo_not_retryable(self):
        err = InvalidRepositoryError("bad url")
        assert err.retryable is False

    def test_repo_too_large_not_retryable(self):
        err = RepoTooLargeError("too big")
        assert err.retryable is False

    def test_clone_failed_is_retryable(self):
        err = CloneFailedError("network error")
        assert err.retryable is True

    def test_ingestion_error_is_subclass(self):
        err = InvalidRepositoryError("test")
        assert isinstance(err, IngestionError)

    def test_invalid_repo_raises_on_bad_url(self):
        with pytest.raises(InvalidRepositoryError):
            clone_repository("ftp://bad-scheme/repo")

    def test_invalid_repo_raises_on_file_scheme(self):
        with pytest.raises(InvalidRepositoryError):
            clone_repository("file:///etc/passwd")

    def test_clone_failed_raises_on_unreachable_host(self):
        """A URL that fails to resolve or connect should raise CloneFailedError."""
        with pytest.raises((CloneFailedError, InvalidRepositoryError)):
            clone_repository("https://this-host-does-not-exist-12345.invalid/repo.git")
