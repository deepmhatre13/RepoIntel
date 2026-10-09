"""Secure repository cloning with typed exceptions and SSRF guard.

Spec §5 mandates:
    * Typed exceptions: InvalidRepositoryError, CloneFailedError, RepoTooLargeError
    * SSRF guard on URL validation (delegated to packages.ingestion.security)
    * Configurable clone timeout
    * Shallow clone into an ephemeral temp workspace
    * Per-file failures are non-fatal; fatal clone failures raise
"""
import os
import signal
import shutil
import subprocess
import tempfile
import urllib.parse
import stat

from packages.ingestion.exceptions import InvalidRepositoryError, CloneFailedError, RepoTooLargeError
from packages.ingestion.security import validate_local_repo_path, validate_repo_url


def _redact_url(value: str, repo_url: str) -> str:
    """Remove credentials and the full repository URL from Git diagnostics."""
    parsed = urllib.parse.urlparse(repo_url)
    if parsed.scheme and parsed.hostname:
        safe_netloc = parsed.hostname
        if parsed.port:
            safe_netloc = f"{safe_netloc}:{parsed.port}"
        safe_url = urllib.parse.urlunparse((
            parsed.scheme, safe_netloc, parsed.path,
            parsed.params, parsed.query, parsed.fragment,
        ))
        value = value.replace(repo_url, safe_url)
        if parsed.username or parsed.password:
            value = value.replace(parsed.netloc, safe_netloc)
    return value.replace(repo_url, "<repository-url>")


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate Git and descendants created for one clone operation."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _remove_tree(path: str) -> None:
    """Remove a clone tree, including read-only Git files on Windows."""
    def handle_remove_error(function, failed_path, exc_info):
        try:
            os.chmod(failed_path, stat.S_IWRITE | stat.S_IREAD)
            function(failed_path)
        except OSError:
            pass

    shutil.rmtree(path, ignore_errors=False, onerror=handle_remove_error)


def _run_git_clone(repo_url: str, target_dir: str, timeout_seconds: float) -> None:
    command = ["git", "clone", "--depth", "1", "--", repo_url, target_dir]
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        stdout, stderr = process.communicate()
        details = _redact_url(
            ((stdout or b"") + (stderr or b"")).decode("utf-8", errors="replace").strip(),
            repo_url,
        )
        suffix = f" Output: {details[:1000]}" if details else ""
        raise CloneFailedError(
            f"Git clone timed out after {timeout_seconds:g} seconds.{suffix}"
        ) from exc

    output = _redact_url(
        ((stdout or b"") + (stderr or b"")).decode("utf-8", errors="replace").strip(),
        repo_url,
    )
    if process.returncode != 0:
        details = f" Output: {output[:1000]}" if output else ""
        raise CloneFailedError(
            f"Git clone failed with exit code {process.returncode}.{details}"
        )


def clone_repository(
    repo_url: str,
    max_repo_size_mb: float = 500.0,
    clone_timeout_seconds: int = 180,
    *,
    allowed_local_root: str | None = None,
) -> str:
    """Clone a repository with size cap and timeout.

    Returns the path to the temporary checkout.
    Raises InvalidRepositoryError, CloneFailedError, RepoTooLargeError.
    """
    if allowed_local_root is not None and os.path.isdir(repo_url):
        parsed = validate_local_repo_path(repo_url, allowed_local_root)
        repo_url = parsed.path
    else:
        parsed = validate_repo_url(repo_url)

    # Local directory — copy instead of clone.
    if parsed.scheme == "local":
        target_dir = tempfile.mkdtemp(prefix="repointel_clone_")
        try:
            shutil.copytree(repo_url, target_dir, dirs_exist_ok=True)
        except Exception as exc:
            _remove_tree(target_dir)
            raise CloneFailedError(f"Local copy failed for {repo_url!r}: {exc}")
        return target_dir

    target_dir = tempfile.mkdtemp(prefix="repointel_clone_")

    try:
        _run_git_clone(repo_url, target_dir, clone_timeout_seconds)
    except CloneFailedError:
        _remove_tree(target_dir)
        raise
    except Exception as exc:
        _remove_tree(target_dir)
        raise CloneFailedError(
            f"Git clone failed: {_redact_url(str(exc), repo_url)[:1000]}"
        ) from exc

    # Calculate size.
    total_bytes = 0
    try:
        for root, _, files in os.walk(target_dir):
            for f in files:
                fp = os.path.join(root, f)
                if not os.path.islink(fp):
                    total_bytes += os.path.getsize(fp)
    except Exception as exc:
        _remove_tree(target_dir)
        raise CloneFailedError(f"Size calculation failed for {repo_url!r}: {exc}")

    size_mb = total_bytes / (1024 * 1024)
    if size_mb > max_repo_size_mb:
        _remove_tree(target_dir)
        raise RepoTooLargeError(
            f"Repository size ({size_mb:.2f} MB) exceeds maximum allowed ({max_repo_size_mb} MB)"
        )

    return target_dir


def cleanup_repository(dir_path: str):
    """Remove a temporary checkout."""
    if os.path.exists(dir_path):
        _remove_tree(dir_path)
