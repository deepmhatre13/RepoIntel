import os
import subprocess

import pytest

from packages.ingestion import cloning
from packages.ingestion.exceptions import CloneFailedError, InvalidRepositoryError


class _CompletedProcess:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.pid = 1234
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.killed = False

    def communicate(self, timeout=None):
        if timeout is not None:
            return self.stdout, self.stderr
        return self.stdout, self.stderr

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True


def test_successful_clone_is_shallow_and_cleans_up(monkeypatch, tmp_path):
    process = _CompletedProcess()
    command = {}

    def fake_popen(args, **kwargs):
        command["args"] = args
        command["kwargs"] = kwargs
        return process

    monkeypatch.setattr(cloning.subprocess, "Popen", fake_popen)
    target = cloning.clone_repository("https://github.com/example/repo", clone_timeout_seconds=7)

    assert command["args"][:5] == ["git", "clone", "--depth", "1", "--"]
    assert command["args"][5:] == ["https://github.com/example/repo", target]
    assert command["kwargs"]["start_new_session"] is (os.name != "nt")
    assert os.path.isdir(target)
    cloning.cleanup_repository(target)
    assert not os.path.exists(target)


def test_git_failure_is_typed_and_cleans_up(monkeypatch):
    process = _CompletedProcess(128, stderr=b"fatal: repository not found")
    monkeypatch.setattr(cloning.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(CloneFailedError, match="exit code 128"):
        cloning.clone_repository("https://github.com/example/missing")


def test_clone_timeout_terminates_process_and_cleans_up(monkeypatch):
    process = _CompletedProcess(stderr=b"remote: secret-token")

    def timeout_communicate(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired(["git", "clone"], timeout)
        return b"", process.stderr

    process.communicate = timeout_communicate
    terminated = []
    monkeypatch.setattr(cloning.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(cloning, "_terminate_process_tree", lambda proc: terminated.append(proc))

    with pytest.raises(CloneFailedError, match="timed out after 1 seconds") as exc_info:
        cloning.clone_repository("https://user:secret@example.com/repo", clone_timeout_seconds=1)

    assert terminated == [process]
    assert "user:secret@" not in str(exc_info.value)


def test_url_validation_still_runs_before_clone(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cloning.subprocess, "Popen", fail_if_called)
    with pytest.raises(InvalidRepositoryError):
        cloning.clone_repository("http://127.0.0.1/private")
    assert called is False