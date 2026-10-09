import pytest

from packages.shared.langfuse_tracing import (
    create_langfuse_callback,
    flush_langfuse,
    langfuse_generation,
)


def test_langfuse_is_disabled_without_both_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert create_langfuse_callback() == (None, None)
    with langfuse_generation("test", "local-model", {"prompt": "safe"}) as generation:
        assert generation is None
    flush_langfuse(None)


def test_langfuse_requires_both_public_and_secret_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert create_langfuse_callback() == (None, None)