from contextlib import contextmanager
import os
from typing import Any, Dict, Optional, Tuple


def get_langfuse_client():
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception:
        return None


def create_langfuse_callback() -> Tuple[Optional[Any], Optional[Any]]:
    client = get_langfuse_client()
    if client is None:
        return None, None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler(), client
    except Exception:
        return None, None


@contextmanager
def langfuse_generation(
    name: str,
    model: str,
    input_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
):
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    try:
        observation = client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_data,
            metadata=metadata or {},
        )
    except Exception:
        yield None
        return
    with observation as generation:
        yield generation


def flush_langfuse(client) -> None:
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        return