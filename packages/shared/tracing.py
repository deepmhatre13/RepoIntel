"""Lightweight OpenTelemetry instrumentation for RepoIntel (Stage 8).

Works locally WITHOUT a hosted observability account: traces are exported to
stderr via ``ConsoleSpanExporter`` when ``opentelemetry`` is installed and an
environment switch is on, and become a disciplined no-op otherwise. The same
module exposes ``get_tracer`` / ``span`` helpers used by every stage (ingestion,
parsing, SCIP enrichment, graph construction, retrieval, planner, tool
execution, Gemini, synthesis, reviewer, reflection, artifact generation,
security analysis).

Security
--------
Attribute values are caller-supplied and MUST already be secret-redacted before
this module is invoked. This module never receives raw secrets; it only passes
through attribute dicts. It also never logs sensitive repository content beyond
the (already curated) attribute set.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger("RepoIntel.Tracing")

TRACE_ENABLED = os.environ.get("REPOINTEL_TRACING", "0") == "1"

# Module-level singleton so we only configure the SDK once.
_configured = False
_tracer = None


def is_tracing_enabled() -> bool:
    return TRACE_ENABLED


def configure_tracing(service_name: str = "repointel") -> bool:
    """Initialise the OpenTelemetry SDK once. Returns True when active.

    No-op (returns False) when ``REPOINTEL_TRACING`` is not ``1`` or when the
    ``opentelemetry`` packages are not installed. Never raises when the SDK is
    absent -- RepoIntel remains fully functional without OTel.
    """
    global _configured, _tracer
    if _configured:
        return _tracer is not None
    _configured = True

    if not TRACE_ENABLED:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    except Exception as exc:  # pragma: no cover - only when OTel is missing
        logger.info("OpenTelemetry SDK unavailable; tracing stays a no-op: %s", exc)
        return False

    try:
        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("OpenTelemetry initialisation failed; tracing no-op: %s", exc)
        return False


def get_tracer(scope: str = "repointel"):
    """Return a Tracer (or None when tracing is disabled/absent)."""
    if _tracer is None and TRACE_ENABLED:
        configure_tracing()
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry.trace import get_tracer as _get_tracer
        return _get_tracer(scope)
    except Exception:  # pragma: no cover
        return None
@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None, **kwargs):
    """Context manager creating a span with given attributes.

    Accepts either a dict via ``attributes``, or keyword attributes directly::

        with span("ingestion", {"repository_id": repo_id, "run_id": run_id}):
            ...
        with span("ingestion", repository_id=repo_id, run_id=run_id):
            ...
    """
    if kwargs:
        merged = dict(attributes or {})
        merged.update(kwargs)
        attributes = merged
    tracer = get_tracer()
    if tracer is None or not TRACE_ENABLED:
        yield None
        return

    try:
        from opentelemetry import trace as _trace
    except Exception:  # pragma: no cover
        yield None
        return

    ctx = _trace.set_span_in_context(_trace.get_current_span())
    with tracer.start_as_current_span(name, context=ctx) as sp:
        if attributes:
            for k, v in attributes.items():
                try:
                    sp.set_attribute(k, _clean(v))
                except Exception:  # pragma: no cover
                    pass
        yield sp


def set_span_attribute(key: str, value: Any) -> None:
    """Best-effort set an attribute on the current span (no-op if absent)."""
    tracer = get_tracer()
    if tracer is None:
        return
    try:
        from opentelemetry import trace as _trace
        span_obj = _trace.get_current_span()
        span_obj.set_attribute(key, _clean(value))
    except Exception:  # pragma: no cover
        pass


def _clean(value: Any) -> Any:
    """Coerce values to OTel-safe primitives (never leak raw objects)."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return str(value)
    except Exception:  # pragma: no cover
        return ""