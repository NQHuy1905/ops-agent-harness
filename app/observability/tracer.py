"""Minimal span tracer persisted to the database, named after the OpenTelemetry GenAI conventions.

Span names / attributes follow https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans
(``invoke_agent {agent}``, ``chat {model}``, ``execute_tool {tool}``; ``gen_ai.*`` attributes,
``error.type`` on failure). Spans are stored in the ``spans`` table and exposed via
``GET /runs/{id}/trace`` so no external collector is needed. When ``OTEL_EXPORTER_OTLP_ENDPOINT``
is set and the optional ``otel`` extra is installed, spans are mirrored to an OTLP collector too.
"""

from __future__ import annotations

import contextvars
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog

from app.core.state import SpanView
from app.storage.models import SpanRow
from app.storage.repository import RunRepository

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("current_span", default=None)


class Span:
    def __init__(self, run_id: str, name: str, kind: str, attributes: dict[str, Any], parent_id: str | None):
        self.id = secrets.token_hex(8)
        self.run_id = run_id
        self.name = name
        self.kind = kind
        self.parent_id = parent_id
        self.attributes: dict[str, Any] = dict(attributes)
        self.status = "ok"
        self.status_message: str | None = None
        self.start_ns = time.time_ns()
        self.end_ns: int | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, values: dict[str, Any]) -> None:
        self.attributes.update(values)

    def set_error(self, error_type: str, message: str | None = None) -> None:
        self.status = "error"
        self.status_message = message
        self.attributes["error.type"] = error_type


class Tracer:
    def __init__(self, repo: RunRepository, otlp_endpoint: str | None = None) -> None:
        self._repo = repo
        self._otel = _OtelMirror.create(otlp_endpoint)

    @asynccontextmanager
    async def span(
        self, run_id: str, name: str, kind: str = "INTERNAL", attributes: dict[str, Any] | None = None
    ) -> AsyncIterator[Span]:
        parent = _current_span.get()
        span = Span(run_id, name, kind, attributes or {}, parent.id if parent else None)
        token = _current_span.set(span)
        structlog.contextvars.bind_contextvars(span_id=span.id)
        try:
            yield span
        except BaseException as exc:
            if span.status != "error":
                span.set_error(getattr(exc, "code", type(exc).__name__), str(exc)[:500])
            raise
        finally:
            span.end_ns = time.time_ns()
            _current_span.reset(token)
            if parent:
                structlog.contextvars.bind_contextvars(span_id=parent.id)
            else:
                structlog.contextvars.unbind_contextvars("span_id")
            await self._repo.add_span(
                SpanRow(
                    id=span.id,
                    run_id=span.run_id,
                    parent_id=span.parent_id,
                    name=span.name,
                    kind=span.kind,
                    status=span.status,
                    status_message=span.status_message,
                    attributes=span.attributes,
                    start_ns=span.start_ns,
                    end_ns=span.end_ns,
                )
            )
            if self._otel:
                self._otel.export(span)


def build_span_tree(rows: list[SpanRow]) -> list[SpanView]:
    """Nest persisted spans by parent_id. Returns root spans ordered by start time."""
    views = {
        r.id: SpanView(
            id=r.id,
            parent_id=r.parent_id,
            name=r.name,
            kind=r.kind,
            status=r.status,
            status_message=r.status_message,
            attributes=r.attributes or {},
            start_ns=r.start_ns,
            end_ns=r.end_ns,
            duration_ms=round((r.end_ns - r.start_ns) / 1e6, 3) if r.end_ns else None,
        )
        for r in sorted(rows, key=lambda r: r.start_ns)
    }
    roots: list[SpanView] = []
    for view in views.values():
        parent = views.get(view.parent_id) if view.parent_id else None
        (parent.children if parent else roots).append(view)
    return roots


class _OtelMirror:
    """Optional OTLP export. Silently disabled when the endpoint or the SDK is missing."""

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    @classmethod
    def create(cls, endpoint: str | None) -> _OtelMirror | None:
        if not endpoint:
            return None
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:  # pragma: no cover - optional extra
            structlog.get_logger("harness").warning("otel.disabled", reason="opentelemetry extra not installed")
            return None
        provider = TracerProvider(resource=Resource.create({"service.name": "ops-agent-harness"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        return cls(provider.get_tracer("ops-agent-harness"))

    def export(self, span: Span) -> None:  # pragma: no cover - needs a collector
        from opentelemetry.trace import Status, StatusCode

        attrs = {k: v if isinstance(v, (str, int, float, bool)) else str(v) for k, v in span.attributes.items()}
        attrs["harness.run_id"] = span.run_id
        otel_span = self._tracer.start_span(span.name, start_time=span.start_ns, attributes=attrs)
        if span.status == "error":
            otel_span.set_status(Status(StatusCode.ERROR, span.status_message))
        otel_span.end(end_time=span.end_ns)
