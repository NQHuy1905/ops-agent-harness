"""Persisted span tree: OTel GenAI naming, parent/child shape, attributes and error spans."""

from __future__ import annotations

from app.core.state import ApprovalDecision, RunStatus, SpanView
from app.llm.scripted import ScriptedLLM, final, malformed, tool
from app.observability.tracer import build_span_tree
from tests.helpers import CREATE_INCIDENT_ARGS


def flatten(spans: list[SpanView]) -> list[SpanView]:
    out: list[SpanView] = []
    for span in spans:
        out.append(span)
        out.extend(flatten(span.children))
    return out


def by_name(spans: list[SpanView], name: str) -> list[SpanView]:
    return [s for s in flatten(spans) if s.name == name]


async def trace_of(c, run_id):
    return build_span_tree(await c.repo.list_spans(run_id))


async def test_success_span_tree_shape_and_attributes(open_container):
    c = await open_container(
        ScriptedLLM([tool("get_service_status", call_id="t1", service_name="auth-service"), final("Healthy.")])
    )
    view = await c.runner.start("Check auth-service")
    roots = await trace_of(c, view.id)

    assert [r.name for r in roots] == ["invoke_agent ops-assistant"]
    root = roots[0]
    assert root.kind == "INTERNAL" and root.status == "ok" and root.parent_id is None
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.agent.name"] == "ops-assistant"
    assert root.attributes["harness.segment"] == "start"
    assert [ch.name for ch in root.children] == ["chat scripted", "execute_tool get_service_status", "chat scripted"]

    chat = root.children[0]
    assert chat.kind == "CLIENT"
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.provider.name"] == "scripted"
    assert chat.attributes["harness.decision"] == "tool_calls"
    assert chat.attributes["harness.step"] == 1

    execute = root.children[1]
    assert execute.attributes["gen_ai.operation.name"] == "execute_tool"
    assert execute.attributes["gen_ai.tool.name"] == "get_service_status"
    assert execute.attributes["gen_ai.tool.call.id"] == "t1"
    assert execute.attributes["gen_ai.tool.type"] == "function"
    assert execute.attributes["harness.ok"] is True and execute.attributes["harness.attempts"] == 1
    assert "error.type" not in execute.attributes
    assert all(s.end_ns is not None and s.duration_ms is not None for s in flatten(roots))


async def test_retried_tool_span_records_attempts_and_ok_status(open_container):
    c = await open_container(
        ScriptedLLM([tool("get_service_status", service_name="flaky-service"), final("Recovered.")])
    )
    view = await c.runner.start("Check flaky-service")
    (span,) = by_name(await trace_of(c, view.id), "execute_tool get_service_status")

    assert span.status == "ok" and span.attributes["harness.attempts"] == 3


async def test_failed_tool_span_has_error_status_and_error_type(open_container):
    c = await open_container(
        ScriptedLLM([tool("get_service_status", service_name="broken-service"), final("Invalid output seen.")])
    )
    view = await c.runner.start("Check broken-service")
    (span,) = by_name(await trace_of(c, view.id), "execute_tool get_service_status")

    assert span.status == "error"
    assert span.attributes["error.type"] == "tool_output_invalid"
    assert span.attributes["harness.ok"] is False
    assert span.status_message


async def test_timeout_span_reports_retries_and_timeout_error(open_container):
    c = await open_container(ScriptedLLM([tool("get_service_status", service_name="slow-service"), final("x")]))
    view = await c.runner.start("Check slow-service")
    (span,) = by_name(await trace_of(c, view.id), "execute_tool get_service_status")

    assert span.status == "error" and span.attributes["error.type"] == "tool_timeout"
    assert span.attributes["harness.attempts"] == 3


async def test_malformed_chat_span_is_marked_as_error(open_container):
    c = await open_container(ScriptedLLM([malformed(), final("Fine now.")]))
    view = await c.runner.start("Check something")
    chats = by_name(await trace_of(c, view.id), "chat scripted")

    assert [s.status for s in chats] == ["error", "ok"]
    assert chats[0].attributes["error.type"] == "malformed_llm_response"


async def test_approval_run_has_one_root_per_segment(open_container):
    c = await open_container(
        ScriptedLLM([tool("create_incident", call_id="a1", **CREATE_INCIDENT_ARGS), final("Opened.")])
    )
    view = await c.runner.start("Open an incident for payments-api")
    await c.runner.resume(view.id, ApprovalDecision(tool_call_id="a1", decision="approve"))
    roots = await trace_of(c, view.id)

    assert [r.attributes["harness.segment"] for r in roots] == ["start", "resume"]
    start_root, resume_root = roots
    assert [s.name for s in start_root.children] == ["chat scripted"]
    assert [s.name for s in resume_root.children] == [
        "execute_tool create_incident",
        "chat scripted",
    ]
    assert resume_root.attributes["harness.approval.decision"] == "approve"
    assert resume_root.attributes["gen_ai.tool.name"] == "create_incident"


async def test_trace_contains_only_spans_of_the_requested_run(open_container):
    c = await open_container(ScriptedLLM([final("one"), final("two")]))
    first = await c.runner.start("first")
    second = await c.runner.start("second")

    assert len(await trace_of(c, first.id)) == 1
    assert len(await trace_of(c, second.id)) == 1
    assert {s.run_id for s in await c.repo.list_spans(first.id)} == {first.id}


async def test_failed_run_marks_root_span_as_error(open_container):
    from app.llm.scripted import provider_error

    c = await open_container(ScriptedLLM([provider_error("upstream 503")]))
    view = await c.runner.start("Check payments-api")
    (root,) = await trace_of(c, view.id)

    assert root.status == "error"
    assert root.attributes["error.type"] == "llm_provider_error"
    assert root.attributes["harness.run.status"] == "failed"


async def test_limit_run_marks_root_span_as_error(open_container):
    c = await open_container(ScriptedLLM([tool("get_service_status", service_name="auth-service")] * 3 + [final("x")]))
    view = await c.runner.start("Check auth-service", max_steps=1)
    (root,) = await trace_of(c, view.id)

    assert view.status == RunStatus.LIMIT_EXCEEDED
    assert root.status == "error" and root.attributes["error.type"] == "step_limit_exceeded"
    assert root.attributes["harness.run.status"] == "limit_exceeded"


async def test_completed_run_root_span_is_ok(open_container):
    c = await open_container(ScriptedLLM([final("fine")]))
    view = await c.runner.start("Say fine")
    (root,) = await trace_of(c, view.id)

    assert root.status == "ok" and root.attributes["harness.run.status"] == "completed"
