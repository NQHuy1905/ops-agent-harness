"""Human approval gate for create_incident: pause, approve / reject / edit, restart, concurrency, idempotency."""

from __future__ import annotations

import asyncio

import pytest

from app.core.errors import ApprovalArgsInvalidError, ApprovalMismatchError, ApprovalNotPendingError, RunBusyError
from app.core.state import ApprovalDecision, RunStatus, RunView, StepType
from app.llm.scripted import ScriptedLLM, final, multi, tool
from app.tools.base import ToolContext
from tests.helpers import CREATE_INCIDENT_ARGS, make_settings, tool_payloads


def incident_script(call_id: str = "c-inc") -> list:
    return [tool("create_incident", call_id=call_id, **CREATE_INCIDENT_ARGS), final("Incident opened.")]


def approve(call_id: str = "c-inc", **kw) -> ApprovalDecision:
    return ApprovalDecision(tool_call_id=call_id, decision="approve", **kw)


def reject(call_id: str = "c-inc", reason: str | None = None) -> ApprovalDecision:
    return ApprovalDecision(tool_call_id=call_id, decision="reject", reason=reason)


async def paused_run(open_container, script=None):
    llm = ScriptedLLM(script or incident_script())
    c = await open_container(llm)
    view = await c.runner.start("Open an incident for payments-api")
    assert view.status == RunStatus.AWAITING_APPROVAL
    return c, llm, view


# ------------------------------------------------------------------ pause
async def test_incident_request_pauses_before_any_side_effect(open_container):
    c, llm, view = await paused_run(open_container)

    assert view.pending_approval is not None
    assert view.pending_approval.tool_call_id == "c-inc"
    assert view.pending_approval.tool == "create_incident"
    assert view.pending_approval.args == CREATE_INCIDENT_ARGS
    assert await c.repo.list_incidents() == []
    assert len(llm.received) == 1  # the LLM is not asked again while paused
    assert (await c.repo.list_steps(view.id))[-1].type == StepType.APPROVAL_REQUESTED.value


# ------------------------------------------------------------------ approve / reject
async def test_approve_creates_exactly_one_incident(open_container):
    c, _, view = await paused_run(open_container)
    done = await c.runner.resume(view.id, approve())

    assert done.status == RunStatus.COMPLETED and done.final_answer == "Incident opened."
    assert done.pending_approval is None
    incidents = await c.repo.list_incidents(view.id)
    assert len(incidents) == 1
    assert incidents[0].severity == "high" and incidents[0].title == CREATE_INCIDENT_ARGS["title"]
    row = (await c.repo.list_tool_calls(view.id))[0]
    assert row.status == "succeeded"
    approvals = await c.repo.list_approvals(view.id)
    assert [(a.call_id, a.decision, a.edited_args) for a in approvals] == [("c-inc", "approve", None)]


async def test_reject_creates_no_incident_and_tells_the_llm(open_container):
    c, llm, view = await paused_run(open_container, incident_script())
    done = await c.runner.resume(view.id, reject(reason="Not confirmed by on-call"))

    assert done.status == RunStatus.COMPLETED and done.final_answer == "Incident opened."
    assert await c.repo.list_incidents() == []
    row = (await c.repo.list_tool_calls(view.id))[0]
    assert row.status == "rejected" and row.error_type == "rejected_by_user"
    obs = tool_payloads(llm.received[1])[0][1]
    assert obs["ok"] is False and obs["error"]["code"] == "rejected_by_user"
    assert obs["error"]["message"] == "Not confirmed by on-call"


async def test_reject_without_reason_uses_default_message(open_container):
    c, llm, view = await paused_run(open_container)
    await c.runner.resume(view.id, reject())
    obs = tool_payloads(llm.received[1])[0][1]
    assert "Do not retry" in obs["error"]["message"]


# ------------------------------------------------------------------ edited args
async def test_valid_edited_args_are_the_ones_executed(open_container):
    c, _, view = await paused_run(open_container)
    edited = {**CREATE_INCIDENT_ARGS, "severity": "critical", "title": "Payments outage (edited)"}
    done = await c.runner.resume(view.id, approve(edited_args=edited))

    assert done.status == RunStatus.COMPLETED
    (incident,) = await c.repo.list_incidents(view.id)
    assert incident.severity == "critical" and incident.title == "Payments outage (edited)"
    row = (await c.repo.list_tool_calls(view.id))[0]
    assert row.args["severity"] == "critical"
    (approval,) = await c.repo.list_approvals(view.id)
    assert approval.edited_args == edited


@pytest.mark.parametrize(
    "bad_edit",
    [
        {**CREATE_INCIDENT_ARGS, "severity": "urgent"},  # not in the enum
        {**CREATE_INCIDENT_ARGS, "priority": "p1"},  # unknown key (extra=forbid)
        {"title": "x"},  # missing required fields
    ],
    ids=["bad-enum", "unknown-key", "missing-fields"],
)
async def test_invalid_edited_args_keep_run_awaiting_and_can_be_retried(open_container, bad_edit):
    c, _, view = await paused_run(open_container)

    with pytest.raises(ApprovalArgsInvalidError) as ei:
        await c.runner.resume(view.id, approve(edited_args=bad_edit))
    assert ei.value.http_status == 422 and ei.value.code == "approval_args_invalid"

    still = await c.runner.get_view(view.id)
    assert still.status == RunStatus.AWAITING_APPROVAL and still.pending_approval is not None
    assert await c.repo.list_incidents() == []
    assert await c.repo.list_approvals(view.id) == []

    done = await c.runner.resume(view.id, approve())
    assert done.status == RunStatus.COMPLETED
    assert len(await c.repo.list_incidents(view.id)) == 1


# ------------------------------------------------------------------ guards on the decision
async def test_wrong_tool_call_id_is_rejected_without_state_change(open_container):
    c, _, view = await paused_run(open_container)
    with pytest.raises(ApprovalMismatchError) as ei:
        await c.runner.resume(view.id, approve(call_id="someone-else"))
    assert ei.value.http_status == 409 and ei.value.code == "approval_mismatch"
    assert (await c.runner.get_view(view.id)).status == RunStatus.AWAITING_APPROVAL
    assert await c.repo.list_approvals(view.id) == []


async def test_second_decision_is_rejected_and_creates_no_extra_incident(open_container):
    c, _, view = await paused_run(open_container)
    await c.runner.resume(view.id, approve())

    with pytest.raises(ApprovalNotPendingError) as ei:
        await c.runner.resume(view.id, approve())
    assert ei.value.http_status == 409 and ei.value.code == "approval_not_pending"
    with pytest.raises(ApprovalNotPendingError):
        await c.runner.resume(view.id, reject())
    assert len(await c.repo.list_incidents(view.id)) == 1


async def test_concurrent_decisions_apply_exactly_one(open_container):
    c, _, view = await paused_run(open_container)
    results = await asyncio.gather(
        c.runner.resume(view.id, approve()), c.runner.resume(view.id, reject()), return_exceptions=True
    )

    successes = [r for r in results if isinstance(r, RunView)]
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 1
    assert all(isinstance(f, (RunBusyError, ApprovalNotPendingError)) for f in failures)
    assert len(await c.repo.list_approvals(view.id)) == 1


# ------------------------------------------------------------------ restart
async def test_decision_applies_from_a_fresh_container_on_same_db(open_container):
    llm = ScriptedLLM(incident_script())
    first = await open_container(llm)
    view = await first.runner.start("Open an incident for payments-api")
    await first.close()  # simulate the process going away while awaiting approval

    second = await open_container(llm)  # same settings fixture -> same SQLite file, new runner/engine
    assert (await second.runner.get_view(view.id)).status == RunStatus.AWAITING_APPROVAL
    done = await second.runner.resume(view.id, approve())

    assert done.status == RunStatus.COMPLETED
    assert len(await second.repo.list_incidents(view.id)) == 1
    assert len(llm.received) == 2  # the LLM resumed exactly where it left off


# ------------------------------------------------------------------ wall time
async def test_approval_wait_is_not_counted_against_wall_time(open_container, tmp_path, fake_clock):
    cfg = make_settings(tmp_path, max_wall_time_s=5)
    c = await open_container(ScriptedLLM(incident_script()), cfg=cfg)
    view = await c.runner.start("Open an incident for payments-api")

    fake_clock.advance(1000)  # human takes a long time to decide
    done = await c.runner.resume(view.id, approve())

    assert done.status == RunStatus.COMPLETED
    run = await c.repo.get_run(view.id)
    assert run.active_elapsed_s == 0.0  # nothing executed on the fake clock


# ------------------------------------------------------------------ multi-call queue
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_calls_queued_after_approval_request_run_only_after_decision(open_container, decision):
    script = [
        multi(
            tool("get_service_status", call_id="c-status", service_name="checkout-web"),
            tool("create_incident", call_id="c-inc", **CREATE_INCIDENT_ARGS),
            tool("search_knowledge_base", call_id="c-kb", query="severity matrix"),
        ),
        final("Done with all three."),
    ]
    llm = ScriptedLLM(script)
    c = await open_container(llm)
    view = await c.runner.start("Check checkout-web and open an incident")

    assert view.status == RunStatus.AWAITING_APPROVAL
    statuses = {t.call_id: t.status for t in await c.repo.list_tool_calls(view.id)}
    assert statuses == {"c-status": "succeeded", "c-inc": "pending_approval", "c-kb": "queued"}
    assert len(llm.received) == 1

    decision_obj = approve() if decision == "approve" else reject()
    done = await c.runner.resume(view.id, decision_obj)

    assert done.status == RunStatus.COMPLETED
    statuses = {t.call_id: t.status for t in await c.repo.list_tool_calls(view.id)}
    assert statuses["c-kb"] == "succeeded"
    assert statuses["c-inc"] == ("succeeded" if decision == "approve" else "rejected")
    assert len(await c.repo.list_incidents()) == (1 if decision == "approve" else 0)
    # the next LLM request sees one tool message per call, in the original order
    assert [cid for cid, _ in tool_payloads(llm.received[1])] == ["c-status", "c-inc", "c-kb"]


# ------------------------------------------------------------------ idempotency
async def test_rerunning_create_incident_with_same_run_and_call_is_deduplicated(open_container):
    c = await open_container(ScriptedLLM([final("unused")]))
    tool_obj = c.registry.get("create_incident")
    args = tool_obj.validate_input(CREATE_INCIDENT_ARGS)
    ctx = ToolContext(run_id="run-x", tool_call_id="call-x", repo=c.repo, settings=c.settings)

    first = await tool_obj.run(args, ctx)
    second = await tool_obj.run(args, ctx)

    assert first["deduplicated"] is False and second["deduplicated"] is True
    assert first["incident_id"] == second["incident_id"]
    assert len(await c.repo.list_incidents("run-x")) == 1


# ------------------------------------------------------------------ approval provenance in observations
async def test_approved_observation_states_operator_approval(open_container):
    c, llm, view = await paused_run(open_container)
    await c.runner.resume(view.id, approve())

    obs = tool_payloads(llm.received[1])[0][1]
    assert obs["ok"] is True and obs["approval"] == "approved by operator"
    assert "executed_args" not in obs


async def test_edited_approval_observation_reports_executed_args(open_container):
    c, llm, view = await paused_run(open_container)
    edited = {**CREATE_INCIDENT_ARGS, "severity": "low"}
    await c.runner.resume(view.id, approve(edited_args=edited))

    obs = tool_payloads(llm.received[1])[0][1]
    assert obs["approval"] == "approved with operator edits"
    assert obs["executed_args"] == edited
    assert obs["data"]["severity"] == "low"


async def test_reject_does_not_store_edited_args_even_if_sent(open_container):
    c, _, view = await paused_run(open_container)
    await c.runner.resume(
        view.id,
        ApprovalDecision(
            tool_call_id="c-inc", decision="reject", edited_args={**CREATE_INCIDENT_ARGS, "severity": "low"}
        ),
    )
    (approval,) = await c.repo.list_approvals(view.id)
    assert approval.decision == "reject" and approval.edited_args is None
