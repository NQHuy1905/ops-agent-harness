"""Malformed LLM output is corrected once and aborts after too many consecutive failures; provider errors fail runs."""

from __future__ import annotations

from app.core.state import RunStatus, StepType
from app.llm.scripted import DemoScenarioLLM, ScriptedLLM, empty, final, malformed, multi, provider_error, tool
from tests.helpers import CREATE_INCIDENT_ARGS, PLAIN_SERVICES, make_settings


def corrective_messages(messages):
    return [m["content"] for m in messages if m["role"] == "user" and "previous response was invalid" in m["content"]]


async def test_one_malformed_then_valid_completes_with_corrective_feedback(open_container):
    llm = ScriptedLLM([malformed(), final("Recovered after correction.")])
    c = await open_container(llm)
    view = await c.runner.start("Check payments-api")

    assert view.status == RunStatus.COMPLETED and view.final_answer == "Recovered after correction."
    feedback = corrective_messages(llm.received[1])
    assert len(feedback) == 1 and "invalid JSON args for get_service_status" in feedback[0]
    # the broken assistant turn is not replayed into the conversation
    assert all("tool_calls" not in m for m in llm.received[1] if m["role"] == "assistant")
    malformed_step = [s for s in await c.repo.list_steps(view.id) if s.type == StepType.LLM_MALFORMED.value]
    assert malformed_step[0].payload["consecutive"] == 1
    assert view.step_count == 2  # the malformed attempt counts as a step


async def test_empty_response_is_malformed(open_container):
    llm = ScriptedLLM([empty(), final("Now answering.")])
    c = await open_container(llm)
    view = await c.runner.start("Say something")

    assert view.status == RunStatus.COMPLETED
    assert "empty response" in corrective_messages(llm.received[1])[0]


async def test_exactly_max_consecutive_malformed_is_still_allowed(open_container):
    llm = ScriptedLLM([malformed(), empty(), final("Finally.")])  # default max_malformed_llm_responses=2
    c = await open_container(llm)
    view = await c.runner.start("Check something")

    assert view.status == RunStatus.COMPLETED and view.final_answer == "Finally."


async def test_one_more_than_max_consecutive_malformed_fails_run(open_container):
    llm = ScriptedLLM([malformed(), malformed(), malformed(), final("never reached")])
    c = await open_container(llm)
    view = await c.runner.start("Check something")

    assert view.status == RunStatus.FAILED
    assert view.error_code == "malformed_llm_response"
    assert "3 consecutive malformed LLM responses" in view.error_detail
    assert len(llm.received) == 3
    assert (await c.repo.list_steps(view.id))[-1].type == StepType.RUN_ERROR.value


async def test_malformed_counter_resets_after_a_valid_decision(open_container):
    llm = ScriptedLLM(
        [
            malformed(),
            tool("get_service_status", service_name="auth-service"),
            malformed(),
            empty(),
            final("Still alive."),
        ]
    )
    c = await open_container(llm)
    view = await c.runner.start("Check auth-service")

    assert view.status == RunStatus.COMPLETED


async def test_zero_tolerance_fails_on_first_malformed(open_container, tmp_path):
    llm = ScriptedLLM([malformed(), final("never reached")])
    c = await open_container(llm, cfg=make_settings(tmp_path, max_malformed_llm_responses=0))
    view = await c.runner.start("Check something")

    assert view.status == RunStatus.FAILED and view.error_code == "malformed_llm_response"
    assert len(llm.received) == 1


async def test_too_many_tool_calls_in_one_response_is_malformed(open_container):
    too_many = multi(*[tool("get_service_status", service_name=s) for s in PLAIN_SERVICES[:6]])
    llm = ScriptedLLM([too_many, final("Trimmed my plan.")])  # ScriptedLLM enforces max_tool_calls=5
    c = await open_container(llm)
    view = await c.runner.start("Check six services")

    assert view.status == RunStatus.COMPLETED
    assert "too many tool calls: 6 > 5" in corrective_messages(llm.received[1])[0]
    assert await c.repo.list_tool_calls(view.id) == []  # none of the six calls ran


async def test_provider_error_fails_run_with_llm_code(open_container):
    llm = ScriptedLLM([provider_error("upstream 503")])
    c = await open_container(llm)
    view = await c.runner.start("Check payments-api")

    assert view.status == RunStatus.FAILED
    assert view.error_code == "llm_provider_error"
    assert view.error_detail == "upstream 503"
    run_error = (await c.repo.list_steps(view.id))[-1]
    assert run_error.type == StepType.RUN_ERROR.value
    assert run_error.payload == {"status": "failed", "code": "llm_provider_error", "detail": "upstream 503"}


async def test_provider_error_mid_run_keeps_earlier_tool_results(open_container):
    llm = ScriptedLLM([tool("get_service_status", service_name="orders-db"), provider_error("connection reset")])
    c = await open_container(llm)
    view = await c.runner.start("Check orders-db")

    assert view.status == RunStatus.FAILED and view.error_code == "llm_provider_error"
    tool_rows = await c.repo.list_tool_calls(view.id)
    assert [r.status for r in tool_rows] == ["succeeded"]


async def test_offline_demo_provider_with_missing_scenario_fails_run(open_container, tmp_path):
    cfg = make_settings(tmp_path, mock_data_dir=tmp_path)  # no scenarios/ directory under tmp_path
    c = await open_container(DemoScenarioLLM(cfg), cfg=cfg)
    view = await c.runner.start("Check payments-api")

    assert view.status == RunStatus.FAILED and view.error_code == "llm_provider_error"
    assert "cannot load demo scenario status_check" in view.error_detail


async def test_demo_provider_completes_offline_status_flow(open_container, settings):
    llm = DemoScenarioLLM(settings)
    c = await open_container(llm)
    view = await c.runner.start("Check the status of auth-service")

    assert view.status == RunStatus.COMPLETED
    assert "auth-service" in view.final_answer
    calls = await c.repo.list_tool_calls(view.id)
    assert [(t.tool_name, t.args) for t in calls] == [("get_service_status", {"service_name": "auth-service"})]


class ExplodingLLM(ScriptedLLM):
    """Raises a non-harness exception on the N-th decision (simulates a client bug or DB failure)."""

    def __init__(self, script, explode_on: int) -> None:
        super().__init__(script)
        self._explode_on = explode_on
        self._calls = 0

    async def decide(self, messages, tools):
        self._calls += 1
        if self._calls == self._explode_on:
            raise RuntimeError("boom")
        return await super().decide(messages, tools)


async def test_unexpected_exception_in_llm_client_fails_run_as_internal_error(open_container):
    c = await open_container(ExplodingLLM([final("never")], explode_on=1))
    view = await c.runner.start("Check payments-api")

    assert view.status == RunStatus.FAILED
    assert view.error_code == "internal_error"
    assert view.error_detail == "RuntimeError: boom"
    assert (await c.repo.list_steps(view.id))[-1].type == StepType.RUN_ERROR.value


async def test_unexpected_exception_during_resume_fails_run_not_stuck(open_container):
    from app.core.state import ApprovalDecision

    llm = ExplodingLLM([tool("create_incident", call_id="c-inc", **CREATE_INCIDENT_ARGS)], explode_on=2)
    c = await open_container(llm)
    view = await c.runner.start("Open an incident for payments-api")
    assert view.status == RunStatus.AWAITING_APPROVAL

    done = await c.runner.resume(view.id, ApprovalDecision(tool_call_id="c-inc", decision="approve"))

    assert done.status == RunStatus.FAILED and done.error_code == "internal_error"
    assert len(await c.repo.list_incidents(view.id)) == 1  # the approved action did run once
