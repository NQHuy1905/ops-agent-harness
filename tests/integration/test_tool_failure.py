"""Tool failures become observations the LLM can act on; transient errors are retried."""

from __future__ import annotations

from app.core.state import RunStatus, StepType
from app.llm.scripted import ScriptedLLM, final, tool
from app.tools.base import ToolContext
from tests.helpers import tool_payloads


def _error_of(observation: dict) -> str:
    return observation["error"]["code"]


async def test_flaky_service_recovers_on_third_attempt(open_container):
    llm = ScriptedLLM(
        [tool("get_service_status", call_id="c-flaky", service_name="flaky-service"), final("Recovered.")]
    )
    c = await open_container(llm)
    view = await c.runner.start("Check flaky-service")

    assert view.status == RunStatus.COMPLETED
    row = (await c.repo.list_tool_calls(view.id))[0]
    assert row.status == "succeeded" and row.attempts == 3 and row.error_type is None
    obs = tool_payloads(llm.received[1])[0][1]
    assert obs["ok"] is True and obs["data"]["name"] == "flaky-service"
    tool_step = [s for s in await c.repo.list_steps(view.id) if s.type == StepType.TOOL_RESULT.value][0]
    assert tool_step.payload["attempts"] == 3


async def test_transient_error_surfaces_when_retries_exhausted(open_container, settings):
    cfg = settings.model_copy(update={"tool_max_retries": 0})
    llm = ScriptedLLM(
        [tool("get_service_status", call_id="c-f", service_name="flaky-service"), final("Gave up on it.")]
    )
    c = await open_container(llm, cfg=cfg)
    view = await c.runner.start("Check flaky-service")

    assert view.status == RunStatus.COMPLETED
    obs = tool_payloads(llm.received[1])[0][1]
    assert obs["ok"] is False and _error_of(obs) == "tool_transient_error"
    row = (await c.repo.list_tool_calls(view.id))[0]
    assert row.status == "failed" and row.attempts == 1 and row.error_type == "tool_transient_error"


async def test_slow_service_times_out_after_all_attempts_and_run_continues(open_container):
    llm = ScriptedLLM(
        [tool("get_service_status", call_id="c-slow", service_name="slow-service"), final("Timed out, noted.")]
    )
    c = await open_container(llm)
    view = await c.runner.start("Check slow-service")

    assert view.status == RunStatus.COMPLETED and view.final_answer == "Timed out, noted."
    obs = tool_payloads(llm.received[1])[0][1]
    assert obs["ok"] is False and _error_of(obs) == "tool_timeout"
    assert "timed out after 0.05s" in obs["error"]["message"]
    row = (await c.repo.list_tool_calls(view.id))[0]
    assert row.attempts == 3 and row.error_type == "tool_timeout"


async def test_broken_service_output_is_rejected_by_schema(open_container):
    llm = ScriptedLLM(
        [tool("get_service_status", call_id="c-b", service_name="broken-service"), final("Output was invalid.")]
    )
    c = await open_container(llm)
    view = await c.runner.start("Check broken-service")

    assert view.status == RunStatus.COMPLETED
    obs = tool_payloads(llm.received[1])[0][1]
    assert obs["ok"] is False and _error_of(obs) == "tool_output_invalid"
    row = (await c.repo.list_tool_calls(view.id))[0]
    assert row.status == "failed" and row.attempts == 1  # validation errors are not retried


async def test_unknown_tool_is_reported_with_available_names(open_container):
    llm = ScriptedLLM([tool("restart_database", call_id="c-x", target="orders-db"), final("No such tool.")])
    c = await open_container(llm)
    view = await c.runner.start("Restart orders-db")

    assert view.status == RunStatus.COMPLETED
    obs = tool_payloads(llm.received[1])[0][1]
    assert _error_of(obs) == "tool_not_found"
    assert "restart_database" in obs["error"]["message"] and "get_service_status" in obs["error"]["message"]


async def test_unknown_service_is_permanent_and_not_retried(open_container):
    llm = ScriptedLLM([tool("get_service_status", call_id="c-u", service_name="ghost-api"), final("Not found.")])
    c = await open_container(llm)
    view = await c.runner.start("Check ghost-api")

    obs = tool_payloads(llm.received[1])[0][1]
    assert _error_of(obs) == "tool_execution_error"
    assert (await c.repo.list_tool_calls(view.id))[0].attempts == 1


async def test_invalid_arguments_are_fed_back_then_corrected(open_container):
    llm = ScriptedLLM(
        [
            tool("get_service_status", call_id="c-bad", service_name="Payments API"),
            tool("get_service_status", call_id="c-good", service_name="payments-api"),
            final("Corrected and checked."),
        ]
    )
    c = await open_container(llm)
    view = await c.runner.start("Check payments")

    assert view.status == RunStatus.COMPLETED
    first_obs = tool_payloads(llm.received[1])[0][1]
    assert _error_of(first_obs) == "tool_input_invalid"
    assert "service_name" in first_obs["error"]["message"]
    second_obs = tool_payloads(llm.received[2])[-1][1]
    assert second_obs["ok"] is True and second_obs["data"]["name"] == "payments-api"
    rows = {t.call_id: t for t in await c.repo.list_tool_calls(view.id)}
    assert rows["c-bad"].status == "failed" and rows["c-bad"].error_type == "tool_input_invalid"
    assert rows["c-good"].status == "succeeded"


async def test_buggy_tool_exception_does_not_crash_the_run(open_container):
    llm = ScriptedLLM([tool("search_knowledge_base", call_id="c-kb", query="latency"), final("Recovered.")])
    c = await open_container(llm)

    async def boom(args, ctx: ToolContext):  # simulates an unexpected tool bug
        raise RuntimeError("database connection lost")

    c.registry.get("search_knowledge_base").run = boom
    view = await c.runner.start("Find latency runbook")

    assert view.status == RunStatus.COMPLETED
    obs = tool_payloads(llm.received[1])[0][1]
    assert _error_of(obs) == "tool_execution_error"
    assert obs["error"]["message"] == "unexpected RuntimeError"
    assert "database connection lost" not in str(obs)  # internals are not leaked to the LLM
