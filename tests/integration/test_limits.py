"""Execution limits end runs with LIMIT_EXCEEDED and a machine-readable code."""

from __future__ import annotations

import time

import pytest

from app.core.state import RunStatus
from app.llm.scripted import ScriptedLLM, final, tool
from tests.helpers import PLAIN_SERVICES, ClockedLLM, make_settings


def status_calls(names) -> list:
    return [tool("get_service_status", service_name=name) for name in names]


async def test_step_limit_ends_run_before_the_next_llm_call(open_container):
    # 8 distinct calls (no final answer) with the default max_steps=8
    llm = ScriptedLLM(status_calls(PLAIN_SERVICES) + [final("never reached")])
    c = await open_container(llm)
    view = await c.runner.start("Check everything")

    assert view.status == RunStatus.LIMIT_EXCEEDED
    assert view.error_code == "step_limit_exceeded"
    assert "8/8 steps" in view.error_detail
    assert view.step_count == 8
    assert len(llm.received) == 8  # no 9th decision was requested


@pytest.mark.parametrize(
    "max_steps,expected",
    [(1, RunStatus.LIMIT_EXCEEDED), (2, RunStatus.COMPLETED)],
    ids=["limit-hit-after-one-step", "final-answer-fits-in-budget"],
)
async def test_step_budget_boundary(open_container, tmp_path, max_steps, expected):
    llm = ScriptedLLM([tool("get_service_status", service_name="auth-service"), final("Done.")])
    c = await open_container(llm, cfg=make_settings(tmp_path, max_steps=max_steps))
    view = await c.runner.start("Check auth-service")
    assert view.status == expected


async def test_request_level_max_steps_overrides_settings(open_container):
    llm = ScriptedLLM(status_calls(["auth-service", "orders-db", "cdn-edge"]) + [final("x")])
    c = await open_container(llm)
    view = await c.runner.start("Check three services", max_steps=2)

    assert view.status == RunStatus.LIMIT_EXCEEDED and view.error_code == "step_limit_exceeded"
    assert view.max_steps == 2 and view.step_count == 2
    assert len(llm.received) == 2


async def test_wall_time_limit_ends_run_with_time_code(open_container, tmp_path, fake_clock):
    inner = ScriptedLLM(status_calls(["auth-service", "orders-db", "cdn-edge"]) + [final("x")])
    cfg = make_settings(tmp_path, max_wall_time_s=5)
    c = await open_container(ClockedLLM(inner, fake_clock, cost_s=2.0), cfg=cfg)
    view = await c.runner.start("Check three services")

    assert view.status == RunStatus.LIMIT_EXCEEDED
    assert view.error_code == "time_limit_exceeded"
    assert len(inner.received) == 3  # third decision pushes elapsed to 6s; its tool call is never run
    rows = await c.repo.list_tool_calls(view.id)
    assert [r.status for r in rows] == ["succeeded", "succeeded", "failed"]
    assert rows[2].error_type == "time_limit_exceeded"  # unfinished call closed out with the abort code


async def test_wall_time_just_under_budget_is_allowed(open_container, tmp_path, fake_clock):
    inner = ScriptedLLM([tool("get_service_status", service_name="auth-service"), final("in time")])
    cfg = make_settings(tmp_path, max_wall_time_s=5)
    c = await open_container(ClockedLLM(inner, fake_clock, cost_s=2.4), cfg=cfg)
    view = await c.runner.start("Check auth-service")

    assert view.status == RunStatus.COMPLETED and view.final_answer == "in time"


async def test_wall_time_exactly_at_budget_is_exceeded(open_container, tmp_path, fake_clock):
    # budget is exhausted when elapsed >= max: after 2 decisions (5.0s) the second tool must not run
    inner = ScriptedLLM(status_calls(["auth-service", "orders-db"]) + [final("never")])
    cfg = make_settings(tmp_path, max_wall_time_s=5)
    c = await open_container(ClockedLLM(inner, fake_clock, cost_s=2.5), cfg=cfg)
    view = await c.runner.start("Check two services")

    assert view.status == RunStatus.LIMIT_EXCEEDED and view.error_code == "time_limit_exceeded"
    assert "5.00s >= 5" in view.error_detail
    statuses = [r.status for r in await c.repo.list_tool_calls(view.id)]
    assert statuses == ["succeeded", "failed"]
    assert len(inner.received) == 2


async def test_hard_deadline_cancels_slow_tool_on_real_clock(open_container, tmp_path):
    # slow-service sleeps 2*tool_timeout+0.05s; the run budget (0.3s) must cut it off well before the tool timeout
    cfg = make_settings(tmp_path, max_wall_time_s=0.3, tool_timeout_s=1.0)
    llm = ScriptedLLM([tool("get_service_status", service_name="slow-service"), final("never")])
    c = await open_container(llm, cfg=cfg, clock=time.monotonic)

    started = time.perf_counter()
    view = await c.runner.start("Check slow-service")
    elapsed = time.perf_counter() - started

    assert view.status == RunStatus.LIMIT_EXCEEDED and view.error_code == "time_limit_exceeded"
    assert "during tool get_service_status" in view.error_detail
    assert elapsed < 1.0
    assert [r.status for r in await c.repo.list_tool_calls(view.id)] == ["failed"]


async def test_repeated_identical_call_is_detected_as_loop(open_container):
    llm = ScriptedLLM(status_calls(["payments-api"] * 5) + [final("x")])
    c = await open_container(llm)
    view = await c.runner.start("Keep checking payments-api")

    assert view.status == RunStatus.LIMIT_EXCEEDED
    assert view.error_code == "loop_detected"
    assert "'get_service_status' called 3 times" in view.error_detail
    executed = [r for r in await c.repo.list_tool_calls(view.id) if r.status == "succeeded"]
    assert len(executed) == 2  # the third identical call is blocked, not executed
    assert len(llm.received) == 3


async def test_loop_threshold_is_configurable(open_container, tmp_path):
    llm = ScriptedLLM(status_calls(["payments-api"] * 3) + [final("x")])
    c = await open_container(llm, cfg=make_settings(tmp_path, loop_detect_threshold=2))
    view = await c.runner.start("Keep checking payments-api")

    assert view.error_code == "loop_detected"
    executed = [r for r in await c.repo.list_tool_calls(view.id) if r.status == "succeeded"]
    assert len(executed) == 1


async def test_identical_calls_below_threshold_are_not_a_loop(open_container):
    names = [
        "payments-api",
        "auth-service",
        "payments-api",
        "auth-service",
        "cdn-edge",
        "orders-db",
        "search-api",
        "checkout-web",
    ]
    c = await open_container(ScriptedLLM(status_calls(names) + [final("x")]))
    view = await c.runner.start("Check services")

    # two identical calls each stay under the threshold; the step budget (8) is what stops the run
    assert view.error_code == "step_limit_exceeded"
    executed = [r for r in await c.repo.list_tool_calls(view.id) if r.status == "succeeded"]
    assert len(executed) == 8
