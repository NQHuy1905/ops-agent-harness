"""Execution guards: step budget, active wall-time budget and repeated-call detection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.errors import LimitExceededError, LoopDetected, StepLimitExceeded, TimeLimitExceeded
from app.core.guards import ExecutionGuard
from app.storage.repository import args_hash


def make_run(step_count=0, max_steps=8, max_wall_time_s=10.0, active_elapsed_s=0.0):
    return SimpleNamespace(
        step_count=step_count,
        max_steps=max_steps,
        max_wall_time_s=max_wall_time_s,
        active_elapsed_s=active_elapsed_s,
    )


class FakeRepo:
    """Only ``count_identical_calls`` is used by the guard."""

    def __init__(self, previous: int) -> None:
        self.previous = previous

    async def count_identical_calls(self, run_id: str, tool_name: str, hash_: str) -> int:
        return self.previous


# ------------------------------------------------------------------ step budget
def test_step_budget_allows_calls_below_limit():
    ExecutionGuard(3, clock=lambda: 0.0).check_budget(make_run(step_count=7, max_steps=8), 0.0)


def test_step_budget_blocks_at_limit():
    with pytest.raises(StepLimitExceeded, match="7/7 steps") as ei:
        ExecutionGuard(3, clock=lambda: 0.0).check_budget(make_run(step_count=7, max_steps=7), 0.0)
    assert ei.value.code == "step_limit_exceeded"
    assert isinstance(ei.value, LimitExceededError)


def test_step_limit_is_checked_before_time_limit():
    guard = ExecutionGuard(3, clock=lambda: 100.0)
    run = make_run(step_count=5, max_steps=5, max_wall_time_s=1.0)
    with pytest.raises(StepLimitExceeded):
        guard.check_budget(run, 0.0)


# ------------------------------------------------------------------ wall time
def test_time_just_under_budget_is_allowed():
    ExecutionGuard(3, clock=lambda: 4.999).check_time(make_run(max_wall_time_s=5.0), 0.0)


def test_time_limit_equal_to_budget_is_exceeded():
    """The budget is exhausted at elapsed == max, so the next call must not start."""
    with pytest.raises(TimeLimitExceeded, match="5.00s >= 5.0s"):
        ExecutionGuard(3, clock=lambda: 5.0).check_time(make_run(max_wall_time_s=5.0), 0.0)


def test_time_limit_exceeded_raises_with_elapsed():
    with pytest.raises(TimeLimitExceeded) as ei:
        ExecutionGuard(3, clock=lambda: 5.01).check_time(make_run(max_wall_time_s=5.0), 0.0)
    assert ei.value.code == "time_limit_exceeded"
    assert ei.value.details["elapsed_s"] == pytest.approx(5.01)


def test_elapsed_adds_previous_active_segments():
    guard = ExecutionGuard(3, clock=lambda: 12.0)
    run = make_run(max_wall_time_s=5.0, active_elapsed_s=4.0)
    assert guard.elapsed(run, segment_started_at=10.0) == pytest.approx(6.0)
    with pytest.raises(TimeLimitExceeded):
        guard.check_time(run, segment_started_at=10.0)


def test_time_passing_outside_segment_is_not_counted():
    """Clock movement before a segment starts (e.g. waiting for approval) must not count."""
    now = {"t": 0.0}
    guard = ExecutionGuard(3, clock=lambda: now["t"])
    run = make_run(max_wall_time_s=5.0, active_elapsed_s=2.0)
    now["t"] = 1000.0  # approval wait
    segment_start = now["t"]
    now["t"] += 1.0
    assert guard.elapsed(run, segment_started_at=segment_start) == pytest.approx(3.0)
    guard.check_time(run, segment_started_at=segment_start)


# ------------------------------------------------------------------ loop detection
async def test_loop_threshold_blocks_nth_identical_call():
    guard = ExecutionGuard(3, clock=lambda: 0.0)
    await guard.check_repeat(FakeRepo(previous=0), "run", "get_service_status", "h")
    await guard.check_repeat(FakeRepo(previous=1), "run", "get_service_status", "h")
    with pytest.raises(LoopDetected, match="'get_service_status' called 3 times") as ei:
        await guard.check_repeat(FakeRepo(previous=2), "run", "get_service_status", "h")
    assert ei.value.code == "loop_detected"
    assert ei.value.details["tool"] == "get_service_status"


async def test_loop_threshold_two_blocks_second_identical_call():
    guard = ExecutionGuard(2, clock=lambda: 0.0)
    await guard.check_repeat(FakeRepo(previous=0), "run", "t", "h")
    with pytest.raises(LoopDetected):
        await guard.check_repeat(FakeRepo(previous=1), "run", "t", "h")


def test_args_hash_is_canonical_and_name_sensitive():
    assert args_hash("t", {"a": 1, "b": 2}) == args_hash("t", {"b": 2, "a": 1})
    assert args_hash("t", {"a": 1}) != args_hash("t", {"a": 2})
    assert args_hash("t", {"a": 1}) != args_hash("u", {"a": 1})
