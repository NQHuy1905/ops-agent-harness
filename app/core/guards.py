"""Execution limits: step budget, wall-time budget, repeated identical tool calls."""

from __future__ import annotations

from collections.abc import Callable

from app.core.errors import LoopDetected, StepLimitExceeded, TimeLimitExceeded
from app.storage.models import RunRow
from app.storage.repository import RunRepository


class ExecutionGuard:
    """Stops runaway runs.

    * step limit   - a step is one LLM decision (valid or malformed); checked before each LLM call.
    * time limit   - active execution time only: ``run.active_elapsed_s`` (previous segments) plus the
                     current segment. Time spent waiting for a human approval is not counted.
    * loop detect  - the ``loop_detect_threshold``-th identical (tool, args) call in a run aborts it.
    """

    def __init__(self, loop_threshold: int, clock: Callable[[], float]) -> None:
        self.loop_threshold = loop_threshold
        self.clock = clock

    def elapsed(self, run: RunRow, segment_started_at: float) -> float:
        return run.active_elapsed_s + (self.clock() - segment_started_at)

    def check_time(self, run: RunRow, segment_started_at: float) -> None:
        elapsed = self.elapsed(run, segment_started_at)
        if elapsed >= run.max_wall_time_s:
            raise TimeLimitExceeded(
                f"wall-time limit exceeded: {elapsed:.2f}s >= {run.max_wall_time_s}s", elapsed_s=elapsed
            )

    def check_budget(self, run: RunRow, segment_started_at: float) -> None:
        if run.step_count >= run.max_steps:
            raise StepLimitExceeded(f"step limit reached: {run.step_count}/{run.max_steps} steps")
        self.check_time(run, segment_started_at)

    async def check_repeat(self, repo: RunRepository, run_id: str, tool_name: str, args_hash: str) -> None:
        previous = await repo.count_identical_calls(run_id, tool_name, args_hash)
        if previous + 1 >= self.loop_threshold:
            raise LoopDetected(
                f"loop detected: '{tool_name}' called {previous + 1} times with identical arguments",
                tool=tool_name,
            )
