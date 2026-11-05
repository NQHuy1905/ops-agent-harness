"""The agent harness: LLM <-> tool execution loop with persisted state, limits and approvals.

Run lifecycle (``RunStatus``)::

    RUNNING --final answer--------------------> COMPLETED
    RUNNING --create_incident requested-------> AWAITING_APPROVAL --approve/reject--> RUNNING
    RUNNING --step / time / loop limit--------> LIMIT_EXCEEDED
    RUNNING --malformed x N / provider down---> FAILED

Execution is split into *segments*: ``start`` runs until the run finishes or pauses for approval;
``resume`` applies the human decision and continues. Everything needed to continue is persisted
(steps, queued tool calls), so a segment can run in a different process than the previous one.
The prototype executes a segment synchronously inside the calling HTTP request / CLI command.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from app.config import Settings
from app.core.errors import (
    ApprovalArgsInvalidError,
    ApprovalMismatchError,
    ApprovalNotPendingError,
    LimitExceededError,
    LLMMalformedResponseError,
    LLMProviderError,
    RunBusyError,
    RunNotFoundError,
    TimeLimitExceeded,
    ToolInputValidationError,
    ToolNotFoundError,
)
from app.core.executor import ToolExecutor
from app.core.guards import ExecutionGuard
from app.core.history_builder import build_messages
from app.core.state import (
    ApprovalDecision,
    PendingApproval,
    RunStatus,
    RunView,
    StepType,
    ToolCallStatus,
    ToolResult,
)
from app.llm.base import FinalAnswerDecision, LLMClient, LLMResponse
from app.llm.prompt import build_system_prompt
from app.observability.logging import get_logger, truncate
from app.observability.tracer import Tracer
from app.storage.models import RunRow, ToolCallRow
from app.storage.repository import RunRepository, args_hash
from app.tools.base import Tool, ToolContext
from app.tools.registry import ToolRegistry

log = get_logger("harness.runner")


class AgentRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        repo: RunRepository,
        registry: ToolRegistry,
        llm: LLMClient,
        tracer: Tracer,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.registry = registry
        self.llm = llm
        self.tracer = tracer
        self.clock = clock
        self.executor = ToolExecutor(settings, tracer, sleep)
        self.guard = ExecutionGuard(settings.loop_detect_threshold, clock)
        self.system_prompt = build_system_prompt(settings.agent_name)
        self._locks: dict[str, asyncio.Lock] = {}

    # ================================================================== public API
    async def start(
        self, objective: str, max_steps: int | None = None, max_wall_time_s: float | None = None
    ) -> RunView:
        run = await self.repo.create_run(
            objective=objective,
            max_steps=max_steps or self.settings.max_steps,
            max_wall_time_s=max_wall_time_s or self.settings.max_wall_time_s,
            llm_provider=self.llm.provider,
            llm_model=self.llm.model,
        )
        structlog.contextvars.bind_contextvars(run_id=run.id)
        log.info("run.started", objective=truncate(objective, 500), max_steps=run.max_steps)
        async with self._lock(run.id):
            async with self._agent_span(run.id, "start") as span:
                await self._advance(run.id, self.clock())
                await self._mark_span_outcome(span, run.id)
        self._locks.pop(run.id, None)
        return await self.get_view(run.id)

    async def resume(self, run_id: str, decision: ApprovalDecision) -> RunView:
        """Apply a human approval decision to the pending tool call, then continue the run."""
        lock = self._lock(run_id)
        if lock.locked():
            raise RunBusyError("run is currently executing")
        async with lock:
            structlog.contextvars.bind_contextvars(run_id=run_id)
            run = await self._get_run(run_id)
            pending = await self.repo.pending_approval_call(run_id)
            if run.status != RunStatus.AWAITING_APPROVAL.value or pending is None:
                raise ApprovalNotPendingError(f"run is '{run.status}', no approval pending")
            if decision.tool_call_id != pending.call_id:
                raise ApprovalMismatchError(
                    f"tool_call_id '{decision.tool_call_id}' does not match pending call '{pending.call_id}'"
                )
            tool = self.registry.get(pending.tool_name)
            args = pending.args
            if decision.decision == "approve" and decision.edited_args is not None:
                try:
                    tool.validate_input(decision.edited_args)
                except ToolInputValidationError as exc:
                    raise ApprovalArgsInvalidError(f"edited_args invalid: {exc.message}") from exc
                args = decision.edited_args
            # Compare-and-set guards against two concurrent decisions for the same call.
            if not await self.repo.transition_status(run_id, RunStatus.AWAITING_APPROVAL, RunStatus.RUNNING):
                raise ApprovalNotPendingError("approval already decided")

            edited = decision.decision == "approve" and decision.edited_args is not None
            segment_start = self.clock()  # executing the approved call counts as active time
            async with self._agent_span(run_id, "resume") as span:
                span.set_attributes({"harness.approval.decision": decision.decision, "gen_ai.tool.name": tool.name})
                try:
                    await self._apply_decision(run_id, pending, tool, decision, args, edited, segment_start)
                except LimitExceededError as exc:
                    await self._finish(run_id, RunStatus.LIMIT_EXCEEDED, exc.code, exc.message)
                except Exception as exc:  # never leave the run stuck in RUNNING
                    await self._fail_internal(run_id, exc)
                else:
                    await self._advance(run_id, segment_start)
                await self._mark_span_outcome(span, run_id)
        self._locks.pop(run_id, None)
        return await self.get_view(run_id)

    async def _apply_decision(
        self,
        run_id: str,
        pending: ToolCallRow,
        tool: Tool,
        decision: ApprovalDecision,
        args: dict[str, Any],
        edited: bool,
        segment_start: float,
    ) -> None:
        await self.repo.record_approval(
            run_id,
            pending.call_id,
            decision.decision,
            decision.edited_args if edited else None,
            decision.reason,
            decision.decided_by,
        )
        await self.repo.append_step(
            run_id,
            StepType.APPROVAL_DECIDED,
            {
                "call_id": pending.call_id,
                "tool": tool.name,
                "decision": decision.decision,
                "edited": edited,
                "args": args,
                "reason": decision.reason,
                "decided_by": decision.decided_by,
            },
        )
        log.info("approval.decided", call_id=pending.call_id, decision=decision.decision, edited=edited)
        if decision.decision == "approve":
            note = None
            if edited:
                await self.repo.update_tool_call(
                    run_id, pending.call_id, args=args, args_hash=args_hash(tool.name, args)
                )
                # Tell the model what was actually executed, so its final answer is truthful.
                note = {"approval": "approved with operator edits", "executed_args": args}
            else:
                note = {"approval": "approved by operator"}
            await self._execute_call(run_id, pending, tool, tool.validate_input(args), segment_start, note)
        else:
            rejected = ToolResult(
                ok=False,
                error={
                    "code": "rejected_by_user",
                    "message": decision.reason or "The user rejected this action. Do not retry it.",
                },
            )
            await self._record_result(run_id, pending, rejected, ToolCallStatus.REJECTED)

    async def get_view(self, run_id: str) -> RunView:
        run = await self._get_run(run_id)
        pending = None
        if run.status == RunStatus.AWAITING_APPROVAL.value:
            call = await self.repo.pending_approval_call(run_id)
            if call:
                pending = PendingApproval(tool_call_id=call.call_id, tool=call.tool_name, args=call.args)
        return RunView(
            id=run.id,
            objective=run.objective,
            status=RunStatus(run.status),
            final_answer=run.final_answer,
            error_code=run.error_code,
            error_detail=run.error_detail,
            step_count=run.step_count,
            max_steps=run.max_steps,
            max_wall_time_s=run.max_wall_time_s,
            llm_provider=run.llm_provider,
            llm_model=run.llm_model,
            pending_approval=pending,
            created_at=run.created_at,
            updated_at=run.updated_at,
            completed_at=run.completed_at,
        )

    # ================================================================== loop
    async def _advance(self, run_id: str, segment_start: float) -> None:
        """Run one segment: drain queued tool calls, then loop LLM decisions until done or paused.

        Every exit path leaves the run in a well-defined status: completed, awaiting_approval,
        limit_exceeded or failed (``internal_error`` for unexpected exceptions).
        """
        consecutive_malformed = 0
        try:
            if await self._drain_queue(run_id, segment_start):
                return
            while True:
                run = await self._get_run(run_id)
                self.guard.check_budget(run, segment_start)
                messages = build_messages(self.system_prompt, run.objective, await self.repo.list_steps(run_id))
                try:
                    response = await self._call_llm(run, messages, segment_start)
                except LLMMalformedResponseError as exc:
                    consecutive_malformed += 1
                    await self.repo.update_run(run_id, step_count=run.step_count + 1)
                    await self.repo.append_step(
                        run_id,
                        StepType.LLM_MALFORMED,
                        {
                            "reason": exc.message,
                            "raw": exc.details.get("raw"),
                            "consecutive": consecutive_malformed,
                        },
                    )
                    log.warning("llm.malformed", reason=exc.message, consecutive=consecutive_malformed)
                    if consecutive_malformed > self.settings.max_malformed_llm_responses:
                        await self._finish(
                            run_id,
                            RunStatus.FAILED,
                            exc.code,
                            f"{consecutive_malformed} consecutive malformed LLM responses; last: {exc.message}",
                        )
                        return
                    continue
                except LLMProviderError as exc:
                    await self._finish(run_id, RunStatus.FAILED, exc.code, exc.message)
                    return
                consecutive_malformed = 0
                if await self._handle_decision(run, response, segment_start):
                    return
        except LimitExceededError as exc:
            log.warning("limit.exceeded", code=exc.code, detail=exc.message)
            await self._finish(run_id, RunStatus.LIMIT_EXCEEDED, exc.code, exc.message)
        except Exception as exc:  # never leave the run stuck in RUNNING
            await self._fail_internal(run_id, exc)
        finally:
            run = await self.repo.get_run(run_id)
            if run is not None:
                await self.repo.update_run(
                    run_id, active_elapsed_s=run.active_elapsed_s + (self.clock() - segment_start)
                )

    async def _handle_decision(self, run: RunRow, response: LLMResponse, segment_start: float) -> bool:
        """Persist a valid LLM decision and act on it. Returns True when the segment must stop."""
        decision = response.decision
        assistant_message = response.assistant_message
        if not isinstance(decision, FinalAnswerDecision):
            assistant_message = await self._dedupe_call_ids(run.id, decision, assistant_message)
        step = await self.repo.append_step(
            run.id,
            StepType.LLM_RESPONSE,
            {
                "kind": decision.kind,
                "assistant_message": assistant_message,
                "model": response.model,
                "usage": response.usage.model_dump(),
                "latency_ms": response.latency_ms,
            },
        )
        await self.repo.update_run(run.id, step_count=run.step_count + 1)
        if isinstance(decision, FinalAnswerDecision):
            await self.repo.append_step(run.id, StepType.FINAL_ANSWER, {"answer": decision.answer})
            if not await self.repo.transition_status(
                run.id, RunStatus.RUNNING, RunStatus.COMPLETED, final_answer=decision.answer
            ):
                log.warning("run.transition_conflict", expected=RunStatus.RUNNING.value)
            log.info("run.finished", status=RunStatus.COMPLETED.value)
            return True
        await self.repo.add_tool_calls(run.id, step.idx, [(c.call_id, c.tool, c.args) for c in decision.calls])
        return await self._drain_queue(run.id, segment_start)

    async def _drain_queue(self, run_id: str, segment_start: float) -> bool:
        """Execute queued tool calls in order. Returns True if the run paused for approval."""
        for call in await self.repo.queued_tool_calls(run_id):
            self.guard.check_time(await self._get_run(run_id), segment_start)
            await self.guard.check_repeat(self.repo, run_id, call.tool_name, call.args_hash)
            try:
                tool = self.registry.get(call.tool_name)
                args = tool.validate_input(call.args)
            except (ToolNotFoundError, ToolInputValidationError) as exc:
                await self._record_result(
                    run_id, call, ToolResult(ok=False, error=exc.to_dict()), ToolCallStatus.FAILED
                )
                continue
            if tool.requires_approval:
                await self._request_approval(run_id, call)
                return True
            await self._execute_call(run_id, call, tool, args, segment_start)
        return False

    async def _remaining_s(self, run_id: str, segment_start: float) -> float:
        """Active wall-time budget left; raises TimeLimitExceeded when exhausted."""
        run = await self._get_run(run_id)
        self.guard.check_time(run, segment_start)
        return run.max_wall_time_s - self.guard.elapsed(run, segment_start)

    async def _within_deadline(
        self, run_id: str, segment_start: float, make_call: Callable[[], Awaitable[Any]], what: str
    ) -> Any:
        """Hard deadline: cancel the in-flight LLM/tool call when the run's time budget runs out.

        ``make_call`` is a factory so no coroutine is created when the budget is already exhausted.
        """
        remaining = await self._remaining_s(run_id, segment_start)
        try:
            return await asyncio.wait_for(make_call(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise TimeLimitExceeded(f"wall-time limit exceeded during {what}") from exc

    # ================================================================== helpers
    async def _call_llm(self, run: RunRow, messages: list[dict[str, Any]], segment_start: float) -> LLMResponse:
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": self.llm.provider,
            "gen_ai.request.model": self.llm.model,
            "harness.step": run.step_count + 1,
        }
        async with self.tracer.span(run.id, f"chat {self.llm.model}", kind="CLIENT", attributes=attrs) as span:
            log.info("llm.request", step=run.step_count + 1, messages=len(messages))
            response = await self._within_deadline(
                run.id, segment_start, lambda: self.llm.decide(messages, self.registry.llm_tool_specs()), "LLM call"
            )
            span.set_attributes(
                {
                    "gen_ai.response.model": response.model,
                    "gen_ai.usage.input_tokens": response.usage.input_tokens,
                    "gen_ai.usage.output_tokens": response.usage.output_tokens,
                    "harness.decision": response.decision.kind,
                }
            )
            log.info(
                "llm.response",
                step=run.step_count + 1,
                decision=response.decision.kind,
                tool_calls=[c.tool for c in getattr(response.decision, "calls", [])],
                latency_ms=response.latency_ms,
            )
            return response

    async def _request_approval(self, run_id: str, call: ToolCallRow) -> None:
        await self.repo.update_tool_call(run_id, call.call_id, status=ToolCallStatus.PENDING_APPROVAL.value)
        await self.repo.append_step(
            run_id,
            StepType.APPROVAL_REQUESTED,
            {"call_id": call.call_id, "tool": call.tool_name, "args": call.args},
        )
        if not await self.repo.transition_status(run_id, RunStatus.RUNNING, RunStatus.AWAITING_APPROVAL):
            log.warning("run.transition_conflict", expected=RunStatus.RUNNING.value)
        log.info("approval.requested", call_id=call.call_id, tool=call.tool_name, args=truncate(call.args))

    async def _execute_call(
        self,
        run_id: str,
        call: ToolCallRow,
        tool: Tool,
        args: Any,
        segment_start: float,
        note: dict[str, Any] | None = None,
    ) -> None:
        await self.repo.update_tool_call(run_id, call.call_id, status=ToolCallStatus.RUNNING.value)
        ctx = ToolContext(run_id=run_id, tool_call_id=call.call_id, repo=self.repo, settings=self.settings)
        result = await self._within_deadline(
            run_id, segment_start, lambda: self.executor.execute(tool, args, ctx), f"tool {tool.name}"
        )
        await self._record_result(
            run_id, call, result, ToolCallStatus.SUCCEEDED if result.ok else ToolCallStatus.FAILED, note
        )

    async def _record_result(
        self,
        run_id: str,
        call: ToolCallRow,
        result: ToolResult,
        status: ToolCallStatus,
        note: dict[str, Any] | None = None,
    ) -> None:
        observation = {**result.observation(), **(note or {})}
        await self.repo.update_tool_call(
            run_id,
            call.call_id,
            status=status.value,
            result=observation,
            error_type=result.error["code"] if result.error else None,
            attempts=result.attempts,
            duration_ms=result.duration_ms,
        )
        await self.repo.append_step(
            run_id,
            StepType.TOOL_RESULT,
            {
                "call_id": call.call_id,
                "tool": call.tool_name,
                "status": status.value,
                "observation": observation,
                "attempts": result.attempts,
                "duration_ms": result.duration_ms,
            },
        )

    async def _finish(self, run_id: str, status: RunStatus, code: str, detail: str) -> None:
        """Terminal transition (failed / limit_exceeded). Unfinished tool calls are closed out."""
        await self.repo.cancel_unfinished_tool_calls(run_id, code)
        await self.repo.append_step(
            run_id, StepType.RUN_ERROR, {"status": status.value, "code": code, "detail": detail}
        )
        if not await self.repo.transition_status(
            run_id, RunStatus.RUNNING, status, error_code=code, error_detail=detail
        ):
            log.warning("run.transition_conflict", expected=RunStatus.RUNNING.value, target=status.value)
        log.info("run.finished", status=status.value, error_code=code, detail=detail)

    async def _fail_internal(self, run_id: str, exc: Exception) -> None:
        log.exception("run.internal_error", error_type=type(exc).__name__)
        await self._finish(run_id, RunStatus.FAILED, "internal_error", f"{type(exc).__name__}: {str(exc)[:300]}")

    async def _mark_span_outcome(self, span: Any, run_id: str) -> None:
        run = await self._get_run(run_id)
        span.set_attribute("harness.run.status", run.status)
        if run.status in (RunStatus.FAILED.value, RunStatus.LIMIT_EXCEEDED.value):
            span.set_error(run.error_code or run.status, run.error_detail)

    async def _dedupe_call_ids(self, run_id: str, decision: Any, assistant_message: dict) -> dict:
        """Tool-call ids must be unique per run. Rename reused ids (vs. earlier steps or within this
        response) in both the decision and the replayed assistant message, matched by position."""
        seen = {c.call_id for c in await self.repo.list_tool_calls(run_id)}
        tool_calls = [dict(tc) for tc in assistant_message.get("tool_calls", [])]
        changed = False
        for position, call in enumerate(decision.calls):
            if call.call_id in seen:
                n = 1
                while f"{call.call_id}_r{n}" in seen:
                    n += 1
                call.call_id = f"{call.call_id}_r{n}"
                tool_calls[position]["id"] = call.call_id
                changed = True
            seen.add(call.call_id)
        if not changed:
            return assistant_message
        return {**assistant_message, "tool_calls": tool_calls}

    async def _get_run(self, run_id: str) -> RunRow:
        run = await self.repo.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"run '{run_id}' not found")
        return run

    def _lock(self, run_id: str) -> asyncio.Lock:
        return self._locks.setdefault(run_id, asyncio.Lock())

    def _agent_span(self, run_id: str, segment: str):
        return self.tracer.span(
            run_id,
            f"invoke_agent {self.settings.agent_name}",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": self.settings.agent_name,
                "gen_ai.provider.name": self.llm.provider,
                "gen_ai.request.model": self.llm.model,
                "harness.segment": segment,
            },
        )
