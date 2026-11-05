"""Tool execution with timeout, transient-error retries and output validation.

The executor never raises for tool problems: every outcome becomes a ``ToolResult`` that the LLM
observes (short error code + message, never a stack trace), so the agent can recover or explain.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from app.config import Settings
from app.core.errors import ToolError, ToolTimeoutError, ToolTransientError
from app.core.retry import retry_async
from app.core.state import ToolResult
from app.observability.logging import get_logger, truncate
from app.observability.tracer import Tracer
from app.tools.base import Tool, ToolContext

log = get_logger("harness.executor")


class ToolExecutor:
    def __init__(
        self,
        settings: Settings,
        tracer: Tracer,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.tracer = tracer
        self.sleep = sleep

    async def execute(self, tool: Tool, args: BaseModel, ctx: ToolContext) -> ToolResult:
        timeout = self.settings.tool_timeout_s
        attempts = 0
        started = time.perf_counter()

        async def attempt() -> object:
            nonlocal attempts
            attempts += 1
            try:
                return await asyncio.wait_for(tool.run(args, ctx), timeout=timeout)
            except asyncio.TimeoutError as exc:  # asyncio.TimeoutError != TimeoutError on 3.10
                raise ToolTimeoutError(f"{tool.name} timed out after {timeout}s") from exc

        def on_retry(n: int, exc: BaseException) -> None:
            log.warning("tool.retry", tool=tool.name, call_id=ctx.tool_call_id, retry=n, error=str(exc))

        attrs = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool.name,
            "gen_ai.tool.call.id": ctx.tool_call_id,
            "gen_ai.tool.type": "function",
        }
        async with self.tracer.span(ctx.run_id, f"execute_tool {tool.name}", attributes=attrs) as span:
            log.info("tool.call", tool=tool.name, call_id=ctx.tool_call_id, args=truncate(args.model_dump()))
            try:
                raw = await retry_async(
                    attempt,
                    retries=self.settings.tool_max_retries,
                    base_delay=self.settings.tool_retry_base_delay_s,
                    retry_on=(ToolTimeoutError, ToolTransientError),
                    sleep=self.sleep,
                    on_retry=on_retry,
                )
                data = tool.validate_output(raw)
                result = ToolResult(ok=True, data=data)
            except ToolError as exc:
                result = ToolResult(ok=False, error=exc.to_dict())
            except Exception as exc:  # defensive: a buggy tool must not crash the run
                log.exception("tool.unexpected_error", tool=tool.name, call_id=ctx.tool_call_id)
                result = ToolResult(
                    ok=False,
                    error={"code": "tool_execution_error", "message": f"unexpected {type(exc).__name__}"},
                )
            result.attempts = attempts
            result.duration_ms = int((time.perf_counter() - started) * 1000)
            span.set_attributes({"harness.attempts": attempts, "harness.ok": result.ok})
            if not result.ok:
                span.set_error(result.error["code"], result.error["message"])
            log.info(
                "tool.result",
                tool=tool.name,
                call_id=ctx.tool_call_id,
                ok=result.ok,
                attempts=attempts,
                duration_ms=result.duration_ms,
                error=result.error,
            )
            return result
