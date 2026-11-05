"""Validate and normalize a raw OpenAI-format assistant message into an LLMDecision."""

from __future__ import annotations

import json
from typing import Any

from app.core.errors import LLMMalformedResponseError
from app.llm.base import (
    FinalAnswerDecision,
    LLMDecision,
    ToolCallRequest,
    ToolCallsDecision,
)

_RAW_LIMIT = 2000


def _raw(message: dict[str, Any]) -> str:
    try:
        text = json.dumps(message, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(message)
    return text[:_RAW_LIMIT]


def normalize_assistant_message(message: dict[str, Any], max_tool_calls: int) -> tuple[LLMDecision, dict[str, Any]]:
    """Return ``(decision, clean_assistant_message)`` or raise LLMMalformedResponseError.

    ``reasoning_content`` is preserved when present: DeepSeek thinking mode requires it to be
    sent back with tool-call turns (it is simply absent when thinking is disabled).
    """

    def bad(reason: str) -> LLMMalformedResponseError:
        return LLMMalformedResponseError(reason, raw=_raw(message))

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise bad("content is not a string")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise bad("tool_calls is not a list")

    clean: dict[str, Any] = {"role": "assistant", "content": content or None}
    reasoning = message.get("reasoning_content")
    if reasoning:
        clean["reasoning_content"] = reasoning

    if raw_calls:
        if len(raw_calls) > max_tool_calls:
            raise bad(f"too many tool calls: {len(raw_calls)} > {max_tool_calls}")
        calls: list[ToolCallRequest] = []
        clean_calls: list[dict[str, Any]] = []
        for raw in raw_calls:
            if not isinstance(raw, dict):
                raise bad("tool call is not an object")
            call_id = raw.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise bad("tool call missing id")
            fn = raw.get("function")
            if not isinstance(fn, dict):
                raise bad("tool call missing function")
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                raise bad("tool call missing function name")
            arguments = fn.get("arguments")
            if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
                arguments = "{}"
            if not isinstance(arguments, str):
                raise bad(f"arguments for {name} is not a JSON string")
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                raise bad(f"invalid JSON args for {name}") from None
            if not isinstance(args, dict):
                raise bad(f"args for {name} is not a JSON object")
            calls.append(ToolCallRequest(call_id=call_id, tool=name, args=args))
            clean_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        clean["tool_calls"] = clean_calls
        return ToolCallsDecision(calls=calls, content=content or None), clean

    if content and content.strip():
        return FinalAnswerDecision(answer=content.strip()), clean
    raise bad("empty response")
