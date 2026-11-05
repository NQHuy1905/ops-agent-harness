"""Deterministic LLM doubles.

* ``ScriptedLLM`` replays a fixed script of OpenAI-format assistant messages through the real
  ``normalize_assistant_message`` so tests exercise the production parsing path.
* ``DemoScenarioLLM`` is a stateless offline provider (``LLM_PROVIDER=scripted``) driven by JSON
  scenarios in ``data/scenarios``; its position is derived from the message history so it works
  across separate HTTP requests and resumes.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.errors import LLMProviderError
from app.llm.base import LLMResponse
from app.llm.parser import normalize_assistant_message

_ids = itertools.count(1)


class _Raise:
    """Script item that makes ``decide`` raise ``LLMProviderError``."""

    def __init__(self, msg: str) -> None:
        self.msg = msg


def _call(name: str, args_json: str, call_id: str | None) -> dict[str, Any]:
    return {
        "id": call_id,  # None -> assigned by the ScriptedLLM instance counter
        "type": "function",
        "function": {"name": name, "arguments": args_json},
    }


def tool(name: str, call_id: str | None = None, **args: Any) -> dict[str, Any]:
    """One response with a single tool call."""
    return {"role": "assistant", "content": None, "tool_calls": [_call(name, json.dumps(args), call_id)]}


def multi(*tool_items: dict[str, Any]) -> dict[str, Any]:
    """One response with several tool calls (merged from ``tool(...)`` items)."""
    calls = [c for item in tool_items for c in item["tool_calls"]]
    return {"role": "assistant", "content": None, "tool_calls": calls}


def final(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def malformed(raw_arguments: str = "{not json", tool_name: str = "get_service_status") -> dict[str, Any]:
    """Tool call whose arguments are not valid JSON."""
    return {"role": "assistant", "content": None, "tool_calls": [_call(tool_name, raw_arguments, None)]}


def empty() -> dict[str, Any]:
    """Response with neither content nor tool calls."""
    return {"role": "assistant", "content": None}


def provider_error(msg: str = "boom") -> _Raise:
    return _Raise(msg)


class ScriptedLLM:
    provider = "scripted"
    model = "scripted"

    def __init__(self, script: list[Any], max_tool_calls: int = 5) -> None:
        self._script = list(script)
        self._max_tool_calls = max_tool_calls
        self._counter = itertools.count(1)
        self.received: list[list[dict[str, Any]]] = []

    async def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.received.append(copy.deepcopy(messages))
        if not self._script:
            raise LLMProviderError("scripted LLM script exhausted")
        item = self._script.pop(0)
        if isinstance(item, _Raise):
            raise LLMProviderError(item.msg)
        message = copy.deepcopy(item)
        for call in message.get("tool_calls") or []:
            if call.get("id") is None:
                call["id"] = f"call_{next(self._counter)}"
        decision, assistant_message = normalize_assistant_message(message, self._max_tool_calls)
        return LLMResponse(decision=decision, assistant_message=assistant_message, model=self.model)


# ------------------------------------------------------------------ demo provider
KNOWN_SERVICES = (
    "payments-api", "auth-service", "checkout-web", "orders-db", "search-api",
    "notification-worker", "inventory-api", "cdn-edge", "flaky-service", "slow-service",
    "broken-service",
)  # fmt: skip
_KB_WORDS = {"runbook", "how", "doc", "docs", "knowledge", "kb"}


def _pick_scenario(objective: str) -> str:
    text = objective.lower()
    if "loop" in text:
        return "loop_demo"
    if "incident" in text:
        return "incident_flow"
    if _KB_WORDS & set(re.findall(r"[a-z]+", text)):
        return "kb_lookup"
    return "status_check"


def _fill(value: Any, mapping: dict[str, str]) -> Any:
    """Recursively replace ``{placeholder}`` tokens in strings."""
    if isinstance(value, str):
        for key, repl in mapping.items():
            value = value.replace("{" + key + "}", repl)
        return value
    if isinstance(value, dict):
        return {k: _fill(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, mapping) for v in value]
    return value


class DemoScenarioLLM:
    provider = "scripted"
    model = "scripted"

    def __init__(self, settings: Settings) -> None:
        self._dir = Path(settings.mock_data_dir) / "scenarios"
        self._max_tool_calls = settings.max_tool_calls_per_step

    def _load(self, name: str) -> list[dict[str, Any]]:
        try:
            return json.loads((self._dir / f"{name}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LLMProviderError(f"cannot load demo scenario {name}") from exc

    async def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        objective = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
        position = sum(1 for m in messages if m.get("role") == "assistant")
        lowered = objective.lower()
        service = next((s for s in KNOWN_SERVICES if s in lowered), "payments-api")
        last_tool = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "tool"), "")
        mapping = {"service": service, "last_tool_result": last_tool[:500]}

        script = self._load(_pick_scenario(objective))
        if position >= len(script):
            item: dict[str, Any] = {"type": "final", "text": "Scenario finished."}
        else:
            item = _fill(script[position], mapping)

        kind = item.get("type")
        if kind == "tool":
            msg = _scenario_call(item, position, 0)
            message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": [msg]}
        elif kind == "multi":
            calls = [_scenario_call(c, position, i) for i, c in enumerate(item["calls"])]
            message = {"role": "assistant", "content": None, "tool_calls": calls}
        else:
            message = {"role": "assistant", "content": item.get("text") or "Scenario finished."}
        decision, assistant_message = normalize_assistant_message(message, self._max_tool_calls)
        return LLMResponse(decision=decision, assistant_message=assistant_message, model=self.model)


def _scenario_call(item: dict[str, Any], position: int, index: int) -> dict[str, Any]:
    """Build a tool call with an id that is deterministic from the scenario position."""
    return _call(item["name"], json.dumps(item.get("args", {})), f"call_demo_{position}_{index}")
