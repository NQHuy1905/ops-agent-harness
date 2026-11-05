"""Rebuild the OpenAI-format LLM conversation from persisted steps.

Steps are the single source of truth, so a run can be resumed by a fresh process (e.g. after an
approval arrives in a later HTTP request) without any in-memory state.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.state import StepType
from app.storage.models import StepRow

CORRECTIVE_TEMPLATE = (
    "Your previous response was invalid and was discarded: {reason}. "
    "Respond either with valid tool call(s) using the provided tools (arguments must be a JSON object) "
    "or with a non-empty final answer."
)


def tool_message(call_id: str, observation: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(observation, default=str)}


def build_messages(system_prompt: str, objective: str, steps: list[StepRow]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": objective},
    ]
    for step in steps:
        p = step.payload
        if step.type == StepType.LLM_RESPONSE.value:
            messages.append(p["assistant_message"])
        elif step.type == StepType.TOOL_RESULT.value:
            messages.append(tool_message(p["call_id"], p["observation"]))
        elif step.type == StepType.LLM_MALFORMED.value:
            # The invalid assistant output is NOT replayed (it may break the tool-call protocol);
            # the model only sees why it was rejected.
            messages.append({"role": "user", "content": CORRECTIVE_TEMPLATE.format(reason=p["reason"])})
    return messages
