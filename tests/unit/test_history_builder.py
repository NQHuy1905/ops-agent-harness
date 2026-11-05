"""History -> OpenAI message reconstruction: ordering, replay rules and corrective feedback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from app.core.history_builder import CORRECTIVE_TEMPLATE, build_messages, tool_message
from app.core.state import StepType


def step(idx: int, type_: StepType, payload: dict):
    return SimpleNamespace(idx=idx, type=type_.value, payload=payload)


ASSISTANT_TOOL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
}


def test_empty_history_has_system_and_objective_only():
    msgs = build_messages("SYS", "Check payments", [])
    assert msgs == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Check payments"}]


def test_llm_response_is_replayed_verbatim_and_order_is_kept():
    obs = {"ok": True, "data": {"status": "operational"}}
    steps = [
        step(0, StepType.LLM_RESPONSE, {"assistant_message": ASSISTANT_TOOL}),
        step(1, StepType.TOOL_RESULT, {"call_id": "c1", "observation": obs}),
        step(2, StepType.LLM_RESPONSE, {"assistant_message": {"role": "assistant", "content": "done"}}),
    ]
    msgs = build_messages("S", "O", steps)
    assert msgs[2] is steps[0].payload["assistant_message"]
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps(obs)}
    assert msgs[4] == {"role": "assistant", "content": "done"}


def test_tool_result_content_round_trips_observation():
    obs = {"ok": False, "error": {"code": "tool_timeout", "message": "slow"}}
    msg = tool_message("call-9", obs)
    assert msg["role"] == "tool" and msg["tool_call_id"] == "call-9"
    assert json.loads(msg["content"]) == obs


def test_tool_message_serialises_non_json_values():
    when = datetime(2026, 11, 5, tzinfo=timezone.utc)
    msg = tool_message("c", {"ok": True, "data": {"created_at": when}})
    assert json.loads(msg["content"])["data"]["created_at"] == str(when)


def test_malformed_response_is_not_replayed_but_reason_is_fed_back():
    raw_bad = {"role": "assistant", "content": None, "tool_calls": [{"raw": "{not json"}]}
    steps = [
        step(0, StepType.LLM_MALFORMED, {"reason": "invalid JSON args for t", "raw": raw_bad, "consecutive": 1}),
    ]
    msgs = build_messages("S", "O", steps)
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == CORRECTIVE_TEMPLATE.format(reason="invalid JSON args for t")
    assert "Your previous response was invalid" in msgs[2]["content"]
    assert all("{not json" not in json.dumps(m) for m in msgs)
    assert len(msgs) == 3


def test_bookkeeping_steps_produce_no_messages():
    steps = [
        step(0, StepType.APPROVAL_REQUESTED, {"call_id": "c1", "tool": "t", "args": {}}),
        step(1, StepType.APPROVAL_DECIDED, {"call_id": "c1", "decision": "approve"}),
        step(2, StepType.FINAL_ANSWER, {"answer": "x"}),
        step(3, StepType.RUN_ERROR, {"status": "failed", "code": "c", "detail": "d"}),
    ]
    assert len(build_messages("S", "O", steps)) == 2
