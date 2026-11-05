from __future__ import annotations

import json

import pytest

from app.core.errors import LLMMalformedResponseError
from app.llm.base import FinalAnswerDecision, ToolCallsDecision
from app.llm.parser import normalize_assistant_message


def _call(name="get_service_status", args='{"service_name": "payments-api"}', cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def _msg(content=None, calls=None):
    m = {"role": "assistant", "content": content}
    if calls is not None:
        m["tool_calls"] = calls
    return m


def test_single_tool_call():
    decision, clean = normalize_assistant_message(_msg(calls=[_call()]), 5)
    assert isinstance(decision, ToolCallsDecision)
    assert decision.calls[0].tool == "get_service_status"
    assert decision.calls[0].args == {"service_name": "payments-api"}
    assert clean["tool_calls"][0]["id"] == "c1"
    assert json.loads(clean["tool_calls"][0]["function"]["arguments"]) == decision.calls[0].args


def test_final_answer():
    decision, clean = normalize_assistant_message(_msg("  all good  "), 5)
    assert isinstance(decision, FinalAnswerDecision)
    assert decision.answer == "all good"
    assert "tool_calls" not in clean


def test_bad_json_args():
    with pytest.raises(LLMMalformedResponseError, match="invalid JSON args") as ei:
        normalize_assistant_message(_msg(calls=[_call(args="{not json")]), 5)
    assert len(ei.value.details["raw"]) <= 2000


def test_args_not_object():
    with pytest.raises(LLMMalformedResponseError, match="not a JSON object"):
        normalize_assistant_message(_msg(calls=[_call(args="[1]")]), 5)


def test_missing_id():
    call = _call()
    del call["id"]
    with pytest.raises(LLMMalformedResponseError, match="missing id"):
        normalize_assistant_message(_msg(calls=[call]), 5)


def test_empty_response():
    for m in (_msg(None), _msg("   "), _msg("", calls=[])):
        with pytest.raises(LLMMalformedResponseError, match="empty response"):
            normalize_assistant_message(m, 5)


def test_multiple_calls_kept_in_order():
    calls = [_call(name=f"t{i}", cid=f"c{i}") for i in range(3)]
    decision, clean = normalize_assistant_message(_msg("thinking", calls), 5)
    assert [c.call_id for c in decision.calls] == ["c0", "c1", "c2"]
    assert decision.content == "thinking"
    assert [c["id"] for c in clean["tool_calls"]] == ["c0", "c1", "c2"]


def test_too_many_calls():
    calls = [_call(cid=f"c{i}") for i in range(3)]
    with pytest.raises(LLMMalformedResponseError, match="too many"):
        normalize_assistant_message(_msg(calls=calls), 2)


def test_empty_arguments_string_is_empty_dict():
    decision, clean = normalize_assistant_message(_msg(calls=[_call(args="")]), 5)
    assert decision.calls[0].args == {}
    assert clean["tool_calls"][0]["function"]["arguments"] == "{}"


def test_reasoning_content_preserved_when_present():
    m = _msg(calls=[_call()])
    m["reasoning_content"] = "hmm"
    _, clean = normalize_assistant_message(m, 5)
    assert clean["reasoning_content"] == "hmm"
