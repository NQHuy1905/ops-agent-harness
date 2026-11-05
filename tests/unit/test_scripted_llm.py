from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.errors import LLMMalformedResponseError, LLMProviderError
from app.llm.base import FinalAnswerDecision, ToolCallsDecision
from app.llm.scripted import (
    DemoScenarioLLM,
    ScriptedLLM,
    empty,
    final,
    malformed,
    multi,
    provider_error,
    tool,
)

MSGS = [{"role": "user", "content": "hi"}]


async def test_sequence_and_auto_ids():
    llm = ScriptedLLM([tool("a", x=1), multi(tool("b"), tool("c", call_id="zz")), final("done")])
    r1 = await llm.decide(MSGS, [])
    r2 = await llm.decide(MSGS, [])
    r3 = await llm.decide(MSGS, [])
    assert isinstance(r1.decision, ToolCallsDecision)
    assert r1.decision.calls[0].call_id == "call_1"
    assert r1.decision.calls[0].args == {"x": 1}
    assert [c.call_id for c in r2.decision.calls] == ["call_2", "zz"]
    assert isinstance(r3.decision, FinalAnswerDecision) and r3.decision.answer == "done"


async def test_received_is_deep_copy():
    llm = ScriptedLLM([final("x")])
    msgs = [{"role": "user", "content": "hi"}]
    await llm.decide(msgs, [])
    msgs[0]["content"] = "changed"
    assert llm.received[0][0]["content"] == "hi"


async def test_exhausted():
    llm = ScriptedLLM([])
    with pytest.raises(LLMProviderError, match="exhausted"):
        await llm.decide(MSGS, [])


async def test_provider_error_and_malformed_items():
    llm = ScriptedLLM([provider_error("bad"), malformed(), empty()])
    with pytest.raises(LLMProviderError, match="bad"):
        await llm.decide(MSGS, [])
    with pytest.raises(LLMMalformedResponseError):
        await llm.decide(MSGS, [])
    with pytest.raises(LLMMalformedResponseError, match="empty"):
        await llm.decide(MSGS, [])


async def test_max_tool_calls_enforced():
    llm = ScriptedLLM([multi(tool("a"), tool("b"))], max_tool_calls=1)
    with pytest.raises(LLMMalformedResponseError):
        await llm.decide(MSGS, [])


# ---------------------------------------------------------------- demo
@pytest.fixture
def demo():
    return DemoScenarioLLM(Settings(mock_data_dir=Path("data")))


async def test_demo_status_check_flow_and_placeholders(demo):
    msgs = [{"role": "user", "content": "Check the status of auth-service"}]
    r = await demo.decide(msgs, [])
    call = r.decision.calls[0]
    assert call.tool == "get_service_status" and call.args == {"service_name": "auth-service"}
    tool_msg = {"role": "tool", "tool_call_id": call.call_id, "content": '{"status":"ok"}'}
    msgs += [r.assistant_message, tool_msg]
    r2 = await demo.decide(msgs, [])
    assert isinstance(r2.decision, FinalAnswerDecision)
    assert '{"status":"ok"}' in r2.decision.answer and "auth-service" in r2.decision.answer
    msgs.append(r2.assistant_message)
    r3 = await demo.decide(msgs, [])
    assert r3.decision.answer == "Scenario finished."


async def test_demo_default_service(demo):
    r = await demo.decide([{"role": "user", "content": "status please"}], [])
    assert r.decision.calls[0].args == {"service_name": "payments-api"}


async def test_demo_incident_flow_tools(demo):
    msgs = [{"role": "user", "content": "Open an incident for checkout-web"}]
    names = []
    for _ in range(3):
        r = await demo.decide(msgs, [])
        names.append(r.decision.calls[0].tool)
        msgs.append(r.assistant_message)
        msgs.append({"role": "tool", "tool_call_id": r.decision.calls[0].call_id, "content": "r"})
    assert names == ["get_service_status", "search_knowledge_base", "create_incident"]
    last = r.decision.calls[0].args
    assert last["severity"] == "high" and "checkout-web" in last["title"]


async def test_demo_kb_routing(demo):
    r = await demo.decide([{"role": "user", "content": "How do I restart it? see runbook"}], [])
    assert r.decision.calls[0].tool == "search_knowledge_base"


async def test_demo_loop_repeats_same_call(demo):
    msgs = [{"role": "user", "content": "loop on flaky-service"}]
    seen = []
    for _ in range(4):
        r = await demo.decide(msgs, [])
        seen.append((r.decision.calls[0].tool, r.decision.calls[0].args))
        msgs += [r.assistant_message, {"role": "tool", "tool_call_id": "x", "content": "r"}]
    assert len(set(map(str, seen))) == 1
