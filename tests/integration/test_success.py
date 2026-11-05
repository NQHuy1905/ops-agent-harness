"""Happy paths through the real runner: status check, knowledge base + status, history and final answer."""

from __future__ import annotations

from app.core.state import RunStatus, StepType
from app.llm.scripted import ScriptedLLM, final, multi, tool
from tests.helpers import tool_payloads


async def test_status_check_completes_with_final_answer(open_container):
    llm_script = [tool("get_service_status", service_name="payments-api"), final("Payments is degraded.")]

    llm = ScriptedLLM(llm_script)
    c = await open_container(llm)
    view = await c.runner.start("Check the status of payments-api")

    assert view.status == RunStatus.COMPLETED
    assert view.final_answer == "Payments is degraded."
    assert view.error_code is None and view.pending_approval is None
    assert view.step_count == 2
    assert (await c.runner.get_view(view.id)).final_answer == "Payments is degraded."

    observed = tool_payloads(llm.received[1])
    assert len(observed) == 1 and observed[0][1]["ok"] is True
    assert observed[0][1]["data"]["name"] == "payments-api"
    assert observed[0][1]["data"]["status"] == "degraded"


async def test_knowledge_base_then_status_completes(open_container):

    llm = ScriptedLLM(
        [
            tool("search_knowledge_base", call_id="c-kb", query="payments latency"),
            tool("get_service_status", call_id="c-st", service_name="payments-api"),
            final("Runbook found and status confirmed."),
        ]
    )
    c = await open_container(llm)
    view = await c.runner.start("How do I handle payments latency?")

    assert view.status == RunStatus.COMPLETED
    calls = await c.repo.list_tool_calls(view.id)
    assert [(t.call_id, t.tool_name, t.status) for t in calls] == [
        ("c-kb", "search_knowledge_base", "succeeded"),
        ("c-st", "get_service_status", "succeeded"),
    ]
    kb_obs = tool_payloads(llm.received[1])[0][1]
    assert kb_obs["data"]["results"][0]["service"] == "payments-api"


async def test_history_steps_are_ordered_and_complete(open_container):

    c = await open_container(
        ScriptedLLM(
            [
                tool("get_service_status", service_name="auth-service"),
                tool("get_service_status", service_name="orders-db"),
                final("All good."),
            ]
        )
    )
    view = await c.runner.start("Check auth-service and orders-db")
    steps = await c.repo.list_steps(view.id)

    assert [s.idx for s in steps] == list(range(len(steps)))
    assert [s.type for s in steps] == [
        StepType.LLM_RESPONSE.value,
        StepType.TOOL_RESULT.value,
        StepType.LLM_RESPONSE.value,
        StepType.TOOL_RESULT.value,
        StepType.LLM_RESPONSE.value,
        StepType.FINAL_ANSWER.value,
    ]
    assert steps[-1].payload == {"answer": "All good."}
    assert [s.payload["tool"] for s in steps if s.type == StepType.TOOL_RESULT.value] == [
        "get_service_status",
        "get_service_status",
    ]


async def test_final_answer_is_stripped_and_persisted(open_container):

    c = await open_container(ScriptedLLM([final("   All clear.  \n")]))
    view = await c.runner.start("Anything to report?")

    assert view.status == RunStatus.COMPLETED
    assert view.final_answer == "All clear."
    assert view.step_count == 1
    persisted = [s for s in await c.repo.list_steps(view.id) if s.type == StepType.FINAL_ANSWER.value]
    assert persisted[0].payload["answer"] == "All clear."


async def test_first_request_contains_system_prompt_and_objective(open_container):

    llm = ScriptedLLM([final("ok")])
    c = await open_container(llm)
    await c.runner.start("Check cdn-edge please")

    first = llm.received[0]
    assert [m["role"] for m in first] == ["system", "user"]
    assert first[1]["content"] == "Check cdn-edge please"


async def test_multiple_calls_in_one_response_run_in_order(open_container):

    llm = ScriptedLLM(
        [
            multi(
                tool("get_service_status", call_id="m1", service_name="auth-service"),
                tool("search_knowledge_base", call_id="m2", query="severity matrix"),
                tool("get_service_status", call_id="m3", service_name="orders-db"),
            ),
            final("Three checks done."),
        ]
    )
    c = await open_container(llm)
    view = await c.runner.start("Run three checks")

    assert view.status == RunStatus.COMPLETED
    seqs = [(t.call_id, t.seq, t.status) for t in await c.repo.list_tool_calls(view.id)]
    assert seqs == [("m1", 0, "succeeded"), ("m2", 1, "succeeded"), ("m3", 2, "succeeded")]
    assert [call_id for call_id, _ in tool_payloads(llm.received[1])] == ["m1", "m2", "m3"]


# ------------------------------------------------------------------ reused tool-call ids
def _assert_protocol_ids(messages):
    """Every assistant tool_calls id must be answered by a tool message with the same id, in order."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            want = [tc["id"] for tc in m["tool_calls"]]
            got = [mm["tool_call_id"] for mm in messages[i + 1 : i + 1 + len(want)]]
            assert got == want


async def test_duplicate_id_within_one_response_is_renamed(open_container):
    llm = ScriptedLLM(
        [
            multi(
                tool("get_service_status", call_id="x", service_name="auth-service"),
                tool("get_service_status", call_id="x", service_name="orders-db"),
            ),
            final("Both checked."),
        ]
    )
    c = await open_container(llm)
    view = await c.runner.start("Check two services")

    assert view.status == RunStatus.COMPLETED
    ids = [t.call_id for t in await c.repo.list_tool_calls(view.id)]
    assert ids == ["x", "x_r1"]
    assert [cid for cid, _ in tool_payloads(llm.received[1])] == ["x", "x_r1"]
    _assert_protocol_ids(llm.received[1])


async def test_id_reused_from_earlier_step_is_renamed_in_history_and_next_request(open_container):
    llm = ScriptedLLM(
        [
            tool("get_service_status", call_id="dup", service_name="auth-service"),
            tool("get_service_status", call_id="dup", service_name="orders-db"),
            final("Done."),
        ]
    )
    c = await open_container(llm)
    view = await c.runner.start("Check two services")

    assert view.status == RunStatus.COMPLETED
    assert [t.call_id for t in await c.repo.list_tool_calls(view.id)] == ["dup", "dup_r1"]
    assert [cid for cid, _ in tool_payloads(llm.received[2])] == ["dup", "dup_r1"]
    _assert_protocol_ids(llm.received[2])


async def test_renamed_id_does_not_collide_with_an_existing_id(open_container):
    llm = ScriptedLLM(
        [
            tool("get_service_status", call_id="a", service_name="auth-service"),
            tool("get_service_status", call_id="a_r1", service_name="orders-db"),
            tool("get_service_status", call_id="a", service_name="cdn-edge"),
            final("Done."),
        ]
    )
    c = await open_container(llm)
    view = await c.runner.start("Check three services")

    assert view.status == RunStatus.COMPLETED
    ids = [t.call_id for t in await c.repo.list_tool_calls(view.id)]
    assert ids == ["a", "a_r1", "a_r2"]
    _assert_protocol_ids(llm.received[3])
