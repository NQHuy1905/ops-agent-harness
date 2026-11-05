"""HTTP surface: run creation (200/202), reads, approvals, error envelope and catalogue endpoints."""

from __future__ import annotations

import pytest

from app.llm.scripted import ScriptedLLM, final, tool
from tests.helpers import CREATE_INCIDENT_ARGS


def incident_script():
    return [tool("create_incident", call_id="c-inc", **CREATE_INCIDENT_ARGS), final("Incident opened.")]


async def test_health_and_tool_catalogue(open_container, make_api):
    c = await open_container(ScriptedLLM([]))
    api = await make_api(c)

    health = await api.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok" and health.json()["llm_provider"] == "scripted"

    tools = {t["name"]: t for t in (await api.get("/tools")).json()}
    assert set(tools) == {"search_knowledge_base", "get_service_status", "create_incident"}
    assert tools["create_incident"]["requires_approval"] is True
    assert tools["get_service_status"]["requires_approval"] is False
    assert "service_name" in tools["get_service_status"]["parameters"]["properties"]


async def test_post_run_completed_returns_200(open_container, make_api):
    c = await open_container(ScriptedLLM([tool("get_service_status", service_name="cdn-edge"), final("CDN ok.")]))
    api = await make_api(c)

    resp = await api.post("/runs", json={"objective": "Check cdn-edge"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed" and body["final_answer"] == "CDN ok."
    assert body["pending_approval"] is None and body["error_code"] is None


async def test_post_run_awaiting_approval_returns_202_with_pending_payload(open_container, make_api):
    c = await open_container(ScriptedLLM(incident_script()))
    api = await make_api(c)

    resp = await api.post("/runs", json={"objective": "Open an incident for payments-api"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "awaiting_approval"
    assert body["pending_approval"] == {
        "tool_call_id": "c-inc",
        "tool": "create_incident",
        "args": CREATE_INCIDENT_ARGS,
    }


async def test_post_run_limit_exceeded_returns_200_with_code(open_container, make_api):
    calls = [tool("get_service_status", service_name=s) for s in ("auth-service", "orders-db", "cdn-edge")]
    c = await open_container(ScriptedLLM(calls + [final("x")]))
    api = await make_api(c)

    resp = await api.post("/runs", json={"objective": "Check three services", "max_steps": 1})
    assert resp.status_code == 200
    assert resp.json()["status"] == "limit_exceeded"
    assert resp.json()["error_code"] == "step_limit_exceeded" and resp.json()["max_steps"] == 1


async def test_post_run_failed_returns_200_with_code(open_container, make_api):
    from app.llm.scripted import provider_error

    c = await open_container(ScriptedLLM([provider_error("upstream down")]))
    api = await make_api(c)

    resp = await api.post("/runs", json={"objective": "Check payments-api"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "failed" and resp.json()["error_code"] == "llm_provider_error"


async def test_get_run_and_unknown_run_envelope(open_container, make_api):
    c = await open_container(ScriptedLLM([final("hello there")]))
    api = await make_api(c)
    run_id = (await api.post("/runs", json={"objective": "Say hello"})).json()["id"]

    assert (await api.get(f"/runs/{run_id}")).json()["final_answer"] == "hello there"
    missing = await api.get("/runs/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "run_not_found"
    assert "does-not-exist" in missing.json()["error"]["message"]


async def test_history_endpoint_returns_ordered_steps_and_tool_calls(open_container, make_api):
    c = await open_container(
        ScriptedLLM([tool("get_service_status", call_id="h1", service_name="flaky-service"), final("Done.")])
    )
    api = await make_api(c)
    run_id = (await api.post("/runs", json={"objective": "Check flaky-service"})).json()["id"]

    history = (await api.get(f"/runs/{run_id}/history")).json()
    idxs = [s["idx"] for s in history["steps"]]
    assert idxs == sorted(set(idxs)) and idxs == list(range(len(idxs)))
    assert [s["type"] for s in history["steps"]] == [
        "llm_response",
        "tool_result",
        "llm_response",
        "final_answer",
    ]
    (call,) = history["tool_calls"]
    assert call["call_id"] == "h1" and call["tool"] == "get_service_status"
    assert call["status"] == "succeeded" and call["attempts"] == 3
    assert history["approvals"] == []
    assert (await api.get("/runs/nope/history")).status_code == 404


async def test_trace_endpoint_and_unknown_run(open_container, make_api):
    c = await open_container(ScriptedLLM([final("ok")]))
    api = await make_api(c)
    run_id = (await api.post("/runs", json={"objective": "Say ok"})).json()["id"]

    trace = (await api.get(f"/runs/{run_id}/trace")).json()
    assert trace["run_id"] == run_id
    assert [s["name"] for s in trace["spans"]] == ["invoke_agent ops-assistant"]
    missing = await api.get("/runs/nope/trace")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "run_not_found"


async def test_approval_approve_over_http(open_container, make_api):
    c = await open_container(ScriptedLLM(incident_script()))
    api = await make_api(c)
    run_id = (await api.post("/runs", json={"objective": "Open an incident for payments-api"})).json()["id"]

    resp = await api.post(f"/runs/{run_id}/approval", json={"tool_call_id": "c-inc", "decision": "approve"})
    assert resp.status_code == 200 and resp.json()["status"] == "completed"

    incidents = (await api.get("/incidents", params={"run_id": run_id})).json()
    assert len(incidents) == 1 and incidents[0]["severity"] == "high" and incidents[0]["run_id"] == run_id
    assert len((await api.get("/incidents")).json()) == 1
    history = (await api.get(f"/runs/{run_id}/history")).json()
    assert [a["decision"] for a in history["approvals"]] == ["approve"]
    assert history["approvals"][0]["decided_by"] == "api-user"


async def test_approval_reject_over_http_creates_no_incident(open_container, make_api):
    c = await open_container(ScriptedLLM(incident_script()))
    api = await make_api(c)
    run_id = (await api.post("/runs", json={"objective": "Open an incident for payments-api"})).json()["id"]

    resp = await api.post(
        f"/runs/{run_id}/approval",
        json={"tool_call_id": "c-inc", "decision": "reject", "reason": "false alarm"},
    )
    assert resp.status_code == 200 and resp.json()["status"] == "completed"
    assert (await api.get("/incidents")).json() == []


async def test_next_incident_after_decision_pauses_again_with_202(open_container, make_api):
    """A second create_incident after the first decision pauses the run again."""
    script = [
        tool("create_incident", call_id="first", **CREATE_INCIDENT_ARGS),
        tool("create_incident", call_id="second", **{**CREATE_INCIDENT_ARGS, "title": "Second incident"}),
        final("done"),
    ]
    c = await open_container(ScriptedLLM(script))
    api = await make_api(c)
    run_id = (await api.post("/runs", json={"objective": "Open incidents"})).json()["id"]

    resp = await api.post(
        f"/runs/{run_id}/approval",
        json={
            "tool_call_id": "first",
            "decision": "approve",
            "edited_args": {**CREATE_INCIDENT_ARGS, "severity": "low"},
        },
    )
    assert resp.status_code == 202
    assert resp.json()["pending_approval"]["tool_call_id"] == "second"


async def test_approval_error_envelopes(open_container, make_api):
    c = await open_container(ScriptedLLM(incident_script()))
    api = await make_api(c)
    run_id = (await api.post("/runs", json={"objective": "Open an incident for payments-api"})).json()["id"]
    url = f"/runs/{run_id}/approval"

    mismatch = await api.post(url, json={"tool_call_id": "wrong", "decision": "approve"})
    assert mismatch.status_code == 409 and mismatch.json()["error"]["code"] == "approval_mismatch"

    invalid = await api.post(
        url,
        json={"tool_call_id": "c-inc", "decision": "approve", "edited_args": {**CREATE_INCIDENT_ARGS, "severity": "x"}},
    )
    assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "approval_args_invalid"
    assert (await api.get(f"/runs/{run_id}")).json()["status"] == "awaiting_approval"

    ok = await api.post(url, json={"tool_call_id": "c-inc", "decision": "approve"})
    assert ok.status_code == 200

    again = await api.post(url, json={"tool_call_id": "c-inc", "decision": "approve"})
    assert again.status_code == 409 and again.json()["error"]["code"] == "approval_not_pending"

    unknown = await api.post("/runs/nope/approval", json={"tool_call_id": "c-inc", "decision": "approve"})
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "run_not_found"


async def test_approval_body_must_name_a_known_decision(open_container, make_api):
    c = await open_container(ScriptedLLM(incident_script()))
    api = await make_api(c)
    run_id = (await api.post("/runs", json={"objective": "Open an incident for payments-api"})).json()["id"]

    resp = await api.post(f"/runs/{run_id}/approval", json={"tool_call_id": "c-inc", "decision": "maybe"})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "body",
    [{"objective": "x"}, {"objective": "ok run", "max_wall_time_s": 0}, {"objective": "ok run", "max_steps": 99}],
    ids=["objective-too-short", "wall-time-not-positive", "max-steps-above-cap"],
)
async def test_run_creation_validates_request_body(open_container, make_api, body):
    c = await open_container(ScriptedLLM([final("x")]))
    api = await make_api(c)
    assert (await api.post("/runs", json=body)).status_code == 422


async def test_run_list_is_newest_first_and_paginated(open_container, make_api):
    c = await open_container(ScriptedLLM([final("a"), final("b"), final("c")]))
    api = await make_api(c)
    ids = []
    for label in ("first run", "second run", "third run"):
        ids.append((await api.post("/runs", json={"objective": label})).json()["id"])

    page = (await api.get("/runs", params={"limit": 2})).json()
    assert [r["objective"] for r in page] == ["third run", "second run"]
    rest = (await api.get("/runs", params={"limit": 2, "offset": 2})).json()
    assert [r["id"] for r in rest] == [ids[0]]
    assert (await api.get("/runs", params={"limit": 0})).status_code == 422


async def test_request_validation_error_uses_error_envelope(open_container, make_api):
    c = await open_container(ScriptedLLM([final("x")]))
    api = await make_api(c)
    body = (await api.post("/runs", json={"objective": "x"})).json()
    assert body["error"]["code"] == "validation_error"
