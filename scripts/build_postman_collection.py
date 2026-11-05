"""Generate the Postman collection + environment in ./postman (run: python scripts/build_postman_collection.py).

Kept as code so the collection stays in sync with the API and is reviewable.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "postman"

SAVE_RUN = """
const body = pm.response.json();
pm.collectionVariables.set("run_id", body.id);
if (body.pending_approval) { pm.collectionVariables.set("tool_call_id", body.pending_approval.tool_call_id); }
"""


def test_script(*lines: str) -> list[dict]:
    return [{"listen": "test", "script": {"type": "text/javascript", "exec": [ln for block in lines for ln in block.strip().splitlines()]}}]


def expect_status(code: int, run_status: str | None = None) -> str:
    js = f'pm.test("HTTP {code}", () => pm.response.to.have.status({code}));'
    if run_status:
        js += f'\npm.test("run status {run_status}", () => pm.expect(pm.response.json().status).to.eql("{run_status}"));'
    return js


def req(name: str, method: str, path: str, body: dict | None = None, tests: list[dict] | None = None, desc: str = ""):
    item = {
        "name": name,
        "request": {
            "method": method,
            "header": [{"key": "Content-Type", "value": "application/json"}],
            "url": {"raw": "{{base_url}}" + path, "host": ["{{base_url}}"], "path": [p for p in path.split("?")[0].split("/") if p]},
            "description": desc,
        },
    }
    if "?" in path:
        q = path.split("?", 1)[1]
        item["request"]["url"]["query"] = [{"key": k, "value": v} for k, v in (kv.split("=") for kv in q.split("&"))]
    if body is not None:
        item["request"]["body"] = {"mode": "raw", "raw": json.dumps(body, indent=2), "options": {"raw": {"language": "json"}}}
    if tests:
        item["event"] = tests
    return item


def run_create(name: str, objective: str, code: int, status: str, extra: dict | None = None, desc: str = ""):
    return req(name, "POST", "/runs", {"objective": objective, **(extra or {})},
               test_script(SAVE_RUN, expect_status(code, status)), desc)


collection = {
    "info": {
        "name": "Ops Agent Harness",
        "description": "Requests for the ops agent harness API. Works offline with LLM_PROVIDER=scripted "
        "(keyword-routed demo scenarios) or live with LLM_PROVIDER=deepseek. Run folders in order: each "
        "POST /runs stores {{run_id}} and {{tool_call_id}} in collection variables.",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "variable": [{"key": "run_id", "value": ""}, {"key": "tool_call_id", "value": ""}],
    "item": [
        {"name": "0. Health & tools", "item": [
            req("Health", "GET", "/health", tests=test_script(expect_status(200))),
            req("List tools", "GET", "/tools", tests=test_script(expect_status(200))),
        ]},
        {"name": "1. Happy path - status check", "item": [
            run_create("Start run: check payments-api", "Check the status of payments-api", 200, "completed"),
            req("Get run", "GET", "/runs/{{run_id}}", tests=test_script(expect_status(200, "completed"))),
            req("History", "GET", "/runs/{{run_id}}/history", tests=test_script(expect_status(200))),
            req("Trace", "GET", "/runs/{{run_id}}/trace", tests=test_script(expect_status(200))),
        ]},
        {"name": "2. Incident - approve", "item": [
            run_create("Start run: incident (pauses)", "checkout-web is down, create an incident", 202, "awaiting_approval",
                       desc="Run pauses before create_incident; response has pending_approval."),
            req("Approve (with edit)", "POST", "/runs/{{run_id}}/approval",
                {"tool_call_id": "{{tool_call_id}}", "decision": "approve",
                 "edited_args": {"title": "checkout-web outage", "description": "Checkout returns 5xx for all users since 12:00 UTC",
                                 "severity": "critical"}},
                test_script(expect_status(200, "completed"))),
            req("Approve again -> 409", "POST", "/runs/{{run_id}}/approval",
                {"tool_call_id": "{{tool_call_id}}", "decision": "approve"}, test_script(expect_status(409))),
            req("Incidents for run", "GET", "/incidents?run_id={{run_id}}", tests=test_script(
                expect_status(200), 'pm.test("exactly one incident", () => pm.expect(pm.response.json().length).to.eql(1));')),
        ]},
        {"name": "3. Incident - reject", "item": [
            run_create("Start run: incident (pauses)", "payments-api is slow, create an incident", 202, "awaiting_approval"),
            req("Wrong tool_call_id -> 409", "POST", "/runs/{{run_id}}/approval",
                {"tool_call_id": "not-the-pending-call", "decision": "approve"}, test_script(expect_status(409))),
            req("Invalid edited args -> 422", "POST", "/runs/{{run_id}}/approval",
                {"tool_call_id": "{{tool_call_id}}", "decision": "approve", "edited_args": {"title": "x", "description": "y", "severity": "urgent"}},
                test_script(expect_status(422))),
            req("Reject", "POST", "/runs/{{run_id}}/approval",
                {"tool_call_id": "{{tool_call_id}}", "decision": "reject", "reason": "Known issue, already tracked"},
                test_script(expect_status(200, "completed"))),
            req("No incident created", "GET", "/incidents?run_id={{run_id}}", tests=test_script(
                expect_status(200), 'pm.test("no incident", () => pm.expect(pm.response.json().length).to.eql(0));')),
        ]},
        {"name": "4. Failure handling", "item": [
            run_create("Transient errors retried (flaky-service)", "Check the status of flaky-service", 200, "completed"),
            run_create("Timeout (slow-service)", "Check the status of slow-service", 200, "completed",
                       desc="Tool times out on every attempt; LLM receives a tool_timeout observation. Takes ~3x TOOL_TIMEOUT_S."),
            run_create("Bad tool output (broken-service)", "Check the status of broken-service", 200, "completed"),
            req("History (see tool errors)", "GET", "/runs/{{run_id}}/history", tests=test_script(expect_status(200))),
        ]},
        {"name": "5. Execution limits", "item": [
            run_create("Step limit (max_steps=1)", "Check the status of payments-api", 200, "limit_exceeded", {"max_steps": 1}),
            run_create("Loop detection", "loop on auth-service status", 200, "limit_exceeded"),
            req("Trace (error spans)", "GET", "/runs/{{run_id}}/trace", tests=test_script(expect_status(200))),
        ]},
        {"name": "6. Inspect", "item": [
            req("List runs", "GET", "/runs?limit=20", tests=test_script(expect_status(200))),
            req("All incidents", "GET", "/incidents", tests=test_script(expect_status(200))),
            req("Unknown run -> 404", "GET", "/runs/does-not-exist", tests=test_script(expect_status(404))),
        ]},
    ],
}

environment = {
    "name": "Ops Agent Harness - local",
    "values": [{"key": "base_url", "value": "http://localhost:8000", "enabled": True}],
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "ops-agent-harness.postman_collection.json").write_text(json.dumps(collection, indent=2))
    (OUT / "ops-agent-harness.postman_environment.json").write_text(json.dumps(environment, indent=2))
    print(f"written to {OUT}")
