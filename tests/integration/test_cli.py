"""CLI over the real runner: interactive approvals, detached run + approve command, edits, inspection commands.

The CLI uses the offline demo provider (``LLM_PROVIDER=scripted``) against a temporary SQLite file.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from typer.testing import CliRunner

import app.cli as cli
from app.container import build_container
from app.llm.scripted import ScriptedLLM
from tests.helpers import CREATE_INCIDENT_ARGS, make_settings

INCIDENT_OBJECTIVE = "Open an incident for checkout-web"


@pytest.fixture
def cli_settings(tmp_path, monkeypatch):
    cfg = make_settings(tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: cfg)
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)
    return cfg


def invoke(*args: str, input: str | None = None):
    return CliRunner().invoke(cli.app, list(args), input=input)


async def incidents_in(cfg) -> list:
    c = await build_container(cfg, llm=ScriptedLLM([]))
    try:
        return await c.repo.list_incidents()
    finally:
        await c.close()


def run_id_of(output: str) -> str:
    match = re.search(r"run (\S+)\s+status=", output)
    assert match, output
    return match.group(1)


def tool_call_id_of(output: str) -> str:
    match = re.search(r"tool_call_id=([^)]+)\)", output)
    assert match, output
    return match.group(1)


def test_run_status_check_completes_and_prints_answer(cli_settings):
    result = invoke("run", "Check the status of auth-service")

    assert result.exit_code == 0, result.output
    assert "status=completed" in result.output
    assert "Final answer" in result.output and "auth-service" in result.output


def test_interactive_approve_creates_one_incident(cli_settings):
    result = invoke("run", INCIDENT_OBJECTIVE, input="y\n")

    assert result.exit_code == 0, result.output
    assert "Approve?" in result.output and "status=completed" in result.output
    incidents = asyncio.run(incidents_in(cli_settings))
    assert len(incidents) == 1 and incidents[0].title == "checkout-web degraded"


def test_interactive_reject_with_reason_creates_no_incident(cli_settings):
    result = invoke("run", INCIDENT_OBJECTIVE, input="n\nfalse alarm\n")

    assert result.exit_code == 0, result.output
    assert "status=completed" in result.output
    assert asyncio.run(incidents_in(cli_settings)) == []


def test_auto_approve_flag_skips_prompt(cli_settings):
    result = invoke("run", "--auto-approve", INCIDENT_OBJECTIVE)

    assert result.exit_code == 0, result.output
    assert "Approve?" not in result.output
    assert len(asyncio.run(incidents_in(cli_settings))) == 1


def test_detached_run_then_approve_command(cli_settings):
    paused = invoke("run", "--no-input", INCIDENT_OBJECTIVE)
    assert paused.exit_code == 0, paused.output
    assert "status=awaiting_approval" in paused.output
    run_id, call_id = run_id_of(paused.output), tool_call_id_of(paused.output)
    assert asyncio.run(incidents_in(cli_settings)) == []

    approved = invoke("approve", run_id, call_id)

    assert approved.exit_code == 0, approved.output
    assert "status=completed" in approved.output
    assert len(asyncio.run(incidents_in(cli_settings))) == 1


def test_approve_with_edit_json_uses_edited_values(cli_settings):
    paused = invoke("run", "--no-input", INCIDENT_OBJECTIVE)
    run_id, call_id = run_id_of(paused.output), tool_call_id_of(paused.output)
    # edited_args replaces the whole argument object, so send every required field
    edit = json.dumps({**CREATE_INCIDENT_ARGS, "title": "checkout-web outage (edited)", "severity": "critical"})

    result = invoke("approve", run_id, call_id, "--edit-json", edit)

    assert result.exit_code == 0, result.output
    (incident,) = asyncio.run(incidents_in(cli_settings))
    assert incident.title == "checkout-web outage (edited)" and incident.severity == "critical"


def test_reject_command_leaves_no_incident(cli_settings):
    paused = invoke("run", "--no-input", INCIDENT_OBJECTIVE)
    run_id, call_id = run_id_of(paused.output), tool_call_id_of(paused.output)

    result = invoke("reject", run_id, call_id, "--reason", "not needed")

    assert result.exit_code == 0, result.output
    assert "status=completed" in result.output
    assert asyncio.run(incidents_in(cli_settings)) == []


def test_decision_on_unknown_run_exits_non_zero(cli_settings):
    result = invoke("approve", "no-such-run", "call-1")

    assert result.exit_code == 1
    assert "run_not_found" in result.output


def test_show_and_trace_inspect_a_finished_run(cli_settings):
    done = invoke("run", "Check the status of cdn-edge")
    run_id = run_id_of(done.output)

    shown = invoke("show", run_id)
    assert shown.exit_code == 0, shown.output
    assert "LLM -> tool calls: get_service_status" in shown.output
    assert "get_service_status: ok (attempts=1)" in shown.output

    traced = invoke("trace", run_id)
    assert traced.exit_code == 0, traced.output
    spans = json.loads(traced.output)
    assert [s["name"] for s in spans] == ["invoke_agent ops-assistant"]


def test_auto_approve_stops_instead_of_looping_when_resume_fails(cli_settings, monkeypatch):
    from app.core.errors import ApprovalNotPendingError
    from app.core.runner import AgentRunner

    async def refuse(self, run_id, decision):
        raise ApprovalNotPendingError("decision refused for test")

    monkeypatch.setattr(AgentRunner, "resume", refuse)
    result = invoke("run", "--auto-approve", INCIDENT_OBJECTIVE)

    assert result.exit_code == 0, result.output
    assert "Approval error: approval_not_pending" in result.output
    assert "status=awaiting_approval" in result.output
    assert asyncio.run(incidents_in(cli_settings)) == []
