"""Unit tests for the tool layer (validation, mock behaviour, registry, specs)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.core.errors import (
    ToolExecutionError,
    ToolInputValidationError,
    ToolNotFoundError,
    ToolOutputValidationError,
    ToolTransientError,
)
from app.storage.db import Database
from app.storage.repository import RunRepository
from app.tools.base import ToolContext
from app.tools.get_service_status import reset_fault_state
from app.tools.registry import ToolRegistry, build_default_registry

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture
async def settings(tmp_path):
    return Settings(database_url=f"sqlite+aiosqlite:///{tmp_path}/t.db", mock_data_dir=DATA_DIR, tool_timeout_s=0.05)


@pytest.fixture
async def repo(settings):
    db = Database(settings.database_url)
    await db.init()
    yield RunRepository(db.session_factory)
    await db.dispose()


@pytest.fixture
def registry(settings):
    reset_fault_state()
    return build_default_registry(settings)


def make_ctx(repo, settings, call_id="call-1", run_id="run-1"):
    return ToolContext(run_id=run_id, tool_call_id=call_id, repo=repo, settings=settings)


async def call(registry, name, raw, ctx):
    tool = registry.get(name)
    out = await tool.run(tool.validate_input(raw), ctx)
    return tool.validate_output(out)


# ------------------------------------------------------------ input validation
async def test_valid_inputs(registry):
    registry.get("search_knowledge_base").validate_input({"query": "  payments  "})
    registry.get("get_service_status").validate_input({"service_name": "payments-api"})
    registry.get("create_incident").validate_input(
        {"title": "Payments slow", "description": "p95 latency above 1s", "severity": "high"}
    )
    assert registry.get("search_knowledge_base").validate_input({"query": "  x "}).query == "x"


@pytest.mark.parametrize(
    "tool,raw,fragment",
    [
        ("search_knowledge_base", {"query": "a", "limit": 3}, "limit"),
        ("search_knowledge_base", {"query": "   "}, "query"),
        ("get_service_status", {"service_name": "Payments API"}, "service_name"),
        (
            "create_incident",
            {"title": "Payments slow", "description": "p95 latency above 1s", "severity": "urgent"},
            "severity: Input should be 'low', 'medium', 'high' or 'critical'",
        ),
    ],
)
async def test_invalid_inputs_rejected(registry, tool, raw, fragment):
    with pytest.raises(ToolInputValidationError) as ei:
        registry.get(tool).validate_input(raw)
    assert fragment in ei.value.message


async def test_non_dict_input_rejected(registry):
    with pytest.raises(ToolInputValidationError):
        registry.get("search_knowledge_base").validate_input("payments")  # type: ignore[arg-type]


# ------------------------------------------------------------ knowledge base
async def test_kb_ranks_payments_runbook(registry, repo, settings):
    ctx = make_ctx(repo, settings)
    out = await call(registry, "search_knowledge_base", {"query": "payments latency"}, ctx)
    assert out["results"][0]["service"] == "payments-api"
    assert len(out["results"]) <= 5 and out["total_matches"] >= 1
    assert len(out["results"][0]["snippet"]) <= 203


async def test_kb_no_match(registry, repo, settings):
    out = await call(registry, "search_knowledge_base", {"query": "zzzzqqq"}, make_ctx(repo, settings))
    assert out == {"results": [], "total_matches": 0}


def test_kb_dataset_shape():
    docs = json.loads((DATA_DIR / "knowledge_base.json").read_text())
    assert 15 <= len(docs) <= 20
    assert all({"id", "title", "service", "tags", "content"} <= d.keys() for d in docs)


# ------------------------------------------------------------ service status
async def test_status_ok(registry, repo, settings):
    ctx = make_ctx(repo, settings)
    out = await call(registry, "get_service_status", {"service_name": "checkout-web"}, ctx)
    assert out["status"] == "outage" and out["checked_at"]


async def test_status_unknown_service(registry, repo, settings):
    tool = registry.get("get_service_status")
    with pytest.raises(ToolExecutionError, match="service 'nope-svc' not found"):
        await tool.run(tool.validate_input({"service_name": "nope-svc"}), make_ctx(repo, settings))


async def test_flaky_fails_twice_then_succeeds_per_call_id(registry, repo, settings):
    tool = registry.get("get_service_status")
    args = tool.validate_input({"service_name": "flaky-service"})
    ctx = make_ctx(repo, settings, "c1")
    for _ in range(2):
        with pytest.raises(ToolTransientError):
            await tool.run(args, ctx)
    assert tool.validate_output(await tool.run(args, ctx))["status"] == "degraded"
    # a new tool_call_id starts its own failure budget
    with pytest.raises(ToolTransientError):
        await tool.run(args, make_ctx(repo, settings, "c2"))


async def test_broken_service_fails_output_validation(registry, repo, settings):
    tool = registry.get("get_service_status")
    raw = await tool.run(tool.validate_input({"service_name": "broken-service"}), make_ctx(repo, settings))
    with pytest.raises(ToolOutputValidationError):
        tool.validate_output(raw)


async def test_slow_service_exceeds_timeout(registry, repo, settings):
    tool = registry.get("get_service_status")
    args = tool.validate_input({"service_name": "slow-service"})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(tool.run(args, make_ctx(repo, settings)), settings.tool_timeout_s)


# ------------------------------------------------------------ create_incident
async def test_create_incident_idempotent(registry, repo, settings):
    raw = {
        "title": "Checkout outage",
        "description": "checkout-web returns 5xx for all users",
        "severity": "critical",
    }
    assert registry.get("create_incident").requires_approval is True
    ctx = make_ctx(repo, settings)
    first = await call(registry, "create_incident", raw, ctx)
    second = await call(registry, "create_incident", raw, ctx)
    assert first["deduplicated"] is False and second["deduplicated"] is True
    assert first["incident_id"] == second["incident_id"]
    assert first["status"] == "open"
    assert len(await repo.list_incidents("run-1")) == 1


# ------------------------------------------------------------ registry & specs
def test_registry_unknown_and_duplicate(registry):
    with pytest.raises(ToolNotFoundError, match="unknown tool 'nope'. available: "):
        registry.get("nope")
    with pytest.raises(ValueError):
        registry.register(registry.get("create_incident"))
    assert registry.names() == ["search_knowledge_base", "get_service_status", "create_incident"]
    assert isinstance(ToolRegistry().all(), list)


def _keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            if k != "properties":
                yield from _keys(v)
            else:
                for pv in v.values():
                    yield from _keys(pv)
    elif isinstance(node, list):
        for i in node:
            yield from _keys(i)


def test_specs_openai_format_without_titles(registry):
    specs = registry.llm_tool_specs()
    assert len(specs) == 3
    for s in specs:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "title" not in set(_keys(fn["parameters"]))
        json.dumps(s)
    inc = next(s for s in specs if s["function"]["name"] == "create_incident")
    assert "title" in inc["function"]["parameters"]["properties"]  # real field named 'title' kept
    assert "approval" in inc["function"]["description"].lower()
