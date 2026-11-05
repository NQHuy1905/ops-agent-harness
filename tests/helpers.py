"""Plain helpers shared by the test suite: fake clock, settings builder, LLM wrapper, payload readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.state import RunView
from app.llm.base import LLMResponse
from app.llm.scripted import ScriptedLLM

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Healthy/degraded services that never trigger fault injection in the mock tools.
PLAIN_SERVICES = (
    "auth-service",
    "orders-db",
    "checkout-web",
    "payments-api",
    "search-api",
    "notification-worker",
    "inventory-api",
    "cdn-edge",
)

CREATE_INCIDENT_ARGS = {
    "title": "Payments latency incident",
    "description": "p95 latency above 1s since 10:00 UTC",
    "severity": "high",
}


class FakeClock:
    """Monotonic-clock stand-in. Time only moves when a test advances it."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def no_sleep(_seconds: float) -> None:
    """Replaces asyncio.sleep so retry back-off costs no wall time."""


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": f"sqlite+aiosqlite:///{tmp_path}/t.db",
        "mock_data_dir": DATA_DIR,
        "tool_timeout_s": 0.05,
        "tool_retry_base_delay_s": 0,
        "max_steps": 8,
        "max_wall_time_s": 30,
        "llm_provider": "scripted",
    }
    values.update(overrides)
    return Settings(**values)


class ClockedLLM:
    """Wraps a ScriptedLLM and advances the fake clock by ``cost_s`` on every decision (simulated latency)."""

    def __init__(self, inner: ScriptedLLM, clock: FakeClock, cost_s: float) -> None:
        self.inner = inner
        self.clock = clock
        self.cost_s = cost_s
        self.provider = inner.provider
        self.model = inner.model

    @property
    def received(self) -> list[list[dict[str, Any]]]:
        return self.inner.received

    async def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.clock.advance(self.cost_s)
        return await self.inner.decide(messages, tools)


def tool_payloads(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """(tool_call_id, observation) for every tool message in an LLM request, in order."""
    return [(m["tool_call_id"], json.loads(m["content"])) for m in messages if m["role"] == "tool"]


def assert_run_error(view: RunView, status: str, code: str) -> None:
    assert view.status.value == status, f"status {view.status.value} (error {view.error_code}: {view.error_detail})"
    assert view.error_code == code
