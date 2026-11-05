"""Mock service-status lookup with deterministic fault injection for harness testing."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import ToolExecutionError, ToolTransientError
from app.tools.base import Tool, ToolContext

_FLAKY_FAILURES = 2
_status_cache: dict[Path, dict[str, dict]] = {}
_flaky_attempts: dict[tuple[str, str], int] = {}


def reset_fault_state() -> None:
    """Clear fault-injection counters (call between tests)."""
    _flaky_attempts.clear()


def _load_statuses(path: Path) -> dict[str, dict]:
    if path not in _status_cache:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.values() if isinstance(raw, dict) else raw
        _status_cache[path] = {item["name"]: item for item in items}
    return _status_cache[path]


class ServiceStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service_name: str = Field(
        pattern=r"^[a-z0-9-]{2,50}$",
        description="Lowercase service identifier, e.g. 'payments-api', 'checkout-web', 'orders-db'.",
    )


class ServiceStatusOutput(BaseModel):
    name: str
    status: Literal["operational", "degraded", "outage"]
    latency_ms: float = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    region: str
    last_incident: str | None = None
    checked_at: datetime


class GetServiceStatusTool(Tool):
    name = "get_service_status"
    description = (
        "Get the current health of one service: status (operational/degraded/outage), p95 latency in ms, "
        "error rate (0-1), region and last incident. Use it to confirm impact before creating an incident."
    )
    input_model = ServiceStatusInput
    output_model = ServiceStatusOutput

    async def run(self, args: BaseModel, ctx: ToolContext) -> dict:
        assert isinstance(args, ServiceStatusInput)
        name = args.service_name
        statuses = _load_statuses(Path(ctx.settings.mock_data_dir) / "service_statuses.json")
        if name not in statuses:
            raise ToolExecutionError(f"service '{name}' not found")
        data = {**statuses[name], "checked_at": datetime.now(timezone.utc).isoformat()}

        if name == "flaky-service":
            key = (ctx.run_id, ctx.tool_call_id)
            _flaky_attempts[key] = _flaky_attempts.get(key, 0) + 1
            if _flaky_attempts[key] <= _FLAKY_FAILURES:
                attempt = _flaky_attempts[key]
                raise ToolTransientError(f"status backend temporarily unavailable (attempt {attempt})")
        elif name == "slow-service":
            await asyncio.sleep(ctx.settings.tool_timeout_s * 2 + 0.05)
        elif name == "broken-service":
            return {"name": name, "status": "on-fire", "latency_ms": -1}
        return data
