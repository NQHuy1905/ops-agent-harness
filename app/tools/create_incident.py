"""Create an incident in the mock incident system (idempotent, requires human approval)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import Tool, ToolContext


class CreateIncidentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=5, max_length=120, description="Short incident summary, 5-120 chars.")
    description: str = Field(
        min_length=10,
        max_length=2000,
        description="Impact, start time, affected service(s) and suspected cause, 10-2000 chars.",
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity chosen using the severity matrix from the knowledge base."
    )


class CreateIncidentOutput(BaseModel):
    incident_id: str
    status: Literal["open"]
    severity: Literal["low", "medium", "high", "critical"]
    title: str
    created_at: datetime
    deduplicated: bool


class CreateIncidentTool(Tool):
    name = "create_incident"
    description = (
        "Create an incident ticket. This is a side-effecting action and REQUIRES HUMAN APPROVAL before it "
        "runs. Only call it after confirming degradation/outage (or when the user asks), and pick the "
        "severity using the severity matrix (search the knowledge base for it first)."
    )
    input_model = CreateIncidentInput
    output_model = CreateIncidentOutput
    requires_approval = True

    async def run(self, args: BaseModel, ctx: ToolContext) -> dict:
        assert isinstance(args, CreateIncidentInput)
        row, created = await ctx.repo.create_incident(
            args.title,
            args.description,
            args.severity,
            idempotency_key=f"{ctx.run_id}:{ctx.tool_call_id}",
            run_id=ctx.run_id,
        )
        return {
            "incident_id": row.id,
            "status": "open",
            "severity": row.severity,
            "title": row.title,
            "created_at": row.created_at,
            "deduplicated": not created,
        }
