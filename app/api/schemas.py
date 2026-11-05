"""HTTP request / response models (domain DTOs live in app.core.state)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.core.state import RunView, SpanView, StepView


class RunCreate(BaseModel):
    objective: str = Field(..., min_length=3, max_length=2000, examples=["Check the status of payments-api"])
    max_steps: int | None = Field(None, ge=1, le=50, description="Override MAX_STEPS for this run")
    max_wall_time_s: float | None = Field(None, gt=0, le=600, description="Override MAX_WALL_TIME_S")


class ToolCallView(BaseModel):
    call_id: str
    step_idx: int
    seq: int
    tool: str
    args: dict[str, Any]
    status: str
    result: dict[str, Any] | None
    error_type: str | None
    attempts: int
    duration_ms: int


class ApprovalView(BaseModel):
    call_id: str
    decision: str
    edited_args: dict[str, Any] | None
    reason: str | None
    decided_by: str
    decided_at: datetime


class RunHistory(BaseModel):
    run: RunView
    steps: list[StepView]
    tool_calls: list[ToolCallView]
    approvals: list[ApprovalView]


class RunTrace(BaseModel):
    run_id: str
    spans: list[SpanView]


class IncidentView(BaseModel):
    id: str
    title: str
    description: str
    severity: str
    status: str
    run_id: str | None
    created_at: datetime


class ToolInfo(BaseModel):
    name: str
    description: str
    requires_approval: bool
    parameters: dict[str, Any]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
