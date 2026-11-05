"""Run state enums and the DTOs shared by the runner, API and CLI."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    LIMIT_EXCEEDED = "limit_exceeded"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.LIMIT_EXCEEDED)


class StepType(str, Enum):
    """Ordered execution-history entries. LLM message history is rebuilt from these."""

    LLM_RESPONSE = "llm_response"  # valid assistant message (content and/or tool calls)
    LLM_MALFORMED = "llm_malformed"  # invalid assistant output + corrective feedback
    TOOL_RESULT = "tool_result"  # observation for one tool call
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    FINAL_ANSWER = "final_answer"
    RUN_ERROR = "run_error"  # terminal failure / limit


class ToolCallStatus(str, Enum):
    QUEUED = "queued"
    PENDING_APPROVAL = "pending_approval"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


class ToolResult(BaseModel):
    """Outcome of one tool execution, exactly what the LLM observes (plus metadata)."""

    ok: bool
    data: dict[str, Any] | None = None
    error: dict[str, str] | None = None  # {"code": ..., "message": ...}
    attempts: int = 1
    duration_ms: int = 0

    def observation(self) -> dict[str, Any]:
        return {"ok": True, "data": self.data} if self.ok else {"ok": False, "error": self.error}


class ApprovalDecision(BaseModel):
    tool_call_id: str = Field(..., min_length=1)
    decision: Literal["approve", "reject"]
    edited_args: dict[str, Any] | None = None
    reason: str | None = Field(None, max_length=500)
    decided_by: str = Field("api-user", max_length=100)


class PendingApproval(BaseModel):
    tool_call_id: str
    tool: str
    args: dict[str, Any]


class RunView(BaseModel):
    id: str
    objective: str
    status: RunStatus
    final_answer: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    step_count: int
    max_steps: int
    max_wall_time_s: float
    llm_provider: str
    llm_model: str
    pending_approval: PendingApproval | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class StepView(BaseModel):
    idx: int
    type: StepType
    payload: dict[str, Any]
    created_at: datetime


class SpanView(BaseModel):
    id: str
    parent_id: str | None
    name: str
    kind: str
    status: str
    status_message: str | None = None
    attributes: dict[str, Any]
    start_ns: int
    end_ns: int | None
    duration_ms: float | None = None
    children: list[SpanView] = Field(default_factory=list)
