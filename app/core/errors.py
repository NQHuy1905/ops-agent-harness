"""Typed error taxonomy for the harness.

Every error carries a stable machine-readable ``code``, whether it is ``retryable``
(transport-level retry is allowed), and the HTTP status used when it surfaces at the API.
"""

from __future__ import annotations


class HarnessError(Exception):
    code: str = "harness_error"
    retryable: bool = False
    http_status: int = 500

    def __init__(self, message: str = "", **details: object) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


# ---------------------------------------------------------------- tools
class ToolError(HarnessError):
    code = "tool_error"


class ToolNotFoundError(ToolError):
    code = "tool_not_found"
    http_status = 404


class ToolInputValidationError(ToolError):
    code = "tool_input_invalid"
    http_status = 422


class ToolOutputValidationError(ToolError):
    code = "tool_output_invalid"


class ToolTimeoutError(ToolError):
    code = "tool_timeout"
    retryable = True


class ToolTransientError(ToolError):
    code = "tool_transient_error"
    retryable = True


class ToolExecutionError(ToolError):
    """Permanent tool failure (e.g. resource not found). Not retried."""

    code = "tool_execution_error"


# ---------------------------------------------------------------- llm
class LLMError(HarnessError):
    code = "llm_error"


class LLMMalformedResponseError(LLMError):
    code = "malformed_llm_response"


class LLMProviderError(LLMError):
    code = "llm_provider_error"
    retryable = True
    http_status = 502


# ---------------------------------------------------------------- limits
class LimitExceededError(HarnessError):
    code = "limit_exceeded"


class StepLimitExceeded(LimitExceededError):
    code = "step_limit_exceeded"


class TimeLimitExceeded(LimitExceededError):
    code = "time_limit_exceeded"


class LoopDetected(LimitExceededError):
    code = "loop_detected"


# ---------------------------------------------------------------- runs / approvals
class RunNotFoundError(HarnessError):
    code = "run_not_found"
    http_status = 404


class RunBusyError(HarnessError):
    code = "run_busy"
    http_status = 409


class ApprovalError(HarnessError):
    code = "approval_error"
    http_status = 409


class ApprovalNotPendingError(ApprovalError):
    code = "approval_not_pending"


class ApprovalMismatchError(ApprovalError):
    code = "approval_mismatch"


class ApprovalArgsInvalidError(ApprovalError):
    code = "approval_args_invalid"
    http_status = 422
