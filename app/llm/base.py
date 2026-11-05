"""Provider-agnostic LLM contract: decisions, response envelope and client protocol."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    call_id: str
    tool: str
    args: dict[str, Any]


class ToolCallsDecision(BaseModel):
    kind: Literal["tool_calls"] = "tool_calls"
    calls: list[ToolCallRequest] = Field(min_length=1)
    content: str | None = None


class FinalAnswerDecision(BaseModel):
    kind: Literal["final"] = "final"
    answer: str = Field(min_length=1)


LLMDecision = Annotated[ToolCallsDecision | FinalAnswerDecision, Field(discriminator="kind")]


class LLMUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMResponse(BaseModel):
    decision: LLMDecision
    # OpenAI-format assistant message, replayed verbatim into history.
    assistant_message: dict[str, Any]
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str
    latency_ms: int = 0


class LLMClient(Protocol):
    provider: str
    model: str

    async def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Return the next decision for plain OpenAI chat-completions ``messages``/``tools``."""
        ...
