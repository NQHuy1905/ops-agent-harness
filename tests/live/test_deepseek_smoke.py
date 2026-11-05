"""Live DeepSeek smoke test; skipped without DEEPSEEK_API_KEY."""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.llm.base import ToolCallsDecision
from app.llm.deepseek_client import DeepSeekClient

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_service_status",
            "description": "Get the current health of one service.",
            "parameters": {
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        },
    }
]


async def test_deepseek_returns_tool_call():
    client = DeepSeekClient(Settings())
    resp = await client.decide(
        [
            {"role": "system", "content": "You are an ops assistant. Use tools; never guess."},
            {"role": "user", "content": "Check the status of payments-api"},
        ],
        TOOLS,
    )
    assert isinstance(resp.decision, ToolCallsDecision)
    assert resp.decision.calls[0].tool == "get_service_status"
