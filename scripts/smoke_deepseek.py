"""Manual DeepSeek smoke check: asks for payments-api status and prints the decision.

Usage: DEEPSEEK_API_KEY=... python scripts/smoke_deepseek.py
"""

from __future__ import annotations

import asyncio
import sys

from app.config import Settings
from app.core.errors import HarnessError
from app.llm.deepseek_client import DeepSeekClient

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


async def main() -> int:
    try:
        client = DeepSeekClient(Settings())
        resp = await client.decide(
            [
                {"role": "system", "content": "You are an ops assistant. Use tools; never guess."},
                {"role": "user", "content": "Check the status of payments-api"},
            ],
            TOOLS,
        )
    except HarnessError as exc:
        print(f"FAILED: {exc.message} {exc.details}")
        return 1
    print(resp.model_dump_json(indent=2))
    ok = resp.decision.kind == "tool_calls" and resp.decision.calls[0].tool == "get_service_status"
    print("OK" if ok else "UNEXPECTED: no get_service_status call")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
