"""Composition root shared by the API and the CLI (one place that wires the harness together)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.config import Settings
from app.core.runner import AgentRunner
from app.llm.base import LLMClient
from app.llm.factory import build_llm
from app.observability.tracer import Tracer
from app.storage.db import Database
from app.storage.repository import RunRepository
from app.tools.registry import ToolRegistry, build_default_registry


@dataclass
class Container:
    settings: Settings
    db: Database
    repo: RunRepository
    registry: ToolRegistry
    llm: LLMClient
    tracer: Tracer
    runner: AgentRunner

    async def close(self) -> None:
        await self.db.dispose()


async def build_container(
    settings: Settings,
    llm: LLMClient | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Container:
    """Build all components. Tests inject a scripted ``llm`` and fake ``clock`` / ``sleep``."""
    db = Database(settings.database_url)
    await db.init()
    repo = RunRepository(db.session_factory)
    registry = build_default_registry(settings)
    llm = llm or build_llm(settings)
    tracer = Tracer(repo, settings.otel_exporter_otlp_endpoint)
    runner = AgentRunner(
        settings=settings, repo=repo, registry=registry, llm=llm, tracer=tracer, clock=clock, sleep=sleep
    )
    return Container(settings, db, repo, registry, llm, tracer, runner)
