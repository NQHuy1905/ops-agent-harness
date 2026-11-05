"""Shared fixtures: isolated settings/DB per test, fake clock, container and HTTP client factories."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.container import Container, build_container
from app.llm.base import LLMClient
from app.main import create_app
from app.tools.get_service_status import reset_fault_state
from tests.helpers import FakeClock, make_settings, no_sleep

OpenContainer = Callable[..., Awaitable[Container]]
MakeApi = Callable[..., Awaitable[httpx.AsyncClient]]


@pytest.fixture(autouse=True)
def _fresh_fault_counters() -> Iterator[None]:
    """flaky-service counters are module-global; clear them around every test."""
    reset_fault_state()
    yield
    reset_fault_state()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def open_container(settings: Settings, fake_clock: FakeClock) -> AsyncIterator[OpenContainer]:
    """Build a container (temp DB, fake clock, no-op sleep). Closed at teardown."""
    opened: list[Container] = []

    async def _open(llm: LLMClient, cfg: Settings | None = None, clock: FakeClock | None = None) -> Container:
        container = await build_container(
            settings if cfg is None else cfg,
            llm=llm,
            clock=fake_clock if clock is None else clock,
            sleep=no_sleep,
        )
        opened.append(container)
        return container

    yield _open
    for container in opened:
        await container.close()


@pytest.fixture
async def make_api(settings: Settings) -> AsyncIterator[MakeApi]:
    """HTTP client bound to the app via ASGITransport. The lifespan does not run, so state is wired here."""
    clients: list[httpx.AsyncClient] = []

    async def _make(container: Container, cfg: Settings | None = None) -> httpx.AsyncClient:
        app = create_app(settings if cfg is None else cfg, container=container)
        app.state.container = container
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client

    yield _make
    for client in clients:
        await client.aclose()
