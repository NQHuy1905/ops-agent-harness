"""Shared async retry helper with exponential backoff (injectable sleep for tests)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int,
    base_delay: float,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call ``fn`` up to ``retries + 1`` times.

    Only exceptions in ``retry_on`` are retried; others propagate immediately. The delay
    before retry ``n`` (1-based) is ``base_delay * 2**(n-1)``. ``on_retry(n, exc)`` runs
    before each sleep. After exhaustion the last exception is re-raised.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except retry_on as exc:
            if attempt >= retries:
                raise
            attempt += 1
            if on_retry is not None:
                on_retry(attempt, exc)
            await sleep(base_delay * 2 ** (attempt - 1))
