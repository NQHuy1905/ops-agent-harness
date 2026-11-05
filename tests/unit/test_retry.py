from __future__ import annotations

import pytest

from app.core.retry import retry_async


class Transient(Exception):
    pass


async def _run(fn, retries=2, retry_on=(Transient,)):
    delays: list[float] = []
    seen: list[tuple[int, BaseException]] = []

    async def fake_sleep(d: float) -> None:
        delays.append(d)

    result = await retry_async(
        fn,
        retries=retries,
        base_delay=0.5,
        retry_on=retry_on,
        sleep=fake_sleep,
        on_retry=lambda n, e: seen.append((n, e)),
    )
    return result, delays, seen


async def test_success_first_try():
    async def fn():
        return "ok"

    result, delays, seen = await _run(fn)
    assert (result, delays, seen) == ("ok", [], [])


async def test_transient_then_success():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise Transient(str(calls))
        return "ok"

    result, delays, seen = await _run(fn)
    assert result == "ok" and calls == 3
    assert [n for n, _ in seen] == [1, 2]
    assert delays == [0.5, 1.0]


async def test_exhaustion_reraises_last():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise Transient(str(calls))

    with pytest.raises(Transient, match="3"):
        await _run(fn, retries=2)
    assert calls == 3


async def test_non_retryable_not_retried():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await _run(fn)
    assert calls == 1


async def test_zero_retries_single_attempt():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise Transient()

    with pytest.raises(Transient):
        await _run(fn, retries=0)
    assert calls == 1
