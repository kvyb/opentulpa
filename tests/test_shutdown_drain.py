from __future__ import annotations

import asyncio

import pytest

from opentulpa.core.shutdown_drain import ShutdownDrain, ShutdownDrainingError


@pytest.mark.asyncio
async def test_shutdown_drain_idle_returns_immediately() -> None:
    drain = ShutdownDrain(timeout_seconds=1)

    assert await drain.drain() is True
    assert drain.status().draining is True
    assert drain.status().active_turns == 0


@pytest.mark.asyncio
async def test_shutdown_drain_waits_for_active_turn() -> None:
    drain = ShutdownDrain(timeout_seconds=1)

    async with drain.active_turn():
        task = asyncio.create_task(drain.drain())
        await asyncio.sleep(0)

        assert not task.done()

    assert await asyncio.wait_for(task, timeout=1) is True
    assert drain.status().active_turns == 0


@pytest.mark.asyncio
async def test_shutdown_drain_rejects_new_turn_after_draining() -> None:
    drain = ShutdownDrain(timeout_seconds=1)
    drain.start_draining()

    with pytest.raises(ShutdownDrainingError):
        async with drain.active_turn():
            raise AssertionError("turn should not start")
