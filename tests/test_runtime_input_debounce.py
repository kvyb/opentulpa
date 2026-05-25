from __future__ import annotations

import asyncio

import pytest

from opentulpa.agent.runtime_input import ThreadInputCoordinator


def _mk_coordinator(*, debounce: float) -> ThreadInputCoordinator:
    return ThreadInputCoordinator(debounce_seconds=debounce)


@pytest.mark.asyncio
async def test_thread_input_enqueues_burst_before_turn_start() -> None:
    coordinator = _mk_coordinator(debounce=0.12)
    results: list[tuple[str, str, str]] = []

    async def _submit(text: str, delay: float) -> None:
        await asyncio.sleep(delay)
        state, merged = await coordinator.begin_turn(thread_id="chat-1", text=text)
        if state is None:
            results.append(("suppressed", text, merged))
            return
        results.append(("active", text, merged))
        await asyncio.sleep(0.01)
        ThreadInputCoordinator.end_turn(state)

    await asyncio.gather(
        _submit("first", 0.0),
        _submit("second", 0.04),
    )

    active = [item for item in results if item[0] == "active"]
    suppressed = [item for item in results if item[0] == "suppressed"]
    assert len(active) == 2
    assert active[0][2] == "first"
    assert active[1][2] == "second"
    assert suppressed == []


@pytest.mark.asyncio
async def test_thread_input_enqueues_while_running() -> None:
    coordinator = _mk_coordinator(debounce=0.06)
    results: list[tuple[str, str, str]] = []

    async def _submit(text: str, delay: float, hold_seconds: float) -> None:
        await asyncio.sleep(delay)
        state, merged = await coordinator.begin_turn(thread_id="chat-2", text=text)
        if state is None:
            results.append(("suppressed", text, merged))
            return
        results.append(("active", text, merged))
        await asyncio.sleep(hold_seconds)
        ThreadInputCoordinator.end_turn(state)

    await asyncio.gather(
        _submit("first", 0.0, 0.25),
        _submit("second", 0.08, 0.01),
        _submit("third", 0.12, 0.01),
    )

    active = [item for item in results if item[0] == "active"]
    suppressed = [item for item in results if item[0] == "suppressed"]
    assert len(active) == 3
    assert active[0][2] == "first"
    assert active[1][2] == "second"
    assert active[2][2] == "third"
    assert suppressed == []


@pytest.mark.asyncio
async def test_thread_input_steering_drain_suppresses_consumed_request() -> None:
    coordinator = _mk_coordinator(debounce=0.0)
    results: list[tuple[str, str, str]] = []

    async def _submit(text: str, delay: float, hold_seconds: float) -> None:
        await asyncio.sleep(delay)
        state, active_text = await coordinator.begin_turn(thread_id="chat-3", text=text)
        if state is None:
            results.append(("suppressed", text, active_text))
            return
        results.append(("active", text, active_text))
        await asyncio.sleep(hold_seconds)
        ThreadInputCoordinator.end_turn(state)

    first = asyncio.create_task(_submit("first", 0.0, 0.12))
    second = asyncio.create_task(_submit("please use this detail", 0.02, 0.0))
    await asyncio.sleep(0.06)

    drained = await coordinator.drain_steering_inputs(thread_id="chat-3")
    await asyncio.gather(first, second)

    assert drained == ["please use this detail"]
    assert results == [
        ("active", "first", "first"),
        ("suppressed", "please use this detail", ""),
    ]
