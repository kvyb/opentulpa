from __future__ import annotations

import asyncio

import pytest

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def _mk_runtime(*, debounce: float) -> OpenTulpaLangGraphRuntime:
    return OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        model_name="gpt-4o-mini",
        checkpoint_db_path=".opentulpa/test-debounce.sqlite",
        input_debounce_seconds=debounce,
    )


@pytest.mark.asyncio
async def test_thread_input_coalesces_burst_before_turn_start() -> None:
    runtime = _mk_runtime(debounce=0.12)
    results: list[tuple[str, str, str]] = []

    async def _submit(text: str, delay: float) -> None:
        await asyncio.sleep(delay)
        state, merged = await runtime._begin_thread_turn(thread_id="chat-1", text=text)
        if state is None:
            results.append(("suppressed", text, merged))
            return
        results.append(("active", text, merged))
        await asyncio.sleep(0.01)
        state.turn_lock.release()

    await asyncio.gather(
        _submit("first", 0.0),
        _submit("second", 0.04),
    )

    active = [item for item in results if item[0] == "active"]
    suppressed = [item for item in results if item[0] == "suppressed"]
    assert len(active) == 1
    assert active[0][2] == "first\n\nsecond"
    assert len(suppressed) == 1


@pytest.mark.asyncio
async def test_thread_input_enqueues_while_running_then_coalesces_next_turn() -> None:
    runtime = _mk_runtime(debounce=0.06)
    results: list[tuple[str, str, str]] = []

    async def _submit(text: str, delay: float, hold_seconds: float) -> None:
        await asyncio.sleep(delay)
        state, merged = await runtime._begin_thread_turn(thread_id="chat-2", text=text)
        if state is None:
            results.append(("suppressed", text, merged))
            return
        results.append(("active", text, merged))
        await asyncio.sleep(hold_seconds)
        state.turn_lock.release()

    await asyncio.gather(
        _submit("first", 0.0, 0.25),
        _submit("second", 0.08, 0.01),
        _submit("third", 0.12, 0.01),
    )

    active = [item for item in results if item[0] == "active"]
    suppressed = [item for item in results if item[0] == "suppressed"]
    assert len(active) == 2
    assert active[0][2] == "first"
    assert active[1][2] == "second\n\nthird"
    assert len(suppressed) == 1
