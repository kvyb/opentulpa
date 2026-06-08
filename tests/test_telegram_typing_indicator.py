from __future__ import annotations

import asyncio

import pytest

from opentulpa.interfaces.telegram import relay as relay_module
from opentulpa.interfaces.telegram.relay import _emit_typing_until_done


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def send_chat_action(
        self,
        *,
        chat_id: int | str,
        action: str = "typing",
    ) -> bool:
        self.calls.append((int(chat_id), action))
        return True


@pytest.mark.asyncio
async def test_emit_typing_until_done_sends_typing_action() -> None:
    relay_module._TELEGRAM_TYPING_LAST_SENT_AT.clear()
    client = _FakeClient()
    stop = asyncio.Event()

    async def _stop_soon() -> None:
        await asyncio.sleep(0.01)
        stop.set()

    stopper = asyncio.create_task(_stop_soon())
    await _emit_typing_until_done(client=client, chat_id=42, stop_event=stop)  # type: ignore[arg-type]
    await stopper

    assert client.calls
    assert all(chat_id == 42 for chat_id, _ in client.calls)
    assert all(action == "typing" for _, action in client.calls)


@pytest.mark.asyncio
async def test_emit_typing_until_done_throttles_concurrent_loops_for_same_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay_module._TELEGRAM_TYPING_LAST_SENT_AT.clear()
    monkeypatch.setattr(relay_module, "TELEGRAM_TYPING_MIN_INTERVAL_SECONDS", 60.0)
    client = _FakeClient()
    stop_a = asyncio.Event()
    stop_b = asyncio.Event()

    async def _stop_soon() -> None:
        await asyncio.sleep(0.01)
        stop_a.set()
        stop_b.set()

    stopper = asyncio.create_task(_stop_soon())
    await asyncio.gather(
        _emit_typing_until_done(client=client, chat_id=42, stop_event=stop_a),  # type: ignore[arg-type]
        _emit_typing_until_done(client=client, chat_id=42, stop_event=stop_b),  # type: ignore[arg-type]
    )
    await stopper

    assert client.calls == [(42, "typing")]
