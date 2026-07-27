from __future__ import annotations

import asyncio
from typing import Any

import pytest

from opentulpa.interfaces.telegram.run_output import TelegramRunOutput


class _Client:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    async def send_message(self, **kwargs: Any) -> dict[str, Any]:
        self.messages.append(kwargs)
        return {"ok": True, "result": {"message_id": len(self.messages)}}

    async def edit_message_text(self, **kwargs: Any) -> bool:
        self.edits.append(kwargs)
        return True

    async def delete_message(self, **kwargs: Any) -> bool:
        self.deleted.append(kwargs)
        return True


@pytest.mark.asyncio
async def test_run_output_coalesces_tool_bursts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "opentulpa.interfaces.telegram.run_output._EDIT_MIN_INTERVAL_SECONDS",
        0.01,
    )
    client = _Client()
    output = TelegramRunOutput(client=client, chat_id=9)  # type: ignore[arg-type]

    await output.delta("Checking")
    await output.tool_started("source_shell")
    await output.tool_started("write_todos")
    await output.tool_started("source_status")
    await asyncio.sleep(0.02)

    assert len(client.edits) == 1
    assert client.edits[0]["text"].endswith("Working: Reviewing OpenTulpa changes...")

    await output.finish("Done")
    assert client.deleted == [{"chat_id": 9, "message_id": 1}]


@pytest.mark.asyncio
async def test_run_output_keeps_latest_text_within_telegram_limit() -> None:
    client = _Client()
    output = TelegramRunOutput(client=client, chat_id=9)  # type: ignore[arg-type]

    await output.delta(("old-" * 1_200) + "LATEST_PROGRESS")

    preview = client.messages[0]["text"]
    assert len(preview) <= 3_500
    assert preview.startswith("[Earlier progress omitted]")
    assert preview.endswith("LATEST_PROGRESS")


@pytest.mark.asyncio
async def test_run_output_preserves_whitespace_between_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "opentulpa.interfaces.telegram.run_output._EDIT_MIN_INTERVAL_SECONDS",
        0,
    )
    client = _Client()
    output = TelegramRunOutput(client=client, chat_id=9)  # type: ignore[arg-type]

    await output.delta("Hello ")
    await output.delta("world")

    assert client.edits[-1]["text"] == "Hello world"
