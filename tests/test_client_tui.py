from __future__ import annotations

import pytest

from opentulpa.client import tui
from opentulpa.client.api import ClientEvent
from opentulpa.client.config import Connection


@pytest.mark.asyncio
async def test_tui_renders_stream_once_and_persists_run_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Connection(
        url="https://tulpa.example",
        token="token",
        thread_id="thread-1",
        credential_storage="file",
    )
    updates: list[tuple[str | None, int]] = []

    def update(value: Connection, **changes: object) -> Connection:
        run_id = str(changes.get("last_run_id") or "") or None
        sequence = int(changes.get("last_sequence") or 0)
        updates.append((run_id, sequence))
        return Connection(
            url=value.url,
            token=value.token,
            thread_id=value.thread_id,
            credential_storage=value.credential_storage,
            last_run_id=run_id,
            last_sequence=sequence,
        )

    monkeypatch.setattr(tui, "update_connection", update)
    interface = tui.OpenTulpaTUI(connection)
    try:
        interface._render(ClientEvent("run.started", "run-1", 1, "now", {}))  # noqa: SLF001
        interface._render(  # noqa: SLF001
            ClientEvent("message.delta", "run-1", 2, "now", {"text": "hello"})
        )
        interface._render(  # noqa: SLF001
            ClientEvent("run.completed", "run-1", 3, "now", {"text": "hello"})
        )
    finally:
        await interface.client.aclose()

    assert interface.output.text.count("hello") == 1
    assert interface.state == "connected"
    assert interface.busy is False
    assert updates[-1] == ("run-1", 3)


@pytest.mark.asyncio
async def test_tui_new_thread_attach_and_regenerate_command_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Connection(
        url="https://tulpa.example",
        token="token",
        thread_id="thread-1",
        credential_storage="file",
    )
    monkeypatch.setattr(tui, "update_connection", lambda value, **changes: value)
    interface = tui.OpenTulpaTUI(connection)
    try:
        assert await interface._command("/help") is True  # noqa: SLF001
        assert await interface._command("/regenerate") is False  # noqa: SLF001
        assert "COMMANDS" in interface.output.text
    finally:
        await interface.client.aclose()


@pytest.mark.asyncio
async def test_tui_shows_activity_and_expandable_sanitized_tool_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Connection(
        url="https://tulpa.example",
        token="token",
        thread_id="thread-1",
        credential_storage="file",
    )
    monkeypatch.setattr(tui, "update_connection", lambda value, **changes: value)
    interface = tui.OpenTulpaTUI(connection)
    try:
        interface.busy = True
        interface._render(ClientEvent("run.started", "run-1", 1, "now", {}))  # noqa: SLF001

        assert "Planning next moves" in interface._activity()[1][1]  # noqa: SLF001

        interface._render(  # noqa: SLF001
            ClientEvent(
                "tool.started",
                "run-1",
                2,
                "now",
                {
                    "name": "read_file",
                    "call_id": "call-1",
                    "arguments": {"path": "workspace/notes.txt"},
                },
            )
        )
        assert "▣ 1" in interface.output.text
        assert "Read File  workspace/notes.txt" in interface.output.text
        assert "arguments" not in interface.output.text

        interface._render(  # noqa: SLF001
            ClientEvent(
                "tool.completed",
                "run-1",
                3,
                "now",
                {
                    "name": "read_file",
                    "call_id": "call-1",
                    "ok": True,
                    "result": {"status": "ok", "data": "hello"},
                },
            )
        )
        assert "✓ 1" in interface.output.text
        assert await interface._command("/tool 1") is True  # noqa: SLF001
        assert "arguments" in interface.output.text
        assert '"path": "workspace/notes.txt"' in interface.output.text
        assert '"status": "ok"' in interface.output.text
    finally:
        await interface.client.aclose()


@pytest.mark.asyncio
async def test_tui_reports_empty_completed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Connection(
        url="https://tulpa.example",
        token="token",
        thread_id="thread-1",
        credential_storage="file",
    )
    monkeypatch.setattr(tui, "update_connection", lambda value, **changes: value)
    interface = tui.OpenTulpaTUI(connection)
    try:
        interface._render(ClientEvent("run.started", "run-1", 1, "now", {}))  # noqa: SLF001
        interface._render(ClientEvent("run.completed", "run-1", 2, "now", {}))  # noqa: SLF001
    finally:
        await interface.client.aclose()

    assert "completed without a response" in interface.output.text
