from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from opentulpa.client import config, tui
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
        collapsed = "".join(item[1] for item in interface._tool_activity())  # noqa: SLF001
        assert "▸ 1 tool" in collapsed
        assert "● Read File  workspace/notes.txt" in collapsed
        assert "input" not in collapsed

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
        collapsed = "".join(item[1] for item in interface._tool_activity())  # noqa: SLF001
        assert "✓ Read File  workspace/notes.txt" in collapsed
        assert await interface._command("/tool 1") is True  # noqa: SLF001
        expanded = "".join(item[1] for item in interface._tool_activity())  # noqa: SLF001
        assert "▾ 1 tool" in expanded
        assert "call  call-1" in expanded
        assert 'input {"path": "workspace/notes.txt"}' in expanded
        assert 'result{"data": "hello","status": "ok"}' in expanded
    finally:
        await interface.client.aclose()


@pytest.mark.asyncio
async def test_tui_compacts_tool_history_and_ignores_stream_fragments(
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
        for sequence, data in enumerate(
            (
                {"name": "web_search", "call_id": "call-1", "arguments": {"query": "cats"}},
                {"name": "", "call_id": "None", "arguments": {"query": "cats"}},
                {
                    "name": "content_fetch",
                    "call_id": "call-2",
                    "arguments": {"url": "https://example.com"},
                },
            ),
            start=1,
        ):
            interface._render(  # noqa: SLF001
                ClientEvent("tool.started", "run-1", sequence, "now", data)
            )

        collapsed = "".join(item[1] for item in interface._tool_activity())  # noqa: SLF001
        assert len(interface._tools) == 2  # noqa: SLF001
        assert "▸ 2 tools" in collapsed
        assert "Content Fetch  https://example.com" in collapsed
        assert "Web Search" not in collapsed

        assert await interface._command("/tool") is True  # noqa: SLF001
        expanded = "".join(item[1] for item in interface._tool_activity())  # noqa: SLF001
        assert "Web Search  cats" in expanded
        assert "Content Fetch  https://example.com" in expanded
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


@pytest.mark.asyncio
async def test_tui_accepts_dragged_file_paths_as_attachments(
    tmp_path: Path,
) -> None:
    connection = Connection(
        url="https://tulpa.example",
        token="token",
        thread_id="thread-1",
        credential_storage="file",
    )
    image = tmp_path / "image with spaces.png"
    image.write_bytes(b"png")
    interface = tui.OpenTulpaTUI(connection)
    try:
        await interface._dispatch(shlex.quote(str(image)))  # noqa: SLF001
    finally:
        await interface.client.aclose()

    assert interface.attachments == [image.resolve()]
    assert "[attached] image with spaces.png" in interface.output.text


def test_tui_parses_multiple_paths_and_file_urls(tmp_path: Path) -> None:
    first = tmp_path / "first image.png"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"png")
    second.write_bytes(b"pdf")

    dropped = tui._dropped_files(  # noqa: SLF001
        f"{first.as_uri()} {shlex.quote(str(second))}"
    )

    assert dropped == [first.resolve(), second.resolve()]


@pytest.mark.asyncio
async def test_tui_creates_lists_and_switches_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_CLIENT_CONFIG", str(tmp_path / "client.json"))
    monkeypatch.setattr(config, "_keyring_set", lambda account, token: False)
    connection = config.save_connection(
        "https://tulpa.example",
        "owner-secret",
        thread_id="thread-main",
    )
    interface = tui.OpenTulpaTUI(connection)
    try:
        assert await interface._command("/new Research") is True  # noqa: SLF001
        assert interface.session_name == "Research"
        assert await interface._command("/sessions") is True  # noqa: SLF001
        assert "Main" in interface.output.text
        assert "Research" in interface.output.text
        interface.attachments.append(tmp_path / "queued.png")
        assert await interface._command("/session 1") is True  # noqa: SLF001
    finally:
        await interface.client.aclose()

    assert interface.session_name == "Main"
    assert interface.connection.thread_id == "thread-main"
    assert interface.attachments == []


@pytest.mark.asyncio
async def test_tui_approval_card_has_clickable_single_submission_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Connection(
        url="https://tulpa.example",
        token="token",
        thread_id="thread-1",
        credential_storage="file",
    )
    interface = tui.OpenTulpaTUI(connection)
    scheduled: list[Any] = []
    monkeypatch.setattr(interface.app, "create_background_task", scheduled.append)
    interface._remember_approval(  # noqa: SLF001
        "run-1",
        {
            "approval_id": "approval-1",
            "tool_name": "integration_invoke",
            "description": "Send the prepared message",
            "arguments": {"action": "send"},
            "allowed_decisions": ["approve", "reject", "edit"],
        },
    )
    interface.busy = False
    actions = interface._approval_actions()  # noqa: SLF001
    approve = next(fragment for fragment in actions if "APPROVE" in fragment[1])
    reject = next(fragment for fragment in actions if "REJECT" in fragment[1])
    mouse_up = MouseEvent(
        position=Point(x=1, y=1),
        event_type=MouseEventType.MOUSE_UP,
        button=MouseButton.LEFT,
        modifiers=frozenset(),
    )

    try:
        assert interface._show_approval_panel() is True  # noqa: SLF001
        assert len(approve) == 3
        assert len(reject) == 3
        approve[2](mouse_up)
        approve[2](mouse_up)

        assert interface._approval_click_pending == {"approval-1"}  # noqa: SLF001
        assert len(scheduled) == 1
        assert interface._show_approval_panel() is False  # noqa: SLF001
    finally:
        for coroutine in scheduled:
            coroutine.close()
        await interface.client.aclose()


@pytest.mark.asyncio
async def test_tui_clicked_approval_uses_existing_resume_command_and_unlocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Connection(
        url="https://tulpa.example",
        token="token",
        thread_id="thread-1",
        credential_storage="file",
    )
    interface = tui.OpenTulpaTUI(connection)
    calls: list[tuple[str, list[str]]] = []

    async def approval_command(command: str, values: list[str]) -> None:
        calls.append((command, values))

    monkeypatch.setattr(interface, "_approval_command", approval_command)
    interface._approval_click_pending.add("approval-1")  # noqa: SLF001
    try:
        await interface._run_clicked_approval("/reject", "approval-1")  # noqa: SLF001
    finally:
        await interface.client.aclose()

    assert calls == [("/reject", ["approval-1"])]
    assert interface._approval_click_pending == set()  # noqa: SLF001
