"""Keyboard-first Grok-style terminal interface for OpenTulpa."""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from opentulpa.client.api import ClientEvent, OpenTulpaClient, RemoteError
from opentulpa.client.config import (
    ClientConfigError,
    Connection,
    clear_connection,
    update_connection,
)
from opentulpa.client.sessions import (
    SessionCatalogError,
    create_session,
    list_sessions,
    switch_session,
)

_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


@dataclass(slots=True)
class _ToolView:
    number: int
    call_id: str
    name: str
    arguments: Any
    ok: bool | None = None
    result: Any = None
    expanded: bool = False


class _TranscriptLexer(Lexer):
    def lex_document(self, document: Document) -> Any:
        lines = document.lines

        def get_line(lineno: int) -> list[tuple[str, str]]:
            line = lines[lineno]
            stripped = line.lstrip()
            if line == "YOU":
                style = "class:transcript.user-label"
            elif line == "OPENTULPA":
                style = "class:transcript.agent-label"
            elif line.startswith("│"):
                style = "class:transcript.user"
            elif stripped.startswith("▣"):
                style = "class:transcript.tool-running"
            elif stripped.startswith("✓"):
                style = "class:transcript.tool-ok"
            elif stripped.startswith("×"):
                style = "class:transcript.tool-error"
            elif line.startswith("      "):
                style = "class:transcript.tool-detail"
            elif stripped.startswith("["):
                style = "class:transcript.notice"
            else:
                style = "class:transcript.text"
            return [(style, line)]

        return get_line


class OpenTulpaTUI:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.client = OpenTulpaClient(connection)
        self.output = TextArea(
            text="",
            read_only=True,
            focusable=False,
            scrollbar=True,
            wrap_lines=True,
            lexer=_TranscriptLexer(),
        )
        self.input = TextArea(
            height=1,
            prompt=FormattedText([("class:prompt", "> ")]),
            multiline=False,
            wrap_lines=True,
        )
        self.state = "connecting"
        self.model = ""
        self.base_url = ""
        self.session_name = ""
        self.server_ready = False
        self.busy = False
        self.active_run_id: str | None = None
        self.last_sequence = 0
        self.attachments: list[Path] = []
        self.approvals: dict[str, dict[str, Any]] = {}
        self._approval_click_pending: set[str] = set()
        self.approval_notifications: dict[str, int] = {}
        self.notification_approvals: dict[int, set[str]] = {}
        self.notification_cursor = 0
        self.seen_run_ids: set[str] = set()
        self.runs_with_text: set[str] = set()
        self._transcript: list[str | _ToolView] = []
        self._tools: list[_ToolView] = []
        self._active_tools: dict[str, _ToolView] = {}
        self._phase = "idle"
        self._spinner_index = 0
        self._closed = False
        self._run_task: asyncio.Task[None] | None = None
        self._notification_task: asyncio.Task[None] | None = None
        self._restore_task: asyncio.Task[None] | None = None
        self._spinner_task: asyncio.Task[None] | None = None
        self.app: Application[None] = Application(
            layout=Layout(self._layout(), focused_element=self.input),
            key_bindings=self._bindings(),
            style=_STYLE,
            full_screen=True,
            mouse_support=True,
        )

    async def run(self) -> None:
        try:
            status = await self.client.host_status()
            config = status.get("config") or {}
            self.model = str(config.get("model") or "")
            self.base_url = str(config.get("base_url") or "")
            runtime = status.get("runtime") or {}
            if not status.get("configured") or runtime.get("status") != "ready":
                self.state = "setup"
            else:
                self.server_ready = True
                self.state = "connected"
        except RemoteError:
            self.state = "reconnecting"
        try:
            sessions = list_sessions(self.connection)
            current = next(item for item in sessions if item.thread_id == self.connection.thread_id)
            self.session_name = current.name
        except (SessionCatalogError, StopIteration) as exc:
            self._append(f"[session catalog unavailable: {exc}]\n")
        self._append("OPENTULPA\n")
        self._append(
            f"{self.model + '  ·  ' if self.model else ''}{self.connection.url}\n"
            f"session {self.session_name or self.connection.thread_id}\n"
            "Type /help for commands.\n\n"
        )
        if self.state == "setup":
            self._append(f"[runtime is not ready; configure it at {self.connection.url}/_host]\n\n")
        if self.connection.last_run_id:
            self.busy = True
            self.state = "reconnecting"
        self._notification_task = asyncio.create_task(self._poll_notifications())
        self._restore_task = asyncio.create_task(self._restore_last_run())
        self._spinner_task = asyncio.create_task(self._animate_activity())
        try:
            await self.app.run_async()
        finally:
            self._closed = True
            tasks = (
                self._run_task,
                self._notification_task,
                self._restore_task,
                self._spinner_task,
            )
            for task in tasks:
                if task is not None:
                    task.cancel()
            for task in tasks:
                if task is not None:
                    with suppress(asyncio.CancelledError):
                        await task
            await self.client.aclose()

    def _layout(self) -> HSplit:
        return HSplit(
            [
                Window(
                    FormattedTextControl(self._header),
                    height=1,
                    style="class:header",
                ),
                Window(height=1, char="-", style="class:rule"),
                self.output,
                Window(height=1, char="-", style="class:rule"),
                Window(
                    FormattedTextControl(self._activity),
                    height=1,
                    style="class:activity",
                ),
                ConditionalContainer(
                    HSplit(
                        [
                            Window(
                                FormattedTextControl(self._approval_summary),
                                height=2,
                                wrap_lines=True,
                                style="class:approval",
                            ),
                            Window(
                                FormattedTextControl(self._approval_actions),
                                height=1,
                                style="class:approval.actions",
                            ),
                        ]
                    ),
                    filter=Condition(self._show_approval_panel),
                ),
                Window(
                    FormattedTextControl(self._status),
                    height=1,
                    style="class:status",
                ),
                self.input,
                Window(
                    FormattedTextControl(
                        FormattedText(
                            [
                                ("class:hint", " enter send  "),
                                ("class:key", "ctrl-c"),
                                ("class:hint", " cancel  "),
                                ("class:key", "ctrl-t"),
                                ("class:hint", " tool details  "),
                                ("class:key", "/help"),
                            ]
                        )
                    ),
                    height=1,
                ),
            ]
        )

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(_: Any) -> None:
            text = self.input.text.strip()
            self.input.text = ""
            if text:
                self.app.create_background_task(self._dispatch(text))

        @bindings.add("c-c")
        def cancel(_: Any) -> None:
            if self.active_run_id:
                self.app.create_background_task(self._cancel_active())
            else:
                self.app.exit()

        @bindings.add("c-d")
        def exit_app(_: Any) -> None:
            self.app.exit()

        @bindings.add("c-t")
        def toggle_tool(_: Any) -> None:
            self._toggle_tool()

        return bindings

    def _header(self) -> FormattedText:
        model = f"  {self.model}" if self.model else ""
        session = self.session_name or self.connection.thread_id
        return FormattedText(
            [
                ("class:brand", " OPENTULPA"),
                ("class:header", model),
                ("class:thread", f"  {session}"),
            ]
        )

    def _activity(self) -> FormattedText:
        if not self.busy or self.state not in {"working", "reconnecting"}:
            return FormattedText([("class:activity", "")])
        spinner = _SPINNER_FRAMES[self._spinner_index % len(_SPINNER_FRAMES)]
        if self._active_tools:
            latest = next(reversed(self._active_tools.values()))
            label = _tool_label(latest.name, latest.arguments)
        elif self.state == "reconnecting":
            label = "Reconnecting to the run"
        elif self._phase == "responding":
            label = "Writing response"
        else:
            label = "Planning next moves"
        return FormattedText(
            [
                ("class:activity.spinner", f" {spinner} "),
                ("class:activity", label),
            ]
        )

    def _show_approval_panel(self) -> bool:
        return bool(self.approvals) and not self.busy and not self._approval_click_pending

    def _approval_summary(self) -> FormattedText:
        current = self._current_approval()
        if current is None:
            return FormattedText()
        approval_id, approval = current
        tool_name = str(approval.get("tool_name") or "action").replace("_", " ")
        description = " ".join(str(approval.get("description") or "").split())
        if len(description) > 100:
            description = f"{description[:97]}..."
        position = next(
            index for index, key in enumerate(self.approvals, start=1) if key == approval_id
        )
        return FormattedText(
            [
                ("class:approval.title", f" APPROVAL REQUIRED  {position}/{len(self.approvals)}\n"),
                ("class:approval.tool", f" {tool_name}"),
                ("class:approval.description", f"  {description}" if description else ""),
            ]
        )

    def _approval_actions(self) -> FormattedText:
        current = self._current_approval()
        if current is None:
            return FormattedText()
        approval_id, approval = current
        allowed = {str(value) for value in approval.get("allowed_decisions") or []}
        fragments: list[tuple[Any, ...]] = [("class:approval.actions", " ")]
        fragments.extend(self._approval_button(approval_id, "/approve", " APPROVE ", allowed))
        fragments.append(("class:approval.actions", "  "))
        fragments.extend(self._approval_button(approval_id, "/reject", " REJECT ", allowed))
        if "edit" in allowed:
            fragments.append(("class:approval.hint", f"   /edit {approval_id} JSON"))
        return FormattedText(fragments)

    def _approval_button(
        self,
        approval_id: str,
        command: str,
        label: str,
        allowed: set[str],
    ) -> list[tuple[Any, ...]]:
        decision = command.removeprefix("/")
        if decision not in allowed:
            return [("class:approval.button.disabled", label)]
        style = (
            "class:approval.button.approve"
            if decision == "approve"
            else "class:approval.button.reject"
        )
        return [(style, label, self._approval_mouse_handler(approval_id, command))]

    def _approval_mouse_handler(
        self,
        approval_id: str,
        command: str,
    ) -> Callable[[MouseEvent], None]:
        def handle(mouse_event: MouseEvent) -> None:
            if (
                mouse_event.event_type != MouseEventType.MOUSE_UP
                or self.busy
                or approval_id in self._approval_click_pending
            ):
                return
            self._approval_click_pending.add(approval_id)
            self._invalidate()
            self.app.create_background_task(self._run_clicked_approval(command, approval_id))

        return handle

    async def _run_clicked_approval(self, command: str, approval_id: str) -> None:
        try:
            await self._approval_command(command, [approval_id])
        finally:
            self._approval_click_pending.discard(approval_id)
            self._invalidate()

    def _current_approval(self) -> tuple[str, dict[str, Any]] | None:
        return next(iter(self.approvals.items()), None)

    def _status(self) -> FormattedText:
        attachment = f"  {len(self.attachments)} attachment(s)" if self.attachments else ""
        approvals = f"  {len(self.approvals)} approval(s)" if self.approvals else ""
        return FormattedText(
            [
                (f"class:state.{self.state}", f" {self.state.upper()} "),
                ("class:status", f"{attachment}{approvals}"),
            ]
        )

    async def _dispatch(self, text: str) -> None:
        dropped = _dropped_files(text)
        if dropped:
            for path in dropped:
                self._attach_path(path)
            return
        if text.startswith("/") and text != "/regenerate" and await self._command(text):
            return
        if self.busy:
            self._append("[busy] Wait for the current run, approve it, or use /cancel.\n")
            return
        self._append(f"YOU\n{_user_block(text)}\n\nOPENTULPA\n")
        self.busy = True
        self.state = "working"
        self._phase = "thinking"
        self._invalidate()
        paths = list(self.attachments)
        self._run_task = asyncio.create_task(self._start_run(text, paths))

    async def _start_run(self, text: str, paths: list[Path]) -> None:
        try:
            file_ids: list[str] = []
            for path in paths:
                self._append(f"[upload] {path.name}\n")
                payload = await self.client.upload(path)
                file_id = str((payload.get("file") or {}).get("id") or "").strip()
                if not file_id:
                    raise RemoteError(f"Upload returned no file ID for {path.name}.")
                file_ids.append(file_id)
            self.attachments = [path for path in self.attachments if path not in paths]
            async for event in self.client.run(
                thread_id=self.connection.thread_id,
                text=text,
                file_ids=file_ids,
            ):
                self._render(event)
        except RemoteError as exc:
            await self._recover_stream(exc)
        finally:
            if self.state == "working" and not self.active_run_id:
                self.busy = False
                self.state = "connected"
            self._invalidate()

    async def _recover_stream(self, error: RemoteError) -> None:
        if not self.active_run_id:
            self._append(f"\n[error] {error}\n\n")
            self.busy = False
            self.state = "error"
            return
        if error.status_code not in {None, 502, 503, 504}:
            self._append(f"\n[stream failed] {error}\n\n")
            self.active_run_id = None
            self.busy = False
            self.state = "error"
            return
        self.state = "reconnecting"
        self._append("\n[connection lost; replaying persisted events]\n")
        self._invalidate()
        while not self._closed and self.active_run_id:
            await asyncio.sleep(1.5)
            try:
                async for event in self.client.run_events(
                    self.active_run_id,
                    after_sequence=self.last_sequence,
                ):
                    self._render(event)
                return
            except RemoteError as exc:
                if exc.status_code not in {None, 502, 503, 504}:
                    self._append(f"\n[replay failed] {exc}\n\n")
                    self.active_run_id = None
                    self.busy = False
                    self.state = "error"
                    return
                continue

    def _render(self, event: ClientEvent) -> None:
        self.active_run_id = event.run_id
        self.last_sequence = max(self.last_sequence, event.sequence)
        self.seen_run_ids.add(event.run_id)
        if event.type == "run.started" or event.terminal:
            try:
                self.connection = update_connection(
                    self.connection,
                    last_run_id=event.run_id,
                    last_sequence=self.last_sequence,
                )
            except ClientConfigError as exc:
                self._append(f"\n[warning: run cursor was not saved] {exc}\n")
        data = event.data
        if event.type == "run.started":
            self.state = "working"
            self._phase = "thinking"
        elif event.type == "message.delta":
            self.runs_with_text.add(event.run_id)
            self._phase = "responding"
            self._append(str(data.get("text") or ""))
        elif event.type == "tool.started":
            self._start_tool(data)
        elif event.type == "tool.completed":
            self._complete_tool(data)
            self._phase = "thinking"
        elif event.type == "artifact.ready":
            self._append(f"[artifact] {data.get('name') or data.get('path') or 'ready'}\n")
        elif event.type == "approval.required":
            self._remember_approval(event.run_id, data)
            self._append("\n[waiting for approval]\n\n")
            self.busy = False
            self.state = "approval"
            self.active_run_id = None
        elif event.type == "run.completed":
            fallback = str(data.get("text") or "")
            if fallback and event.run_id not in self.runs_with_text:
                self._append(fallback)
            elif not fallback and event.run_id not in self.runs_with_text:
                self._append("[the agent completed without a response; use /regenerate or /logs]\n")
            self._append("\n\n")
            self.busy = False
            self.state = "connected"
            self._phase = "idle"
            self._active_tools.clear()
            self.active_run_id = None
        elif event.type == "run.failed":
            self._append(f"\n[failed] {data.get('message') or 'Agent run failed.'}\n\n")
            self.busy = False
            self.state = "error"
            self._phase = "idle"
            self._active_tools.clear()
            self.active_run_id = None
        self._invalidate()

    def _start_tool(self, data: dict[str, Any]) -> None:
        call_id = str(data.get("call_id") or "").strip()
        name = str(data.get("name") or "tool").strip() or "tool"
        tool = _ToolView(
            number=len(self._tools) + 1,
            call_id=call_id or f"{name}:{len(self._tools) + 1}",
            name=name,
            arguments=data.get("arguments") or {},
        )
        self._tools.append(tool)
        self._active_tools[tool.call_id] = tool
        self._transcript.append(tool)
        self._refresh_output()

    def _complete_tool(self, data: dict[str, Any]) -> None:
        call_id = str(data.get("call_id") or "").strip()
        name = str(data.get("name") or "tool").strip() or "tool"
        tool = self._active_tools.pop(call_id, None) if call_id else None
        if tool is None:
            tool = next(
                (item for item in reversed(self._tools) if item.ok is None and item.name == name),
                None,
            )
        if tool is None:
            tool = _ToolView(
                number=len(self._tools) + 1,
                call_id=call_id or f"{name}:{len(self._tools) + 1}",
                name=name,
                arguments={},
            )
            self._tools.append(tool)
            self._transcript.append(tool)
        self._active_tools.pop(tool.call_id, None)
        tool.ok = data.get("ok") is not False
        tool.result = data.get("result") if tool.ok else data.get("error")
        self._refresh_output()

    def _toggle_tool(self, number: int | None = None) -> None:
        if not self._tools:
            self._append("[no tool calls in this session]\n")
            return
        tool = (
            self._tools[-1]
            if number is None
            else next(
                (item for item in self._tools if item.number == number),
                None,
            )
        )
        if tool is None:
            self._append(f"[unknown tool call: {number}]\n")
            return
        tool.expanded = not tool.expanded
        self._refresh_output()

    async def _animate_activity(self) -> None:
        while not self._closed:
            await asyncio.sleep(0.08)
            if self.busy:
                self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
                self._invalidate()

    def _remember_approval(self, run_id: str, data: dict[str, Any]) -> None:
        approval_id = str(data.get("approval_id") or "").strip()
        if not approval_id or approval_id in self.approvals:
            return
        approval = dict(data)
        approval["run_id"] = run_id
        self.approvals[approval_id] = approval
        decisions = ", ".join(str(item) for item in data.get("allowed_decisions") or [])
        arguments = data.get("arguments")
        rendered_arguments = (
            f"Arguments: {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}\n"
            if isinstance(arguments, dict) and arguments
            else ""
        )
        self._append(
            f"\nAPPROVAL {approval_id}\n"
            f"{data.get('tool_name') or 'action'}: {data.get('description') or ''}\n"
            f"{rendered_arguments}"
            f"Decisions: {decisions}\n"
            f"Use /approve {approval_id}, /reject {approval_id}, or /edit {approval_id} JSON.\n"
        )

    async def _command(self, raw: str) -> bool:
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            self._append(f"[command] {exc}\n")
            return True
        if not parts:
            return True
        command = parts[0].casefold()
        if command in {"/quit", "/exit"}:
            self.app.exit()
        elif command == "/help":
            self._append(_HELP)
        elif command == "/new":
            if self.busy:
                self._append("[busy] Finish or cancel the current run first.\n")
            else:
                try:
                    name = " ".join(parts[1:]).strip() or None
                    self.connection = create_session(self.connection, name=name)
                    sessions = list_sessions(self.connection)
                    selected = next(
                        item for item in sessions if item.thread_id == self.connection.thread_id
                    )
                except (ClientConfigError, SessionCatalogError) as exc:
                    self._append(f"[new thread failed] {exc}\n")
                    return True
                self._enter_session(selected.name)
        elif command == "/sessions":
            self._list_sessions()
        elif command == "/session":
            if self.busy:
                self._append("[busy] Finish or cancel the current run first.\n")
            else:
                try:
                    selector = " ".join(parts[1:])
                    self.connection, selected = switch_session(self.connection, selector)
                except (ClientConfigError, SessionCatalogError) as exc:
                    self._append(f"[session switch failed] {exc}\n")
                    return True
                self._enter_session(selected.name)
        elif command == "/attach":
            self._attach(parts[1:])
        elif command == "/approvals":
            self._list_approvals()
        elif command in {"/approve", "/reject", "/edit"}:
            await self._approval_command(command, parts[1:])
        elif command == "/cancel":
            await self._cancel_active()
        elif command == "/logs":
            await self._show_logs()
        elif command in {"/tool", "/tools"}:
            if len(parts) > 2 or (len(parts) == 2 and not parts[1].isdigit()):
                self._append("Usage: /tool [NUMBER]\n")
            else:
                self._toggle_tool(int(parts[1]) if len(parts) == 2 else None)
        elif command == "/settings":
            self._append(
                f"SERVER  {self.connection.url}\n"
                f"SESSION {self.session_name or self.connection.thread_id}\n"
                f"THREAD  {self.connection.thread_id}\n"
                f"MODEL   {self.model or 'not configured'}\n"
                f"API     {self.base_url or 'not configured'}\n"
                f"TOKEN   {self.connection.credential_storage}\n"
            )
        elif command == "/disconnect":
            try:
                clear_connection()
            except ClientConfigError as exc:
                self._append(f"[disconnect failed] {exc}\n")
                return True
            self._append("[disconnected; local credentials removed]\n")
            self.app.exit()
        else:
            return False
        self._invalidate()
        return True

    def _attach(self, values: list[str]) -> None:
        if not values:
            if not self.attachments:
                self._append("Usage: /attach PATH\n")
            else:
                self._append("Attachments:\n" + "".join(f"  {item}\n" for item in self.attachments))
            return
        path = Path(" ".join(values)).expanduser().resolve()
        if not path.is_file():
            self._append(f"[attachment not found] {path}\n")
            return
        self._attach_path(path)

    def _attach_path(self, path: Path) -> None:
        if path not in self.attachments:
            self.attachments.append(path)
            self._append(f"[attached] {path.name}\n")
        else:
            self._append(f"[already attached] {path.name}\n")

    def _list_sessions(self) -> None:
        try:
            sessions = list_sessions(self.connection)
        except SessionCatalogError as exc:
            self._append(f"[sessions unavailable] {exc}\n")
            return
        self._append("SESSIONS\n")
        for index, session in enumerate(sessions, start=1):
            marker = "*" if session.thread_id == self.connection.thread_id else " "
            self._append(f"{marker} {index:<2} {session.name}\n")
        self._append("Use /session NUMBER_OR_NAME or /new [NAME].\n")

    def _enter_session(self, name: str) -> None:
        self.session_name = name
        self.last_sequence = self.connection.last_sequence
        self.active_run_id = None
        self.runs_with_text.clear()
        self.attachments.clear()
        self._tools.clear()
        self._active_tools.clear()
        self._transcript.clear()
        self._refresh_output()
        self._append(
            "OPENTULPA\n"
            f"{self.model + '  ·  ' if self.model else ''}{name}\n"
            f"thread {self.connection.thread_id}\n"
            "[session context restored]\n\n"
        )
        if self.connection.last_run_id:
            self.busy = True
            self.state = "reconnecting"
            self._restore_task = asyncio.create_task(self._restore_last_run())
        else:
            self.busy = False
            self.state = "connected"

    def _list_approvals(self) -> None:
        if not self.approvals:
            self._append("No pending approvals.\n")
            return
        for approval_id, approval in self.approvals.items():
            self._append(
                f"{approval_id}  {approval.get('tool_name') or 'action'}  "
                f"{approval.get('description') or ''}\n"
            )

    async def _approval_command(self, command: str, values: list[str]) -> None:
        if not self.approvals:
            self._append("No pending approvals.\n")
            return
        approval_id = values[0] if values else next(iter(self.approvals))
        approval = self.approvals.get(approval_id)
        if approval is None:
            self._append(f"Unknown approval: {approval_id}\n")
            return
        decision = command.removeprefix("/")
        edited_arguments: dict[str, Any] | None = None
        if decision == "edit":
            if len(values) < 2:
                self._append(f'Usage: /edit {approval_id} \'{{"argument": "value"}}\'\n')
                return
            try:
                parsed = json.loads(" ".join(values[1:]))
            except ValueError:
                self._append("Edited arguments must be one valid JSON object.\n")
                return
            if not isinstance(parsed, dict):
                self._append("Edited arguments must be one valid JSON object.\n")
                return
            edited_arguments = parsed
        if decision not in set(approval.get("allowed_decisions") or []):
            self._append(f"Decision {decision} is not allowed for {approval_id}.\n")
            return
        self.busy = True
        self.state = "working"
        self._append(f"[{decision}] {approval_id}\n\nOPENTULPA\n")
        try:
            async for event in self.client.resume(
                str(approval["run_id"]),
                approval_id=approval_id,
                decision=decision,
                edited_arguments=edited_arguments,
            ):
                self._render(event)
        except RemoteError as exc:
            self._append(f"[approval failed] {exc}\n")
            self.busy = False
            self.state = "error"
            return
        self.approvals.pop(approval_id, None)
        await self._resolve_approval_notification(approval_id)

    async def _cancel_active(self) -> None:
        if not self.active_run_id:
            self._append("No active run to cancel.\n")
            return
        run_id = self.active_run_id
        try:
            await self.client.cancel(run_id)
        except RemoteError as exc:
            self._append(f"[cancel failed] {exc}\n")
            return
        self.active_run_id = None
        self.busy = False
        self.state = "connected"
        self._append(f"[cancelled] {run_id}\n")

    async def _show_logs(self) -> None:
        try:
            entries = await self.client.logs()
        except RemoteError as exc:
            self._append(f"[logs unavailable] {exc}\n")
            return
        for entry in entries[-40:]:
            self._append(
                f"{entry.get('sequence', ''):>4} {entry.get('stream', ''):<6} {entry.get('text', '')}\n"
            )

    async def _restore_last_run(self) -> None:
        run_id = self.connection.last_run_id
        if not run_id:
            return
        try:
            try:
                snapshot = await self.client.get_run(run_id)
            except RemoteError:
                return
            status = str(snapshot.get("status") or "")
            if status in {"running", "queued", "resume_pending"}:
                self.active_run_id = run_id
                self.state = "reconnecting"
                self._append(f"[restoring run] {run_id}\n\nOPENTULPA\n")
                try:
                    async for event in self.client.run_events(
                        run_id,
                        after_sequence=self.connection.last_sequence,
                    ):
                        self._render(event)
                except RemoteError as exc:
                    await self._recover_stream(exc)
            elif status == "interrupted":
                for approval in snapshot.get("pending_approvals") or []:
                    if isinstance(approval, dict):
                        self._remember_approval(run_id, approval)
                self.state = "approval"
        finally:
            if self.active_run_id is None and self.state != "approval":
                self.busy = False
                if self.state == "reconnecting":
                    self.state = "connected" if self.server_ready else "setup"
            self._invalidate()

    async def _poll_notifications(self) -> None:
        while not self._closed:
            try:
                payload = await self.client.notifications(
                    after_id=self.notification_cursor,
                    wait_seconds=20,
                )
                for item in payload.get("notifications") or []:
                    if not isinstance(item, dict):
                        continue
                    await self._notification(item)
                    self.notification_cursor = max(
                        self.notification_cursor,
                        int(item.get("id") or 0),
                    )
                if self.state == "reconnecting" and not self.busy:
                    self.state = "connected"
            except RemoteError as exc:
                if not self.busy:
                    if exc.status_code == 503:
                        self.state = "setup"
                    elif exc.status_code in {401, 403}:
                        self.state = "error"
                    else:
                        self.state = "reconnecting"
                    self._invalidate()
                await asyncio.sleep(2)
            except ValueError:
                await asyncio.sleep(2)

    async def _notification(self, item: dict[str, Any]) -> None:
        notification_id = int(item.get("id") or 0)
        approvals = [value for value in item.get("approvals") or [] if isinstance(value, dict)]
        run_id = str(item.get("run_id") or "")
        if run_id not in self.seen_run_ids:
            self._append(f"\n[notification] {item.get('text') or item.get('kind') or 'update'}\n")
        if approvals:
            pending: set[str] = set()
            for approval in approvals:
                approval_id = str(approval.get("approval_id") or "")
                if not approval_id:
                    continue
                pending.add(approval_id)
                self.approval_notifications[approval_id] = notification_id
                self._remember_approval(run_id, approval)
            if pending:
                self.notification_approvals[notification_id] = pending
                self.state = "approval"
                self._invalidate()
                return
        if notification_id > 0:
            await self.client.acknowledge_notification(notification_id)

    async def _resolve_approval_notification(self, approval_id: str) -> None:
        notification_id = self.approval_notifications.pop(approval_id, None)
        if notification_id is None:
            return
        pending = self.notification_approvals.get(notification_id)
        if pending is None:
            return
        pending.discard(approval_id)
        if pending:
            return
        await self.client.acknowledge_notification(notification_id)
        self.notification_approvals.pop(notification_id, None)

    def _append(self, value: str) -> None:
        rendered = str(value)
        if self._transcript and isinstance(self._transcript[-1], str):
            self._transcript[-1] += rendered
        else:
            self._transcript.append(rendered)
        self._refresh_output()

    def _refresh_output(self) -> None:
        updated = "".join(
            item if isinstance(item, str) else _render_tool(item) for item in self._transcript
        )
        document = Document(updated, cursor_position=len(updated))
        self.output.buffer.set_document(document, bypass_readonly=True)
        self._invalidate()

    def _invalidate(self) -> None:
        app = getattr(self, "app", None)
        if app is not None:
            app.invalidate()


def _user_block(text: str) -> str:
    return "\n".join(f"│ {line}" for line in text.splitlines() or [""])


def _dropped_files(value: str) -> list[Path]:
    try:
        tokens = shlex.split(value)
    except ValueError:
        return []
    if not tokens:
        return []
    paths: list[Path] = []
    for token in tokens:
        candidate = token
        if candidate.startswith("file://"):
            parsed = urlsplit(candidate)
            if parsed.netloc not in {"", "localhost"}:
                return []
            candidate = unquote(parsed.path)
        path = Path(candidate).expanduser()
        if not path.is_file():
            return []
        resolved = path.resolve()
        if resolved not in paths:
            paths.append(resolved)
    return paths


def _tool_label(name: str, arguments: Any) -> str:
    display_name = name.replace("_", " ").strip().title() or "Tool"
    if not isinstance(arguments, dict):
        return display_name
    for key in ("command", "path", "query", "url", "instruction", "description", "action"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            summary = " ".join(value.strip().split())
            if len(summary) > 72:
                summary = f"{summary[:69]}..."
            return f"{display_name}  {summary}"
    return display_name


def _render_tool(tool: _ToolView) -> str:
    if tool.ok is None:
        marker = "▣"
    elif tool.ok:
        marker = "✓"
    else:
        marker = "×"
    value = f"\n  {marker} {tool.number:<2} {_tool_label(tool.name, tool.arguments)}"
    if not tool.expanded:
        return f"{value}\n"
    value += "  [ctrl-t to collapse]\n"
    value += _detail_block("arguments", tool.arguments)
    if tool.ok is not None:
        value += _detail_block("result" if tool.ok else "error", tool.result)
    return value


def _detail_block(label: str, value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    lines = rendered.splitlines() or [""]
    return f"      {label}\n" + "".join(f"        {line}\n" for line in lines)


_HELP = """COMMANDS
  /new [NAME]          create and enter a remembered session
  /sessions            list remembered sessions
  /session NAME_OR_NUM switch to a previous session
  /regenerate          regenerate the previous agent answer
  /attach PATH         attach a file, or drag files into the terminal
  /approvals           list pending approvals
  /approve [ID]        approve an action
  /reject [ID]         reject an action
  /edit ID JSON        approve with edited arguments
  /cancel              cancel the active run
  /tool [NUMBER]       expand or collapse sanitized tool details
  /logs                show recent server runtime logs
  /settings            show connection and thread settings
  /disconnect          forget this server and credential
  /quit                close the terminal

"""

_STYLE = Style.from_dict(
    {
        "": "bg:#000000 #e0e0e0",
        "header": "bg:#000000 #666666",
        "brand": "bg:#000000 #ffffff bold",
        "thread": "bg:#000000 #444444",
        "rule": "#222222",
        "prompt": "#ffffff bold",
        "activity": "bg:#000000 #777777",
        "activity.spinner": "bg:#000000 #5c9cf5 bold",
        "approval": "bg:#111111 #e0e0e0",
        "approval.title": "bg:#111111 #e5c07b bold",
        "approval.tool": "bg:#111111 #ffffff bold",
        "approval.description": "bg:#111111 #888888",
        "approval.actions": "bg:#111111 #666666",
        "approval.button.approve": "bg:#1d6b47 #ffffff bold",
        "approval.button.reject": "bg:#792f38 #ffffff bold",
        "approval.button.disabled": "bg:#222222 #555555",
        "approval.hint": "bg:#111111 #666666",
        "status": "bg:#000000 #666666",
        "state.connected": "bg:#10291f #62d99c bold",
        "state.working": "bg:#13243a #5c9cf5 bold",
        "state.approval": "bg:#3b2d10 #f6ca66 bold",
        "state.reconnecting": "bg:#2c243a #c89cff bold",
        "state.setup": "bg:#302610 #f6ca66 bold",
        "state.connecting": "bg:#1a1a1a #999999 bold",
        "state.error": "bg:#3b1619 #ff737d bold",
        "hint": "bg:#000000 #444444",
        "key": "bg:#000000 #777777 bold",
        "transcript.text": "bg:#000000 #e0e0e0",
        "transcript.user-label": "bg:#000000 #777777 bold",
        "transcript.agent-label": "bg:#000000 #ffffff bold",
        "transcript.user": "bg:#111111 #e0e0e0",
        "transcript.tool-running": "bg:#000000 #777777",
        "transcript.tool-ok": "bg:#000000 #66d9c2",
        "transcript.tool-error": "bg:#000000 #ff737d",
        "transcript.tool-detail": "bg:#000000 #666666",
        "transcript.notice": "bg:#000000 #e5c07b",
    }
)


async def run_tui(connection: Connection) -> None:
    await OpenTulpaTUI(connection).run()


__all__ = ["OpenTulpaTUI", "run_tui"]
