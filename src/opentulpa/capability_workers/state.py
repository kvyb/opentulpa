"""Atomic durable state for the standalone Telegram interface worker."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from uuid import uuid4


class TelegramStateError(RuntimeError):
    """The worker state is unsafe or corrupt."""


def _default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "next_update_id": 0,
        "paired": None,
        "threads": {},
        "codex_logins": {},
        "update_inbox": {},
        "seen_source_events": [],
        "pending_runs": {},
        "approvals": {},
        "awaiting_edits": {},
        "notification_cursor": 0,
        "pending_notification_acks": [],
    }


class TelegramWorkerState:
    """Single-worker JSON state with atomic replacement and bounded dedupe history."""

    def __init__(self, path: Path, *, max_seen_source_events: int = 5_000) -> None:
        if max_seen_source_events < 100:
            raise ValueError("max_seen_source_events must be at least 100")
        expanded = path.expanduser()
        self.path = (
            expanded if expanded.is_absolute() else Path.cwd() / expanded
        ).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._max_seen = max_seen_source_events
        self._lock = threading.RLock()
        self._state = self._read()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return _default_state()
        if self.path.is_symlink() or not self.path.is_file():
            raise TelegramStateError("Telegram worker state path must be a regular file")
        if self.path.stat().st_size > 2_000_000:
            raise TelegramStateError("Telegram worker state exceeds the size limit")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TelegramStateError("Telegram worker state is unreadable") from exc
        if not isinstance(value, dict) or value.get("version") != 1:
            raise TelegramStateError("Telegram worker state has an unsupported schema")
        default = _default_state()
        default.update(value)
        for key in (
            "threads",
            "codex_logins",
            "update_inbox",
            "pending_runs",
            "approvals",
            "awaiting_edits",
        ):
            if not isinstance(default[key], dict):
                raise TelegramStateError(f"Telegram worker state field {key!r} is invalid")
        if not isinstance(default["seen_source_events"], list):
            raise TelegramStateError("Telegram worker dedupe history is invalid")
        if not isinstance(default["pending_notification_acks"], list):
            raise TelegramStateError("Telegram notification acknowledgements are invalid")
        try:
            if int(default["notification_cursor"]) < 0 or any(
                int(item) < 1 for item in default["pending_notification_acks"]
            ):
                raise ValueError
        except (TypeError, ValueError):
            raise TelegramStateError("Telegram notification cursor is invalid") from None
        return default

    def _write(self) -> None:
        payload = json.dumps(
            self._state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary)
            raise

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return cast(dict[str, Any], json.loads(json.dumps(self._state)))

    @property
    def next_update_id(self) -> int:
        with self._lock:
            return max(0, int(self._state["next_update_id"]))

    def paired_identity(self) -> tuple[int, int] | None:
        with self._lock:
            paired = self._state.get("paired")
            if not isinstance(paired, dict):
                return None
            try:
                return int(paired["user_id"]), int(paired["chat_id"])
            except (KeyError, TypeError, ValueError):
                raise TelegramStateError("Telegram paired identity is invalid") from None

    def pair(self, *, user_id: int, chat_id: int) -> bool:
        with self._lock:
            existing = self.paired_identity()
            identity = (int(user_id), int(chat_id))
            if existing is not None and existing != identity:
                return False
            if existing is None:
                self._state["paired"] = {"user_id": identity[0], "chat_id": identity[1]}
                self._write()
            return True

    def thread_id(self, chat_id: int) -> str:
        key = str(int(chat_id))
        with self._lock:
            value = str(self._state["threads"].get(key) or "").strip()
            if value:
                return value
            value = self.new_thread_id(chat_id)
            self._state["threads"][key] = value
            self._write()
            return value

    @staticmethod
    def new_thread_id(chat_id: int) -> str:
        return f"telegram_{int(chat_id)}_{uuid4().hex}"

    def replace_thread(
        self,
        chat_id: int,
        *,
        expected_thread_id: str,
        replacement_thread_id: str,
    ) -> None:
        key = str(int(chat_id))
        replacement = str(replacement_thread_id or "").strip()
        if not replacement:
            raise ValueError("replacement_thread_id must not be empty")
        with self._lock:
            current = str(self._state["threads"].get(key) or "").strip()
            if current != str(expected_thread_id or "").strip():
                raise TelegramStateError("Telegram conversation changed during replacement")
            self._state["threads"][key] = replacement
            self._write()

    def codex_login(self, chat_id: int) -> str:
        with self._lock:
            return str(self._state["codex_logins"].get(str(int(chat_id))) or "").strip()

    def set_codex_login(self, chat_id: int, login_id: str | None) -> None:
        key = str(int(chat_id))
        with self._lock:
            if login_id:
                self._state["codex_logins"][key] = str(login_id)
            else:
                self._state["codex_logins"].pop(key, None)
            self._write()

    def accept_update(self, *, update_id: int, update: dict[str, Any]) -> bool:
        """Persist an update before advancing the Telegram long-poll cursor."""

        identifier = int(update_id)
        detached = json.loads(json.dumps(update, ensure_ascii=False, allow_nan=False))
        key = str(identifier)
        with self._lock:
            existing = self._state["update_inbox"].get(key)
            if existing is not None and existing != detached:
                raise TelegramStateError("Telegram update id payload conflict")
            self._state["update_inbox"][key] = detached
            self._state["next_update_id"] = max(self.next_update_id, identifier + 1)
            self._write()
            return existing is None

    def pending_updates(self) -> list[tuple[int, dict[str, Any]]]:
        with self._lock:
            return sorted(
                (
                    int(key),
                    cast(dict[str, Any], json.loads(json.dumps(value))),
                )
                for key, value in self._state["update_inbox"].items()
                if isinstance(value, dict)
            )

    def update_pending(self, update_id: int) -> bool:
        with self._lock:
            return str(int(update_id)) in self._state["update_inbox"]

    def coalesce_updates(
        self,
        *,
        update_ids: Sequence[int],
        merged_update: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically replace consecutive pending updates with one merged update."""

        identifiers = tuple(dict.fromkeys(int(item) for item in update_ids))
        if not identifiers:
            raise ValueError("update_ids must not be empty")
        detached = json.loads(
            json.dumps(merged_update, ensure_ascii=False, allow_nan=False)
        )
        primary = str(identifiers[0])
        with self._lock:
            missing = [
                identifier
                for identifier in identifiers
                if str(identifier) not in self._state["update_inbox"]
            ]
            if missing:
                raise TelegramStateError("cannot coalesce updates that are not pending")
            self._state["update_inbox"][primary] = detached
            for identifier in identifiers[1:]:
                self._state["update_inbox"].pop(str(identifier), None)
            self._write()
        return cast(dict[str, Any], json.loads(json.dumps(detached)))

    @property
    def notification_cursor(self) -> int:
        with self._lock:
            return max(0, int(self._state["notification_cursor"]))

    def pending_notification_acks(self) -> list[int]:
        with self._lock:
            return sorted({int(item) for item in self._state["pending_notification_acks"]})

    def mark_notification_delivered(self, notification_id: int) -> None:
        identifier = int(notification_id)
        if identifier < 1:
            raise ValueError("notification_id must be positive")
        with self._lock:
            pending = {int(item) for item in self._state["pending_notification_acks"]}
            pending.add(identifier)
            self._state["pending_notification_acks"] = sorted(pending)
            self._state["notification_cursor"] = max(self.notification_cursor, identifier)
            self._write()

    def mark_notification_acknowledged(self, notification_id: int) -> None:
        identifier = int(notification_id)
        with self._lock:
            pending = [
                int(item)
                for item in self._state["pending_notification_acks"]
                if int(item) != identifier
            ]
            if len(pending) != len(self._state["pending_notification_acks"]):
                self._state["pending_notification_acks"] = pending
                self._write()

    def source_seen(self, source_event_id: str) -> bool:
        with self._lock:
            return source_event_id in self._state["seen_source_events"]

    def complete_update(
        self,
        *,
        update_id: int,
        source_event_id: str,
        consumed_approval_token: str | None = None,
    ) -> None:
        with self._lock:
            history = [
                str(item)
                for item in self._state["seen_source_events"]
                if str(item) != source_event_id
            ]
            history.append(source_event_id)
            self._state["seen_source_events"] = history[-self._max_seen :]
            self._state["next_update_id"] = max(
                self.next_update_id,
                int(update_id) + 1,
            )
            self._state["update_inbox"].pop(str(int(update_id)), None)
            self._state["pending_runs"].pop(source_event_id, None)
            if consumed_approval_token:
                self._state["approvals"].pop(consumed_approval_token, None)
                self._state["awaiting_edits"] = {
                    key: value
                    for key, value in self._state["awaiting_edits"].items()
                    if value != consumed_approval_token
                }
            self._write()

    def save_pending_run(
        self,
        *,
        source_event_id: str,
        update_id: int,
        run_id: str,
        chat_id: int,
        sequence: int,
        accumulated_text: str,
        response_message_id: int | None = None,
        rendered_text: str = "",
    ) -> None:
        with self._lock:
            self._state["pending_runs"][source_event_id] = {
                "source_event_id": source_event_id,
                "update_id": int(update_id),
                "run_id": run_id,
                "chat_id": int(chat_id),
                "sequence": max(0, int(sequence)),
                "accumulated_text": accumulated_text[-200_000:],
                "response_message_id": (
                    int(response_message_id)
                    if response_message_id is not None and int(response_message_id) > 0
                    else None
                ),
                "rendered_text": rendered_text[-200_000:],
            }
            self._write()

    def save_pending_delivery(
        self,
        *,
        source_event_id: str,
        response_message_id: int | None,
        rendered_text: str,
    ) -> None:
        with self._lock:
            record = self._state["pending_runs"].get(source_event_id)
            if not isinstance(record, dict):
                return
            record["response_message_id"] = (
                int(response_message_id)
                if response_message_id is not None and int(response_message_id) > 0
                else None
            )
            record["rendered_text"] = rendered_text[-200_000:]
            self._write()

    def pending_run(self, source_event_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._state["pending_runs"].get(source_event_id)
            return json.loads(json.dumps(value)) if isinstance(value, dict) else None

    def pending_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                json.loads(json.dumps(value))
                for value in self._state["pending_runs"].values()
                if isinstance(value, dict)
            ]

    def remove_pending_run(self, source_event_id: str) -> None:
        with self._lock:
            if self._state["pending_runs"].pop(source_event_id, None) is not None:
                self._write()

    def save_approval(
        self,
        *,
        token: str,
        run_id: str,
        approval_id: str,
        chat_id: int,
        user_id: int,
        allowed_decisions: list[str],
        tool_name: str,
        description: str,
    ) -> None:
        with self._lock:
            self._state["approvals"][token] = {
                "run_id": run_id,
                "approval_id": approval_id,
                "chat_id": int(chat_id),
                "user_id": int(user_id),
                "allowed_decisions": list(allowed_decisions),
                "tool_name": tool_name,
                "description": description,
                "delivered": False,
            }
            self._write()

    def approval(self, token: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._state["approvals"].get(token)
            return json.loads(json.dumps(value)) if isinstance(value, dict) else None

    def remove_approval(self, token: str) -> None:
        with self._lock:
            removed = self._state["approvals"].pop(token, None)
            awaiting = [
                key for key, value in self._state["awaiting_edits"].items() if value == token
            ]
            for key in awaiting:
                self._state["awaiting_edits"].pop(key, None)
            if removed is not None or awaiting:
                self._write()

    def find_approval(self, *, run_id: str, approval_id: str) -> tuple[str, dict[str, Any]] | None:
        with self._lock:
            for token, value in self._state["approvals"].items():
                if not isinstance(value, dict):
                    continue
                if value.get("run_id") == run_id and value.get("approval_id") == approval_id:
                    return str(token), json.loads(json.dumps(value))
        return None

    def mark_approval_delivered(self, token: str) -> None:
        with self._lock:
            value = self._state["approvals"].get(token)
            if isinstance(value, dict) and not bool(value.get("delivered")):
                value["delivered"] = True
                self._write()

    def undelivered_approvals(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            return [
                (str(token), json.loads(json.dumps(value)))
                for token, value in self._state["approvals"].items()
                if isinstance(value, dict) and not bool(value.get("delivered"))
            ]

    def await_edit(self, *, chat_id: int, token: str) -> None:
        with self._lock:
            self._state["awaiting_edits"][str(int(chat_id))] = token
            self._write()

    def awaiting_edit(self, chat_id: int) -> str | None:
        with self._lock:
            value = self._state["awaiting_edits"].get(str(int(chat_id)))
            return str(value) if value else None

    def clear_awaiting_edit(self, *, chat_id: int, token: str) -> bool:
        key = str(int(chat_id))
        with self._lock:
            if self._state["awaiting_edits"].get(key) != token:
                return False
            self._state["awaiting_edits"].pop(key, None)
            self._write()
            return True


__all__ = ["TelegramStateError", "TelegramWorkerState"]
