"""Durable Telegram session/admin state storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, cast


class TelegramStateStore:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path.resolve()
        self.backup_path = self.state_path.with_suffix(f"{self.state_path.suffix}.bak")
        self._lock = RLock()

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {
            "admin_user_id": None,
            "sessions": {},
            "pending_key_by_chat": {},
            "support_bindings": {},
            "support_audit": [],
            "support_command_chats": {},
            "codex_logins": {},
            "owner_update_inbox": {},
            "owner_update_completed": [],
        }

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.state_path.exists() and not self.backup_path.exists():
            return self._default_state()

        for path in (self.state_path, self.backup_path):
            try:
                raw = path.read_bytes()
                data = json.loads(raw)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            if path == self.backup_path:
                self._atomic_write(self.state_path, raw)
            return data
        raise RuntimeError("Telegram state is corrupt and no valid backup is available")

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
        current = self._valid_payload(self.state_path)
        if current is not None:
            self._atomic_write(self.backup_path, current)
        self._atomic_write(self.state_path, payload)

    @staticmethod
    def _valid_payload(path: Path) -> bytes | None:
        try:
            payload = path.read_bytes()
            parsed = json.loads(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(parsed, dict) else None

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(raw_temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            with suppress(OSError):
                os.chmod(path, 0o600)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = -1
            if directory_fd >= 0:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary.unlink()

    def load(self) -> dict[str, Any]:
        with self._lock:
            return self._load_unlocked()

    def save(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._save_unlocked(state)

    def update(self, mutator: Any) -> Any:
        """
        Atomically load-modify-save state in one lock scope.
        `mutator` receives mutable state dict and can return any value.
        """
        with self._lock:
            state = self._load_unlocked()
            result = mutator(state)
            self._save_unlocked(state)
            return result

    def enqueue_owner_update(self, body: dict[str, Any]) -> tuple[str, bool]:
        """Persist an owner webhook update before acknowledgement and deduplicate retries."""

        detached = json.loads(json.dumps(body, ensure_ascii=False, allow_nan=False))
        encoded = json.dumps(
            detached,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        raw_update_id = body.get("update_id")
        if raw_update_id is None:
            update_id = 0
        elif isinstance(raw_update_id, int) and not isinstance(raw_update_id, bool):
            update_id = raw_update_id
        elif isinstance(raw_update_id, str) and raw_update_id.isdecimal():
            update_id = int(raw_update_id)
        else:
            raise ValueError("Telegram update_id is invalid")
        if update_id < 0:
            raise ValueError("Telegram update_id is invalid")
        key = (
            f"telegram:update:{update_id}"
            if body.get("update_id") is not None
            else f"telegram:update:sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"
        )

        def enqueue(state: dict[str, Any]) -> tuple[str, bool]:
            completed = state.get("owner_update_completed")
            completed_values = completed if isinstance(completed, list) else []
            if key in completed_values:
                return key, False
            inbox = state.get("owner_update_inbox")
            if not isinstance(inbox, dict):
                inbox = {}
            existing = inbox.get(key)
            if isinstance(existing, dict):
                existing_body = existing.get("body")
                existing_encoded = json.dumps(
                    existing_body,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if existing_encoded != encoded:
                    raise ValueError("Telegram update_id payload conflict")
                return key, True
            inbox[key] = {
                "update_id": update_id,
                "body": detached,
                "accepted_at": datetime.now(UTC).isoformat(),
            }
            state["owner_update_inbox"] = inbox
            return key, True

        return cast("tuple[str, bool]", self.update(enqueue))

    def owner_update(self, ingress_key: str) -> dict[str, Any] | None:
        state = self.load()
        inbox = state.get("owner_update_inbox")
        item = inbox.get(ingress_key) if isinstance(inbox, dict) else None
        body = item.get("body") if isinstance(item, dict) else None
        return dict(body) if isinstance(body, dict) else None

    def owner_update_preparation(self, ingress_key: str) -> dict[str, Any] | None:
        state = self.load()
        inbox = state.get("owner_update_inbox")
        item = inbox.get(ingress_key) if isinstance(inbox, dict) else None
        preparation = item.get("preparation") if isinstance(item, dict) else None
        return dict(preparation) if isinstance(preparation, dict) else None

    def prepare_owner_update(
        self,
        ingress_key: str,
        preparation: dict[str, Any],
    ) -> dict[str, Any]:
        detached = json.loads(json.dumps(preparation, ensure_ascii=False, allow_nan=False))

        def prepare(state: dict[str, Any]) -> dict[str, Any]:
            inbox = state.get("owner_update_inbox")
            item = inbox.get(ingress_key) if isinstance(inbox, dict) else None
            if not isinstance(item, dict):
                raise KeyError(f"Telegram owner update is unavailable: {ingress_key}")
            existing = item.get("preparation")
            if isinstance(existing, dict):
                return dict(existing)
            item["preparation"] = detached
            return dict(detached)

        return cast("dict[str, Any]", self.update(prepare))

    def pending_owner_updates(self, *, limit: int = 100) -> list[tuple[str, dict[str, Any]]]:
        state = self.load()
        inbox = state.get("owner_update_inbox")
        if not isinstance(inbox, dict):
            return []
        pending: list[tuple[int, str, dict[str, Any]]] = []
        for key, item in inbox.items():
            body = item.get("body") if isinstance(item, dict) else None
            if not isinstance(body, dict):
                continue
            try:
                update_id = int(item.get("update_id"))
            except (TypeError, ValueError):
                continue
            pending.append((update_id, str(key), dict(body)))
        pending.sort(key=lambda item: (item[0], item[1]))
        return [(key, body) for _, key, body in pending[: max(1, min(limit, 1_000))]]

    def complete_owner_update(self, ingress_key: str) -> bool:
        def complete(state: dict[str, Any]) -> bool:
            inbox = state.get("owner_update_inbox")
            if not isinstance(inbox, dict) or ingress_key not in inbox:
                return False
            inbox.pop(ingress_key, None)
            completed = state.get("owner_update_completed")
            completed_values = completed if isinstance(completed, list) else []
            completed_values = [
                str(value) for value in completed_values if str(value) != ingress_key
            ]
            completed_values.append(ingress_key)
            state["owner_update_inbox"] = inbox
            state["owner_update_completed"] = completed_values[-2_048:]
            return True

        return bool(self.update(complete))

    def find_session_slots(self, customer_id: str) -> list[dict[str, Any]]:
        state = self.load()
        sessions = state.get("sessions", {})
        slots: list[dict[str, Any]] = []
        for chat_id, slot in sessions.items():
            if str(slot.get("customer_id", "")) == customer_id:
                with suppress(Exception):
                    slots.append(
                        {
                            "chat_id": int(chat_id),
                            "user_id": slot.get("user_id"),
                            "thread_id": slot.get("thread_id"),
                            "wake_thread_id": slot.get("wake_thread_id"),
                            "customer_id": slot.get("customer_id"),
                            "role": slot.get("role") or "owner",
                            "username": slot.get("username"),
                            "last_user_message_at": slot.get("last_user_message_at"),
                            "last_assistant_message_at": slot.get("last_assistant_message_at"),
                        }
                    )
        if customer_id.startswith("telegram_"):
            uid = customer_id.removeprefix("telegram_").strip()
            for chat_id, slot in sessions.items():
                if str(slot.get("user_id", "")) == uid:
                    with suppress(Exception):
                        cid = int(chat_id)
                        if not any(s.get("chat_id") == cid for s in slots):
                            slots.append(
                                {
                                    "chat_id": cid,
                                    "user_id": slot.get("user_id"),
                                    "thread_id": slot.get("thread_id"),
                                    "wake_thread_id": slot.get("wake_thread_id"),
                                    "customer_id": slot.get("customer_id"),
                                    "role": slot.get("role") or "owner",
                                    "username": slot.get("username"),
                                    "last_user_message_at": slot.get("last_user_message_at"),
                                    "last_assistant_message_at": slot.get("last_assistant_message_at"),
                                }
                            )
        bindings = state.get("support_bindings", {})
        if isinstance(bindings, dict):
            for chat_id, binding in bindings.items():
                if not isinstance(binding, dict):
                    continue
                if str(binding.get("bound_customer_id", "")).strip() != customer_id:
                    continue
                with suppress(Exception):
                    cid = int(chat_id)
                    if any(s.get("chat_id") == cid for s in slots):
                        continue
                    slots.append(
                        {
                            "chat_id": cid,
                            "user_id": binding.get("support_user_id"),
                            "thread_id": binding.get("thread_id"),
                            "wake_thread_id": binding.get("wake_thread_id"),
                            "customer_id": binding.get("bound_customer_id"),
                            "role": "support",
                            "username": binding.get("support_username"),
                            "last_user_message_at": binding.get("last_user_message_at"),
                            "last_assistant_message_at": binding.get("last_assistant_message_at"),
                        }
                    )
        return slots

    def get_session_slot(self, chat_id: int | str) -> dict[str, Any] | None:
        state = self.load()
        sessions = state.get("sessions", {})
        key = str(chat_id)
        slot = sessions.get(key) if isinstance(sessions, dict) else None
        if isinstance(slot, dict):
            return {
                "chat_id": int(chat_id),
                "user_id": slot.get("user_id"),
                "thread_id": slot.get("thread_id"),
                "wake_thread_id": slot.get("wake_thread_id"),
                "customer_id": slot.get("customer_id"),
                "role": slot.get("role") or "owner",
                "username": slot.get("username"),
                "last_user_message_at": slot.get("last_user_message_at"),
                "last_assistant_message_at": slot.get("last_assistant_message_at"),
            }
        bindings = state.get("support_bindings", {})
        binding = bindings.get(key) if isinstance(bindings, dict) else None
        if not isinstance(binding, dict):
            return None
        return {
            "chat_id": int(chat_id),
            "user_id": binding.get("support_user_id"),
            "thread_id": binding.get("thread_id"),
            "wake_thread_id": binding.get("wake_thread_id"),
            "customer_id": binding.get("bound_customer_id"),
            "role": "support",
            "username": binding.get("support_username"),
            "last_user_message_at": binding.get("last_user_message_at"),
            "last_assistant_message_at": binding.get("last_assistant_message_at"),
        }

    def list_owner_customer_summaries(self) -> list[dict[str, Any]]:
        state = self.load()
        sessions = state.get("sessions", {})
        if not isinstance(sessions, dict):
            return []
        by_customer: dict[str, dict[str, Any]] = {}
        for chat_id, slot in sessions.items():
            if not isinstance(slot, dict):
                continue
            customer_id = str(slot.get("customer_id", "") or "").strip()
            if not customer_id:
                continue
            current = by_customer.get(customer_id)
            last_user = str(slot.get("last_user_message_at", "") or "").strip()
            last_assistant = str(slot.get("last_assistant_message_at", "") or "").strip()
            last_activity = max(last_user, last_assistant)
            if current is not None and str(current.get("last_activity", "")) >= last_activity:
                continue
            by_customer[customer_id] = {
                "customer_id": customer_id,
                "owner_chat_id": str(chat_id),
                "owner_user_id": str(slot.get("user_id", "") or ""),
                "owner_username": str(slot.get("username", "") or ""),
                "last_activity": last_activity,
            }
        return sorted(
            by_customer.values(),
            key=lambda item: (str(item.get("last_activity", "")), str(item.get("customer_id", ""))),
            reverse=True,
        )

    def touch_assistant_message(self, chat_id: int | str) -> None:
        now_utc_iso = datetime.now(UTC).isoformat()
        key = str(chat_id)

        def _touch(state: dict[str, Any]) -> None:
            sessions = state.get("sessions")
            if not isinstance(sessions, dict):
                sessions = {}
            slot = sessions.get(key)
            if not isinstance(slot, dict):
                bindings = state.get("support_bindings")
                if isinstance(bindings, dict):
                    binding = bindings.get(key)
                    if isinstance(binding, dict):
                        binding["last_assistant_message_at"] = now_utc_iso
                        bindings[key] = binding
                        state["support_bindings"] = bindings
                return
            slot["last_assistant_message_at"] = now_utc_iso
            sessions[key] = slot
            state["sessions"] = sessions

        self.update(_touch)
