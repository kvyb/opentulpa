"""Durable Telegram session/admin state storage."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any


class TelegramStateStore:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path.resolve()
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
        }

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else self._default_state()
        except Exception:
            return self._default_state()

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        with suppress(Exception):
            self.state_path.chmod(0o600)

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
