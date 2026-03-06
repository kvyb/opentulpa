"""Durable Telegram session/admin state storage."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


class TelegramStateStore:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path.resolve()
        self._lock = RLock()

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {"admin_user_id": None, "sessions": {}, "pending_key_by_chat": {}}

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
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
                                    "last_user_message_at": slot.get("last_user_message_at"),
                                    "last_assistant_message_at": slot.get("last_assistant_message_at"),
                                }
                            )
        return slots

    def get_session_slot(self, chat_id: int | str) -> dict[str, Any] | None:
        state = self.load()
        sessions = state.get("sessions", {})
        key = str(chat_id)
        slot = sessions.get(key) if isinstance(sessions, dict) else None
        if not isinstance(slot, dict):
            return None
        return {
            "chat_id": int(chat_id),
            "user_id": slot.get("user_id"),
            "thread_id": slot.get("thread_id"),
            "wake_thread_id": slot.get("wake_thread_id"),
            "customer_id": slot.get("customer_id"),
            "last_user_message_at": slot.get("last_user_message_at"),
            "last_assistant_message_at": slot.get("last_assistant_message_at"),
        }

    def touch_assistant_message(self, chat_id: int | str) -> None:
        now_utc_iso = datetime.now(timezone.utc).isoformat()
        key = str(chat_id)

        def _touch(state: dict[str, Any]) -> None:
            sessions = state.get("sessions")
            if not isinstance(sessions, dict):
                sessions = {}
            slot = sessions.get(key)
            if not isinstance(slot, dict):
                return
            slot["last_assistant_message_at"] = now_utc_iso
            sessions[key] = slot
            state["sessions"] = sessions

        self.update(_touch)
