"""Browser Use task/session registry state."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

TERMINAL_STATUSES = {"finished", "stopped", "failed"}
DEFAULT_CUSTOMER_ID = "default"


def safe_profile_name(session_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_id or "").strip())
    value = value.strip("._-")
    return value[:80] or "default"


def normalize_customer_id(customer_id: str | None) -> str:
    raw = str(customer_id or "").strip()
    return raw or DEFAULT_CUSTOMER_ID


def normalize_optional_customer_id(customer_id: str | None) -> str | None:
    raw = str(customer_id or "").strip()
    if not raw:
        return None
    return raw


def session_key(customer_id: str, session_id: str) -> str:
    customer = normalize_customer_id(customer_id)
    session = safe_profile_name(session_id)
    return f"{customer}\0{session}"


@dataclass(slots=True)
class BrowserUseTaskState:
    task_id: str
    session_id: str | None
    task: str
    llm: str
    customer_id: str = DEFAULT_CUSTOMER_ID
    status: str = "queued"
    is_success: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    output: str | None = None
    output_files: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    image_candidates: list[dict[str, Any]] = field(default_factory=list)
    network_image_resources: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    created_monotonic: float = field(default_factory=time.monotonic)
    updated_monotonic: float = field(default_factory=time.monotonic)
    runner: asyncio.Task[Any] | None = None
    browser_session: Any = None
    allow_owner_input: bool = True
    owner_input_prompt: str | None = None
    owner_input_type: str | None = None
    owner_input_requested_at: str | None = None
    owner_input_future: asyncio.Future[str] | None = None
    stop_requested: bool = False
    close_session_when_done: bool = False


@dataclass(slots=True)
class BrowserUseSessionState:
    session: Any
    customer_id: str
    session_id: str
    backend: str = "local"
    cloud_profile_id: str | None = None
    cloud_browser_session_id: str | None = None
    live_url: str | None = None
    updated_monotonic: float = field(default_factory=time.monotonic)


class BrowserUseSessionRegistry:
    """Own in-memory Browser Use task/session indexes and reuse rules."""

    def __init__(self) -> None:
        self.tasks: dict[str, BrowserUseTaskState] = {}
        self.sessions: dict[str, BrowserUseSessionState] = {}

    def set_task(self, state: BrowserUseTaskState) -> None:
        assert state.task_id.strip()
        self.tasks[state.task_id] = state

    def set_session(self, state: BrowserUseSessionState) -> None:
        assert state.customer_id.strip()
        assert state.session_id.strip()
        self.sessions[session_key(state.customer_id, state.session_id)] = state

    def session_state(
        self,
        *,
        customer_id: str,
        session_id: str | None,
    ) -> BrowserUseSessionState | None:
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return None
        return self.sessions.get(session_key(customer_id, safe_session))

    def related_tasks(
        self,
        *,
        customer_id: str,
        session_id: str,
    ) -> list[BrowserUseTaskState]:
        safe_customer = normalize_customer_id(customer_id)
        safe_session = str(session_id or "").strip()
        related = [
            state
            for state in self.tasks.values()
            if state.customer_id == safe_customer
            and str(state.session_id or "").strip() == safe_session
        ]
        related.sort(
            key=lambda item: float(item.updated_monotonic or item.created_monotonic),
            reverse=True,
        )
        return related

    def active_task_for_session(
        self,
        *,
        customer_id: str,
        session_id: str,
    ) -> BrowserUseTaskState | None:
        active = [
            state
            for state in self.related_tasks(customer_id=customer_id, session_id=session_id)
            if state.status not in TERMINAL_STATUSES
        ]
        return active[0] if active else None

    def session_has_active_tasks(self, *, customer_id: str, session_id: str | None) -> bool:
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return False
        return self.active_task_for_session(customer_id=customer_id, session_id=safe_session) is not None

    def live_session_count_for_customer(self, customer_id: str | None = None) -> int:
        safe_customer = normalize_customer_id(customer_id)
        return sum(1 for item in self.sessions.values() if item.customer_id == safe_customer)

    def pick_reusable_session_id(self, customer_id: str) -> str | None:
        safe_customer = normalize_customer_id(customer_id)
        reusable: list[tuple[float, str]] = []
        for session_state in self.sessions.values():
            if session_state.customer_id != safe_customer:
                continue
            if self.session_has_active_tasks(
                customer_id=session_state.customer_id,
                session_id=session_state.session_id,
            ):
                continue
            reusable.append((float(session_state.updated_monotonic or 0.0), session_state.session_id))
        if not reusable:
            return None
        reusable.sort(reverse=True)
        return reusable[0][1]

    def detach_session_if_unused(
        self,
        *,
        customer_id: str,
        session_id: str | None,
    ) -> BrowserUseSessionState | None:
        safe_customer = normalize_customer_id(customer_id)
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return None
        if self.session_has_active_tasks(customer_id=safe_customer, session_id=safe_session):
            return None
        return self.sessions.pop(session_key(safe_customer, safe_session), None)

    def touch_session(self, *, customer_id: str, session_id: str | None) -> None:
        state = self.session_state(customer_id=customer_id, session_id=session_id)
        if state is not None:
            state.updated_monotonic = time.monotonic()

    def session_summaries(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        safe_customer = normalize_customer_id(customer_id)
        out: list[dict[str, Any]] = []
        for session_state in self.sessions.values():
            if session_state.customer_id != safe_customer:
                continue
            active_task = self.active_task_for_session(
                customer_id=session_state.customer_id,
                session_id=session_state.session_id,
            )
            out.append(
                {
                    "session_id": session_state.session_id,
                    "customer_id": session_state.customer_id,
                    "reusable": active_task is None,
                    "active_task_id": active_task.task_id if active_task is not None else None,
                    "last_used_monotonic": float(session_state.updated_monotonic or 0.0),
                }
            )
        out.sort(key=lambda item: item["last_used_monotonic"], reverse=True)
        return out

    def pop_expired_terminal_tasks(self, *, now: float, retention_seconds: int) -> list[str]:
        expired = [
            task_id
            for task_id, state in self.tasks.items()
            if state.status in TERMINAL_STATUSES
            and now - float(state.updated_monotonic or state.created_monotonic) >= retention_seconds
        ]
        for task_id in expired:
            self.tasks.pop(task_id, None)
        return expired

    def pop_expired_idle_sessions(
        self,
        *,
        now: float,
        idle_timeout_seconds: int,
    ) -> list[BrowserUseSessionState]:
        expired: list[str] = []
        for key, session_state in self.sessions.items():
            if self.session_has_active_tasks(
                customer_id=session_state.customer_id,
                session_id=session_state.session_id,
            ):
                continue
            age = now - float(session_state.updated_monotonic or now)
            if age >= idle_timeout_seconds:
                expired.append(key)
        out: list[BrowserUseSessionState] = []
        for key in expired:
            if key in self.sessions:
                out.append(self.sessions.pop(key))
        return out
