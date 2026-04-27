"""Interactive Telegram chat inbox/session coordination."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class InteractiveSubmissionResult:
    fragment: str | None = None
    direct_reply: str | None = None


@dataclass(slots=True)
class _QueuedSubmission:
    result: InteractiveSubmissionResult = field(default_factory=InteractiveSubmissionResult)
    ready: bool = False


class InteractiveSession:
    """Ordered per-chat mailbox for interactive Telegram submissions."""

    def __init__(
        self,
        *,
        chat_id: int,
        customer_id: str,
        thread_id: str,
    ) -> None:
        self.chat_id = int(chat_id)
        self.customer_id = str(customer_id or "").strip()
        self.thread_id = str(thread_id or "").strip()
        self._runner_active = False
        self._condition = asyncio.Condition()
        self._queue: deque[_QueuedSubmission] = deque()

    async def enqueue(self) -> tuple[_QueuedSubmission, bool]:
        async with self._condition:
            submission = _QueuedSubmission()
            self._queue.append(submission)
            became_runner = not self._runner_active
            if became_runner:
                self._runner_active = True
            self._condition.notify_all()
            return submission, became_runner

    async def publish(
        self,
        submission: _QueuedSubmission,
        *,
        fragment: str | None = None,
        direct_reply: str | None = None,
    ) -> None:
        async with self._condition:
            submission.result = InteractiveSubmissionResult(
                fragment=str(fragment or "").strip() or None,
                direct_reply=str(direct_reply or "").strip() or None,
            )
            submission.ready = True
            self._condition.notify_all()

    async def wait_for_ready_head(self) -> bool:
        async with self._condition:
            while True:
                if not self._queue:
                    return False
                if self._queue[0].ready:
                    return True
                await self._condition.wait()

    async def consume_ready_batch(self) -> list[InteractiveSubmissionResult]:
        async with self._condition:
            ready: list[InteractiveSubmissionResult] = []
            while self._queue and self._queue[0].ready:
                ready.append(self._queue.popleft().result)
            return ready

    async def drain_graph_fragments(self) -> list[str]:
        async with self._condition:
            fragments: list[str] = []
            while self._queue and self._queue[0].ready:
                head = self._queue[0]
                fragment = str(head.result.fragment or "").strip()
                if not fragment:
                    break
                self._queue.popleft()
                fragments.append(fragment)
            return fragments

    async def has_pending_items(self) -> bool:
        async with self._condition:
            return bool(self._queue)

    async def finish_runner_if_idle(self) -> bool:
        async with self._condition:
            if self._queue:
                return False
            self._runner_active = False
            self._condition.notify_all()
            return True

    async def is_idle(self) -> bool:
        async with self._condition:
            return (not self._runner_active) and (not self._queue)


class TelegramInteractiveInbox:
    """Owns per-chat interactive Telegram sessions."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, InteractiveSession] = {}

    async def submit(
        self,
        *,
        chat_id: int,
        customer_id: str,
        thread_id: str,
    ) -> tuple[InteractiveSession, _QueuedSubmission, bool]:
        key = str(int(chat_id))
        async with self._lock:
            session = self._sessions.get(key)
            if (
                session is None
                or session.thread_id != str(thread_id or "").strip()
                or session.customer_id != str(customer_id or "").strip()
            ):
                session = InteractiveSession(
                    chat_id=chat_id,
                    customer_id=customer_id,
                    thread_id=thread_id,
                )
                self._sessions[key] = session
        submission, became_runner = await session.enqueue()
        return session, submission, became_runner

    async def prune_if_idle(self, session: InteractiveSession) -> None:
        key = str(int(session.chat_id))
        if not await session.is_idle():
            return
        async with self._lock:
            current = self._sessions.get(key)
            if current is session and await session.is_idle():
                self._sessions.pop(key, None)

    async def reset_chat(self, chat_id: int) -> None:
        async with self._lock:
            self._sessions.pop(str(int(chat_id)), None)
