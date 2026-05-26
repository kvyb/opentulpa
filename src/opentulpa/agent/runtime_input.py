"""Thread-level turn input coordination for runtime debounce and steering."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field


class MergedInputSuppressedError(Exception):
    """Raised when a queued input was already consumed by a previous turn."""


@dataclass
class _ThreadInputState:
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_inputs: list[tuple[str, str]] = field(default_factory=list)


class ThreadInputCoordinator:
    """Per-thread input queue with start-time debounce and checkpoint steering."""

    def __init__(self, *, debounce_seconds: float = 0.65) -> None:
        self._debounce_seconds = max(0.0, min(float(debounce_seconds), 3.0))
        self._states_lock = asyncio.Lock()
        self._states: dict[str, _ThreadInputState] = {}

    async def _get_state(self, thread_id: str) -> _ThreadInputState:
        tid = str(thread_id or "").strip() or "__default__"
        async with self._states_lock:
            state = self._states.get(tid)
            if state is None:
                state = _ThreadInputState()
                self._states[tid] = state
            return state

    async def begin_turn(
        self, *, thread_id: str, text: str
    ) -> tuple[_ThreadInputState | None, str]:
        """
        Returns `(state, active_text)`.

        Each request keeps its own active text. Other pending requests wait for
        their own turn instead of being merged into the current prompt.
        """
        state = await self._get_state(thread_id)
        request_id = f"req_{id(asyncio.current_task())}"
        safe_text = str(text or "").strip()
        async with state.pending_lock:
            state.pending_inputs.append((request_id, safe_text))

        await state.turn_lock.acquire()
        try:
            if self._debounce_seconds > 0:
                await asyncio.sleep(self._debounce_seconds)
            async with state.pending_lock:
                ids = [rid for rid, _ in state.pending_inputs]
                if request_id not in ids:
                    state.turn_lock.release()
                    return None, ""
                active_text = safe_text
                state.pending_inputs = [
                    (rid, chunk) for rid, chunk in state.pending_inputs if rid != request_id
                ]
            return state, active_text
        except Exception:
            with suppress(Exception):
                state.turn_lock.release()
            raise

    async def drain_steering_inputs(self, *, thread_id: str) -> list[str]:
        """Consume queued inputs so an active graph turn can see them at its next checkpoint."""
        state = await self._get_state(thread_id)
        async with state.pending_lock:
            if not state.pending_inputs:
                return []
            drained = [text for _, text in state.pending_inputs if text]
            state.pending_inputs.clear()
        assert isinstance(drained, list)
        assert all(isinstance(item, str) for item in drained)
        return drained

    @staticmethod
    def end_turn(state: _ThreadInputState | None) -> None:
        if state is None:
            return
        with suppress(Exception):
            state.turn_lock.release()
