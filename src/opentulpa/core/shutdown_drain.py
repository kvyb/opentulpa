"""Graceful shutdown drain coordination for active conversation turns."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


class ShutdownDrainingError(RuntimeError):
    """Raised when new work is rejected because the instance is draining."""


@dataclass(frozen=True)
class DrainStatus:
    draining: bool
    active_turns: int


class ShutdownDrain:
    """Tracks active turns so shutdown can wait without delaying idle deploys."""

    def __init__(self, *, timeout_seconds: float = 300.0) -> None:
        assert timeout_seconds >= 0
        self._timeout_seconds = float(timeout_seconds)
        self._active_turns = 0
        self._draining = False
        self._condition = asyncio.Condition()

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def draining(self) -> bool:
        return self._draining

    def status(self) -> DrainStatus:
        assert self._active_turns >= 0
        return DrainStatus(draining=self._draining, active_turns=self._active_turns)

    def start_draining(self) -> None:
        self._draining = True

    @asynccontextmanager
    async def active_turn(self) -> Any:
        async with self._condition:
            if self._draining:
                raise ShutdownDrainingError("instance is draining")
            self._active_turns += 1
            assert self._active_turns > 0
        try:
            yield
        finally:
            async with self._condition:
                self._active_turns -= 1
                assert self._active_turns >= 0
                if self._active_turns == 0:
                    self._condition.notify_all()

    async def drain(self) -> bool:
        self.start_draining()
        async with self._condition:
            if self._active_turns == 0:
                return True
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._active_turns == 0),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError:
                return False
        return True
