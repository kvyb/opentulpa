"""Atomic release pointer consumed by deployment-specific restart adapters."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Protocol

from opentulpa.evolution.models import Release


class ReleasePointer(Protocol):
    async def current(self) -> Release | None: ...

    async def activate(self, release: Release) -> None: ...

    async def clear(self) -> None: ...


class AtomicReleasePointer:
    """Persist the desired immutable release using write, fsync, and rename."""

    def __init__(self, path: str | Path) -> None:
        raw = Path(path).expanduser()
        if raw.is_symlink():
            raise ValueError("release pointer cannot be a symlink")
        self._path = raw.resolve(strict=False)
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def current(self) -> Release | None:
        async with self._lock:
            return await asyncio.to_thread(self._read)

    async def activate(self, release: Release) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write, release)

    async def clear(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._clear)

    def _read(self) -> Release | None:
        if not self._path.exists():
            return None
        if self._path.is_symlink() or not self._path.is_file():
            raise RuntimeError("release pointer is not a regular file")
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
            return Release.model_validate(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("release pointer is invalid") from exc

    def _write(self, release: Release) -> None:
        parent = self._path.parent
        if parent.is_symlink():
            raise RuntimeError("release pointer parent cannot be a symlink")
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise RuntimeError("release pointer parent is invalid")
        temporary = parent / f".{self._path.name}.{release.id}.tmp"
        payload = json.dumps(
            release.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _clear(self) -> None:
        if self._path.exists():
            if self._path.is_symlink() or not self._path.is_file():
                raise RuntimeError("release pointer is not a regular file")
            self._path.unlink()
            directory = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


__all__ = ["AtomicReleasePointer", "ReleasePointer"]
