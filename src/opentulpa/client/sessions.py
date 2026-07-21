"""Private local catalog of remembered Deep Agent threads."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from opentulpa.client.config import Connection, config_path, update_connection

_VERSION = 1


class SessionCatalogError(RuntimeError):
    """The local session catalog is invalid or cannot be updated safely."""


@dataclass(frozen=True, slots=True)
class Session:
    thread_id: str
    name: str
    created_at: str
    updated_at: str
    last_run_id: str | None = None
    last_sequence: int = 0


def sessions_path() -> Path:
    return config_path().with_name("sessions.json")


def list_sessions(connection: Connection) -> tuple[Session, ...]:
    payload = _read_catalog()
    server = _server(payload, connection.url)
    sessions = _sessions(server)
    if not any(item.thread_id == connection.thread_id for item in sessions):
        sessions.append(_current_session(connection, name="Main" if not sessions else _next_name(sessions)))
        _save_server(payload, connection.url, sessions)
    return tuple(sessions)


def create_session(connection: Connection, *, name: str | None = None) -> Connection:
    payload = _read_catalog()
    server = _server(payload, connection.url)
    sessions = _sessions(server)
    _remember_current(sessions, connection)
    safe_name = str(name or "").strip() or _next_name(sessions)
    _validate_name(safe_name)
    if any(item.name.casefold() == safe_name.casefold() for item in sessions):
        raise SessionCatalogError(f"A session named {safe_name!r} already exists.")
    now = _now()
    created = Session(
        thread_id=f"cli-{uuid4()}",
        name=safe_name,
        created_at=now,
        updated_at=now,
    )
    sessions.append(created)
    _save_server(payload, connection.url, sessions)
    return update_connection(
        connection,
        thread_id=created.thread_id,
        last_run_id=None,
        last_sequence=0,
    )


def switch_session(connection: Connection, selector: str) -> tuple[Connection, Session]:
    payload = _read_catalog()
    server = _server(payload, connection.url)
    sessions = _sessions(server)
    _remember_current(sessions, connection)
    selected = _select(sessions, selector)
    _save_server(payload, connection.url, sessions)
    updated = update_connection(
        connection,
        thread_id=selected.thread_id,
        last_run_id=selected.last_run_id,
        last_sequence=selected.last_sequence,
    )
    return updated, selected


def _select(sessions: list[Session], selector: str) -> Session:
    safe_selector = str(selector or "").strip()
    if not safe_selector:
        raise SessionCatalogError("Usage: /session NUMBER_OR_NAME")
    if safe_selector.isdigit():
        index = int(safe_selector) - 1
        if 0 <= index < len(sessions):
            return sessions[index]
    folded = safe_selector.casefold()
    matches = [
        item
        for item in sessions
        if item.name.casefold() == folded or item.thread_id == safe_selector
    ]
    if len(matches) == 1:
        return matches[0]
    raise SessionCatalogError(f"Unknown session: {safe_selector}")


def _remember_current(sessions: list[Session], connection: Connection) -> None:
    for index, item in enumerate(sessions):
        if item.thread_id != connection.thread_id:
            continue
        sessions[index] = Session(
            thread_id=item.thread_id,
            name=item.name,
            created_at=item.created_at,
            updated_at=_now(),
            last_run_id=connection.last_run_id,
            last_sequence=connection.last_sequence,
        )
        return
    sessions.append(
        _current_session(connection, name="Main" if not sessions else _next_name(sessions))
    )


def _current_session(connection: Connection, *, name: str) -> Session:
    now = _now()
    return Session(
        thread_id=connection.thread_id,
        name=name,
        created_at=now,
        updated_at=now,
        last_run_id=connection.last_run_id,
        last_sequence=connection.last_sequence,
    )


def _next_name(sessions: list[Session]) -> str:
    used = {item.name.casefold() for item in sessions}
    number = len(sessions) + 1
    while f"session {number}".casefold() in used:
        number += 1
    return f"Session {number}"


def _validate_name(name: str) -> None:
    if not name or len(name) > 80 or any(ord(character) < 32 for character in name):
        raise SessionCatalogError("Session names must be 1-80 printable characters.")


def _read_catalog() -> dict[str, Any]:
    path = sessions_path()
    if path.is_symlink():
        raise SessionCatalogError("Session catalog cannot be a symlink.")
    if not path.exists():
        return {"version": _VERSION, "servers": {}}
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise SessionCatalogError("Session catalog must be a private mode-0600 file.")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except SessionCatalogError:
        raise
    except (OSError, ValueError) as exc:
        raise SessionCatalogError("Remembered OpenTulpa sessions are invalid.") from exc
    if not isinstance(payload, dict) or payload.get("version") != _VERSION:
        raise SessionCatalogError("Remembered OpenTulpa sessions are unsupported.")
    if not isinstance(payload.get("servers"), dict):
        raise SessionCatalogError("Remembered OpenTulpa sessions are invalid.")
    return payload


def _server(payload: dict[str, Any], url: str) -> dict[str, Any]:
    servers = payload.setdefault("servers", {})
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    raw = servers.get(key)
    if raw is None:
        return {"url": url, "sessions": []}
    if not isinstance(raw, dict) or raw.get("url") != url:
        raise SessionCatalogError("Remembered OpenTulpa server sessions are invalid.")
    return raw


def _sessions(server: dict[str, Any]) -> list[Session]:
    raw_sessions = server.get("sessions", [])
    if not isinstance(raw_sessions, list):
        raise SessionCatalogError("Remembered OpenTulpa sessions are invalid.")
    sessions: list[Session] = []
    try:
        for raw in raw_sessions:
            if not isinstance(raw, dict):
                raise ValueError
            item = Session(
                thread_id=str(raw["thread_id"]),
                name=str(raw["name"]),
                created_at=str(raw["created_at"]),
                updated_at=str(raw["updated_at"]),
                last_run_id=str(raw.get("last_run_id") or "").strip() or None,
                last_sequence=max(0, int(raw.get("last_sequence", 0))),
            )
            if not item.thread_id or len(item.thread_id) > 200:
                raise ValueError
            _validate_name(item.name)
            sessions.append(item)
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionCatalogError("Remembered OpenTulpa sessions are invalid.") from exc
    if len({item.thread_id for item in sessions}) != len(sessions):
        raise SessionCatalogError("Remembered OpenTulpa sessions contain duplicate threads.")
    return sessions


def _save_server(payload: dict[str, Any], url: str, sessions: list[Session]) -> None:
    servers = payload.setdefault("servers", {})
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    servers[key] = {"url": url, "sessions": [asdict(item) for item in sessions]}
    _write_catalog(payload)


def _write_catalog(payload: dict[str, Any]) -> None:
    path = sessions_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise SessionCatalogError("Session catalog directory is unsafe.")
    os.chmod(path.parent, 0o700)
    if path.is_symlink() or (path.exists() and path.stat().st_mode & 0o077):
        raise SessionCatalogError("Session catalog must be a private mode-0600 file.")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "Session",
    "SessionCatalogError",
    "create_session",
    "list_sessions",
    "sessions_path",
    "switch_session",
]
