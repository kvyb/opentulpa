"""Private local connection state with OS-keychain credential storage."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

_KEYRING_SERVICE = "opentulpa.owner"
_CONFIG_VERSION = 2


class ClientConfigError(RuntimeError):
    """The remembered client connection is unavailable or unsafe."""


@dataclass(frozen=True, slots=True)
class Connection:
    url: str
    token: str
    thread_id: str
    credential_storage: str
    last_run_id: str | None = None
    last_sequence: int = 0


def config_path() -> Path:
    return Path(
        os.environ.get("OPENTULPA_CLIENT_CONFIG") or "~/.config/opentulpa/client.json"
    ).expanduser()


def normalize_url(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ClientConfigError(
            "Server URL must be an http(s) origin without a path, credentials, or query."
        )
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def save_connection(
    url: str,
    token: str,
    *,
    thread_id: str | None = None,
    last_run_id: str | None = None,
    last_sequence: int = 0,
) -> Connection:
    normalized = normalize_url(url)
    safe_token = str(token or "").strip()
    safe_thread_id = str(thread_id or f"cli-{uuid4()}").strip()
    if not safe_thread_id or len(safe_thread_id) > 200:
        raise ClientConfigError("OpenTulpa thread is invalid.")
    account = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    try:
        previous = _read_payload()
    except ClientConfigError:
        previous = {}
    storage = "none"
    payload_token: str | None = None
    if safe_token:
        if _keyring_set(account, safe_token):
            storage = "keyring"
        else:
            storage = "file"
            payload_token = safe_token
    connection = Connection(
        url=normalized,
        token=safe_token,
        thread_id=safe_thread_id,
        credential_storage=storage,
        last_run_id=str(last_run_id or "").strip() or None,
        last_sequence=max(0, int(last_sequence)),
    )
    try:
        _write_payload(_payload(connection, account=account, fallback_token=payload_token))
    except Exception:
        if storage == "keyring":
            _keyring_delete(account)
        raise
    previous_account = str(previous.get("credential_account") or "")
    if (
        str(previous.get("credential_storage") or "") == "keyring"
        and previous_account
        and (storage != "keyring" or previous_account != account)
    ):
        _keyring_delete(previous_account)
    return connection


def load_connection() -> Connection:
    payload = _read_payload()
    try:
        version = int(payload.get("version", 1))
    except (TypeError, ValueError) as exc:
        raise ClientConfigError("Remembered OpenTulpa connection is invalid.") from exc
    if version == 1 and "token" in payload:
        return save_connection(str(payload.get("url") or ""), str(payload.get("token") or ""))
    if version != _CONFIG_VERSION:
        raise ClientConfigError("Remembered OpenTulpa connection version is unsupported.")
    try:
        url = normalize_url(str(payload["url"]))
        thread_id = str(payload["thread_id"]).strip()
        storage = str(payload.get("credential_storage") or "none")
        account = str(payload.get("credential_account") or "").strip()
        last_sequence = max(0, int(payload.get("last_sequence", 0)))
    except (KeyError, TypeError, ValueError) as exc:
        raise ClientConfigError("Remembered OpenTulpa connection is invalid.") from exc
    if not thread_id or len(thread_id) > 200:
        raise ClientConfigError("Remembered OpenTulpa thread is invalid.")
    if storage == "keyring":
        token = _keyring_get(account)
        if token is None:
            raise ClientConfigError(
                "The owner credential is missing from the OS keychain. Run `opentulpa connect` again."
            )
    elif storage == "file":
        token = str(payload.get("token") or "").strip()
        if not token:
            raise ClientConfigError("Remembered owner credential is missing.")
    elif storage == "none":
        token = ""
    else:
        raise ClientConfigError("Remembered credential storage is invalid.")
    last_run_id = str(payload.get("last_run_id") or "").strip() or None
    if last_run_id is not None and len(last_run_id) > 300:
        raise ClientConfigError("Remembered OpenTulpa run is invalid.")
    return Connection(
        url=url,
        token=token,
        thread_id=thread_id,
        credential_storage=storage,
        last_run_id=last_run_id,
        last_sequence=last_sequence,
    )


def update_connection(
    connection: Connection,
    *,
    thread_id: str | None = None,
    last_run_id: str | None | object = ...,
    last_sequence: int | None = None,
) -> Connection:
    next_thread_id = connection.thread_id if thread_id is None else str(thread_id).strip()
    if not next_thread_id or len(next_thread_id) > 200:
        raise ClientConfigError("OpenTulpa thread is invalid.")
    updated = replace(
        connection,
        thread_id=next_thread_id,
        last_run_id=(
            connection.last_run_id if last_run_id is ... else str(last_run_id or "").strip() or None
        ),
        last_sequence=(
            connection.last_sequence if last_sequence is None else max(0, int(last_sequence))
        ),
    )
    payload = _read_payload()
    payload.update(
        {
            "thread_id": updated.thread_id,
            "last_run_id": updated.last_run_id,
            "last_sequence": updated.last_sequence,
        }
    )
    _write_payload(payload)
    return updated


def clear_connection() -> None:
    path = config_path()
    try:
        payload = _read_payload()
    except ClientConfigError:
        payload = {}
    if str(payload.get("credential_storage") or "") == "keyring":
        _keyring_delete(str(payload.get("credential_account") or ""))
    try:
        if path.is_symlink():
            raise ClientConfigError("Client config cannot be a symlink.")
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise ClientConfigError("Could not remove the remembered connection.") from exc


def _payload(
    connection: Connection,
    *,
    account: str,
    fallback_token: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": _CONFIG_VERSION,
        "url": connection.url,
        "thread_id": connection.thread_id,
        "credential_storage": connection.credential_storage,
        "credential_account": account if connection.credential_storage == "keyring" else None,
        "last_run_id": connection.last_run_id,
        "last_sequence": connection.last_sequence,
    }
    if fallback_token is not None:
        payload["token"] = fallback_token
    return payload


def _read_payload() -> dict[str, Any]:
    path = config_path()
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ClientConfigError("Client config must be a private mode-0600 regular file.")
        value = json.loads(path.read_text(encoding="utf-8"))
    except ClientConfigError:
        raise
    except (OSError, ValueError) as exc:
        raise ClientConfigError("Run `opentulpa connect SERVER_URL` first.") from exc
    if not isinstance(value, dict):
        raise ClientConfigError("Remembered OpenTulpa connection is invalid.")
    return value


def _write_payload(payload: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ClientConfigError("Client config directory is unsafe.")
    os.chmod(path.parent, 0o700)
    if path.exists() and (path.is_symlink() or path.stat().st_mode & 0o077):
        raise ClientConfigError("Client config must be a private mode-0600 regular file.")
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


def _keyring_set(account: str, token: str) -> bool:
    try:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, account, token)
        return keyring.get_password(_KEYRING_SERVICE, account) == token
    except Exception:
        return False


def _keyring_get(account: str) -> str | None:
    if not account:
        return None
    try:
        import keyring

        value = keyring.get_password(_KEYRING_SERVICE, account)
    except Exception:
        return None
    return str(value).strip() if value else None


def _keyring_delete(account: str) -> None:
    if not account:
        return
    try:
        import keyring

        keyring.delete_password(_KEYRING_SERVICE, account)
    except Exception:
        return


__all__ = [
    "ClientConfigError",
    "Connection",
    "clear_connection",
    "config_path",
    "load_connection",
    "normalize_url",
    "save_connection",
    "update_connection",
]
