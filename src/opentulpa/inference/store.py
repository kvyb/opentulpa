"""Encrypted SQLite persistence for Codex credentials and device logins."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from opentulpa.secrets.cipher import EncryptedSecret, HostKeySecretCipher


@dataclass(frozen=True, slots=True)
class CodexCredential:
    access_token: str
    refresh_token: str
    id_token: str | None
    account_id: str | None
    expires_at: datetime
    revision: int = 1

    def __post_init__(self) -> None:
        if not self.access_token or not self.refresh_token:
            raise ValueError("Codex credential is missing a token")
        if self.expires_at.tzinfo is None:
            raise ValueError("Codex credential expiry must be timezone-aware")

    def expires_soon(self, *, skew: timedelta = timedelta(minutes=5)) -> bool:
        return datetime.now(UTC) >= self.expires_at.astimezone(UTC) - skew


@dataclass(frozen=True, slots=True)
class DeviceLogin:
    id: str
    tenant_id: str
    status: str
    verification_url: str
    user_code: str
    device_auth_id: str
    interval_seconds: float
    next_poll_at: datetime
    expires_at: datetime
    error_code: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS codex_credentials (
    tenant_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    key_id TEXT NOT NULL,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS codex_credential_epochs (
    tenant_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS codex_device_logins (
    login_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    status TEXT NOT NULL,
    key_id TEXT NOT NULL,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_codex_device_login_tenant
ON codex_device_logins (tenant_id, updated_at DESC);
"""


class InferenceCredentialStore:
    """Keep provider secrets encrypted; the host key never enters SQLite."""

    def __init__(
        self,
        path: str | Path,
        *,
        cipher: HostKeySecretCipher,
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = cipher
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def connected(self, tenant_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM codex_credentials WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return row is not None

    def credential_revision(self, tenant_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT revision FROM codex_credential_epochs WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def load_credential(self, tenant_id: str) -> CodexCredential | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM codex_credentials WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return self._credential_from_row(tenant_id, row) if row is not None else None

    def save_credential(self, tenant_id: str, credential: CodexCredential) -> CodexCredential:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT revision FROM codex_credential_epochs WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            revision = (int(current[0]) if current is not None else 0) + 1
            saved = CodexCredential(
                access_token=credential.access_token,
                refresh_token=credential.refresh_token,
                id_token=credential.id_token,
                account_id=credential.account_id,
                expires_at=credential.expires_at.astimezone(UTC),
                revision=revision,
            )
            self._write_credential(connection, tenant_id, saved)
            connection.commit()
        return saved

    def refresh_credential(
        self,
        tenant_id: str,
        refresh: Callable[[CodexCredential], CodexCredential],
        *,
        force: bool = False,
    ) -> CodexCredential:
        """Serialize refresh-token rotation with a database write lock."""

        with self._connect(timeout=45.0) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM codex_credentials WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise FileNotFoundError("Codex is not connected")
            current = self._credential_from_row(tenant_id, row)
            if not force and not current.expires_soon():
                connection.commit()
                return current
            refreshed = refresh(current)
            saved = CodexCredential(
                access_token=refreshed.access_token,
                refresh_token=refreshed.refresh_token,
                id_token=refreshed.id_token,
                account_id=refreshed.account_id,
                expires_at=refreshed.expires_at.astimezone(UTC),
                revision=current.revision + 1,
            )
            self._write_credential(connection, tenant_id, saved)
            connection.commit()
            return saved

    def delete_credential(self, tenant_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT revision FROM codex_credential_epochs WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            cursor = connection.execute(
                "DELETE FROM codex_credentials WHERE tenant_id = ?",
                (tenant_id,),
            )
            if cursor.rowcount == 1:
                connection.execute(
                    """
                    INSERT INTO codex_credential_epochs (tenant_id, revision)
                    VALUES (?, ?)
                    ON CONFLICT (tenant_id) DO UPDATE SET revision = excluded.revision
                    """,
                    (tenant_id, (int(current[0]) if current is not None else 0) + 1),
                )
            connection.commit()
        return cursor.rowcount == 1

    def create_device_login(self, login: DeviceLogin) -> None:
        payload = self._device_payload(login)
        encrypted = self._encrypt(
            payload,
            associated_data=self._device_aad(login.tenant_id, login.id),
        )
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO codex_device_logins (
                    login_id, tenant_id, status, key_id, nonce, ciphertext,
                    expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    login.id,
                    login.tenant_id,
                    login.status,
                    encrypted.key_id,
                    encrypted.nonce,
                    encrypted.ciphertext,
                    login.expires_at.astimezone(UTC).isoformat(),
                    now,
                ),
            )
            connection.commit()

    def load_device_login(self, tenant_id: str, login_id: str) -> DeviceLogin | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM codex_device_logins
                WHERE tenant_id = ? AND login_id = ?
                """,
                (tenant_id, login_id),
            ).fetchone()
        return self._device_from_row(row) if row is not None else None

    def update_device_login(self, login: DeviceLogin) -> None:
        encrypted = self._encrypt(
            self._device_payload(login),
            associated_data=self._device_aad(login.tenant_id, login.id),
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE codex_device_logins
                SET status = ?, key_id = ?, nonce = ?, ciphertext = ?,
                    expires_at = ?, updated_at = ?
                WHERE tenant_id = ? AND login_id = ?
                """,
                (
                    login.status,
                    encrypted.key_id,
                    encrypted.nonce,
                    encrypted.ciphertext,
                    login.expires_at.astimezone(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                    login.tenant_id,
                    login.id,
                ),
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise KeyError("device login not found")

    def delete_device_login(self, tenant_id: str, login_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM codex_device_logins
                WHERE tenant_id = ? AND login_id = ?
                """,
                (tenant_id, login_id),
            )
            connection.commit()
        return cursor.rowcount == 1

    def _write_credential(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        credential: CodexCredential,
    ) -> None:
        payload = {
            "access_token": credential.access_token,
            "refresh_token": credential.refresh_token,
            "id_token": credential.id_token,
            "account_id": credential.account_id,
            "expires_at": credential.expires_at.astimezone(UTC).isoformat(),
        }
        encrypted = self._encrypt(payload, associated_data=self._credential_aad(tenant_id))
        now = datetime.now(UTC).isoformat()
        connection.execute(
            """
            INSERT INTO codex_credentials (
                tenant_id, revision, key_id, nonce, ciphertext, expires_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (tenant_id) DO UPDATE SET
                revision = excluded.revision,
                key_id = excluded.key_id,
                nonce = excluded.nonce,
                ciphertext = excluded.ciphertext,
                expires_at = excluded.expires_at,
                updated_at = excluded.updated_at
            """,
            (
                tenant_id,
                credential.revision,
                encrypted.key_id,
                encrypted.nonce,
                encrypted.ciphertext,
                credential.expires_at.astimezone(UTC).isoformat(),
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO codex_credential_epochs (tenant_id, revision)
            VALUES (?, ?)
            ON CONFLICT (tenant_id) DO UPDATE SET revision = excluded.revision
            """,
            (tenant_id, credential.revision),
        )

    def _credential_from_row(
        self,
        tenant_id: str,
        row: sqlite3.Row,
    ) -> CodexCredential:
        payload = self._decrypt(row, associated_data=self._credential_aad(tenant_id))
        return CodexCredential(
            access_token=str(payload["access_token"]),
            refresh_token=str(payload["refresh_token"]),
            id_token=str(payload["id_token"]) if payload.get("id_token") else None,
            account_id=str(payload["account_id"]) if payload.get("account_id") else None,
            expires_at=datetime.fromisoformat(str(payload["expires_at"])).astimezone(UTC),
            revision=int(row["revision"]),
        )

    def _device_from_row(self, row: sqlite3.Row) -> DeviceLogin:
        tenant_id = str(row["tenant_id"])
        login_id = str(row["login_id"])
        payload = self._decrypt(row, associated_data=self._device_aad(tenant_id, login_id))
        return DeviceLogin(
            id=login_id,
            tenant_id=tenant_id,
            status=str(row["status"]),
            verification_url=str(payload["verification_url"]),
            user_code=str(payload["user_code"]),
            device_auth_id=str(payload["device_auth_id"]),
            interval_seconds=float(payload["interval_seconds"]),
            next_poll_at=datetime.fromisoformat(str(payload["next_poll_at"])).astimezone(UTC),
            expires_at=datetime.fromisoformat(str(payload["expires_at"])).astimezone(UTC),
            error_code=str(payload["error_code"]) if payload.get("error_code") else None,
        )

    @staticmethod
    def _device_payload(login: DeviceLogin) -> dict[str, Any]:
        return {
            "verification_url": login.verification_url,
            "user_code": login.user_code,
            "device_auth_id": login.device_auth_id,
            "interval_seconds": login.interval_seconds,
            "next_poll_at": login.next_poll_at.astimezone(UTC).isoformat(),
            "expires_at": login.expires_at.astimezone(UTC).isoformat(),
            "error_code": login.error_code,
        }

    def _encrypt(self, payload: dict[str, Any], *, associated_data: bytes) -> EncryptedSecret:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return self._cipher.encrypt(raw, associated_data=associated_data)

    def _decrypt(self, row: sqlite3.Row, *, associated_data: bytes) -> dict[str, Any]:
        raw = self._cipher.decrypt(
            EncryptedSecret(
                key_id=str(row["key_id"]),
                nonce=bytes(row["nonce"]),
                ciphertext=bytes(row["ciphertext"]),
            ),
            associated_data=associated_data,
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("encrypted inference record is invalid")
        return payload

    def _connect(self, *, timeout: float = 10.0) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @staticmethod
    def _credential_aad(tenant_id: str) -> bytes:
        return f"opentulpa:codex-credential:{tenant_id}".encode()

    @staticmethod
    def _device_aad(tenant_id: str, login_id: str) -> bytes:
        return f"opentulpa:codex-device:{tenant_id}:{login_id}".encode()


__all__ = ["CodexCredential", "DeviceLogin", "InferenceCredentialStore"]
