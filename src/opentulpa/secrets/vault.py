"""Encrypted tenant secret vault with pending handles and ephemeral grants."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from pydantic import SecretStr

from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.secrets.cipher import EncryptedSecret, HostKeySecretCipher
from opentulpa.secrets.models import (
    IssuedSecretGrant,
    SecretGrantReceipt,
    SecretHandle,
    SecretMaterial,
    SecretState,
)


class SecretVaultError(RuntimeError):
    """Sanitized base error for secret-vault operations."""


class SecretVaultConflictError(SecretVaultError):
    """A secret revision changed before a requested transition."""


class SecretVaultNotFoundError(SecretVaultError, KeyError):
    """A tenant-owned secret handle does not exist."""


class SecretGrantError(SecretVaultError):
    """A grant is expired, consumed, out of scope, or otherwise invalid."""


class SecretVault:
    """Keep plaintext out of storage and release it only through one-time grants."""

    def __init__(
        self,
        db_path: Path,
        *,
        cipher: HostKeySecretCipher,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = cipher
        self._clock = clock or (lambda: datetime.now(UTC))
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = connect_sqlite(self.db_path, wal=True)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("secret vault clock must return an aware datetime")
        return value.astimezone(UTC)

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS secret_revisions (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    name TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'active', 'revoked')),
                    scopes_json TEXT NOT NULL,
                    key_id TEXT,
                    nonce BLOB,
                    ciphertext BLOB,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id, revision),
                    CHECK (
                        (state = 'active' AND key_id IS NOT NULL
                         AND nonce IS NOT NULL AND ciphertext IS NOT NULL)
                        OR
                        (state != 'active' AND key_id IS NULL
                         AND nonce IS NULL AND ciphertext IS NULL)
                    )
                );

                CREATE TABLE IF NOT EXISTS secret_current_refs (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id),
                    FOREIGN KEY (tenant_id, id, revision)
                        REFERENCES secret_revisions (tenant_id, id, revision)
                );

                CREATE TABLE IF NOT EXISTS secret_grants (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    secret_id TEXT NOT NULL,
                    secret_revision INTEGER NOT NULL CHECK (secret_revision >= 1),
                    capability_id TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT,
                    FOREIGN KEY (tenant_id, secret_id, secret_revision)
                        REFERENCES secret_revisions (tenant_id, id, revision)
                );

                CREATE INDEX IF NOT EXISTS idx_secret_current_tenant
                ON secret_current_refs (tenant_id, id);

                CREATE INDEX IF NOT EXISTS idx_secret_grants_expiry
                ON secret_grants (expires_at, consumed_at);
                """
            )
            conn.commit()

    @staticmethod
    def _scopes(value: Sequence[str]) -> tuple[str, ...]:
        scopes = tuple(str(item or "").strip() for item in value)
        if not scopes or any(not item for item in scopes):
            raise ValueError("at least one non-empty secret scope is required")
        if len(scopes) != len(set(scopes)):
            raise ValueError("secret scopes must be unique")
        return scopes

    @staticmethod
    def _handle(row: sqlite3.Row) -> SecretHandle:
        return SecretHandle(
            tenant_id=row["tenant_id"],
            id=row["id"],
            revision=row["revision"],
            name=row["name"],
            state=row["state"],
            scopes=tuple(json.loads(str(row["scopes_json"]))),
            created_at=row["created_at"],
            created_by=row["created_by"],
        )

    @staticmethod
    def _aad(*, tenant_id: str, secret_id: str, revision: int, name: str) -> bytes:
        return f"opentulpa-secret-v1\0{tenant_id}\0{secret_id}\0{revision}\0{name}".encode()

    @staticmethod
    def _clean_value(value: str | SecretStr) -> bytes:
        plaintext = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        encoded = plaintext.encode("utf-8")
        if not encoded:
            raise ValueError("secret value must not be empty")
        if len(encoded) > 1_048_576:
            raise ValueError("secret value must be at most 1 MiB")
        return encoded

    def create_pending(
        self,
        *,
        tenant_id: str,
        name: str,
        scopes: Sequence[str],
        created_by: str,
        secret_id: str | None = None,
    ) -> SecretHandle:
        handle_id = str(secret_id or "").strip() or new_short_id("sec", suffix_chars=12)
        now = self._now()
        handle = SecretHandle(
            tenant_id=tenant_id,
            id=handle_id,
            revision=1,
            name=name,
            state=SecretState.PENDING,
            scopes=self._scopes(scopes),
            created_at=now,
            created_by=created_by,
        )
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            exists = conn.execute(
                "SELECT 1 FROM secret_current_refs WHERE tenant_id = ? AND id = ?",
                (handle.tenant_id, handle.id),
            ).fetchone()
            if exists is not None:
                conn.rollback()
                raise SecretVaultConflictError("secret handle already exists")
            conn.execute(
                """
                INSERT INTO secret_revisions (
                    tenant_id, id, revision, name, state, scopes_json,
                    created_at, created_by
                ) VALUES (?, ?, 1, ?, 'pending', ?, ?, ?)
                """,
                (
                    handle.tenant_id,
                    handle.id,
                    handle.name,
                    json.dumps(handle.scopes, separators=(",", ":")),
                    handle.created_at.isoformat(),
                    handle.created_by,
                ),
            )
            conn.execute(
                """
                INSERT INTO secret_current_refs (
                    tenant_id, id, revision, updated_at, updated_by
                ) VALUES (?, ?, 1, ?, ?)
                """,
                (
                    handle.tenant_id,
                    handle.id,
                    handle.created_at.isoformat(),
                    handle.created_by,
                ),
            )
            conn.commit()
        return handle

    def get_handle(self, *, tenant_id: str, secret_id: str) -> SecretHandle | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT revisions.* FROM secret_current_refs AS current
                JOIN secret_revisions AS revisions
                  ON revisions.tenant_id = current.tenant_id
                 AND revisions.id = current.id
                 AND revisions.revision = current.revision
                WHERE current.tenant_id = ? AND current.id = ?
                """,
                (tenant_id, secret_id),
            ).fetchone()
        return self._handle(row) if row is not None else None

    def list_handles(self, *, tenant_id: str) -> list[SecretHandle]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT revisions.* FROM secret_current_refs AS current
                JOIN secret_revisions AS revisions
                  ON revisions.tenant_id = current.tenant_id
                 AND revisions.id = current.id
                 AND revisions.revision = current.revision
                WHERE current.tenant_id = ?
                ORDER BY revisions.name ASC, revisions.id ASC
                """,
                (tenant_id,),
            ).fetchall()
        return [self._handle(row) for row in rows]

    def fulfill(
        self,
        *,
        tenant_id: str,
        secret_id: str,
        expected_revision: int,
        value: str | SecretStr,
        updated_by: str,
    ) -> SecretHandle:
        return self._write_active_revision(
            tenant_id=tenant_id,
            secret_id=secret_id,
            expected_revision=expected_revision,
            expected_state=SecretState.PENDING,
            value=value,
            scopes=None,
            updated_by=updated_by,
        )

    def rotate(
        self,
        *,
        tenant_id: str,
        secret_id: str,
        expected_revision: int,
        value: str | SecretStr,
        updated_by: str,
        scopes: Sequence[str] | None = None,
    ) -> SecretHandle:
        return self._write_active_revision(
            tenant_id=tenant_id,
            secret_id=secret_id,
            expected_revision=expected_revision,
            expected_state=SecretState.ACTIVE,
            value=value,
            scopes=self._scopes(scopes) if scopes is not None else None,
            updated_by=updated_by,
        )

    def _write_active_revision(
        self,
        *,
        tenant_id: str,
        secret_id: str,
        expected_revision: int,
        expected_state: SecretState,
        value: str | SecretStr,
        scopes: tuple[str, ...] | None,
        updated_by: str,
    ) -> SecretHandle:
        plaintext = self._clean_value(value)
        now = self._now()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._current_row(conn, tenant_id=tenant_id, secret_id=secret_id)
            handle = self._handle(current)
            if handle.revision != expected_revision:
                conn.rollback()
                raise SecretVaultConflictError(
                    f"secret revision is {handle.revision}, expected {expected_revision}"
                )
            if handle.state is not expected_state:
                conn.rollback()
                raise SecretVaultConflictError(
                    f"secret must be {expected_state.value} for this operation"
                )
            revision = handle.revision + 1
            next_scopes = scopes or handle.scopes
            encrypted = self._cipher.encrypt(
                plaintext,
                associated_data=self._aad(
                    tenant_id=handle.tenant_id,
                    secret_id=handle.id,
                    revision=revision,
                    name=handle.name,
                ),
            )
            next_handle = SecretHandle(
                tenant_id=handle.tenant_id,
                id=handle.id,
                revision=revision,
                name=handle.name,
                state=SecretState.ACTIVE,
                scopes=next_scopes,
                created_at=now,
                created_by=updated_by,
            )
            conn.execute(
                """
                INSERT INTO secret_revisions (
                    tenant_id, id, revision, name, state, scopes_json,
                    key_id, nonce, ciphertext, created_at, created_by
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)
                """,
                (
                    next_handle.tenant_id,
                    next_handle.id,
                    next_handle.revision,
                    next_handle.name,
                    json.dumps(next_handle.scopes, separators=(",", ":")),
                    encrypted.key_id,
                    encrypted.nonce,
                    encrypted.ciphertext,
                    next_handle.created_at.isoformat(),
                    next_handle.created_by,
                ),
            )
            self._advance_current_ref(
                conn,
                handle=next_handle,
                expected_revision=handle.revision,
                updated_at=now,
                updated_by=updated_by,
            )
            conn.commit()
            return next_handle

    def revoke(
        self,
        *,
        tenant_id: str,
        secret_id: str,
        expected_revision: int,
        updated_by: str,
    ) -> SecretHandle:
        now = self._now()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._current_row(conn, tenant_id=tenant_id, secret_id=secret_id)
            handle = self._handle(current)
            if handle.revision != expected_revision:
                conn.rollback()
                raise SecretVaultConflictError(
                    f"secret revision is {handle.revision}, expected {expected_revision}"
                )
            if handle.state is SecretState.REVOKED:
                conn.rollback()
                return handle
            revoked = SecretHandle(
                tenant_id=handle.tenant_id,
                id=handle.id,
                revision=handle.revision + 1,
                name=handle.name,
                state=SecretState.REVOKED,
                scopes=handle.scopes,
                created_at=now,
                created_by=updated_by,
            )
            conn.execute(
                """
                INSERT INTO secret_revisions (
                    tenant_id, id, revision, name, state, scopes_json,
                    created_at, created_by
                ) VALUES (?, ?, ?, ?, 'revoked', ?, ?, ?)
                """,
                (
                    revoked.tenant_id,
                    revoked.id,
                    revoked.revision,
                    revoked.name,
                    json.dumps(revoked.scopes, separators=(",", ":")),
                    revoked.created_at.isoformat(),
                    revoked.created_by,
                ),
            )
            self._advance_current_ref(
                conn,
                handle=revoked,
                expected_revision=handle.revision,
                updated_at=now,
                updated_by=updated_by,
            )
            conn.commit()
            return revoked

    @staticmethod
    def _current_row(
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        secret_id: str,
    ) -> sqlite3.Row:
        row = conn.execute(
            """
            SELECT revisions.* FROM secret_current_refs AS current
            JOIN secret_revisions AS revisions
              ON revisions.tenant_id = current.tenant_id
             AND revisions.id = current.id
             AND revisions.revision = current.revision
            WHERE current.tenant_id = ? AND current.id = ?
            """,
            (tenant_id, secret_id),
        ).fetchone()
        if row is None:
            raise SecretVaultNotFoundError(secret_id)
        return cast(sqlite3.Row, row)

    @staticmethod
    def _advance_current_ref(
        conn: sqlite3.Connection,
        *,
        handle: SecretHandle,
        expected_revision: int,
        updated_at: datetime,
        updated_by: str,
    ) -> None:
        cursor = conn.execute(
            """
            UPDATE secret_current_refs
            SET revision = ?, updated_at = ?, updated_by = ?
            WHERE tenant_id = ? AND id = ? AND revision = ?
            """,
            (
                handle.revision,
                updated_at.isoformat(),
                updated_by,
                handle.tenant_id,
                handle.id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise SecretVaultConflictError("secret changed during revision transition")

    def issue_grant(
        self,
        *,
        tenant_id: str,
        secret_id: str,
        capability_id: str,
        scopes: Sequence[str],
        ttl_seconds: int = 60,
    ) -> IssuedSecretGrant:
        requested_scopes = self._scopes(scopes)
        if ttl_seconds < 1 or ttl_seconds > 3_600:
            raise ValueError("secret grant TTL must be between 1 and 3600 seconds")
        now = self._now()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._current_row(conn, tenant_id=tenant_id, secret_id=secret_id)
            handle = self._handle(row)
            if handle.state is not SecretState.ACTIVE:
                conn.rollback()
                raise SecretGrantError("secret is not active")
            if not set(requested_scopes).issubset(handle.scopes):
                conn.rollback()
                raise SecretGrantError("requested scope is not allowed for this secret")
            grant_id = new_short_id("grant", suffix_chars=12)
            raw_token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
            receipt = SecretGrantReceipt(
                id=grant_id,
                tenant_id=handle.tenant_id,
                secret_id=handle.id,
                secret_revision=handle.revision,
                capability_id=capability_id,
                scopes=requested_scopes,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
            conn.execute(
                """
                INSERT INTO secret_grants (
                    id, tenant_id, secret_id, secret_revision, capability_id,
                    scopes_json, token_hash, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.id,
                    receipt.tenant_id,
                    receipt.secret_id,
                    receipt.secret_revision,
                    receipt.capability_id,
                    json.dumps(receipt.scopes, separators=(",", ":")),
                    token_hash,
                    receipt.expires_at.isoformat(),
                    now.isoformat(),
                ),
            )
            conn.commit()
        return IssuedSecretGrant(receipt=receipt, token=SecretStr(raw_token))

    def redeem_grant(
        self,
        *,
        token: str | SecretStr,
        capability_id: str,
        scope: str,
    ) -> SecretMaterial:
        raw_token = token.get_secret_value() if isinstance(token, SecretStr) else str(token)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        now = self._now()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT grants.*, revisions.name, revisions.state,
                       revisions.key_id, revisions.nonce, revisions.ciphertext,
                       current.revision AS current_revision
                FROM secret_grants AS grants
                JOIN secret_revisions AS revisions
                  ON revisions.tenant_id = grants.tenant_id
                 AND revisions.id = grants.secret_id
                 AND revisions.revision = grants.secret_revision
                JOIN secret_current_refs AS current
                  ON current.tenant_id = grants.tenant_id
                 AND current.id = grants.secret_id
                WHERE grants.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise SecretGrantError("secret grant is invalid")
            if row["consumed_at"] is not None:
                conn.rollback()
                raise SecretGrantError("secret grant was already consumed")
            if datetime.fromisoformat(str(row["expires_at"])) <= now:
                conn.rollback()
                raise SecretGrantError("secret grant has expired")
            if str(row["capability_id"]) != str(capability_id or "").strip():
                conn.rollback()
                raise SecretGrantError("secret grant belongs to another capability")
            granted_scopes = tuple(json.loads(str(row["scopes_json"])))
            if scope not in granted_scopes:
                conn.rollback()
                raise SecretGrantError("secret grant does not include the requested scope")
            if int(row["current_revision"]) != int(row["secret_revision"]):
                conn.rollback()
                raise SecretGrantError("secret grant was invalidated by a newer revision")
            if str(row["state"]) != SecretState.ACTIVE.value:
                conn.rollback()
                raise SecretGrantError("secret is not active")
            encrypted = EncryptedSecret(
                key_id=str(row["key_id"]),
                nonce=bytes(row["nonce"]),
                ciphertext=bytes(row["ciphertext"]),
            )
            plaintext = self._cipher.decrypt(
                encrypted,
                associated_data=self._aad(
                    tenant_id=str(row["tenant_id"]),
                    secret_id=str(row["secret_id"]),
                    revision=int(row["secret_revision"]),
                    name=str(row["name"]),
                ),
            ).decode("utf-8")
            cursor = conn.execute(
                """
                UPDATE secret_grants SET consumed_at = ?
                WHERE id = ? AND consumed_at IS NULL
                """,
                (now.isoformat(), row["id"]),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise SecretGrantError("secret grant was already consumed")
            conn.commit()
        return SecretMaterial(
            grant_id=row["id"],
            secret_id=row["secret_id"],
            name=row["name"],
            scope=scope,
            value=SecretStr(plaintext),
        )

    def redact_text(self, *, tenant_id: str, text: str) -> str:
        """Replace current and historical tenant secrets without exposing them."""

        return self._redact_text(str(text), self._plaintext_values(tenant_id=tenant_id))

    @staticmethod
    def _redact_text(text: str, values: Sequence[str]) -> str:
        redacted = text
        for value in sorted(values, key=len, reverse=True):
            if value:
                redacted = redacted.replace(value, "[redacted]")
        return redacted

    def redact_payload(self, *, tenant_id: str, value: Any) -> Any:
        """Recursively redact exact tenant secret values from log or event payloads."""

        return self._redact_payload(value, self._plaintext_values(tenant_id=tenant_id))

    @classmethod
    def _redact_payload(cls, value: Any, sensitive_values: Sequence[str]) -> Any:
        if isinstance(value, str):
            return cls._redact_text(value, sensitive_values)
        if isinstance(value, Mapping):
            return {
                str(key): cls._redact_payload(item, sensitive_values)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(cls._redact_payload(item, sensitive_values) for item in value)
        if isinstance(value, list):
            return [cls._redact_payload(item, sensitive_values) for item in value]
        return value

    def _plaintext_values(self, *, tenant_id: str) -> tuple[str, ...]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM secret_revisions
                WHERE tenant_id = ? AND state = 'active'
                """,
                (tenant_id,),
            ).fetchall()
        values: list[str] = []
        for row in rows:
            plaintext = self._cipher.decrypt(
                EncryptedSecret(
                    key_id=str(row["key_id"]),
                    nonce=bytes(row["nonce"]),
                    ciphertext=bytes(row["ciphertext"]),
                ),
                associated_data=self._aad(
                    tenant_id=str(row["tenant_id"]),
                    secret_id=str(row["id"]),
                    revision=int(row["revision"]),
                    name=str(row["name"]),
                ),
            ).decode("utf-8")
            values.append(plaintext)
        return tuple(values)


__all__ = [
    "SecretGrantError",
    "SecretVault",
    "SecretVaultConflictError",
    "SecretVaultError",
    "SecretVaultNotFoundError",
]
