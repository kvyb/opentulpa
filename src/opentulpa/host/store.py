"""Crash-safe encrypted configuration store for the stable host."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from opentulpa.host.models import HostConfig, HostConfigInput, HostConfigView
from opentulpa.secrets.cipher import EncryptedSecret, HostKeySecretCipher

_SCHEMA_VERSION = 1


class HostConfigError(RuntimeError):
    """Stable host configuration could not be read or changed safely."""


class HostConfigConflictError(HostConfigError):
    """The caller edited a stale configuration revision."""


class HostStore:
    """Persist host ownership and encrypted, revisioned runtime configuration."""

    def __init__(self, path: Path, *, cipher: HostKeySecretCipher) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._cipher = cipher
        self._lock = threading.RLock()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        for path in (self.path, self.path.with_name(f"{self.path.name}-wal")):
            with suppress(FileNotFoundError):
                os.chmod(path, 0o600)
        return connection

    def _migrate(self) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS host_schema (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        version INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS host_auth (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        owner_token_hash TEXT,
                        setup_token_hash TEXT,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS host_config_revisions (
                        revision INTEGER PRIMARY KEY AUTOINCREMENT,
                        status TEXT NOT NULL CHECK (
                            status IN ('staged', 'active', 'inactive', 'failed')
                        ),
                        base_url TEXT NOT NULL,
                        model TEXT NOT NULL,
                        telegram_user_id INTEGER,
                        secrets_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        error TEXT
                    );
                    CREATE UNIQUE INDEX IF NOT EXISTS host_one_active_config
                    ON host_config_revisions(status) WHERE status = 'active';
                    """
                )
                row = connection.execute(
                    "SELECT version FROM host_schema WHERE singleton = 1"
                ).fetchone()
                if row is not None and int(row["version"]) > _SCHEMA_VERSION:
                    raise HostConfigError("host database uses a newer schema")
                connection.execute(
                    """
                    INSERT INTO host_schema(singleton, version, updated_at) VALUES(1, ?, ?)
                    ON CONFLICT(singleton) DO UPDATE SET version=excluded.version,
                        updated_at=excluded.updated_at
                    """,
                    (_SCHEMA_VERSION, now),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO host_auth(singleton, updated_at) VALUES(1, ?)",
                    (now,),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _token_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def configure_setup_token(self, token: str) -> None:
        value = token.strip()
        if len(value) < 16:
            raise ValueError("setup token must contain at least 16 characters")
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE host_auth SET setup_token_hash=COALESCE(setup_token_hash, ?), updated_at=?
                WHERE singleton=1 AND owner_token_hash IS NULL
                """,
                (self._token_hash(value), datetime.now(UTC).isoformat()),
            )
            connection.commit()

    @property
    def claimed(self) -> bool:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT owner_token_hash FROM host_auth WHERE singleton=1"
            ).fetchone()
        return bool(row and row["owner_token_hash"])

    def claim(self, *, setup_token: str | None, owner_token: str) -> None:
        owner = owner_token.strip()
        if len(owner) < 32:
            raise ValueError("owner token must contain at least 32 characters")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_token_hash, setup_token_hash FROM host_auth WHERE singleton=1"
            ).fetchone()
            if row is None:
                raise HostConfigError("host authentication state is missing")
            if row["owner_token_hash"]:
                if not hmac.compare_digest(self._token_hash(owner), str(row["owner_token_hash"])):
                    raise HostConfigConflictError("host is already claimed")
                connection.commit()
                return
            expected = str(row["setup_token_hash"] or "")
            supplied = self._token_hash(str(setup_token or "").strip())
            if not expected or not hmac.compare_digest(supplied, expected):
                raise PermissionError("invalid setup token")
            connection.execute(
                """
                UPDATE host_auth SET owner_token_hash=?, setup_token_hash=NULL, updated_at=?
                WHERE singleton=1
                """,
                (self._token_hash(owner), datetime.now(UTC).isoformat()),
            )
            connection.commit()

    def authorize_owner(self, token: str | None) -> bool:
        value = str(token or "").strip()
        if not value:
            return False
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT owner_token_hash FROM host_auth WHERE singleton=1"
            ).fetchone()
        expected = str(row["owner_token_hash"] or "") if row else ""
        return bool(expected and hmac.compare_digest(self._token_hash(value), expected))

    def stage(self, value: HostConfigInput) -> HostConfig:
        active = self.active()
        if active is None and value.expected_revision is not None:
            raise HostConfigConflictError("no active configuration exists")
        if active is not None and value.expected_revision != active.revision:
            raise HostConfigConflictError("active configuration revision changed")
        api_key = value.api_key.get_secret_value() if value.api_key else None
        if not api_key and active is not None:
            api_key = active.api_key.get_secret_value()
        if not api_key:
            raise ValueError("api_key is required for the first configuration")
        telegram_token = (
            value.telegram_bot_token.get_secret_value()
            if value.telegram_bot_token is not None
            else None
        )
        if telegram_token is None and active is not None:
            telegram_token = (
                active.telegram_bot_token.get_secret_value()
                if active.telegram_bot_token is not None
                else None
            )
        if telegram_token is None and value.telegram_user_id is not None:
            raise ValueError("Telegram bot token is required when a user ID is configured")
        internal_token = secrets.token_urlsafe(48)
        pairing_code = (
            telegram_token[-8:]
            if telegram_token and value.telegram_user_id is None
            else secrets.token_urlsafe(9)
            if telegram_token
            else None
        )
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO host_config_revisions(
                    status, base_url, model, telegram_user_id, secrets_json, created_at
                ) VALUES('staged', ?, ?, ?, '{}', ?)
                """,
                (
                    value.base_url,
                    value.model,
                    value.telegram_user_id,
                    datetime.now(UTC).isoformat(),
                ),
            )
            if cursor.lastrowid is None:
                raise HostConfigError("configuration revision was not allocated")
            revision = cursor.lastrowid
            encrypted = {
                "api_key": self._encrypt(revision, "api_key", api_key),
                "internal_runtime_token": self._encrypt(
                    revision, "internal_runtime_token", internal_token
                ),
                "telegram_bot_token": self._encrypt(revision, "telegram_bot_token", telegram_token),
                "telegram_pairing_code": self._encrypt(
                    revision, "telegram_pairing_code", pairing_code
                ),
            }
            connection.execute(
                "UPDATE host_config_revisions SET secrets_json=? WHERE revision=?",
                (json.dumps(encrypted, sort_keys=True, separators=(",", ":")), revision),
            )
            connection.commit()
        config = self.get(revision)
        if config is None:
            raise HostConfigError("staged configuration disappeared")
        return config

    def activate(self, revision: int) -> HostConfig:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM host_config_revisions WHERE revision=?", (revision,)
            ).fetchone()
            if row is None or row["status"] != "staged":
                raise HostConfigConflictError("configuration is not staged")
            connection.execute(
                "UPDATE host_config_revisions SET status='inactive' WHERE status='active'"
            )
            connection.execute(
                "UPDATE host_config_revisions SET status='active', error=NULL WHERE revision=?",
                (revision,),
            )
            connection.commit()
        config = self.get(revision)
        if config is None:
            raise HostConfigError("activated configuration disappeared")
        return config

    def fail(self, revision: int, error: str) -> None:
        safe_error = str(error or "configuration activation failed").strip()[:1_000]
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE host_config_revisions SET status='failed', error=?
                WHERE revision=? AND status='staged'
                """,
                (safe_error, revision),
            )
            connection.commit()

    def active(self) -> HostConfig | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM host_config_revisions WHERE status='active'"
            ).fetchone()
        return self._decode(row) if row else None

    def get(self, revision: int) -> HostConfig | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM host_config_revisions WHERE revision=?", (revision,)
            ).fetchone()
        return self._decode(row) if row else None

    def list_views(self) -> list[HostConfigView]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM host_config_revisions ORDER BY revision DESC LIMIT 20"
            ).fetchall()
        return [self.view(self._decode(row)) for row in rows]

    @staticmethod
    def view(config: HostConfig) -> HostConfigView:
        return HostConfigView(
            revision=config.revision,
            status=config.status,
            base_url=config.base_url,
            model=config.model,
            api_key_configured=True,
            telegram_configured=config.telegram_bot_token is not None,
            telegram_user_id=config.telegram_user_id,
            telegram_pairing_required=(
                config.telegram_bot_token is not None and config.telegram_user_id is None
            ),
            created_at=config.created_at,
            error=config.error,
        )

    def _associated_data(self, revision: int, name: str) -> bytes:
        return f"opentulpa-host-config:{revision}:{name}".encode()

    def _encrypt(self, revision: int, name: str, value: str | None) -> dict[str, str] | None:
        if value is None:
            return None
        encrypted = self._cipher.encrypt(
            value.encode(), associated_data=self._associated_data(revision, name)
        )
        return {
            "key_id": encrypted.key_id,
            "nonce": base64.urlsafe_b64encode(encrypted.nonce).decode(),
            "ciphertext": base64.urlsafe_b64encode(encrypted.ciphertext).decode(),
        }

    def _decrypt(self, revision: int, name: str, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise HostConfigError("encrypted host configuration is corrupt")
        try:
            encrypted = EncryptedSecret(
                key_id=str(value["key_id"]),
                nonce=base64.urlsafe_b64decode(value["nonce"]),
                ciphertext=base64.urlsafe_b64decode(value["ciphertext"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HostConfigError("encrypted host configuration is corrupt") from exc
        return self._cipher.decrypt(
            encrypted, associated_data=self._associated_data(revision, name)
        ).decode()

    def _decode(self, row: sqlite3.Row) -> HostConfig:
        revision = int(row["revision"])
        try:
            payload = json.loads(row["secrets_json"])
            api_key = self._decrypt(revision, "api_key", payload["api_key"])
            internal = self._decrypt(
                revision, "internal_runtime_token", payload["internal_runtime_token"]
            )
            telegram = self._decrypt(
                revision, "telegram_bot_token", payload.get("telegram_bot_token")
            )
            pairing = self._decrypt(
                revision, "telegram_pairing_code", payload.get("telegram_pairing_code")
            )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise HostConfigError("host configuration is corrupt") from exc
        if not api_key or not internal:
            raise HostConfigError("host configuration is incomplete")
        return HostConfig(
            revision=revision,
            status=row["status"],
            api_key=SecretStr(api_key),
            base_url=str(row["base_url"]),
            model=str(row["model"]),
            telegram_bot_token=SecretStr(telegram) if telegram else None,
            telegram_user_id=row["telegram_user_id"],
            internal_runtime_token=SecretStr(internal),
            telegram_pairing_code=SecretStr(pairing) if pairing else None,
            created_at=datetime.fromisoformat(str(row["created_at"])),
            error=row["error"],
        )


__all__ = ["HostConfigConflictError", "HostConfigError", "HostStore"]
