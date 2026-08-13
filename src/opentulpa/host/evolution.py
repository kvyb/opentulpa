"""Trusted source editing and activation owned by the stable host."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import sys
import threading
from collections.abc import Mapping
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import JsonValue

from opentulpa.evolution.context import EvolutionAuditContext
from opentulpa.evolution.git_security import (
    discover_git_directories,
    repository_mutation_lock,
    run_hardened_git,
)
from opentulpa.evolution.models import EvolutionEvent
from opentulpa.evolution.process import run_bounded_process
from opentulpa.host.runtime import RuntimeLiveSourceSpec, RuntimeSupervisor
from opentulpa.host.runtime_environment import (
    LiveSourceRuntimeEnvironmentStore,
    RuntimeEnvFileManager,
    RuntimeEnvironmentError,
)

_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_SECRET_SOURCE_NAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "credentials.toml",
        "credentials.yaml",
        "credentials.yml",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "secrets.toml",
        "secrets.yaml",
        "secrets.yml",
    }
)
_SECRET_SOURCE_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_SECRET_SOURCE_PATHS = frozenset({".aws/credentials", ".docker/config.json"})
_TRUSTED_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class SourceEvolutionError(RuntimeError):
    """A trusted source operation failed without changing the active release."""


class _ActivationJournal:
    """One crash-safe source of truth for releases and activation attempts."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        for path in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            with suppress(FileNotFoundError):
                path.chmod(0o600)
        return connection

    def _migrate(self) -> None:
        now = self._now()
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS source_journal_schema (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        version INTEGER NOT NULL CHECK (version >= 1),
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS source_releases (
                        id TEXT PRIMARY KEY,
                        source_commit TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS source_state (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        active_release_id TEXT NOT NULL,
                        previous_release_id TEXT,
                        last_good_release_id TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (active_release_id) REFERENCES source_releases(id),
                        FOREIGN KEY (previous_release_id) REFERENCES source_releases(id),
                        FOREIGN KEY (last_good_release_id) REFERENCES source_releases(id)
                    );
                    CREATE TABLE IF NOT EXISTS source_activations (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        request_hash TEXT NOT NULL,
                        kind TEXT NOT NULL CHECK (kind IN ('activate', 'rollback')),
                        target_release_id TEXT NOT NULL,
                        previous_release_id TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('preparing', 'active', 'failed', 'rolled_back')
                        ),
                        reason TEXT NOT NULL,
                        audit_json TEXT NOT NULL,
                        result_json TEXT,
                        notified INTEGER NOT NULL DEFAULT 0 CHECK (notified IN (0, 1)),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE (tenant_id, idempotency_key),
                        FOREIGN KEY (target_release_id) REFERENCES source_releases(id),
                        FOREIGN KEY (previous_release_id) REFERENCES source_releases(id)
                    );
                    CREATE UNIQUE INDEX IF NOT EXISTS source_one_activation_in_progress
                    ON source_activations(status) WHERE status = 'preparing';
                    """
                )
                row = connection.execute(
                    "SELECT version FROM source_journal_schema WHERE singleton = 1"
                ).fetchone()
                if row is not None and int(row["version"]) > 1:
                    raise SourceEvolutionError("source journal uses a newer schema")
                connection.execute(
                    """
                    INSERT INTO source_journal_schema(singleton, version, updated_at)
                    VALUES(1, 1, ?)
                    ON CONFLICT(singleton) DO UPDATE SET version=1, updated_at=excluded.updated_at
                    """,
                    (now,),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def initialize(self, source_commit: str) -> dict[str, Any]:
        release = self.release_for_commit(source_commit)
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO source_state(
                    singleton, active_release_id, previous_release_id,
                    last_good_release_id, updated_at
                ) VALUES(1, ?, NULL, ?, ?)
                """,
                (release["id"], release["id"], self._now()),
            )
            connection.commit()
        return self.state()

    def release_for_commit(self, source_commit: str) -> dict[str, str]:
        commit = _commit(source_commit)
        release_id = f"release_{commit[:64]}"
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO source_releases(id, source_commit, created_at) VALUES(?, ?, ?)",
                (release_id, commit, self._now()),
            )
            row = connection.execute(
                "SELECT id, source_commit, created_at FROM source_releases WHERE id = ?",
                (release_id,),
            ).fetchone()
            connection.commit()
        if row is None or str(row["source_commit"]) != commit:
            raise SourceEvolutionError("source release identity conflicts with persisted history")
        return {key: str(row[key]) for key in ("id", "source_commit", "created_at")}

    def release(self, release_id: str | None) -> dict[str, str] | None:
        if release_id is None:
            return None
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, source_commit, created_at FROM source_releases WHERE id = ?",
                (release_id,),
            ).fetchone()
        if row is None:
            return None
        return {key: str(row[key]) for key in ("id", "source_commit", "created_at")}

    def state(self) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM source_state WHERE singleton = 1").fetchone()
        if row is None:
            raise SourceEvolutionError("source activation state is unavailable")
        return dict(row)

    def begin(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        request_hash: str,
        kind: str,
        target_release_id: str,
        previous_release_id: str,
        reason: str,
        audit: Mapping[str, JsonValue],
    ) -> tuple[dict[str, Any], bool]:
        activation_id = "activation_" + hashlib.sha256(
            f"{tenant_id}\0{idempotency_key}".encode()
        ).hexdigest()[:48]
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    """
                    SELECT * FROM source_activations
                    WHERE tenant_id = ? AND idempotency_key = ?
                    """,
                    (tenant_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    operation = self._operation(existing)
                    if operation["request_hash"] != request_hash:
                        raise SourceEvolutionError(
                            "source activation idempotency key was used for another request"
                        )
                    connection.commit()
                    return operation, True
                state = connection.execute(
                    "SELECT active_release_id FROM source_state WHERE singleton = 1"
                ).fetchone()
                if state is None or str(state["active_release_id"]) != previous_release_id:
                    raise SourceEvolutionError("active source changed before activation")
                now = self._now()
                connection.execute(
                    """
                    INSERT INTO source_activations(
                        id, tenant_id, idempotency_key, request_hash, kind,
                        target_release_id, previous_release_id, status, reason,
                        audit_json, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, 'preparing', ?, ?, ?, ?)
                    """,
                    (
                        activation_id,
                        tenant_id,
                        idempotency_key,
                        request_hash,
                        kind,
                        target_release_id,
                        previous_release_id,
                        reason,
                        json.dumps(dict(audit), sort_keys=True, separators=(",", ":")),
                        now,
                        now,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM source_activations WHERE id = ?", (activation_id,)
                ).fetchone()
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise SourceEvolutionError("another source activation is already running") from exc
            except BaseException:
                connection.rollback()
                raise
        assert row is not None
        return self._operation(row), False

    def operation(self, activation_id: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM source_activations WHERE id = ?", (activation_id,)
            ).fetchone()
        return self._operation(row) if row is not None else None

    def find(self, *, tenant_id: str, idempotency_key: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM source_activations
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (tenant_id, idempotency_key),
            ).fetchone()
        return self._operation(row) if row is not None else None

    def pending(self) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM source_activations WHERE status = 'preparing' ORDER BY created_at"
            ).fetchall()
        return [self._operation(row) for row in rows]

    def latest(self) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM source_activations ORDER BY created_at DESC, id DESC LIMIT 1"
            ).fetchone()
        return self._operation(row) if row is not None else None

    def complete_success(
        self,
        activation_id: str,
        *,
        result: Mapping[str, JsonValue],
    ) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM source_activations WHERE id = ?", (activation_id,)
                ).fetchone()
                if row is None:
                    raise SourceEvolutionError("source activation disappeared")
                operation = self._operation(row)
                if operation["status"] != "preparing":
                    connection.commit()
                    return operation
                state = connection.execute(
                    "SELECT active_release_id FROM source_state WHERE singleton = 1"
                ).fetchone()
                if state is None or str(state["active_release_id"]) != operation["previous_release_id"]:
                    raise SourceEvolutionError("active source changed during activation")
                now = self._now()
                connection.execute(
                    """
                    UPDATE source_state SET active_release_id=?, previous_release_id=?,
                        last_good_release_id=?, updated_at=? WHERE singleton=1
                    """,
                    (
                        operation["target_release_id"],
                        operation["previous_release_id"],
                        operation["target_release_id"],
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE source_activations SET status='active', result_json=?, updated_at=?
                    WHERE id=? AND status='preparing'
                    """,
                    (json.dumps(dict(result), sort_keys=True, separators=(",", ":")), now, activation_id),
                )
                updated = connection.execute(
                    "SELECT * FROM source_activations WHERE id = ?", (activation_id,)
                ).fetchone()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        assert updated is not None
        return self._operation(updated)

    def complete_failure(
        self,
        activation_id: str,
        *,
        rolled_back: bool,
        result: Mapping[str, JsonValue],
    ) -> dict[str, Any]:
        status = "rolled_back" if rolled_back else "failed"
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE source_activations SET status=?, result_json=?, updated_at=?
                WHERE id=? AND status='preparing'
                """,
                (
                    status,
                    json.dumps(dict(result), sort_keys=True, separators=(",", ":")),
                    self._now(),
                    activation_id,
                ),
            )
            connection.commit()
        operation = self.operation(activation_id)
        if operation is None:
            raise SourceEvolutionError("source activation disappeared")
        return operation

    def recovery_candidate(
        self, failed_source_commit: str, *, allow_pending: bool
    ) -> dict[str, str] | None:
        failed_commit = _commit(failed_source_commit)
        with self._lock, closing(self._connect()) as connection:
            state = connection.execute(
                """
                SELECT state.active_release_id, state.previous_release_id,
                    active.source_commit AS active_source_commit
                FROM source_state AS state
                JOIN source_releases AS active ON active.id = state.active_release_id
                WHERE state.singleton = 1
                """
            ).fetchone()
            if (
                state is None
                or str(state["active_source_commit"]) != failed_commit
                or state["previous_release_id"] is None
            ):
                return None
            if not allow_pending and connection.execute(
                "SELECT 1 FROM source_activations WHERE status='preparing' LIMIT 1"
            ).fetchone() is not None:
                return None
            latest = connection.execute(
                """
                SELECT kind FROM source_activations
                WHERE target_release_id = ? AND status = 'active'
                ORDER BY updated_at DESC, id DESC LIMIT 1
                """,
                (state["active_release_id"],),
            ).fetchone()
            if latest is not None and str(latest["kind"]) == "rollback":
                return None
            candidate = connection.execute(
                "SELECT id, source_commit, created_at FROM source_releases WHERE id = ?",
                (state["previous_release_id"],),
            ).fetchone()
        if candidate is None or str(candidate["source_commit"]) == failed_commit:
            return None
        return {key: str(candidate[key]) for key in ("id", "source_commit", "created_at")}

    def complete_recovery(
        self,
        *,
        failed_release_id: str,
        selected_release_id: str,
        invalidate_pending: bool,
    ) -> None:
        now = self._now()
        result = {
            "status": "rolled_back",
            "reason": "runtime source failed operational checks",
            "active_release_id": selected_release_id,
            "failed_release_id": failed_release_id,
        }
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = connection.execute(
                    "SELECT * FROM source_state WHERE singleton = 1"
                ).fetchone()
                if state is None or str(state["active_release_id"]) != failed_release_id:
                    raise SourceEvolutionError("active source changed during runtime recovery")
                if str(state["previous_release_id"] or "") != selected_release_id:
                    raise SourceEvolutionError("runtime recovery source changed before activation")
                pending = connection.execute(
                    "SELECT id FROM source_activations WHERE status='preparing' LIMIT 1"
                ).fetchone()
                if pending is not None and not invalidate_pending:
                    raise SourceEvolutionError("source activation changed during runtime recovery")
                connection.execute(
                    """
                    UPDATE source_state SET active_release_id=?, previous_release_id=NULL,
                        last_good_release_id=?, updated_at=? WHERE singleton=1
                    """,
                    (selected_release_id, selected_release_id, now),
                )
                connection.execute(
                    """
                    UPDATE source_activations SET status='rolled_back', result_json=?, updated_at=?
                    WHERE id = (
                        SELECT id FROM source_activations
                        WHERE target_release_id=? AND status='active'
                        ORDER BY updated_at DESC, id DESC LIMIT 1
                    )
                    """,
                    (encoded, now, failed_release_id),
                )
                if invalidate_pending:
                    connection.execute(
                        """
                        UPDATE source_activations SET status='failed', result_json=?, updated_at=?
                        WHERE status='preparing' AND previous_release_id=?
                        """,
                        (encoded, now, failed_release_id),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def pending_notifications(self) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM source_activations
                WHERE status != 'preparing' AND notified = 0 AND result_json IS NOT NULL
                ORDER BY updated_at
                """
            ).fetchall()
        return [self._operation(row) for row in rows]

    def mark_notified(self, activation_id: str) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                "UPDATE source_activations SET notified=1, updated_at=? WHERE id=?",
                (self._now(), activation_id),
            )
            connection.commit()

    @staticmethod
    def _operation(row: sqlite3.Row) -> dict[str, Any]:
        operation = dict(row)
        operation["audit"] = json.loads(str(operation.pop("audit_json")))
        raw_result = operation.pop("result_json")
        operation["result"] = json.loads(str(raw_result)) if raw_result is not None else None
        operation["notified"] = bool(operation["notified"])
        return operation

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()


class _TrustedSourceWorkspace:
    """Persistent independent repository edited directly by the trusted model."""

    def __init__(
        self,
        *,
        source_repository: Path,
        path: Path,
        max_output_bytes: int,
    ) -> None:
        self._source = source_repository.expanduser().resolve(strict=True)
        self.path = path.expanduser().absolute()
        self._max_output_bytes = max_output_bytes

    def prepare(self) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        bundled = _git(self._source, "rev-parse", "--verify", "HEAD^{commit}").strip()
        bundled = _commit(bundled)
        if not os.path.lexists(self.path):
            try:
                _git(
                    self.path.parent,
                    "-c",
                    "protocol.file.allow=always",
                    "clone",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(self._source),
                    str(self.path),
                )
                _git(self.path, "remote", "remove", "origin")
                _git(self.path, "config", "--local", "user.name", "OpenTulpa")
                _git(self.path, "config", "--local", "user.email", "opentulpa@localhost")
                _git(self.path, "checkout", "--detach", bundled)
            except BaseException:
                shutil.rmtree(self.path, ignore_errors=True)
                raise
        self._validate()
        self.path.chmod(0o700)
        if _git(
            self.path,
            "cat-file",
            "-e",
            f"{bundled}^{{commit}}",
            check=False,
        ).returncode != 0:
            fetched = _git(
                self.path,
                "-c",
                "protocol.file.allow=always",
                "fetch",
                "--no-tags",
                str(self._source),
                bundled,
            )
            del fetched
            if _git(self.path, "rev-parse", "--verify", "FETCH_HEAD^{commit}").strip() != bundled:
                raise SourceEvolutionError("bundled source import selected the wrong commit")
        _git(self.path, "update-ref", "refs/opentulpa/bundled", bundled)
        return bundled

    def _validate(self) -> None:
        if self.path.is_symlink() or not self.path.is_dir():
            raise SourceEvolutionError("trusted source workspace is unavailable")
        git_directory, common_directory = discover_git_directories(self.path)
        if git_directory != common_directory:
            raise SourceEvolutionError("trusted source workspace must be an independent repository")
        if not (self.path / "uv.lock").is_file() or not (
            self.path / "src" / "opentulpa" / "__init__.py"
        ).is_file():
            raise SourceEvolutionError("trusted source workspace is not an OpenTulpa checkout")

    def head(self) -> str:
        self._validate()
        return _commit(_git(self.path, "rev-parse", "--verify", "HEAD^{commit}").strip())

    def changes(self) -> tuple[str, ...]:
        raw = _git(
            self.path,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ).stdout
        return tuple(entry for entry in raw.split("\0") if entry)

    def require_clean_head(self, source_commit: str) -> None:
        if self.head() != _commit(source_commit) or self.changes():
            raise SourceEvolutionError("trusted source changed after activation was requested")

    def commit(self, message: str) -> tuple[str, bool]:
        safe_message = " ".join(str(message or "").split())[:500] or "OpenTulpa self-update"
        self._validate()
        _git(self.path, "add", "--all")
        unmerged = _git(self.path, "diff", "--name-only", "--diff-filter=U", "-z").stdout
        if unmerged:
            raise SourceEvolutionError("trusted source has unresolved merge conflicts")
        changed = tuple(
            path
            for path in _git(self.path, "diff", "--cached", "--name-only", "-z").stdout.split("\0")
            if path
        )
        forbidden = tuple(path for path in changed if self._forbidden_source_path(path))
        if forbidden:
            _git(self.path, "reset", "--", ".")
            raise SourceEvolutionError("trusted source contains a credential file")
        if not changed:
            return self.head(), False
        _git(
            self.path,
            "commit",
            "--no-gpg-sign",
            "--no-verify",
            "-m",
            safe_message,
        )
        commit = self.head()
        if self.changes():
            raise SourceEvolutionError("trusted source remained dirty after commit")
        _git(self.path, "update-ref", "refs/opentulpa/workspace", commit)
        return commit, True

    def import_into_live_repository(self, source_commit: str) -> None:
        commit = _commit(source_commit)
        resolved = _git(self.path, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
        if resolved != commit:
            raise SourceEvolutionError("trusted source commit is unavailable")
        _, common_directory = discover_git_directories(self._source)
        with repository_mutation_lock(common_directory):
            status = _status_without_untracked_runtime_env(
                _git(
                    self._source,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "-z",
                ).output
            )
            if status:
                raise SourceEvolutionError("live source checkout is dirty")
            _git(
                self._source,
                "-c",
                "protocol.file.allow=always",
                "fetch",
                "--no-tags",
                str(self.path),
                commit,
            )
            if _git(self._source, "rev-parse", "--verify", "FETCH_HEAD^{commit}").strip() != commit:
                raise SourceEvolutionError("trusted source import selected the wrong commit")
            source_tree = _git(self.path, "show", "--no-patch", "--format=%T", commit).strip()
            imported_tree = _git(
                self._source, "show", "--no-patch", "--format=%T", commit
            ).strip()
            if imported_tree != source_tree:
                raise SourceEvolutionError("trusted source import changed the commit tree")
            _git(self._source, "update-ref", f"refs/opentulpa/releases/{commit}", commit)

    def read(self, path: str, *, offset: int, limit: int) -> dict[str, JsonValue]:
        target, relative = self._source_path(path, must_exist=True)
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SourceEvolutionError("source file is not readable UTF-8 text") from exc
        lines = text.splitlines(keepends=True)
        start = max(0, offset - 1)
        selected = lines[start : start + limit]
        return {
            "path": relative,
            "content": "".join(selected),
            "offset": offset,
            "lines": len(selected),
            "total_lines": len(lines),
            "truncated": start + len(selected) < len(lines),
        }

    def write(self, path: str, content: str) -> dict[str, JsonValue]:
        target, relative = self._source_path(path, must_exist=False, create_parent=True)
        encoded = str(content).encode("utf-8")
        previous_mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
        temporary = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
        try:
            temporary.write_bytes(encoded)
            temporary.chmod(previous_mode)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": relative, "bytes_written": len(encoded)}

    def edit(
        self,
        path: str,
        *,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> dict[str, JsonValue]:
        target, relative = self._source_path(path, must_exist=True)
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SourceEvolutionError("source file is not readable UTF-8 text") from exc
        old = str(old_text)
        if not old:
            raise ValueError("source edit old_text is required")
        count = content.count(old)
        if count == 0:
            raise SourceEvolutionError("source edit text was not found")
        if count > 1 and not replace_all:
            raise SourceEvolutionError("source edit text is not unique")
        updated = content.replace(old, str(new_text), -1 if replace_all else 1)
        self.write(relative, updated)
        return {"path": relative, "replacements": count if replace_all else 1}

    def bash(self, command: str, *, timeout_seconds: int) -> dict[str, JsonValue]:
        safe_command = str(command or "").strip()
        if not safe_command or "\x00" in safe_command or len(safe_command) > 100_000:
            raise ValueError("source bash command is invalid")
        timeout = int(timeout_seconds)
        if timeout < 1 or timeout > 600:
            raise ValueError("source bash timeout must be between 1 and 600 seconds")
        result = run_bounded_process(
            ("/bin/sh", "-c", safe_command),
            cwd=self.path,
            env={
                "PATH": os.environ.get("PATH", _TRUSTED_SYSTEM_PATH),
                "HOME": "/tmp",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "GIT_TERMINAL_PROMPT": "0",
            },
            timeout_seconds=timeout,
            max_output_bytes=self._max_output_bytes,
        )
        output = result.output.decode("utf-8", errors="replace")
        return {
            "exit_code": result.returncode,
            "output": output,
            "output_truncated": result.truncated,
            "timed_out": result.timed_out,
        }

    def _source_path(
        self,
        raw_path: str,
        *,
        must_exist: bool,
        create_parent: bool = False,
    ) -> tuple[Path, str]:
        value = str(raw_path or "").strip().replace("\\", "/")
        relative = PurePosixPath(value)
        if (
            not value
            or "\x00" in value
            or relative.is_absolute()
            or any(part in {"..", ".git"} for part in relative.parts)
        ):
            raise ValueError("source path is invalid")
        current = self.path
        for component in relative.parts[:-1]:
            current /= component
            if os.path.lexists(current):
                metadata = current.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise SourceEvolutionError("source path has an unsafe parent")
            elif create_parent:
                current.mkdir(mode=0o755)
            else:
                raise SourceEvolutionError("source file does not exist")
        target = self.path.joinpath(*relative.parts)
        if os.path.lexists(target) and target.is_symlink():
            raise SourceEvolutionError("source path cannot be a symbolic link")
        if must_exist and not target.is_file():
            raise SourceEvolutionError("source file does not exist")
        if not must_exist and target.exists() and not target.is_file():
            raise SourceEvolutionError("source path is not a regular file")
        return target, relative.as_posix()

    @staticmethod
    def _forbidden_source_path(path: str) -> bool:
        candidate = PurePosixPath(path)
        normalized = candidate.as_posix().casefold()
        name = candidate.name.casefold()
        return (
            name == ".env"
            or name.startswith(".env.")
            or name in _SECRET_SOURCE_NAMES
            or candidate.suffix.lower() in _SECRET_SOURCE_SUFFIXES
            or normalized in _SECRET_SOURCE_PATHS
        )


class HostEvolutionControlService:
    """Stable host service for direct source editing, activation, and rollback."""

    def __init__(
        self,
        *,
        runtime: RuntimeSupervisor,
        workspace: _TrustedSourceWorkspace,
        journal: _ActivationJournal,
        runtime_environment_store: LiveSourceRuntimeEnvironmentStore,
        runtime_env_file_manager: RuntimeEnvFileManager | None = None,
        check_timeout_seconds: int = 120,
        max_output_bytes: int = 1_000_000,
    ) -> None:
        self._runtime = runtime
        self._workspace = workspace
        self._journal = journal
        self._runtime_environment_store = runtime_environment_store
        self._runtime_env_file_manager = runtime_env_file_manager
        self._check_timeout_seconds = check_timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._operation_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._prepared = False
        self._started = False

    @property
    def service(self) -> HostEvolutionControlService:
        return self

    @property
    def source_mutation_enabled(self) -> bool:
        return True

    async def prepare(self) -> None:
        if self._prepared:
            return
        bundled = await asyncio.to_thread(self._workspace.prepare)
        state = await asyncio.to_thread(self._journal.initialize, bundled)
        release = await asyncio.to_thread(self._journal.release, state["active_release_id"])
        if release is None:
            raise SourceEvolutionError("active source release is unavailable")
        self._runtime.configure_source_recovery(self._recovery_source, self._complete_recovery)
        try:
            spec = await self._prepare_spec(str(release["source_commit"]))
        except Exception:
            spec = RuntimeLiveSourceSpec(source_commit=str(release["source_commit"]))
        self._runtime.set_live_source(spec)
        self._prepared = True

    async def _recovery_source(
        self, failed: RuntimeLiveSourceSpec, allow_pending: bool
    ) -> RuntimeLiveSourceSpec | None:
        candidate = await asyncio.to_thread(
            self._journal.recovery_candidate,
            failed.source_commit,
            allow_pending=allow_pending,
        )
        if candidate is None:
            return None
        try:
            return await self._prepare_spec(candidate["source_commit"])
        except Exception:
            return None

    async def _complete_recovery(
        self,
        failed: RuntimeLiveSourceSpec,
        selected: RuntimeLiveSourceSpec,
        invalidate_pending: bool,
    ) -> None:
        failed_release = await asyncio.to_thread(
            self._journal.release_for_commit, failed.source_commit
        )
        selected_release = await asyncio.to_thread(
            self._journal.release_for_commit, selected.source_commit
        )
        await asyncio.to_thread(
            self._journal.complete_recovery,
            failed_release_id=failed_release["id"],
            selected_release_id=selected_release["id"],
            invalidate_pending=invalidate_pending,
        )

    async def start(self) -> None:
        if self._started:
            return
        if not self._prepared:
            await self.prepare()
        self._started = True
        for operation in await asyncio.to_thread(self._journal.pending):
            self._schedule(str(operation["id"]))
        await self._flush_notifications()

    async def shutdown(self) -> None:
        self._started = False
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def source_status(
        self,
        *,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        del audit_context
        self._require_prepared()
        state, head, changes, latest = await asyncio.gather(
            asyncio.to_thread(self._journal.state),
            asyncio.to_thread(self._workspace.head),
            asyncio.to_thread(self._workspace.changes),
            asyncio.to_thread(self._journal.latest),
        )
        active = await asyncio.to_thread(self._journal.release, state["active_release_id"])
        previous = await asyncio.to_thread(self._journal.release, state["previous_release_id"])
        last_good = await asyncio.to_thread(self._journal.release, state["last_good_release_id"])
        return {
            "available": True,
            "workspace_head": head,
            "dirty": bool(changes),
            "changes": list(changes[:200]),
            "changes_truncated": len(changes) > 200,
            "active_release_id": str(state["active_release_id"]),
            "active_source_commit": str(active["source_commit"]) if active else None,
            "previous_release_id": str(state["previous_release_id"])
            if state["previous_release_id"]
            else None,
            "previous_source_commit": str(previous["source_commit"]) if previous else None,
            "last_good_release_id": str(state["last_good_release_id"]),
            "last_good_source_commit": str(last_good["source_commit"]) if last_good else None,
            "runtime_status": self._runtime.status,
            "activation": self._public_operation(latest) if latest is not None else None,
        }

    async def source_read(
        self,
        *,
        path: str,
        offset: int = 1,
        limit: int = 2_000,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        del audit_context
        self._require_started()
        return await asyncio.to_thread(
            self._workspace.read,
            path,
            offset=max(1, int(offset)),
            limit=max(1, min(int(limit), 2_000)),
        )

    async def source_write(
        self,
        *,
        path: str,
        content: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        del audit_context
        self._require_started()
        async with self._operation_lock:
            return await asyncio.to_thread(self._workspace.write, path, content)

    async def source_edit(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        del audit_context
        self._require_started()
        async with self._operation_lock:
            return await asyncio.to_thread(
                self._workspace.edit,
                path,
                old_text=old_text,
                new_text=new_text,
                replace_all=bool(replace_all),
            )

    async def source_bash(
        self,
        *,
        command: str,
        timeout_seconds: int = 300,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        del audit_context
        self._require_started()
        async with self._operation_lock:
            return await asyncio.to_thread(
                self._workspace.bash,
                command,
                timeout_seconds=timeout_seconds,
            )

    async def source_activate(
        self,
        *,
        idempotency_key: str,
        message: str = "OpenTulpa self-update",
        reason: str = "Trusted source activation",
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        self._require_started()
        key = self._idempotency_key(idempotency_key)
        audit = EvolutionAuditContext.from_mapping(audit_context)
        tenant_id = audit.tenant_id or "owner"
        safe_reason = " ".join(str(reason or "").split())[:4_000]
        safe_message = " ".join(str(message or "").split())[:500]
        async with self._operation_lock:
            existing = await asyncio.to_thread(
                self._journal.find,
                tenant_id=tenant_id,
                idempotency_key=key,
            )
            if existing is not None:
                expected_hash = self._request_hash(
                    kind="activate",
                    target_release_id=str(existing["target_release_id"]),
                    reason=safe_reason,
                    message=safe_message,
                )
                if existing["request_hash"] != expected_hash:
                    raise SourceEvolutionError(
                        "source activation idempotency key was used for another request"
                    )
                public = self._public_operation(existing)
                public["replayed"] = True
                return public
            commit, changed = await asyncio.to_thread(self._workspace.commit, message)
            state = await asyncio.to_thread(self._journal.state)
            active = await asyncio.to_thread(self._journal.release, state["active_release_id"])
            if active is not None and active["source_commit"] == commit:
                return {
                    "status": "already_active",
                    "active_release_id": str(active["id"]),
                    "source_commit": commit,
                    "committed": changed,
                }
            target = await asyncio.to_thread(self._journal.release_for_commit, commit)
            request_hash = self._request_hash(
                kind="activate",
                target_release_id=target["id"],
                reason=safe_reason,
                message=safe_message,
            )
            operation, replayed = await asyncio.to_thread(
                self._journal.begin,
                tenant_id=tenant_id,
                idempotency_key=key,
                request_hash=request_hash,
                kind="activate",
                target_release_id=target["id"],
                previous_release_id=str(state["active_release_id"]),
                reason=safe_reason,
                audit=audit.as_metadata(),
            )
            if operation["status"] == "preparing":
                self._schedule(str(operation["id"]))
            public = self._public_operation(operation)
            public["replayed"] = replayed
            public["committed"] = changed
            return public

    async def source_rollback(
        self,
        *,
        idempotency_key: str,
        expected_active_release_id: str,
        reason: str = "Owner requested rollback",
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        self._require_started()
        key = self._idempotency_key(idempotency_key)
        audit = EvolutionAuditContext.from_mapping(audit_context)
        tenant_id = audit.tenant_id or "owner"
        safe_reason = " ".join(str(reason or "").split())[:4_000]
        async with self._operation_lock:
            existing = await asyncio.to_thread(
                self._journal.find,
                tenant_id=tenant_id,
                idempotency_key=key,
            )
            if existing is not None:
                expected_hash = self._request_hash(
                    kind="rollback",
                    target_release_id=str(existing["target_release_id"]),
                    reason=safe_reason,
                    message="",
                )
                if existing["request_hash"] != expected_hash:
                    raise SourceEvolutionError(
                        "source activation idempotency key was used for another request"
                    )
                public = self._public_operation(existing)
                public["replayed"] = True
                return public
            state = await asyncio.to_thread(self._journal.state)
            if str(state["active_release_id"]) != str(expected_active_release_id or "").strip():
                raise SourceEvolutionError("active source changed before rollback")
            target_id = state["previous_release_id"]
            if target_id is None:
                raise SourceEvolutionError("no previous source release is available")
            request_hash = self._request_hash(
                kind="rollback",
                target_release_id=str(target_id),
                reason=safe_reason,
                message="",
            )
            operation, replayed = await asyncio.to_thread(
                self._journal.begin,
                tenant_id=tenant_id,
                idempotency_key=key,
                request_hash=request_hash,
                kind="rollback",
                target_release_id=str(target_id),
                previous_release_id=str(state["active_release_id"]),
                reason=safe_reason,
                audit=audit.as_metadata(),
            )
            if operation["status"] == "preparing":
                self._schedule(str(operation["id"]))
            public = self._public_operation(operation)
            public["replayed"] = replayed
            return public

    async def source_runtime_env_get(
        self,
        *,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        del audit_context
        manager = self._runtime_env_file_manager
        if manager is None:
            return {"available": False, "variables": [], "count": 0}
        return await manager.read()

    async def source_set_runtime_env(
        self,
        *,
        name: str,
        value: str,
        idempotency_key: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        manager = self._runtime_env_file_manager
        if manager is None:
            raise SourceEvolutionError("runtime environment updates are unavailable")
        return await manager.set(
            name=name,
            value=value,
            idempotency_key=idempotency_key,
            audit_context=audit_context,
        )

    def _schedule(self, activation_id: str) -> None:
        if not self._started or activation_id in self._tasks:
            return
        self._tasks[activation_id] = asyncio.create_task(
            self._run_activation(activation_id),
            name=f"source-activation-{activation_id}",
        )

    async def _run_activation(self, activation_id: str) -> None:
        try:
            async with self._operation_lock:
                operation = await asyncio.to_thread(self._journal.operation, activation_id)
                if operation is None or operation["status"] != "preparing":
                    return
                target = await asyncio.to_thread(
                    self._journal.release, operation["target_release_id"]
                )
                previous = await asyncio.to_thread(
                    self._journal.release, operation["previous_release_id"]
                )
                if target is None or previous is None:
                    raise SourceEvolutionError("source activation release history is incomplete")
                checks: list[JsonValue] = []
                try:
                    if operation["kind"] == "activate":
                        await asyncio.to_thread(
                            self._workspace.require_clean_head, target["source_commit"]
                        )
                    target_spec = await self._prepare_spec(target["source_commit"])
                    if operation["kind"] == "activate":
                        checks = await self._run_checks()
                    previous_spec = self._runtime.live_source
                    if (
                        previous_spec is None
                        or previous_spec.source_commit != previous["source_commit"]
                    ):
                        previous_spec = await self._prepare_spec(previous["source_commit"])
                    await self._runtime.replace_live_source(target_spec, rollback=previous_spec)
                    result: dict[str, JsonValue] = {
                        "status": "rolled_back"
                        if operation["kind"] == "rollback"
                        else "active",
                        "activation_id": activation_id,
                        "kind": str(operation["kind"]),
                        "active_release_id": str(target["id"]),
                        "source_commit": str(target["source_commit"]),
                        "previous_release_id": str(previous["id"]),
                        "checks": checks,
                    }
                    operation = await asyncio.to_thread(
                        self._journal.complete_success,
                        activation_id,
                        result=result,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    active_spec = self._runtime.live_source
                    restored = bool(
                        active_spec is not None
                        and active_spec.source_commit == previous["source_commit"]
                        and self._runtime.status in {"ready", "probation"}
                    )
                    result = {
                        "status": "rolled_back" if restored else "failed",
                        "activation_id": activation_id,
                        "kind": str(operation["kind"]),
                        "active_release_id": str(previous["id"]) if restored else None,
                        "source_commit": str(previous["source_commit"]) if restored else None,
                        "target_release_id": str(target["id"]),
                        "target_source_commit": str(target["source_commit"]),
                        "checks": checks,
                        "error": self._safe_error(exc),
                    }
                    operation = await asyncio.to_thread(
                        self._journal.complete_failure,
                        activation_id,
                        rolled_back=restored,
                        result=result,
                    )
                await self._notify(operation)
        finally:
            self._tasks.pop(activation_id, None)

    async def _prepare_spec(self, source_commit: str) -> RuntimeLiveSourceSpec:
        await asyncio.to_thread(self._workspace.import_into_live_repository, source_commit)
        environment = await asyncio.to_thread(
            self._runtime_environment_store.prepare, source_commit
        )
        return RuntimeLiveSourceSpec.from_release_metadata(
            environment.release_metadata(),
            source_commit=source_commit,
        )

    async def _run_checks(self) -> list[JsonValue]:
        scripts = (
            (
                "python.compile",
                "from pathlib import Path; "
                "files=(p for p in Path('src').rglob('*.py')); "
                "[compile(p.read_bytes(),str(p),'exec') for p in files]",
            ),
        )
        results: list[JsonValue] = []
        for name, script in scripts:
            result = await asyncio.to_thread(
                run_bounded_process,
                (sys.executable, "-I", "-S", "-B", "-c", script),
                cwd=self._workspace.path,
                env={
                    "PATH": _TRUSTED_SYSTEM_PATH,
                    "HOME": "/tmp",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                },
                timeout_seconds=self._check_timeout_seconds,
                max_output_bytes=self._max_output_bytes,
            )
            output = result.output.decode("utf-8", errors="replace").strip()
            passed = result.returncode == 0 and not result.timed_out and not result.truncated
            results.append(
                {
                    "name": name,
                    "passed": passed,
                    "exit_code": result.returncode,
                    "output": output[:2_000],
                    "output_truncated": result.truncated,
                }
            )
            if not passed:
                raise SourceEvolutionError(f"source activation check {name} failed")
        return results

    async def _flush_notifications(self) -> None:
        for operation in await asyncio.to_thread(self._journal.pending_notifications):
            await self._notify(operation)

    async def _notify(self, operation: Mapping[str, Any]) -> None:
        result = operation.get("result")
        if not isinstance(result, Mapping):
            return
        kind = str(operation["kind"])
        status = str(operation["status"])
        if status == "active":
            event_type = "build.rolled_back" if kind == "rollback" else "build.active"
        else:
            event_type = "rollback.failed" if kind == "rollback" else "promotion.failed"
        event = EvolutionEvent(
            event_key=f"source:{operation['id']}:{status}",
            event_type=event_type,
            release_id=str(operation["target_release_id"]),
            origin=dict(operation.get("audit") or {}),
            payload=dict(result),
        )
        try:
            await self._runtime.deliver_evolution_event(event)
        except Exception:
            return
        await asyncio.to_thread(self._journal.mark_notified, operation["id"])

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise SourceEvolutionError("trusted source service is not prepared")

    def _require_started(self) -> None:
        if not self._started:
            raise SourceEvolutionError("trusted source service is not started")

    @staticmethod
    def _public_operation(operation: Mapping[str, Any] | None) -> dict[str, JsonValue]:
        if operation is None:
            return {}
        result = operation.get("result")
        if isinstance(result, Mapping):
            return dict(result)
        return {
            "status": str(operation["status"]),
            "activation_id": str(operation["id"]),
            "kind": str(operation["kind"]),
            "target_release_id": str(operation["target_release_id"]),
        }

    @staticmethod
    def _idempotency_key(value: str) -> str:
        key = str(value or "").strip()
        if not key or len(key) > 200:
            raise ValueError("source activation idempotency key is invalid")
        return key

    @staticmethod
    def _request_hash(*, kind: str, target_release_id: str, reason: str, message: str) -> str:
        payload = json.dumps(
            {
                "kind": kind,
                "target_release_id": target_release_id,
                "reason": reason,
                "message": message,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _safe_error(error: Exception) -> str:
        if isinstance(error, (SourceEvolutionError, RuntimeEnvironmentError)):
            message = str(error).strip()
            return message[:1_000] if message else "source activation failed"
        return "source activation failed"


def prepare_live_source_repository(repository: Path) -> Path:
    """Validate the mounted source checkout used by the stable host."""

    source = repository.expanduser().resolve(strict=True)
    if source.is_symlink() or not (source / "uv.lock").is_file():
        raise RuntimeError("live source repository is unavailable")
    if not (source / "src" / "opentulpa" / "__init__.py").is_file():
        raise RuntimeError("live source repository is not an OpenTulpa checkout")
    try:
        _, common_directory = discover_git_directories(source)
    except Exception as exc:
        raise RuntimeError("live source Git metadata is unavailable") from exc
    with repository_mutation_lock(common_directory):
        if _status_without_untracked_runtime_env(
            _git(source, "status", "--porcelain=v1", "--untracked-files=all", "-z").output
        ):
            raise RuntimeError("live source repository is unexpectedly dirty")
        head = _git(source, "rev-parse", "--verify", "HEAD^{commit}").strip()
        if _git(source, "cat-file", "-e", f"{head}:.env", check=False).returncode == 0:
            raise RuntimeError("live source repository must not commit .env")
    return source


def _commit(value: str) -> str:
    commit = str(value or "").strip().lower()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise SourceEvolutionError("source commit identity is invalid")
    return commit


def _status_without_untracked_runtime_env(status: bytes) -> bytes:
    dirty: list[bytes] = []
    parts = status.split(b"\0")
    index = 0
    while index < len(parts):
        entry = parts[index]
        index += 1
        if not entry:
            continue
        code = entry[:2]
        path = entry[3:] if len(entry) > 3 else b""
        if code == b"??" and path == b".env":
            continue
        dirty.append(entry)
        if (code[:1] in {b"R", b"C"} or code[1:2] in {b"R", b"C"}) and index < len(parts):
            dirty.append(parts[index])
            index += 1
    return b"\0".join(dirty)


class _GitResult:
    def __init__(self, returncode: int, output: bytes) -> None:
        self.returncode = returncode
        self.output = output
        self.stdout = output.decode("utf-8", errors="replace")

    def strip(self) -> str:
        return self.stdout.strip()


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
    max_output_bytes: int = 50 * 1024 * 1024,
) -> _GitResult:
    result = run_hardened_git(
        repository,
        tuple(arguments),
        env={
            "GIT_AUTHOR_NAME": "OpenTulpa",
            "GIT_AUTHOR_EMAIL": "opentulpa@localhost",
            "GIT_COMMITTER_NAME": "OpenTulpa Host",
            "GIT_COMMITTER_EMAIL": "host@opentulpa.local",
        },
        timeout_seconds=300,
        max_output_bytes=max_output_bytes,
    )
    if check and (result.returncode != 0 or result.truncated or result.timed_out):
        raise SourceEvolutionError("trusted source Git operation failed")
    return _GitResult(result.returncode, result.output)


__all__ = [
    "HostEvolutionControlService",
    "SourceEvolutionError",
    "prepare_live_source_repository",
]
