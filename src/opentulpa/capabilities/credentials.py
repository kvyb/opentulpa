"""Durable, revocable credentials for capability access to the public V2 API."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import SecretStr

from opentulpa.capabilities.models import (
    AgentInterfaceBinding,
    CapabilityManifest,
    SecretSource,
    WorkerKind,
)
from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.specs import AgentRunBinding, AgentSpecRef

CAPABILITY_CREDENTIAL_PREFIX = "otcap_"
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_TOKEN_RE = re.compile(r"^otcap_[A-Za-z0-9_-]{40,200}$")
_AGENT_API_TOKEN_ENVIRONMENT = "OPENTULPA_AGENT_API_TOKEN"


class CapabilityAPIScope(StrEnum):
    """The only public API operations a capability worker may receive."""

    AGENT_RUN_SUBMIT = "agent.runs.submit"
    AGENT_RUN_REPLAY = "agent.runs.replay"
    AGENT_RUN_RESUME = "agent.runs.resume"
    AGENT_RUN_CANCEL = "agent.runs.cancel"
    FILE_UPLOAD = "files.upload"
    NOTIFICATIONS_READ = "notifications.read"
    NOTIFICATIONS_ACK = "notifications.ack"


CAPABILITY_API_SCOPES = frozenset(scope.value for scope in CapabilityAPIScope)
_OWNER_ONLY_SCOPES = frozenset(
    {
        CapabilityAPIScope.AGENT_RUN_RESUME.value,
        CapabilityAPIScope.AGENT_RUN_CANCEL.value,
        CapabilityAPIScope.NOTIFICATIONS_READ.value,
        CapabilityAPIScope.NOTIFICATIONS_ACK.value,
    }
)


@dataclass(frozen=True, slots=True)
class CapabilityCredential:
    """Public-safe metadata for one capability-generation bearer."""

    id: str
    tenant_id: str
    actor_id: str
    capability_name: str
    capability_instance_id: str
    interface: str
    source_id: str
    channel: str
    agent_binding: AgentRunBinding
    scopes: frozenset[str]
    issued_at: datetime
    revoked_at: datetime | None = None

    @property
    def agent_spec(self) -> AgentSpecRef:
        return self.agent_binding.agent_spec

    @property
    def run_kind(self) -> str:
        return self.agent_binding.run_kind

    @property
    def trust_class(self) -> Literal["owner", "background", "external"]:
        return self.agent_binding.trust_class


@dataclass(frozen=True, slots=True)
class IssuedCapabilityCredential:
    """One-time plaintext returned only to the trusted worker host."""

    credential: CapabilityCredential
    token: SecretStr


class CapabilityCredentialStore:
    """Persist hashed capability bearers and their generation-bound policy."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capability credential clock must return an aware datetime")
        return value.astimezone(UTC)

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS capability_api_credentials (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    capability_name TEXT NOT NULL,
                    capability_instance_id TEXT NOT NULL,
                    interface TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    agent_spec_id TEXT NOT NULL,
                    agent_spec_revision INTEGER NOT NULL CHECK (agent_spec_revision >= 1),
                    run_kind TEXT NOT NULL,
                    trust_class TEXT NOT NULL CHECK (
                        trust_class IN ('owner', 'background', 'external')
                    ),
                    scopes_json TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    issued_at TEXT NOT NULL,
                    revoked_at TEXT,
                    revoked_reason TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_capability_api_credential_active_instance
                ON capability_api_credentials (tenant_id, capability_instance_id)
                WHERE revoked_at IS NULL;

                CREATE INDEX IF NOT EXISTS idx_capability_api_credential_token
                ON capability_api_credentials (token_hash, revoked_at);
                """
            )
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(capability_api_credentials)")
            }
            missing_binding = any(
                column not in columns
                for column in ("agent_spec_id", "agent_spec_revision", "run_kind")
            )
            for column, definition in (
                ("agent_spec_id", "TEXT"),
                ("agent_spec_revision", "INTEGER"),
                ("run_kind", "TEXT"),
            ):
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE capability_api_credentials ADD COLUMN {column} {definition}"
                    )
            if missing_binding:
                # A pre-binding credential has ambiguous agent authority. It cannot be
                # upgraded safely, so force the capability host to rotate it.
                conn.execute(
                    """
                    UPDATE capability_api_credentials
                    SET revoked_at = ?, revoked_reason = 'agent_binding_upgrade'
                    WHERE revoked_at IS NULL
                    """,
                    (self._now().isoformat(),),
                )
            conn.commit()

    def issue(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        capability_instance_id: str,
        interface: str,
        source_id: str,
        channel: str,
        agent_binding: AgentRunBinding,
        scopes: Iterable[str],
    ) -> IssuedCapabilityCredential:
        tenant = _identity(tenant_id, "tenant_id")
        actor = _identity(actor_id, "actor_id")
        instance = _identity(capability_instance_id, "capability_instance_id")
        source = _identity(source_id, "source_id")
        capability = _slug(capability_name, "capability_name")
        interface_name = _slug(interface, "interface")
        channel_name = _slug(channel, "channel")
        if agent_binding.agent_spec.tenant_id != tenant:
            raise ValueError("agent binding belongs to a different tenant")
        safe_scopes = _scopes(scopes)
        _validate_binding_scopes(agent_binding, safe_scopes)
        now = self._now()
        credential_id = new_short_id("capcred", suffix_chars=12)
        token = f"{CAPABILITY_CREDENTIAL_PREFIX}{secrets.token_urlsafe(36)}"
        token_hash = _token_hash(token)
        credential = CapabilityCredential(
            id=credential_id,
            tenant_id=tenant,
            actor_id=actor,
            capability_name=capability,
            capability_instance_id=instance,
            interface=interface_name,
            source_id=source,
            channel=channel_name,
            agent_binding=agent_binding,
            scopes=safe_scopes,
            issued_at=now,
        )
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE capability_api_credentials
                SET revoked_at = ?, revoked_reason = 'rotated'
                WHERE tenant_id = ? AND capability_instance_id = ? AND revoked_at IS NULL
                """,
                (now.isoformat(), tenant, instance),
            )
            conn.execute(
                """
                INSERT INTO capability_api_credentials (
                    id, tenant_id, actor_id, capability_name, capability_instance_id,
                    interface, source_id, channel, agent_spec_id, agent_spec_revision,
                    run_kind, trust_class, scopes_json, token_hash, issued_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    credential.id,
                    credential.tenant_id,
                    credential.actor_id,
                    credential.capability_name,
                    credential.capability_instance_id,
                    credential.interface,
                    credential.source_id,
                    credential.channel,
                    credential.agent_spec.spec_id,
                    credential.agent_spec.revision,
                    credential.run_kind,
                    credential.trust_class,
                    json.dumps(sorted(credential.scopes), separators=(",", ":")),
                    token_hash,
                    credential.issued_at.isoformat(),
                ),
            )
            conn.commit()
        return IssuedCapabilityCredential(
            credential=credential,
            token=SecretStr(token),
        )

    def authenticate(self, token: str) -> CapabilityCredential | None:
        candidate = str(token or "").strip()
        if not _TOKEN_RE.fullmatch(candidate):
            return None
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM capability_api_credentials
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (_token_hash(candidate),),
            ).fetchone()
        return _credential(row) if row is not None else None

    def revoke_instance(
        self,
        *,
        tenant_id: str,
        capability_instance_id: str,
        reason: str = "generation_stopped",
    ) -> int:
        tenant = _identity(tenant_id, "tenant_id")
        instance = _identity(capability_instance_id, "capability_instance_id")
        safe_reason = str(reason or "generation_stopped").strip()[:200]
        now = self._now().isoformat()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE capability_api_credentials
                SET revoked_at = ?, revoked_reason = ?
                WHERE tenant_id = ? AND capability_instance_id = ? AND revoked_at IS NULL
                """,
                (now, safe_reason, tenant, instance),
            )
            conn.commit()
            return cursor.rowcount

    def revoke(
        self,
        *,
        tenant_id: str,
        credential_id: str,
        reason: str = "revoked",
    ) -> bool:
        tenant = _identity(tenant_id, "tenant_id")
        identifier = _identity(credential_id, "credential_id")
        safe_reason = str(reason or "revoked").strip()[:200]
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE capability_api_credentials
                SET revoked_at = ?, revoked_reason = ?
                WHERE tenant_id = ? AND id = ? AND revoked_at IS NULL
                """,
                (self._now().isoformat(), safe_reason, tenant, identifier),
            )
            conn.commit()
            return cursor.rowcount == 1


class CapabilityAPICredentialService:
    """Issue only the API scopes declared by an interface worker manifest."""

    def __init__(
        self,
        store: CapabilityCredentialStore,
        *,
        resolve_agent_spec: Callable[[str, str], AgentSpecRef] | None = None,
    ) -> None:
        self._store = store
        self._resolve_agent_spec = resolve_agent_spec

    def resolve_agent_binding(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> AgentRunBinding | None:
        """Resolve authority once, before a capability generation is committed."""

        contract = self._agent_api_contract(manifest)
        if contract is None:
            return None
        _, declared_binding = contract
        if self._resolve_agent_spec is None:
            raise RuntimeError("AgentSpec resolution is unavailable for interface credentials")
        agent_spec = self._resolve_agent_spec(tenant_id, declared_binding.agent_spec_id)
        if (
            agent_spec.tenant_id != tenant_id
            or agent_spec.spec_id != declared_binding.agent_spec_id
        ):
            raise ValueError("resolved AgentSpec does not match the reviewed interface binding")
        return AgentRunBinding(
            agent_spec=agent_spec,
            run_kind=declared_binding.run_kind,
            trust_class=declared_binding.trust_class,
        )

    def issue_for_capability(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        agent_binding: AgentRunBinding | None,
    ) -> IssuedCapabilityCredential | None:
        """Issue a bearer for the exact persisted generation binding."""

        contract = self._agent_api_contract(manifest)
        if contract is None:
            if agent_binding is not None:
                raise ValueError("capability without Agent API access cannot have an agent binding")
            return None
        scopes, declared_binding = contract
        if agent_binding is None:
            raise ValueError("interface generation is missing its persisted agent binding")
        if agent_binding.agent_spec.tenant_id != tenant_id:
            raise ValueError("agent binding belongs to a different tenant")
        if (
            agent_binding.agent_spec.spec_id != declared_binding.agent_spec_id
            or agent_binding.run_kind != declared_binding.run_kind
            or agent_binding.trust_class != declared_binding.trust_class
        ):
            raise ValueError("persisted AgentSpec binding does not match the interface manifest")
        return self._store.issue(
            tenant_id=tenant_id,
            actor_id=f"capability:{instance_id}",
            capability_name=manifest.name,
            capability_instance_id=instance_id,
            interface=manifest.name,
            source_id=instance_id,
            channel=manifest.name,
            agent_binding=agent_binding,
            scopes=scopes,
        )

    @staticmethod
    def _agent_api_contract(
        manifest: CapabilityManifest,
    ) -> tuple[frozenset[str], AgentInterfaceBinding] | None:
        workers = tuple(
            worker
            for worker in manifest.workers
            if any(secret.name == _AGENT_API_TOKEN_ENVIRONMENT for secret in worker.secrets)
        )
        if not workers:
            return None
        if any(worker.kind is not WorkerKind.INTERFACE for worker in workers):
            raise ValueError("Agent API credentials are only supported for interface workers")
        requirements = tuple(
            secret
            for worker in workers
            for secret in worker.secrets
            if secret.name == _AGENT_API_TOKEN_ENVIRONMENT
        )
        if any(secret.source is not SecretSource.ISSUED for secret in requirements):
            raise ValueError("Agent API credentials must use the issued secret source")
        scopes = {
            permission
            for worker in workers
            for permission in worker.permissions
            if permission in CAPABILITY_API_SCOPES
        }
        if not scopes:
            raise ValueError("interface worker declares no supported Agent API scopes")
        declared_scopes = {scope for secret in requirements for scope in secret.scopes}
        if declared_scopes != scopes:
            raise ValueError("Agent API credential scopes must match worker permissions")
        bindings = {worker.agent_binding for worker in workers}
        if None in bindings or len(bindings) != 1:
            raise ValueError("interface workers must declare one consistent agent binding")
        declared_binding = next(iter(bindings))
        assert declared_binding is not None
        _validate_declared_binding_scopes(declared_binding, frozenset(scopes))
        return frozenset(scopes), declared_binding

    def authenticate(self, token: str) -> CapabilityCredential | None:
        return self._store.authenticate(token)

    def revoke_instance(self, *, tenant_id: str, instance_id: str) -> int:
        return self._store.revoke_instance(
            tenant_id=tenant_id,
            capability_instance_id=instance_id,
        )


def _identity(value: str, label: str) -> str:
    safe = str(value or "").strip()
    if not safe or len(safe) > 200 or any(ord(char) < 32 for char in safe):
        raise ValueError(f"{label} is invalid")
    return safe


def _slug(value: str, label: str) -> str:
    safe = str(value or "").strip()
    if not _SLUG_RE.fullmatch(safe):
        raise ValueError(f"{label} is invalid")
    return safe


def _scopes(values: Iterable[str]) -> frozenset[str]:
    scopes = frozenset(str(value or "").strip() for value in values)
    if not scopes or "" in scopes or not scopes.issubset(CAPABILITY_API_SCOPES):
        raise ValueError("capability API scopes are invalid")
    return scopes


def _validate_declared_binding_scopes(
    binding: AgentInterfaceBinding,
    scopes: frozenset[str],
) -> None:
    if binding.trust_class != "owner" and scopes.intersection(_OWNER_ONLY_SCOPES):
        raise ValueError(
            "restricted interfaces cannot receive approval or owner notification scopes"
        )


def _validate_binding_scopes(
    binding: AgentRunBinding,
    scopes: frozenset[str],
) -> None:
    if binding.trust_class != "owner" and scopes.intersection(_OWNER_ONLY_SCOPES):
        raise ValueError(
            "restricted interfaces cannot receive approval or owner notification scopes"
        )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _credential(row: sqlite3.Row) -> CapabilityCredential:
    return CapabilityCredential(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        actor_id=str(row["actor_id"]),
        capability_name=str(row["capability_name"]),
        capability_instance_id=str(row["capability_instance_id"]),
        interface=str(row["interface"]),
        source_id=str(row["source_id"]),
        channel=str(row["channel"]),
        agent_binding=AgentRunBinding(
            agent_spec=AgentSpecRef(
                tenant_id=str(row["tenant_id"]),
                spec_id=str(row["agent_spec_id"]),
                revision=int(row["agent_spec_revision"]),
            ),
            run_kind=str(row["run_kind"]),
            trust_class=str(row["trust_class"]),  # type: ignore[arg-type]
        ),
        scopes=frozenset(json.loads(str(row["scopes_json"]))),
        issued_at=datetime.fromisoformat(str(row["issued_at"])).astimezone(UTC),
        revoked_at=(
            datetime.fromisoformat(str(row["revoked_at"])).astimezone(UTC)
            if row["revoked_at"] is not None
            else None
        ),
    )


__all__ = [
    "CAPABILITY_API_SCOPES",
    "CAPABILITY_CREDENTIAL_PREFIX",
    "CapabilityAPICredentialService",
    "CapabilityAPIScope",
    "CapabilityCredential",
    "CapabilityCredentialStore",
    "IssuedCapabilityCredential",
]
