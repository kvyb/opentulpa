from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from opentulpa.api.principal import (
    CapabilityPrincipalResolver,
    OwnerOrCapabilityPrincipalResolver,
    OwnerPrincipalResolver,
)
from opentulpa.api.routes.v2_agent import register_v2_agent_routes
from opentulpa.api.routes.v2_files import register_v2_file_routes
from opentulpa.capabilities import (
    CAPABILITY_API_SCOPES,
    CAPABILITY_CREDENTIAL_PREFIX,
    CapabilityAPICredentialService,
    CapabilityAPIScope,
    CapabilityCredentialStore,
)
from opentulpa.deep_agent.contracts import (
    AgentApproval,
    AgentRunEvent,
    AgentRunRequest,
    AgentRunSnapshot,
    ApprovalDecision,
)
from opentulpa.persistence.idempotency import IdempotencyStore
from opentulpa.specs import AgentRunBinding, AgentSpecRef, OriginRef
from opentulpa.tooling.contract import AgentRunContext


@dataclass
class _Agent:
    requests: list[AgentRunRequest] = field(default_factory=list)
    snapshots: dict[str, AgentRunSnapshot] = field(default_factory=dict)
    decisions: list[tuple[str, ApprovalDecision]] = field(default_factory=list)

    async def open_stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        return self.stream(request)

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        yield AgentRunEvent(
            type="run.completed",
            run_id="run_capability",
            sequence=1,
            timestamp="2026-07-20T00:00:00+00:00",
            data={"text": "ok"},
        )

    async def get_run(self, run_id: str) -> AgentRunSnapshot | None:
        return self.snapshots.get(run_id)

    async def events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[AgentRunEvent]:
        del run_id, after_sequence
        if False:
            yield

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        self.decisions.append((run_id, decision))
        yield AgentRunEvent(
            type="run.completed",
            run_id=run_id,
            sequence=2,
            timestamp="2026-07-20T00:00:01+00:00",
            data={"text": "approved"},
        )

    async def open_resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        return self.resume(run_id, decision)

    async def cancel(self, run_id: str) -> AgentRunSnapshot:
        return self.snapshots[run_id]


class _Files:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], dict[str, Any]] = {}

    def search(self, customer_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        del customer_id, query, limit
        return []

    def get_file(self, customer_id: str, file_id: str) -> dict[str, Any] | None:
        return self._records.get((customer_id, file_id))

    def ingest_file(self, **kwargs: Any) -> dict[str, Any]:
        record = {
            "id": "file_capability",
            "kind": kwargs["kind"],
            "original_filename": kwargs["original_filename"],
            "mime_type": kwargs["mime_type"],
            "size_bytes": len(kwargs["raw_bytes"]),
            "caption": kwargs["caption"],
            "summary": "",
            "text_excerpt": "",
            "created_at": "2026-07-20T00:00:00+00:00",
        }
        self._records[(str(kwargs["customer_id"]), str(record["id"]))] = record
        return record

    def delete_file(self, customer_id: str, file_id: str) -> bool:
        del customer_id, file_id
        return False


def _issue(
    store: CapabilityCredentialStore,
    *,
    tenant_id: str = "tenant-a",
    instance_id: str = "cap_tenant_a_telegram_g1",
    scopes: frozenset[str] = CAPABILITY_API_SCOPES,
    agent_spec_id: str = "owner",
    agent_spec_revision: int = 1,
    run_kind: str = "owner",
    trust_class: str = "owner",
) -> tuple[str, str]:
    issued = store.issue(
        tenant_id=tenant_id,
        actor_id=f"capability:{instance_id}",
        capability_name="telegram",
        capability_instance_id=instance_id,
        interface="telegram",
        source_id=instance_id,
        channel="telegram",
        agent_binding=AgentRunBinding(
            agent_spec=AgentSpecRef(
                tenant_id=tenant_id,
                spec_id=agent_spec_id,
                revision=agent_spec_revision,
            ),
            run_kind=run_kind,
            trust_class=trust_class,
        ),
        scopes=scopes,
    )
    return issued.credential.id, issued.token.get_secret_value()


def _client(
    store: CapabilityCredentialStore,
    agent: _Agent | None = None,
    *,
    secret_ingress: Any | None = None,
) -> tuple[TestClient, _Agent]:
    service = agent or _Agent()
    files = _Files()
    idempotency = IdempotencyStore(store.db_path.with_name("capability_file_idempotency.db"))
    app = FastAPI()
    capability_credentials = CapabilityAPICredentialService(store)
    principal = OwnerOrCapabilityPrincipalResolver(
        owner=OwnerPrincipalResolver(
            token="owner-token",
            tenant_id="tenant-owner",
        ),
        capability=CapabilityPrincipalResolver(capability_credentials),
    )
    register_v2_agent_routes(
        app,
        get_agent_service=lambda: service,
        resolve_principal=principal,
        secret_ingress=secret_ingress,
    )
    register_v2_file_routes(
        app,
        get_file_vault=lambda: files,
        get_idempotency_store=lambda: idempotency,
        resolve_principal=principal,
        max_upload_bytes=1_000,
    )
    return TestClient(app), service


def test_capability_credential_is_durable_rotated_and_tenant_scoped(tmp_path: Path) -> None:
    db_path = tmp_path / "capability_credentials.db"
    first_store = CapabilityCredentialStore(db_path)
    first_id, first_token = _issue(first_store)

    restarted_store = CapabilityCredentialStore(db_path)
    restored = restarted_store.authenticate(first_token)
    assert restored is not None
    assert restored.id == first_id
    assert restored.scopes == CAPABILITY_API_SCOPES
    assert first_token.encode() not in db_path.read_bytes()

    second_id, second_token = _issue(restarted_store)
    assert second_id != first_id
    assert second_token != first_token
    assert restarted_store.authenticate(first_token) is None
    assert restarted_store.authenticate(second_token) is not None

    assert (
        restarted_store.revoke_instance(
            tenant_id="tenant-b",
            capability_instance_id="cap_tenant_a_telegram_g1",
        )
        == 0
    )
    assert restarted_store.authenticate(second_token) is not None
    assert (
        restarted_store.revoke_instance(
            tenant_id="tenant-a",
            capability_instance_id="cap_tenant_a_telegram_g1",
        )
        == 1
    )
    assert restarted_store.authenticate(second_token) is None


def test_legacy_unbound_credential_is_revoked_instead_of_defaulting_to_owner(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy_credentials.db"
    token = f"{CAPABILITY_CREDENTIAL_PREFIX}{'x' * 48}"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE capability_api_credentials (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                capability_name TEXT NOT NULL,
                capability_instance_id TEXT NOT NULL,
                interface TEXT NOT NULL,
                source_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                trust_class TEXT NOT NULL,
                scopes_json TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                issued_at TEXT NOT NULL,
                revoked_at TEXT,
                revoked_reason TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO capability_api_credentials VALUES (
                'legacy', 'tenant-a', 'capability:legacy', 'telegram', 'legacy-g1',
                'telegram', 'legacy-g1', 'telegram', 'owner', '["agent.runs.submit"]',
                ?, '2026-07-20T00:00:00+00:00', NULL, NULL
            )
            """,
            (hashlib.sha256(token.encode()).hexdigest(),),
        )
        conn.commit()

    store = CapabilityCredentialStore(db_path)

    assert store.authenticate(token) is None
    with sqlite3.connect(db_path) as conn:
        revoked_at, reason = conn.execute(
            "SELECT revoked_at, revoked_reason FROM capability_api_credentials"
        ).fetchone()
    assert revoked_at is not None
    assert reason == "agent_binding_upgrade"


def test_capability_principal_preserves_origin_and_cannot_cross_tenants(tmp_path: Path) -> None:
    store = CapabilityCredentialStore(tmp_path / "capability_credentials.db")
    _, token = _issue(store)
    other_context = _owner_context("tenant-b")
    agent = _Agent(
        snapshots={
            "run_own": AgentRunSnapshot(
                run_id="run_own",
                context=_owner_context("tenant-a"),
                status="completed",
                created_at="2026-07-20T00:00:00+00:00",
                updated_at="2026-07-20T00:00:00+00:00",
            ),
            "run_waiting": AgentRunSnapshot(
                run_id="run_waiting",
                context=_owner_context("tenant-a"),
                status="interrupted",
                approvals=(
                    AgentApproval(
                        id="approval-1",
                        tool_name="integration_invoke",
                        description="Send",
                        arguments={},
                        allowed_decisions=("approve", "reject"),
                    ),
                ),
                created_at="2026-07-20T00:00:00+00:00",
                updated_at="2026-07-20T00:00:00+00:00",
            ),
            "run_other": AgentRunSnapshot(
                run_id="run_other",
                context=other_context,
                status="completed",
                created_at="2026-07-20T00:00:00+00:00",
                updated_at="2026-07-20T00:00:00+00:00",
            ),
        }
    )
    client, service = _client(store, agent)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Correlation-ID": "telegram:42:99",
        "X-OpenTulpa-Origin-Conversation-ID": "telegram-chat-42",
        "X-OpenTulpa-Origin-Message-ID": "telegram-update-99",
    }

    response = client.post(
        "/v2/agent/runs",
        headers=headers,
        json={"thread_id": "telegram-thread-42", "text": "hello", "file_ids": []},
    )

    assert response.status_code == 200
    context = service.requests[0].context
    assert context.tenant_id == "tenant-a"
    assert context.actor_id == "capability:cap_tenant_a_telegram_g1"
    assert context.channel == "telegram"
    assert context.trust_class == "owner"
    assert context.origin.interface == "telegram"
    assert context.origin.source_id == "cap_tenant_a_telegram_g1"
    assert context.origin.conversation_id == "telegram-chat-42"
    assert context.origin.message_id == "telegram-update-99"

    uploaded = client.post(
        "/v2/files",
        headers={**headers, "Idempotency-Key": "telegram:42:99:file:0"},
        files={"upload": ("notes.txt", b"hello", "text/plain")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["file"]["id"] == "file_capability"
    assert client.get("/v2/files", headers=headers).status_code == 403

    assert client.get("/v2/agent/runs/run_own", headers=headers).status_code == 200
    resumed = client.post(
        "/v2/agent/runs/run_waiting/resume",
        headers=headers,
        json={"approval_id": "approval-1", "decision": "approve"},
    )
    assert resumed.status_code == 200
    assert service.decisions[0][0] == "run_waiting"

    hidden = client.get("/v2/agent/runs/run_other", headers=headers)
    assert hidden.status_code == 404


def test_external_interface_uses_credential_bound_spec_kind_and_trust(tmp_path: Path) -> None:
    store = CapabilityCredentialStore(tmp_path / "capability_credentials.db")
    _, token = _issue(
        store,
        instance_id="cap_tenant_a_public_chat_g1",
        scopes=frozenset(
            {
                CapabilityAPIScope.AGENT_RUN_SUBMIT.value,
                CapabilityAPIScope.AGENT_RUN_REPLAY.value,
            }
        ),
        agent_spec_id="public-intake",
        agent_spec_revision=7,
        run_kind="intake",
        trust_class="external",
    )
    owner_run = AgentRunSnapshot(
        run_id="run_owner",
        context=_owner_context("tenant-a"),
        status="completed",
    )
    ingress_calls: list[dict[str, str]] = []
    client, service = _client(
        store,
        _Agent(snapshots={"run_owner": owner_run}),
        secret_ingress=lambda **kwargs: ingress_calls.append(kwargs) or kwargs["text"],
    )
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post(
        "/v2/agent/runs",
        headers=headers,
        json={"thread_id": "visitor-42", "text": "hello", "file_ids": []},
    )

    assert response.status_code == 200
    context = service.requests[0].context
    assert context.agent_spec == AgentSpecRef(
        tenant_id="tenant-a",
        spec_id="public-intake",
        revision=7,
    )
    assert context.run_kind == "intake"
    assert context.trust_class == "external"
    assert context.agent_spec.spec_id != "owner"
    assert ingress_calls == []

    for override in (
        {"run_kind": "owner"},
        {"trust_class": "owner"},
        {"agent_spec": {"tenant_id": "tenant-a", "spec_id": "owner", "revision": 1}},
        {"channel": "web"},
    ):
        denied = client.post(
            "/v2/agent/runs",
            headers=headers,
            json={"thread_id": "visitor-42", "text": "hello", "file_ids": [], **override},
        )
        assert denied.status_code == 422
    assert len(service.requests) == 1
    assert client.get("/v2/agent/runs/run_owner", headers=headers).status_code == 404


def test_capability_route_scopes_and_revocation_fail_closed(tmp_path: Path) -> None:
    store = CapabilityCredentialStore(tmp_path / "capability_credentials.db")
    credential_id, token = _issue(
        store,
        scopes=frozenset({CapabilityAPIScope.AGENT_RUN_SUBMIT.value}),
    )
    client, _ = _client(store)
    headers = {"Authorization": f"Bearer {token}"}

    allowed = client.post(
        "/v2/agent/runs",
        headers=headers,
        json={"thread_id": "thread-1", "text": "hello", "file_ids": []},
    )
    assert allowed.status_code == 200
    assert client.get("/v2/files", headers=headers).status_code == 403
    assert (
        client.post(
            "/v2/files",
            headers={**headers, "Idempotency-Key": "scope-denied:file:0"},
            files={"upload": ("notes.txt", b"hello", "text/plain")},
        ).status_code
        == 403
    )
    assert client.post("/v2/agent/runs/run-1/cancel", headers=headers).status_code == 403

    assert store.revoke(tenant_id="tenant-a", credential_id=credential_id)
    denied = client.post(
        "/v2/agent/runs",
        headers=headers,
        json={"thread_id": "thread-1", "text": "hello", "file_ids": []},
    )
    assert denied.status_code == 401


@pytest.mark.parametrize(
    ("method", "path", "required_scope"),
    [
        ("PUT", "/v2/agent/threads/thread-1", CapabilityAPIScope.AGENT_RUN_SUBMIT),
        (
            "GET",
            "/v2/agent/threads/thread-1/inference",
            CapabilityAPIScope.AGENT_RUN_REPLAY,
        ),
        (
            "PATCH",
            "/v2/agent/threads/thread-1/inference",
            CapabilityAPIScope.AGENT_RUN_SUBMIT,
        ),
        ("GET", "/v2/inference", CapabilityAPIScope.AGENT_RUN_REPLAY),
        ("GET", "/v2/inference/models", CapabilityAPIScope.AGENT_RUN_REPLAY),
        (
            "POST",
            "/v2/inference/codex/device-logins",
            CapabilityAPIScope.AGENT_RUN_SUBMIT,
        ),
        (
            "GET",
            "/v2/inference/codex/device-logins/login-1",
            CapabilityAPIScope.AGENT_RUN_REPLAY,
        ),
        (
            "POST",
            "/v2/agent/runs/run-1/cancel",
            CapabilityAPIScope.AGENT_RUN_CANCEL,
        ),
        (
            "POST",
            "/v2/agent/threads/thread-1/cancel",
            CapabilityAPIScope.AGENT_RUN_CANCEL,
        ),
    ],
)
def test_owner_capability_control_routes_require_their_declared_scope(
    tmp_path: Path,
    method: str,
    path: str,
    required_scope: CapabilityAPIScope,
) -> None:
    store = CapabilityCredentialStore(tmp_path / "capability_credentials.db")
    _, allowed_token = _issue(
        store,
        instance_id="cap_tenant_a_telegram_allowed",
        scopes=frozenset({required_scope.value}),
    )
    wrong_scope = (
        CapabilityAPIScope.AGENT_RUN_REPLAY
        if required_scope is not CapabilityAPIScope.AGENT_RUN_REPLAY
        else CapabilityAPIScope.AGENT_RUN_SUBMIT
    )
    _, denied_token = _issue(
        store,
        instance_id="cap_tenant_a_telegram_denied",
        scopes=frozenset({wrong_scope.value}),
    )
    resolver = CapabilityPrincipalResolver(CapabilityAPICredentialService(store))

    allowed = resolver(_capability_request(method, path, allowed_token))

    assert required_scope.value in allowed.scopes
    with pytest.raises(HTTPException, match="credential scope") as exc_info:
        resolver(_capability_request(method, path, denied_token))
    assert exc_info.value.status_code == 403


def _capability_request(method: str, path: str, token: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "scheme": "http",
            "server": ("testserver", 80),
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {token}".encode("ascii"))],
        }
    )


def _owner_context(tenant_id: str) -> AgentRunContext:
    return AgentRunContext(
        tenant_id=tenant_id,
        actor_id="owner",
        thread_id="thread",
        channel="web",
        run_kind="owner",
        correlation_id="correlation",
        origin=OriginRef(interface="web", source_id="owner-web"),
        agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=1),
        trust_class="owner",
    )


def test_capability_token_prefix_is_distinct_from_owner_bearer(tmp_path: Path) -> None:
    store = CapabilityCredentialStore(tmp_path / "capability_credentials.db")
    _, token = _issue(store)
    assert token.startswith(CAPABILITY_CREDENTIAL_PREFIX)
