from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from opentulpa.api.routes.v2_files import register_v2_file_routes
from opentulpa.api.routes.v2_integrations import register_v2_integration_routes
from opentulpa.api.routes.v2_principal import V2Principal
from opentulpa.context.file_vault import FileVaultService
from opentulpa.persistence.idempotency import IdempotencyStore


@dataclass
class _Principal:
    tenant_id: str
    actor_id: str


def _resolver(request: Request) -> V2Principal:
    return _Principal(
        tenant_id=request.headers.get("x-tenant-id", ""),
        actor_id=request.headers.get("x-actor-id", ""),
    )


def _headers(
    tenant_id: str = "tenant-a",
    *,
    idempotency_key: str | None = None,
) -> dict[str, str]:
    headers = {"x-tenant-id": tenant_id, "x-actor-id": "actor-1"}
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _file_client(tmp_path: Path) -> tuple[TestClient, FileVaultService]:
    vault = FileVaultService(
        root_dir=tmp_path / "vault",
        db_path=tmp_path / "files.db",
    )
    idempotency = IdempotencyStore(tmp_path / "file-idempotency.db")
    app = FastAPI()
    register_v2_file_routes(
        app,
        get_file_vault=lambda: vault,
        get_idempotency_store=lambda: idempotency,
        resolve_principal=_resolver,
        max_upload_bytes=100,
    )
    return TestClient(app), vault


def _ingest(vault: FileVaultService, tenant_id: str, name: str, content: bytes) -> dict[str, Any]:
    return vault.ingest_file(
        customer_id=tenant_id,
        chat_id=123,
        kind="document",
        telegram_file_id="telegram-secret-id",
        original_filename=name,
        mime_type="text/plain",
        caption="caption",
        raw_bytes=content,
    )


def test_v2_files_are_tenant_scoped_and_never_expose_storage_paths(tmp_path: Path) -> None:
    client, vault = _file_client(tmp_path)
    own = _ingest(vault, "tenant-a", "own.txt", b"owner content")
    other = _ingest(vault, "tenant-b", "other.txt", b"other content")

    listed = client.get("/v2/files", headers=_headers())
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["files"]] == [own["id"]]
    public = listed.json()["files"][0]
    assert "stored_path" not in public
    assert "vault_path" not in public
    assert "local_path" not in public
    assert "customer_id" not in public
    assert "chat_id" not in public
    assert "telegram_file_id" not in public

    fetched = client.get(f"/v2/files/{own['id']}", headers=_headers())
    assert fetched.status_code == 200
    assert fetched.json()["file"]["text_excerpt"] == "owner content"
    assert "stored_path" not in fetched.json()["file"]

    hidden = client.get(f"/v2/files/{other['id']}", headers=_headers())
    assert hidden.status_code == 404


def test_v2_file_upload_uses_principal_and_rejects_identity_fields(tmp_path: Path) -> None:
    client, vault = _file_client(tmp_path)

    missing_key = client.post(
        "/v2/files",
        headers=_headers(),
        files={"upload": ("notes.txt", b"hello", "text/plain")},
    )
    assert missing_key.status_code == 422

    created = client.post(
        "/v2/files",
        headers=_headers(idempotency_key="upload-notes"),
        data={"kind": "document", "caption": "notes"},
        files={"upload": ("notes.txt", b"hello", "text/plain")},
    )
    assert created.status_code == 201
    file_id = created.json()["file"]["id"]
    assert vault.get_file("tenant-a", file_id) is not None
    assert vault.get_file("tenant-b", file_id) is None

    forged = client.post(
        "/v2/files",
        headers=_headers(idempotency_key="upload-forged"),
        data={"kind": "document", "tenant_id": "tenant-b"},
        files={"upload": ("forged.txt", b"hello", "text/plain")},
    )
    assert forged.status_code == 422

    empty = client.post(
        "/v2/files",
        headers=_headers(idempotency_key="upload-empty"),
        files={"upload": ("empty.txt", b"", "text/plain")},
    )
    assert empty.status_code == 400
    too_large = client.post(
        "/v2/files",
        headers=_headers(idempotency_key="upload-large"),
        files={"upload": ("large.txt", b"x" * 101, "text/plain")},
    )
    assert too_large.status_code == 413


def test_v2_file_upload_is_durable_idempotent_and_tenant_scoped(tmp_path: Path) -> None:
    client, first_vault = _file_client(tmp_path)
    headers = _headers(idempotency_key="lost-response-upload")
    request = {
        "headers": headers,
        "data": {"kind": "document", "caption": "notes"},
        "files": {"upload": ("notes.txt", b"hello", "text/plain")},
    }

    created = client.post("/v2/files", **request)
    replayed = client.post("/v2/files", **request)

    assert created.status_code == replayed.status_code == 201
    file_id = created.json()["file"]["id"]
    assert replayed.json()["file"]["id"] == file_id
    assert len(first_vault.search("tenant-a", query="", limit=20)) == 1

    restarted_client, restarted_vault = _file_client(tmp_path)
    replayed_after_restart = restarted_client.post("/v2/files", **request)
    assert replayed_after_restart.status_code == 201
    assert replayed_after_restart.json()["file"]["id"] == file_id
    assert len(restarted_vault.search("tenant-a", query="", limit=20)) == 1

    changed_payload = restarted_client.post(
        "/v2/files",
        headers=headers,
        data={"kind": "document", "caption": "changed"},
        files={"upload": ("notes.txt", b"changed", "text/plain")},
    )
    assert changed_payload.status_code == 409
    assert changed_payload.json()["detail"] == "idempotency key conflict"

    other_tenant = restarted_client.post(
        "/v2/files",
        headers=_headers("tenant-b", idempotency_key="lost-response-upload"),
        data={"kind": "document", "caption": "notes"},
        files={"upload": ("notes.txt", b"hello", "text/plain")},
    )
    assert other_tenant.status_code == 201
    assert other_tenant.json()["file"]["id"] != file_id
    assert len(restarted_vault.search("tenant-b", query="", limit=20)) == 1

    assert restarted_vault.delete_file("tenant-a", file_id)
    missing_result = restarted_client.post("/v2/files", **request)
    assert missing_result.status_code == 409
    assert missing_result.json()["detail"] == "idempotent file result is no longer available"


def test_v2_file_delete_checks_ownership_and_removes_bytes(tmp_path: Path) -> None:
    client, vault = _file_client(tmp_path)
    own = _ingest(vault, "tenant-a", "own.txt", b"owner content")
    other = _ingest(vault, "tenant-b", "other.txt", b"other content")
    own_path = Path(str(own["stored_path"]))

    hidden = client.delete(f"/v2/files/{other['id']}", headers=_headers())
    assert hidden.status_code == 404
    assert vault.get_file("tenant-b", str(other["id"])) is not None

    deleted = client.delete(f"/v2/files/{own['id']}", headers=_headers())
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "file_id": own["id"]}
    assert vault.get_file("tenant-a", str(own["id"])) is None
    assert not own_path.exists()


class _IntegrationService:
    enabled = True

    def __init__(self) -> None:
        self.authorize_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.fail_search = False
        self.redirect_url = "https://auth.example/connect?id=pending-1"

    def list_connections(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["tenant_id"] == "tenant-a"
        return {
            "tenant_id": "tenant-a",
            "items": [
                {
                    "id": "conn-own",
                    "status": "ACTIVE",
                    "user_id": "tenant-a",
                    "integration_id": "gmail",
                    "integration_name": "Gmail",
                    "auth_config_id": "auth-secret",
                    "access_token": "must-not-leak",
                },
                {
                    "id": "conn-other",
                    "status": "ACTIVE",
                    "user_id": "tenant-b",
                    "integration_id": "slack",
                    "integration_name": "Slack",
                },
            ],
        }

    def get_connection(self, *, tenant_id: str, connection_id: str) -> dict[str, Any]:
        items = self.list_connections(tenant_id=tenant_id, integration_id=None)["items"]
        return next(item for item in items if item["id"] == connection_id)

    def list_integrations(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["tenant_id"] == "tenant-a"
        return {
            "tenant_id": "tenant-a",
            "items": [
                {
                    "id": "gmail",
                    "name": "Gmail",
                    "connected": True,
                    "connection_id": "conn-own",
                    "connection_status": "ACTIVE",
                    "requires_authentication": True,
                    "auth_config_id": "auth-secret",
                },
                {
                    "id": "slack",
                    "name": "Slack",
                    "connected": False,
                    "connection_id": None,
                    "connection_status": None,
                    "requires_authentication": True,
                },
            ],
        }

    def connect(self, **kwargs: Any) -> dict[str, Any]:
        self.authorize_calls.append(kwargs)
        return {
            "tenant_id": kwargs["tenant_id"],
            "user_id": kwargs["tenant_id"],
            "connection_id": "pending-1",
            "authorization_url": self.redirect_url,
            "status": "authorization_required"
            if self.redirect_url.startswith("https://")
            else "pending",
            "api_key": "must-not-leak",
        }

    def disconnect(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_calls.append(str(kwargs["connection_id"]))
        return {
            "tenant_id": kwargs["tenant_id"],
            "user_id": kwargs["tenant_id"],
            "connection_id": kwargs["connection_id"],
            "deleted": True,
            "refresh_token": "must-not-leak",
        }

    def search_actions(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        if self.fail_search:
            raise RuntimeError("api_key=must-not-leak")
        return {
            "items": [
                {
                    "name": "GMAIL_SEND_EMAIL",
                    "title": "Send email",
                    "description": "Send one email",
                    "integration_id": "gmail",
                    "integration_name": "Gmail",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "access_token": {"type": "string", "default": "secret"},
                        },
                    },
                    "credential": "must-not-leak",
                }
            ]
        }

    def get_action(self, *, tenant_id: str, action_name: str) -> dict[str, Any]:
        payload = self.search_actions(tenant_id=tenant_id)
        tool = dict(payload["items"][0])
        tool["name"] = action_name
        return {"tenant_id": tenant_id, "action": tool}


def _integration_client() -> tuple[TestClient, _IntegrationService]:
    service = _IntegrationService()
    app = FastAPI()
    register_v2_integration_routes(
        app,
        get_integration_service=lambda: service,
        resolve_principal=_resolver,
    )
    return TestClient(app), service


def test_v2_integrations_and_connections_filter_ownership_and_provider_secrets() -> None:
    client, _ = _integration_client()

    integrations = client.get("/v2/integrations", headers=_headers())
    assert integrations.status_code == 200
    assert integrations.json() == {
        "integrations": [
            {
                "id": "gmail",
                "name": "Gmail",
                "connected": True,
                "connection_id": "conn-own",
                "connection_status": "ACTIVE",
                "requires_authentication": True,
            },
            {
                "id": "slack",
                "name": "Slack",
                "connected": False,
                "connection_id": None,
                "connection_status": None,
                "requires_authentication": True,
            },
        ]
    }
    serialized = str(integrations.json())
    assert "tenant-a" not in serialized
    assert "auth-secret" not in serialized

    connections = client.get("/v2/integrations/connections", headers=_headers())
    assert connections.status_code == 200
    assert connections.json() == {
        "connections": [
            {
                "id": "conn-own",
                "status": "ACTIVE",
                "integration_id": "gmail",
                "integration_name": "Gmail",
            }
        ]
    }


def test_v2_integration_connect_uses_principal_and_strict_typed_body() -> None:
    client, service = _integration_client()

    missing_key = client.post(
        "/v2/integrations/connections",
        headers=_headers(),
        json={"integration_id": "gmail"},
    )
    assert missing_key.status_code == 422

    response = client.post(
        "/v2/integrations/connections",
        headers=_headers(idempotency_key="connect-1"),
        json={"integration_id": "gmail"},
    )
    assert response.status_code == 201
    assert service.authorize_calls == [
        {
            "tenant_id": "tenant-a",
            "actor_id": "actor-1",
            "integration_id": "gmail",
            "redirect_url": None,
            "idempotency_key": "connect-1",
        }
    ]
    assert response.json() == {
        "connection": {
            "id": "pending-1",
            "integration_id": "gmail",
            "authorization_url": "https://auth.example/connect?id=pending-1",
            "status": "authorization_required",
        }
    }
    assert "callback" not in str(response.json())
    assert "api_key" not in str(response.json())

    forged = client.post(
        "/v2/integrations/connections",
        headers=_headers(idempotency_key="connect-2"),
        json={"integration_id": "gmail", "tenant_id": "tenant-b"},
    )
    assert forged.status_code == 422

    service.redirect_url = "javascript:alert(1)"
    unsafe_redirect = client.post(
        "/v2/integrations/connections",
        headers=_headers(idempotency_key="connect-3"),
        json={"integration_id": "gmail"},
    )
    assert unsafe_redirect.status_code == 201
    assert unsafe_redirect.json()["connection"]["authorization_url"] is None


def test_v2_integration_disconnect_is_fail_closed_on_connection_ownership() -> None:
    client, service = _integration_client()

    missing_key = client.delete(
        "/v2/integrations/connections/conn-own",
        headers=_headers(),
    )
    assert missing_key.status_code == 422

    hidden = client.delete(
        "/v2/integrations/connections/conn-other",
        headers=_headers(idempotency_key="disconnect-other"),
    )
    assert hidden.status_code == 404
    assert service.delete_calls == []

    deleted = client.delete(
        "/v2/integrations/connections/conn-own",
        headers=_headers(idempotency_key="disconnect-own"),
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "connection_id": "conn-own"}
    assert service.delete_calls == ["conn-own"]
    assert "refresh_token" not in str(deleted.json())


def test_v2_integration_actions_are_authenticated_and_sanitized() -> None:
    client, service = _integration_client()
    unauthorized = client.get("/v2/integrations/actions", params={"query": "email"})
    assert unauthorized.status_code == 401

    response = client.get(
        "/v2/integrations/actions",
        headers=_headers(),
        params={"query": "email"},
    )
    assert response.status_code == 200
    action = response.json()["actions"][0]
    assert action["name"] == "GMAIL_SEND_EMAIL"
    assert action["input_schema"]["properties"]["access_token"] == "[redacted]"
    assert "must-not-leak" not in str(response.json())

    detail = client.get(
        "/v2/integrations/actions/GMAIL_SEND_EMAIL",
        headers=_headers(),
    )
    assert detail.status_code == 200
    assert detail.json()["action"]["name"] == "GMAIL_SEND_EMAIL"

    service.fail_search = True
    failed = client.get(
        "/v2/integrations/actions",
        headers=_headers(),
        params={"query": "email"},
    )
    assert failed.status_code == 502
    assert "must-not-leak" not in failed.text
