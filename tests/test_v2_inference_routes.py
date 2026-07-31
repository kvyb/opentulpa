from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from opentulpa.api.routes.v2_inference import register_v2_inference_routes
from opentulpa.inference.models import (
    InferenceModel,
    InferenceSelection,
    InferenceServiceTier,
)


@dataclass(frozen=True)
class _Principal:
    tenant_id: str
    actor_id: str = "owner-1"
    trust_class: str = "owner"


@dataclass
class _Inference:
    connected: bool = True
    deleted: list[str] = field(default_factory=list)

    async def status(self, tenant_id: str) -> dict[str, Any]:
        return {
            "api_default": {
                "provider": "api",
                "model": "kimi",
                "reasoning_effort": "low",
                "service_tier": None,
                "fallback_to_api": False,
            },
            "codex": {"connected": self.connected, "credential_revision": 1},
            "tenant_seen": tenant_id,
        }

    async def models(self, tenant_id: str, provider: str, *, query: str = "") -> tuple[Any, ...]:
        del tenant_id
        models = (
            InferenceModel(
                provider=provider,
                id="gpt-test" if provider == "codex" else "kimi",
                reasoning_efforts=("low", "high"),
                service_tiers=(
                    InferenceServiceTier(
                        id="priority",
                        name="Fast",
                        description="1.5x speed, increased usage",
                    ),
                )
                if provider == "codex"
                else (),
            ),
        )
        return tuple(model for model in models if query.casefold() in model.id.casefold())

    async def start_device_login(self, tenant_id: str) -> dict[str, Any]:
        del tenant_id
        return {
            "login_id": "login-1",
            "status": "pending",
            "verification_url": "https://auth.openai.com/codex/device",
            "user_code": "ABCD-EFGH",
            "interval_seconds": 5,
            "expires_at": "2026-01-01T00:10:00+00:00",
            "error": None,
        }

    async def get_device_login(self, tenant_id: str, login_id: str) -> dict[str, Any] | None:
        if tenant_id != "tenant-a" or login_id != "login-1":
            return None
        return await self.start_device_login(tenant_id)

    async def cancel_device_login(self, tenant_id: str, login_id: str) -> bool:
        return tenant_id == "tenant-a" and login_id == "login-1"

    def delete_credential(self, tenant_id: str) -> bool:
        self.deleted.append(tenant_id)
        self.connected = False
        return True


@dataclass
class _Threads:
    selection: InferenceSelection | None = None
    revision: int = 0
    resets: int = 0

    async def get_owner_inference(self, *, tenant_id: str) -> dict[str, Any]:
        effective = self.selection or InferenceSelection(
            provider="api",
            model="kimi",
            reasoning_effort="low",
        )
        return {
            "scope": "owner",
            "tenant_seen": tenant_id,
            "revision": self.revision,
            "selection": self.selection.model_dump(mode="json") if self.selection else None,
            "effective": effective.model_dump(mode="json"),
        }

    async def update_owner_inference(
        self,
        *,
        tenant_id: str,
        expected_revision: int,
        selection: InferenceSelection | None,
    ) -> dict[str, Any]:
        if expected_revision != self.revision:
            raise RuntimeError("unexpected test revision")
        self.revision += 1
        self.selection = selection
        return await self.get_owner_inference(tenant_id=tenant_id)

    async def get_thread_inference(
        self, *, tenant_id: str, thread_id: str
    ) -> dict[str, Any] | None:
        if tenant_id != "tenant-a" or thread_id != "thread-1":
            return None
        effective = self.selection or InferenceSelection(
            provider="api",
            model="kimi",
            reasoning_effort="low",
        )
        return {
            "revision": self.revision,
            "selection": self.selection.model_dump(mode="json") if self.selection else None,
            "effective": effective.model_dump(mode="json"),
        }

    async def update_thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        expected_revision: int,
        selection: InferenceSelection | None,
    ) -> dict[str, Any] | None:
        if expected_revision != self.revision:
            raise RuntimeError("unexpected test revision")
        self.revision += 1
        self.selection = selection
        return await self.get_thread_inference(tenant_id=tenant_id, thread_id=thread_id)

    async def codex_preference_count(self, tenant_id: str) -> int:
        del tenant_id
        return 1

    async def reset_codex_preferences(self, tenant_id: str) -> int:
        del tenant_id
        self.resets += 1
        self.selection = None
        return 1


def _client(*, trust_class: str = "owner") -> tuple[TestClient, _Inference, _Threads]:
    inference = _Inference()
    threads = _Threads()
    app = FastAPI()

    async def principal(request: Request) -> _Principal:
        return _Principal(
            tenant_id=request.headers.get("x-tenant-id", "tenant-a"),
            trust_class=trust_class,
        )

    register_v2_inference_routes(
        app,
        get_inference=lambda: inference,  # type: ignore[arg-type]
        get_threads=lambda: threads,
        resolve_principal=principal,
    )
    return TestClient(app), inference, threads


def test_inference_routes_use_authenticated_tenant_and_revisioned_global_selection() -> None:
    client, _, threads = _client()

    status = client.get("/v2/inference").json()
    models = client.get("/v2/inference/models?provider=codex&query=gpt").json()
    updated = client.patch(
        "/v2/inference/selection",
        json={
            "expected_revision": 0,
            "selection": {
                "provider": "codex",
                "model": "gpt-test",
                "reasoning_effort": "high",
                "service_tier": "priority",
                "fallback_to_api": False,
            },
        },
    )

    assert status["tenant_seen"] == "tenant-a"
    assert models["models"][0]["id"] == "gpt-test"
    assert updated.status_code == 200
    assert updated.json()["revision"] == 1
    assert updated.json()["scope"] == "owner"
    assert threads.selection is not None and threads.selection.provider == "codex"
    assert threads.selection.service_tier == "priority"
    other_tenant = client.get(
        "/v2/inference/selection",
        headers={"x-tenant-id": "tenant-b"},
    )
    assert other_tenant.status_code == 200
    assert other_tenant.json()["tenant_seen"] == "tenant-b"

    legacy = client.get("/v2/agent/threads/thread-1/inference")
    assert legacy.status_code == 200
    assert legacy.json()["revision"] == 1


def test_device_login_is_sanitized_and_non_owner_is_denied() -> None:
    client, _, _ = _client()
    login = client.post("/v2/inference/codex/device-logins")

    assert login.status_code == 201
    assert login.json()["user_code"] == "ABCD-EFGH"
    assert "token" not in " ".join(login.json())
    denied, _, _ = _client(trust_class="background")
    assert denied.get("/v2/inference").status_code == 403


def test_codex_logout_requires_explicit_thread_reset() -> None:
    client, inference, threads = _client()

    rejected = client.delete("/v2/inference/codex/credential")
    confirmed = client.delete("/v2/inference/codex/credential?reset_threads=true")

    assert rejected.status_code == 409
    assert confirmed.status_code == 200
    assert confirmed.json() == {"disconnected": True, "reset_threads": 1}
    assert inference.deleted == ["tenant-a"]
    assert threads.resets == 1
