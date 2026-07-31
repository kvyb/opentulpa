from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import SecretStr

from opentulpa.host.app import _resume_log_cursor, create_host_app
from opentulpa.host.models import HostConfigInput
from opentulpa.host.store import HostStore
from opentulpa.secrets.cipher import AesGcmHostKeyCipher


class _Runtime:
    status = "stopped"
    revision = None
    error = None
    endpoint = None
    log_stream_id = "stream-current"

    def logs(self, *, after: int = 0) -> list[Any]:
        return []


class _Service:
    def __init__(self, store: HostStore) -> None:
        self.store = store
        self.runtime = _Runtime()
        self.applied: HostConfigInput | None = None
        self.started = False
        self.activating = False

    async def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.started = False

    async def apply(self, value: HostConfigInput) -> Any:
        self.applied = value
        staged = self.store.stage(value)
        active = self.store.activate(staged.revision)
        self.runtime.status = "ready"
        self.runtime.revision = active.revision
        self.runtime.endpoint = "http://127.0.0.1:65534"
        return self.store.view(active)

    async def restart(self) -> None:
        self.runtime.status = "ready"


class _Evolution:
    async def source_status(self, *, audit_context: dict[str, str]) -> dict[str, Any]:
        return {"active": False, "audit": audit_context}


class _SandboxSupervisor:
    def __init__(self, *, ok: bool = True, failures_before_ready: int = 0) -> None:
        self.ok = ok
        self.failures_before_ready = failures_before_ready
        self.started = False
        self.stopped = False
        self.start_attempts = 0
        self.status_calls = 0
        self.require_ready_calls = 0

    async def start(self) -> None:
        self.start_attempts += 1
        self.started = True
        await self.require_ready()

    async def shutdown(self) -> None:
        self.stopped = True

    async def require_ready(self) -> None:
        self.require_ready_calls += 1
        if self.failures_before_ready > 0:
            self.failures_before_ready -= 1
            raise RuntimeError("sandbox worker canary failed")
        if not self.ok:
            raise RuntimeError("sandbox worker canary failed")

    async def status(self) -> dict[str, Any]:
        self.status_calls += 1
        return {
            "ok": self.ok and self.failures_before_ready <= 0,
            "step": "ready" if self.ok and self.failures_before_ready <= 0 else "health",
            "tier": "test",
            "checks": {
                "execute": self.ok and self.failures_before_ready <= 0,
                "ssh": self.ok and self.failures_before_ready <= 0,
            },
            "error": (
                None
                if self.ok and self.failures_before_ready <= 0
                else "sandbox worker canary failed"
            ),
        }


def _parts(tmp_path: Path) -> tuple[HostStore, _Service]:
    store = HostStore(tmp_path / "host.db", cipher=AesGcmHostKeyCipher(b"h" * 32))
    store.configure_setup_token("setup-token-with-enough-entropy")
    return store, _Service(store)


def test_unconfigured_host_stays_healthy_and_redirects_chat_to_setup(tmp_path: Path) -> None:
    store, service = _parts(tmp_path)
    app = create_host_app(
        store=store,
        service=service,
        setup_token="setup-token-with-enough-entropy",
        sandbox_supervisor=_SandboxSupervisor(),
    )  # type: ignore[arg-type]

    with TestClient(app) as client:
        assert client.get("/healthz").json() == {
            "ok": True,
            "host": "ready",
            "runtime": "stopped",
            "configured": False,
            "sandbox": {
                "ok": True,
                "step": "ready",
                "tier": "test",
                "checks": {"execute": True, "ssh": True},
                "error": None,
            },
        }
        assert client.get("/agent/healthz").status_code == 503
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/_host"
        console = client.get("/_host")
        assert "Make the runtime yours" in console.text
        assert "LOCAL TERMINAL" in console.text
        assert "OPEN CHAT" not in console.text
        script = client.get("/_host/assets/app.js")
        assert "opentulpa connect" in script.text
        assert "stream_id" in script.text
        assert "open-chat" not in script.text


def test_log_cursor_resumes_same_host_and_resets_after_host_replacement() -> None:
    assert (
        _resume_log_cursor(
            after=5,
            last_event_id="9",
            requested_stream_id="stream-current",
            current_stream_id="stream-current",
        )
        == 9
    )
    assert (
        _resume_log_cursor(
            after=9,
            last_event_id="12",
            requested_stream_id="stream-previous",
            current_stream_id="stream-current",
        )
        == 0
    )


def test_host_health_fails_closed_when_sandbox_worker_is_unavailable(tmp_path: Path) -> None:
    store, service = _parts(tmp_path)
    app = create_host_app(
        store=store,
        service=service,  # type: ignore[arg-type]
        sandbox_supervisor=_SandboxSupervisor(ok=False),
    )

    with TestClient(app) as client:
        health = client.get("/healthz")
        status = client.get("/_host/api/status").json()
        assert service.started is False

    assert health.status_code == 503
    assert health.json()["ok"] is False
    assert health.json()["sandbox"]["checks"] == {"execute": False, "ssh": False}
    assert status["sandbox"]["ok"] is False


def test_host_health_uses_cached_sandbox_status(tmp_path: Path) -> None:
    store, service = _parts(tmp_path)
    sandbox = _SandboxSupervisor()
    app = create_host_app(
        store=store,
        service=service,  # type: ignore[arg-type]
        sandbox_supervisor=sandbox,
    )

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz").status_code == 200
        assert client.get("/_host/api/status").json()["sandbox"]["ok"] is True

    assert sandbox.start_attempts == 1
    assert sandbox.require_ready_calls == 1
    assert sandbox.status_calls == 3


def test_host_lifespan_does_not_start_runtime_when_sandbox_start_fails(
    tmp_path: Path,
) -> None:
    store, service = _parts(tmp_path)
    app = create_host_app(
        store=store,
        service=service,  # type: ignore[arg-type]
        sandbox_supervisor=_SandboxSupervisor(ok=False),
    )

    with TestClient(app):
        assert service.started is False


def test_host_lifespan_starts_runtime_after_sandbox_recovers(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENTULPA_SANDBOX_START_RETRY_SECONDS", "0.01")
    store, service = _parts(tmp_path)
    sandbox = _SandboxSupervisor(failures_before_ready=1)
    app = create_host_app(
        store=store,
        service=service,  # type: ignore[arg-type]
        sandbox_supervisor=sandbox,
    )

    with TestClient(app):
        deadline = time.monotonic() + 1.0
        while not service.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.started is True

    assert sandbox.start_attempts >= 2


def test_owner_config_activation_is_blocked_when_sandbox_worker_is_unavailable(
    tmp_path: Path,
) -> None:
    store, service = _parts(tmp_path)
    store.claim(
        setup_token="setup-token-with-enough-entropy",
        owner_token="owner-token-with-at-least-thirty-two-characters",
    )
    app = create_host_app(
        store=store,
        service=service,  # type: ignore[arg-type]
        sandbox_supervisor=_SandboxSupervisor(ok=False),
    )

    with TestClient(app) as client:
        response = client.put(
            "/_host/api/config",
            headers={"Authorization": "Bearer owner-token-with-at-least-thirty-two-characters"},
            json={
                "api_key": "provider-secret",
                "model": "moonshotai/kimi-k3",
                "base_url": "https://openrouter.ai/api/v1",
            },
        )

    assert response.status_code == 503
    assert "Sandbox worker failed" in response.json()["detail"]
    assert service.applied is None


def test_remote_claim_requires_setup_token_and_sets_owner_session(tmp_path: Path) -> None:
    store, service = _parts(tmp_path)
    app = create_host_app(
        store=store,
        service=service,
        setup_token="setup-token-with-enough-entropy",
        sandbox_supervisor=_SandboxSupervisor(),
    )  # type: ignore[arg-type]

    with TestClient(app) as client:
        assert client.post("/_host/api/claim", json={}).status_code == 403
        response = client.post(
            "/_host/api/claim",
            json={"setup_token": "setup-token-with-enough-entropy"},
        )
        assert response.status_code == 200
        owner_token = response.json()["owner_token"]
        assert len(owner_token) >= 32
        assert response.cookies.get("opentulpa_host_session") == owner_token
        status = client.get("/_host/api/status").json()
        assert status["claimed"] is True
        assert status["authenticated"] is True
        assert status["sandbox"]["ok"] is True
        assert client.get("/_host/api/logs").json() == {
            "stream_id": "stream-current",
            "logs": [],
        }


def test_owner_can_activate_first_config_without_exposing_secrets(tmp_path: Path) -> None:
    store, service = _parts(tmp_path)
    store.claim(
        setup_token="setup-token-with-enough-entropy",
        owner_token="owner-token-with-at-least-thirty-two-characters",
    )
    app = create_host_app(
        store=store,
        service=service,  # type: ignore[arg-type]
        sandbox_supervisor=_SandboxSupervisor(),
    )

    with TestClient(app) as client:
        assert (
            client.put("/_host/api/config", json={"api_key": "provider-secret"}).status_code == 401
        )
        response = client.put(
            "/_host/api/config",
            headers={"Authorization": "Bearer owner-token-with-at-least-thirty-two-characters"},
            json={
                "api_key": "provider-secret",
                "model": "moonshotai/kimi-k3",
                "base_url": "https://openrouter.ai/api/v1",
            },
        )
        assert response.status_code == 200
        assert response.json()["config"]["api_key_configured"] is True
        assert "provider-secret" not in response.text
        assert service.applied is not None
        assert service.applied.api_key == SecretStr("provider-secret")
        root = client.get("/", follow_redirects=False)
        assert root.status_code == 307
        assert root.headers["location"] == "/_host"


def test_evolution_control_route_is_registered_before_runtime_proxy(tmp_path: Path) -> None:
    store, service = _parts(tmp_path)
    token = "evolution-token-with-at-least-thirty-two-characters"
    app = create_host_app(
        store=store,
        service=service,  # type: ignore[arg-type]
        evolution_service=_Evolution(),
        evolution_token=token,
        sandbox_supervisor=_SandboxSupervisor(),
    )

    with TestClient(app) as client:
        denied = client.post(
            "/bootstrap/internal/v1/evolution/source/status",
            json={"audit_context": {}},
        )
        response = client.post(
            "/bootstrap/internal/v1/evolution/source/status",
            headers={"X-OpenTulpa-Evolution-Token": token},
            json={"audit_context": {"tenant_id": "owner"}},
        )

    assert denied.status_code == 401
    assert response.status_code == 200
    assert response.json() == {"active": False, "audit": {"tenant_id": "owner"}}
