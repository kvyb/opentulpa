from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from deepagents.backends.protocol import ExecuteResponse
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from opentulpa.bootstrap.gateway import (
    ActiveReleaseTransport,
    BootstrapGateway,
    _production_environment,
    _validate_release_workspace,
    create_gateway_app,
)
from opentulpa.bootstrap.host import InMemoryReleaseHost
from opentulpa.bootstrap.models import ReleaseRecord
from opentulpa.bootstrap.store import BootstrapStore
from opentulpa.bootstrap.supervisor import BootstrapSupervisor, SupervisorPolicy
from opentulpa.core.config import Settings


def _release() -> ReleaseRecord:
    return ReleaseRecord(
        id="release_blue",
        candidate_id="candidate_blue",
        source_commit="a" * 40,
        artifact_digest=f"sha256:{'b' * 64}",
        manifest_digest=f"sha256:{'c' * 64}",
        entrypoint=("python", "-m", "opentulpa"),
    )


def _policy() -> SupervisorPolicy:
    return SupervisorPolicy(
        stage_probe_attempts=1,
        production_probe_attempts=1,
        probe_interval_seconds=0,
        probation_seconds=0,
        probation_probe_interval_seconds=1,
    )


@pytest.mark.asyncio
async def test_stable_sandbox_endpoint_is_fenced_to_active_release_lease(
    tmp_path: Path,
) -> None:
    class SandboxService:
        async def execute(self, **_: Any) -> ExecuteResponse:
            return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    release = _release()
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost(endpoint="http://127.0.0.1:49153", control_token="c" * 48)
    transport = ActiveReleaseTransport(store=store, host=host)
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await supervisor.start()
    lease = await supervisor.install_initial(release)
    app = create_gateway_app(
        supervisor=supervisor,
        gateway=BootstrapGateway(store=store, transport=transport),
        recovery_token="r" * 48,
        ingress_token="i" * 48,
        sandbox_execution=SandboxService(),  # type: ignore[arg-type]
        sandbox_token="s" * 48,
    )
    request = {"tenant_id": "tenant-a", "command": "pwd", "timeout": 5}
    headers = {
        "X-OpenTulpa-Sandbox-Token": "s" * 48,
        "X-OpenTulpa-Release-ID": release.id,
        "X-OpenTulpa-Lease-Epoch": str(lease.epoch),
        "X-OpenTulpa-Control-Token": "c" * 48,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://bootstrap",
    ) as client:
        accepted = await client.post(
            "/bootstrap/internal/v1/sandbox/execute",
            headers=headers,
            json=request,
        )
        stale = await client.post(
            "/bootstrap/internal/v1/sandbox/execute",
            headers={**headers, "X-OpenTulpa-Lease-Epoch": str(lease.epoch + 1)},
            json=request,
        )
        forged = await client.post(
            "/bootstrap/internal/v1/sandbox/execute",
            headers={**headers, "X-OpenTulpa-Control-Token": "x" * 48},
            json=request,
        )

    assert accepted.status_code == 200
    assert stale.status_code == 401
    assert forged.status_code == 401
    await transport.aclose()


@pytest.mark.asyncio
async def test_gateway_reconciles_both_stable_authorities_for_each_lease(
    tmp_path: Path,
) -> None:
    class SandboxAuthority:
        def __init__(self) -> None:
            self.leases: list[tuple[str, int] | None] = []

        async def reconcile_lease(self, lease: Any) -> None:
            self.leases.append(
                None if lease is None else (lease.release_id, lease.lease_epoch)
            )

        async def execute(self, **_: Any) -> ExecuteResponse:
            return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    class CapabilityAuthority:
        def __init__(self) -> None:
            self.leases: list[tuple[str, int] | None] = []
            self.closed = False

        async def reconcile_lease(self, lease: Any) -> None:
            self.leases.append(
                None if lease is None else (lease.release_id, lease.lease_epoch)
            )

        async def aclose(self) -> None:
            self.closed = True

    release = _release()
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost(endpoint="http://127.0.0.1:49153", control_token="c" * 48)
    transport = ActiveReleaseTransport(store=store, host=host)
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    sandbox_authority = SandboxAuthority()
    capability_authority = CapabilityAuthority()
    app = create_gateway_app(
        supervisor=supervisor,
        gateway=BootstrapGateway(store=store, transport=transport),
        recovery_token="r" * 48,
        ingress_token="i" * 48,
        sandbox_execution=sandbox_authority,  # type: ignore[arg-type]
        sandbox_token="s" * 48,
        capability_workers=capability_authority,  # type: ignore[arg-type]
        capability_worker_token="w" * 48,
    )

    async with app.router.lifespan_context(app):
        initial = await supervisor.install_initial(release)
        await supervisor.enter_safe_mode()
        recovered = await supervisor.recover_last_known_good()

    expected = [
        None,
        (release.id, initial.epoch),
        None,
        (release.id, recovered.epoch),
    ]
    assert sandbox_authority.leases == expected
    assert capability_authority.leases == expected
    assert capability_authority.closed is True


@pytest.mark.asyncio
async def test_gateway_reconciles_all_authorities_before_failing_lease_transition(
    tmp_path: Path,
) -> None:
    class FailingSandboxAuthority:
        def __init__(self) -> None:
            self.leases: list[tuple[str, int] | None] = []

        async def reconcile_lease(self, lease: Any) -> None:
            value = None if lease is None else (lease.release_id, lease.lease_epoch)
            self.leases.append(value)
            if value is not None:
                raise RuntimeError("sandbox reconciliation failed")

        async def execute(self, **_: Any) -> ExecuteResponse:
            return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    class RecordingCapabilityAuthority:
        def __init__(self) -> None:
            self.leases: list[tuple[str, int] | None] = []

        async def reconcile_lease(self, lease: Any) -> None:
            self.leases.append(
                None if lease is None else (lease.release_id, lease.lease_epoch)
            )

        async def aclose(self) -> None:
            return None

    release = _release()
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost(endpoint="http://127.0.0.1:49153", control_token="c" * 48)
    transport = ActiveReleaseTransport(store=store, host=host)
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    sandbox_authority = FailingSandboxAuthority()
    capability_authority = RecordingCapabilityAuthority()
    app = create_gateway_app(
        supervisor=supervisor,
        gateway=BootstrapGateway(store=store, transport=transport),
        recovery_token="r" * 48,
        ingress_token="i" * 48,
        sandbox_execution=sandbox_authority,  # type: ignore[arg-type]
        sandbox_token="s" * 48,
        capability_workers=capability_authority,  # type: ignore[arg-type]
        capability_worker_token="w" * 48,
    )

    async with app.router.lifespan_context(app):
        with pytest.raises(RuntimeError, match="stable lease authority reconciliation failed"):
            await supervisor.install_initial(release)

    attempted = (release.id, 1)
    assert sandbox_authority.leases == [None, attempted, None]
    assert capability_authority.leases == [None, attempted, None]
    assert store.get_state().safe_mode is True


@pytest.mark.asyncio
async def test_gateway_persists_ingress_and_proxies_only_to_fenced_release(
    tmp_path: Path,
) -> None:
    release = _release()
    received_ingress: list[dict[str, Any]] = []
    received_events: list[dict[str, Any]] = []
    upstream = FastAPI()

    @upstream.post(release.ingress_path)
    async def ingress(request: Request) -> Response:
        received_ingress.append(
            {
                "body": await request.json(),
                "release": request.headers.get("x-opentulpa-release-id"),
                "epoch": request.headers.get("x-opentulpa-lease-epoch"),
                "idempotency": request.headers.get("idempotency-key"),
            }
        )
        return Response(status_code=204)

    @upstream.post(release.event_path)
    async def event(request: Request) -> Response:
        received_events.append(await request.json())
        return Response(status_code=204)

    @upstream.post("/echo")
    async def echo(request: Request) -> JSONResponse:
        response = JSONResponse(
            {
                "body": (await request.body()).decode(),
                "release": request.headers.get("x-opentulpa-release-id"),
                "epoch": request.headers.get("x-opentulpa-lease-epoch"),
                "idempotency": request.headers.get("idempotency-key"),
                "ingress_secret": request.headers.get("x-opentulpa-ingress-token"),
                "authorization": request.headers.get("authorization"),
                "gateway_authenticated": request.headers.get(
                    "x-opentulpa-control-token"
                )
                == "t" * 32,
            }
        )
        response.set_cookie("first", "1")
        response.set_cookie("second", "2")
        return response

    @upstream.get("/stream")
    async def stream() -> StreamingResponse:
        async def chunks():  # type: ignore[no-untyped-def]
            yield b"one"
            yield b"two"

        return StreamingResponse(chunks(), media_type="text/plain")

    @upstream.get("/recovery")
    @upstream.get("/recovery/{path:path}")
    async def mutable_recovery_asset(path: str = "") -> JSONResponse:
        return JSONResponse({"proxied": True, "path": path})

    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost(endpoint="http://127.0.0.1:49153")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream),
        base_url="http://127.0.0.1:49153",
    ) as release_client:
        transport = ActiveReleaseTransport(store=store, host=host, client=release_client)
        supervisor = BootstrapSupervisor(
            store=store,
            host=host,
            policy=_policy(),
            outbox_sink=transport,
        )
        await supervisor.start()
        lease = await supervisor.install_initial(release)
        gateway = BootstrapGateway(store=store, transport=transport, retry_interval_seconds=60)
        recovery_token = "r" * 32
        ingress_token = "i" * 32
        app = create_gateway_app(
            supervisor=supervisor,
            gateway=gateway,
            recovery_token=recovery_token,
            ingress_token=ingress_token,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://bootstrap",
        ) as client:
            body = {
                "tenant_id": "tenant_1",
                "thread_id": "thread_1",
                "payload": {"text": "hello"},
            }
            assert (await client.post("/bootstrap/v1/ingress/telegram", json=body)).status_code == 401
            headers = {
                "X-OpenTulpa-Ingress-Token": ingress_token,
                "Idempotency-Key": "telegram:update:42",
            }
            accepted = await client.post(
                "/bootstrap/v1/ingress/telegram",
                json=body,
                headers=headers,
            )
            assert accepted.status_code == 202
            ingress_id = accepted.json()["ingress_id"]
            duplicate = await client.post(
                "/bootstrap/v1/ingress/telegram",
                json=body,
                headers=headers,
            )
            assert duplicate.json()["ingress_id"] == ingress_id
            conflict = await client.post(
                "/bootstrap/v1/ingress/telegram",
                json={**body, "payload": {"text": "changed"}},
                headers=headers,
            )
            assert conflict.status_code == 409

            assert await gateway.dispatch_once() == 1
            status = await client.get(
                f"/bootstrap/v1/ingress/{ingress_id}",
                headers={"X-OpenTulpa-Ingress-Token": ingress_token},
            )
            assert status.json()["status"] == "processed"
            assert received_ingress == [
                {
                    "body": {
                        "id": ingress_id,
                        "tenant_id": "tenant_1",
                        "thread_id": "thread_1",
                        "channel": "telegram",
                        "idempotency_key": "telegram:update:42",
                        "payload": {"text": "hello"},
                        "status": "claimed",
                        "claimed_epoch": lease.epoch,
                        "attempt_count": 1,
                        "created_at": status.json()["created_at"],
                        "updated_at": received_ingress[0]["body"]["updated_at"],
                    },
                    "release": release.id,
                    "epoch": str(lease.epoch),
                    "idempotency": "telegram:update:42",
                }
            ]
            assert any(event["event_type"] == "release.active" for event in received_events)

            proxied = await client.post(
                "/echo?value=1",
                content=b"payload",
                headers={
                    "Idempotency-Key": "client-request-7",
                    "X-OpenTulpa-Release-ID": "forged",
                    "X-OpenTulpa-Lease-Epoch": "999",
                    "X-OpenTulpa-Ingress-Token": ingress_token,
                    "X-OpenTulpa-Control-Token": "forged",
                    "Authorization": f"Bearer {recovery_token}",
                },
            )
            assert proxied.status_code == 200
            assert proxied.json() == {
                "body": "payload",
                "release": release.id,
                "epoch": str(lease.epoch),
                "idempotency": "client-request-7",
                "ingress_secret": None,
                "authorization": None,
                "gateway_authenticated": True,
            }
            assert len(proxied.headers.get_list("set-cookie")) == 2
            assert (await client.get("/stream")).text == "onetwo"
            assert (await client.get("/recovery")).status_code == 404
            assert (await client.get("/recovery/index.html")).status_code == 404
            assert (await client.get("/bootstrap/not-a-real-route")).status_code == 404

            await supervisor.enter_safe_mode()
            assert (await client.get("/echo")).status_code == 503
            queued = await client.post(
                "/bootstrap/v1/ingress/telegram",
                json={**body, "payload": {"text": "while offline"}},
                headers={**headers, "Idempotency-Key": "telegram:update:43"},
            )
            assert queued.status_code == 202
            assert store.get_ingress(queued.json()["ingress_id"]).status == "pending"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_gateway_rejects_oversized_durable_ingress(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    transport = ActiveReleaseTransport(store=store, host=host)
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await supervisor.start()
    gateway = BootstrapGateway(store=store, transport=transport, max_ingress_bytes=1_024)
    token = "i" * 32
    app = create_gateway_app(
        supervisor=supervisor,
        gateway=gateway,
        recovery_token="r" * 32,
        ingress_token=token,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://bootstrap",
    ) as client:
        response = await client.post(
            "/bootstrap/v1/ingress/web",
            json={
                "tenant_id": "tenant_1",
                "thread_id": "thread_1",
                "payload": {"text": "x" * 2_000},
            },
            headers={
                "X-OpenTulpa-Ingress-Token": token,
                "Idempotency-Key": "large:1",
            },
        )
    await transport.aclose()

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_gateway_cleans_up_when_managed_evolution_start_fails(tmp_path: Path) -> None:
    class FailingEvolution:
        def __init__(self) -> None:
            self.service = object()
            self.shutdown_called = False

        async def start(self) -> None:
            raise RuntimeError("evolution startup failed")

        async def shutdown(self) -> None:
            self.shutdown_called = True

    class RecordingGateway(BootstrapGateway):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.shutdown_called = False

        async def shutdown(self) -> None:
            self.shutdown_called = True
            await super().shutdown()

    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    transport = ActiveReleaseTransport(store=store, host=host)
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    gateway = RecordingGateway(store=store, transport=transport)
    evolution = FailingEvolution()
    app = create_gateway_app(
        supervisor=supervisor,
        gateway=gateway,
        recovery_token="r" * 32,
        ingress_token="i" * 32,
        managed_evolution=evolution,  # type: ignore[arg-type]
        evolution_token="e" * 32,
    )

    with pytest.raises(RuntimeError, match="evolution startup failed"):
        async with app.router.lifespan_context(app):
            pass

    assert evolution.shutdown_called is True
    assert gateway.shutdown_called is True


def test_release_workspace_must_not_overlap_stable_state_or_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    state = tmp_path / "state"
    source.mkdir()
    state.mkdir()
    workspace = tmp_path / "workspace"

    assert _validate_release_workspace(
        workspace,
        state_root=state,
        source_root=source,
    ) == workspace.resolve()
    for forbidden in (state, state / "release-data", source, source / "data", tmp_path):
        with pytest.raises(RuntimeError, match="cannot expose"):
            _validate_release_workspace(
                forbidden,
                state_root=state,
                source_root=source,
            )


def test_release_environment_never_injects_stable_recovery_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovery_token = "recovery-secret-" + "r" * 32
    monkeypatch.setenv("OPENTULPA_RECOVERY_TOKEN", recovery_token)
    monkeypatch.setenv("OPENTULPA_INGRESS_TOKEN", "ingress-secret-" + "i" * 32)

    environment = dict(
        _production_environment(
            Settings(openai_compatible_api_key="model-key"),
            evolution_url="http://host.docker.internal:8000/evolution",
            evolution_token="runtime-evolution-token-" + "e" * 32,
            sandbox_url="http://host.docker.internal:8000/sandbox",
            sandbox_token="runtime-sandbox-token-" + "s" * 32,
            capability_worker_url=(
                "http://host.docker.internal:8000/capability-workers"
            ),
            capability_worker_token="runtime-worker-token-" + "w" * 32,
            internal_agent_api_url="http://host.docker.internal:8000",
        )
    )

    assert "OPENTULPA_RECOVERY_TOKEN" not in environment
    assert "OPENTULPA_INGRESS_TOKEN" not in environment
    assert recovery_token not in environment.values()
    assert environment["OPENTULPA_BOOTSTRAP_EVOLUTION_TOKEN"].startswith(
        "runtime-evolution-token-"
    )
    assert environment["OPENTULPA_BOOTSTRAP_SANDBOX_TOKEN"].startswith(
        "runtime-sandbox-token-"
    )
    assert environment["OPENTULPA_BOOTSTRAP_CAPABILITY_WORKER_TOKEN"].startswith(
        "runtime-worker-token-"
    )
    assert environment["OPENTULPA_INTERNAL_AGENT_API_URL"] == (
        "http://host.docker.internal:8000"
    )
    assert environment["LLM_FALLBACK_MODELS"] == (
        '["z-ai/glm-5.2","google/gemini-3.1-pro-preview"]'
    )
    assert "SANDBOX_IMAGE" not in environment
