from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from opentulpa.api.routes.v2_capabilities import register_v2_capability_routes
from opentulpa.api.routes.v2_principal import V2Principal
from opentulpa.capabilities import (
    CapabilityControlService,
    CapabilityManifest,
    CapabilityRevisionStore,
    CapabilityTestCheck,
    CapabilityTestStatus,
    CapabilityWorkerManager,
    EvalCommand,
    WorkerHandle,
    WorkerKind,
    WorkerLaunch,
    WorkerSpec,
)


@dataclass
class _Principal:
    tenant_id: str
    actor_id: str


class _Evaluator:
    async def evaluate(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> Sequence[CapabilityTestCheck]:
        del tenant_id, manifest
        return (
            CapabilityTestCheck(
                name="pytest",
                status=CapabilityTestStatus.PASSED,
            ),
        )


class _Host:
    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        return WorkerHandle(
            id=f"handle-{launch.instance_id}",
            instance_id=launch.instance_id,
            capability_name=launch.manifest.name,
            capability_revision=launch.manifest.revision,
            manifest_digest=launch.manifest.content_digest,
            worker_name=launch.worker.name,
        )

    async def healthy(self, handle: WorkerHandle) -> bool:
        return True

    async def stop(self, handle: WorkerHandle) -> None:
        return None


def _manifest() -> CapabilityManifest:
    return CapabilityManifest(
        name="example",
        version="1.0.0",
        artifact_digest=f"sha256:{'1' * 64}",
        workers=(
            WorkerSpec(
                name="example_interface",
                kind=WorkerKind.INTERFACE,
                protocol="agent-interface-v1",
                command=("example-worker",),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )


def _app(tmp_path: Path) -> tuple[FastAPI, CapabilityControlService]:
    service = CapabilityControlService(
        revisions=CapabilityRevisionStore(tmp_path / "capabilities.sqlite3"),
        evaluator=_Evaluator(),
        workers=CapabilityWorkerManager(_Host()),
        bundled=(_manifest().model_copy(update={"seed": True}),),
    )
    app = FastAPI()

    def principal(request: Request) -> V2Principal:
        return _Principal(
            tenant_id=request.headers.get("X-Tenant", "tenant-a"),
            actor_id="owner-a",
        )

    register_v2_capability_routes(
        app,
        get_capabilities=lambda: service,
        resolve_principal=principal,
    )
    return app, service


@pytest.mark.asyncio
async def test_v2_bundled_capability_test_activate_and_deactivate(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        seeded = await client.post("/v2/capabilities/seed-bundled")
        blocked = await client.post(
            "/v2/capabilities/example/activate",
            json={"revision": 1, "expected_generation": None},
        )
        tested = await client.post(
            "/v2/capabilities/example/test",
            json={"revision": 1},
        )
        activated = await client.post(
            "/v2/capabilities/example/activate",
            json={"revision": 1, "expected_generation": None},
        )
        listed = await client.get("/v2/capabilities")
        other_tenant = await client.get(
            "/v2/capabilities",
            headers={"X-Tenant": "tenant-b"},
        )
        deleted = await client.delete(
            "/v2/capabilities/example",
            params={"expected_generation": 1},
        )

    assert seeded.status_code == 201
    assert blocked.status_code == 409
    assert tested.json()["test"]["status"] == "passed"
    assert activated.json()["activation"]["generation"] == 1
    assert listed.json()["capabilities"][0]["manifest"]["name"] == "example"
    assert other_tenant.json() == {"capabilities": []}
    assert deleted.json()["deactivated"] is True


@pytest.mark.asyncio
async def test_v2_does_not_accept_tenant_capability_manifests(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v2/capabilities",
            json={"manifest": _manifest().model_dump(mode="json")},
        )

    assert response.status_code == 405


@pytest.mark.asyncio
async def test_v2_seed_bundled_is_tenant_scoped_and_idempotent(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        first = await client.post("/v2/capabilities/seed-bundled")
        second = await client.post("/v2/capabilities/seed-bundled")
        other = await client.get(
            "/v2/capabilities",
            headers={"X-Tenant": "tenant-b"},
        )

    assert first.status_code == second.status_code == 201
    assert first.json() == second.json()
    assert other.json() == {"capabilities": []}
