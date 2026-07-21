from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from opentulpa.bootstrap.capability_worker_api import (
    CapabilityWorkerAPIError,
    CapabilityWorkerClient,
    CapabilityWorkerLease,
    CapabilityWorkerStartRequest,
    StableCapabilityWorkerService,
    register_capability_worker_api,
)
from opentulpa.bootstrap.models import ReleaseRecord
from opentulpa.capabilities import (
    CapabilityManifest,
    EvalCommand,
    SecretRequirement,
    WorkerHandle,
    WorkerKind,
    WorkerLaunch,
    WorkerLifecycleError,
    WorkerRuntime,
    WorkerSpec,
    WorkerTransport,
)

DIGEST = f"sha256:{'a' * 64}"


class _Host:
    def __init__(
        self,
        root: Path,
        *,
        runtime: dict[str, tuple[str, int, str, str]] | None = None,
    ) -> None:
        self.root = root
        self.runtime = runtime if runtime is not None else {}
        self.launches: list[WorkerLaunch] = []
        self.stopped: list[str] = []
        self.adopted: list[WorkerHandle] = []
        self.fenced: list[tuple[str, str]] = []
        self.fail_stops = 0
        self.fail_fences = 0
        self.fail_starts_after_launch = 0
        self.closed = False

    def adopt(
        self,
        *,
        handle: WorkerHandle,
        worker: WorkerSpec,
        ready_path: Path | None,
        tenant_id: str = "default",
        release_id: str | None = None,
        lease_epoch: int | None = None,
    ) -> None:
        del worker, ready_path, tenant_id, release_id, lease_epoch
        self.adopted.append(handle)

    def ready_path(self, handle: WorkerHandle) -> Path:
        return self.root / f"{handle.worker_name}.ready"

    async def start(
        self,
        launch: WorkerLaunch,
        *,
        release_id: str | None = None,
        lease_epoch: int | None = None,
    ) -> WorkerHandle:
        assert release_id is not None and lease_epoch is not None
        self.launches.append(launch)
        container_id = f"{len(self.runtime) + 1:064x}"
        self.runtime[container_id] = (
            release_id,
            lease_epoch,
            launch.tenant_id,
            launch.manifest.name,
        )
        if self.fail_starts_after_launch:
            self.fail_starts_after_launch -= 1
            raise WorkerLifecycleError("injected worker launch cleanup failure")
        return WorkerHandle(
            id=f"oci:{container_id}",
            instance_id=launch.instance_id,
            capability_name=launch.manifest.name,
            capability_revision=launch.manifest.revision,
            manifest_digest=launch.manifest.content_digest,
            worker_name=launch.worker.name,
            endpoint=(
                "http://127.0.0.1:49152/mcp"
                if launch.worker.transport is WorkerTransport.STREAMABLE_HTTP
                else None
            ),
        )

    async def healthy(self, handle: WorkerHandle) -> bool:
        return handle.id.removeprefix("oci:") in self.runtime

    async def stop(self, handle: WorkerHandle) -> None:
        if self.fail_stops:
            self.fail_stops -= 1
            raise WorkerLifecycleError("injected worker removal failure")
        self.stopped.append(handle.id)
        self.runtime.pop(handle.id.removeprefix("oci:"), None)

    async def fence(self, *, tenant_id: str, capability_name: str) -> None:
        if self.fail_fences:
            self.fail_fences -= 1
            raise WorkerLifecycleError("injected worker fence failure")
        self.fenced.append((tenant_id, capability_name))
        for container_id, (_, _, owner, capability) in tuple(self.runtime.items()):
            if owner == tenant_id and capability == capability_name:
                self.stopped.append(f"oci:{container_id}")
                self.runtime.pop(container_id, None)

    async def reconcile_managed_workers(
        self,
        *,
        release_id: str | None,
        lease_epoch: int | None,
        keep_container_ids: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        keep = {
            container_id
            for container_id in keep_container_ids
            if self.runtime.get(container_id, ())[:2] == (release_id, lease_epoch)
        }
        for container_id in tuple(self.runtime):
            if container_id not in keep:
                self.stopped.append(f"oci:{container_id}")
                self.runtime.pop(container_id, None)
        return tuple(sorted(keep))

    async def aclose(self) -> None:
        self.closed = True


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()
        self._unblock = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.entered.set()
        await self._unblock.wait()
        yield b"after-close"

    async def aclose(self) -> None:
        self.closed.set()
        self._unblock.set()


def _release() -> ReleaseRecord:
    return ReleaseRecord(
        id="release-test",
        candidate_id="candidate-test",
        source_commit="b" * 40,
        artifact_digest=DIGEST,
        manifest_digest=f"sha256:{'c' * 64}",
        entrypoint=("python", "-m", "opentulpa"),
    )


def _manifest(*, mcp: bool = False) -> CapabilityManifest:
    worker = WorkerSpec(
        name="example_mcp" if mcp else "example_interface",
        kind=WorkerKind.MCP if mcp else WorkerKind.INTERFACE,
        protocol="mcp-v1" if mcp else "agent-interface-v1",
        transport=WorkerTransport.STREAMABLE_HTTP if mcp else WorkerTransport.STDIO,
        endpoint="http://127.0.0.1:8080/mcp" if mcp else None,
        command=("python", "-m", "opentulpa.capability_workers.example"),
        secrets=(
            SecretRequirement(
                name="EXAMPLE_TOKEN",
                scopes=("example.use",),
            ),
        ),
    )
    return CapabilityManifest(
        name="example",
        version="1.0.0",
        workers=(worker,),
        secrets=worker.secrets,
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
        seed=True,
    )


def test_worker_start_request_rejects_runtime_environment_secret_override() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="runtime environment"):
        CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="cap-1",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"PATH": "/untrusted/bin"},
        )


@pytest.mark.asyncio
async def test_stable_service_derives_release_image_and_persists_no_secrets(
    tmp_path: Path,
) -> None:
    host = _Host(tmp_path)
    state_path = tmp_path / "bootstrap" / "workers.json"
    release = _release()
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda release_id: release if release_id == release.id else None,
        state_path=state_path,
    )
    manifest = _manifest()
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=7)
    await service.reconcile_lease(lease)
    result = await service.start(
        lease=lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g1",
            manifest=manifest,
            worker=manifest.workers[0],
            config={"state_path": "/state/example.json"},
            secret_environment={"EXAMPLE_TOKEN": "never-persist-this"},
        ),
    )

    launch = host.launches[0]
    assert launch.tenant_id == "tenant-a"
    assert launch.worker.runtime is WorkerRuntime.OCI
    assert launch.worker.image == DIGEST
    assert launch.manifest.artifact_digest == DIGEST
    assert result.manifest_digest == manifest.content_digest
    assert "never-persist-this" not in state_path.read_text(encoding="utf-8")
    assert "/workspace" not in state_path.read_text(encoding="utf-8")

    adopted_host = _Host(tmp_path, runtime=host.runtime)
    restored = StableCapabilityWorkerService(
        host=adopted_host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=state_path,
    )
    await restored.reconcile_lease(lease)
    assert [item.id for item in adopted_host.adopted] == [f"oci:{1:064x}"]
    await restored.stop(result.handle_id, lease=lease)
    await restored.aclose()
    await service.aclose()


@pytest.mark.asyncio
async def test_stable_fence_failure_keeps_durable_record_for_restart_retry(
    tmp_path: Path,
) -> None:
    host = _Host(tmp_path)
    state_path = tmp_path / "workers.json"
    release = _release()
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=3)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=state_path,
    )
    await service.reconcile_lease(lease)
    manifest = _manifest()
    result = await service.start(
        lease=lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g1",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "secret"},
        ),
    )
    host.fail_fences = 1

    with pytest.raises(WorkerLifecycleError, match="fence failure"):
        await service.fence(
            lease=lease,
            tenant_id="tenant-a",
            capability_name="example",
        )

    assert result.handle_id in state_path.read_text(encoding="utf-8")

    restarted_host = _Host(tmp_path, runtime=host.runtime)
    restarted = StableCapabilityWorkerService(
        host=restarted_host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=state_path,
    )
    await restarted.reconcile_lease(lease)
    await restarted.fence(
        lease=lease,
        tenant_id="tenant-a",
        capability_name="example",
    )

    assert result.handle_id not in state_path.read_text(encoding="utf-8")
    assert host.runtime == {}


@pytest.mark.asyncio
async def test_restart_fences_persisted_worker_from_stale_release_lease(tmp_path: Path) -> None:
    host = _Host(tmp_path)
    state_path = tmp_path / "workers.json"
    release = _release()
    old_lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=4)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=state_path,
    )
    await service.reconcile_lease(old_lease)
    manifest = _manifest()
    result = await service.start(
        lease=old_lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g1",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "secret"},
        ),
    )

    restarted_host = _Host(tmp_path, runtime=host.runtime)
    restarted = StableCapabilityWorkerService(
        host=restarted_host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=state_path,
    )
    await restarted.reconcile_lease(
        CapabilityWorkerLease(release_id=release.id, lease_epoch=5)
    )

    assert restarted_host.adopted == []
    assert host.runtime == {}
    assert result.handle_id not in state_path.read_text(encoding="utf-8")
    with pytest.raises(CapabilityWorkerAPIError, match="not found"):
        await restarted.healthy(result.handle_id)


@pytest.mark.asyncio
async def test_startup_reconciliation_removes_unrecorded_current_lease_orphan(
    tmp_path: Path,
) -> None:
    release = _release()
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=5)
    container_id = f"{9:064x}"
    runtime = {
        container_id: (release.id, lease.lease_epoch, "tenant-a", "example")
    }
    host = _Host(tmp_path, runtime=runtime)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
    )

    await service.reconcile_lease(lease)

    assert runtime == {}
    assert host.adopted == []
    assert host.stopped == [f"oci:{container_id}"]


@pytest.mark.asyncio
async def test_persistence_and_cleanup_failure_retains_provisional_until_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _Host(tmp_path)
    release = _release()
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=6)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
    )
    await service.reconcile_lease(lease)
    original_write = service._write_state
    fail_write = True

    def fail_once() -> None:
        nonlocal fail_write
        if fail_write:
            fail_write = False
            raise OSError("injected persistence failure")
        original_write()

    monkeypatch.setattr(service, "_write_state", fail_once)
    host.fail_stops = 1
    manifest = _manifest()

    with pytest.raises(WorkerLifecycleError, match="persistence cleanup failed"):
        await service.start(
            lease=lease,
            request=CapabilityWorkerStartRequest(
                tenant_id="tenant-a",
                instance_id="example-g1",
                manifest=manifest,
                worker=manifest.workers[0],
                secret_environment={"EXAMPLE_TOKEN": "secret"},
            ),
        )

    provisional_id = next(iter(service._records))
    assert host.runtime
    with pytest.raises(CapabilityWorkerAPIError, match="not found"):
        await service.healthy(provisional_id)

    await service.reconcile_lease(lease)

    assert host.runtime == {}
    assert service._records == {}
    assert provisional_id not in (tmp_path / "workers.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_launch_cleanup_failure_forces_orphan_reconciliation(tmp_path: Path) -> None:
    host = _Host(tmp_path)
    release = _release()
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=6)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
    )
    await service.reconcile_lease(lease)
    host.fail_starts_after_launch = 1
    manifest = _manifest()
    request = CapabilityWorkerStartRequest(
        tenant_id="tenant-a",
        instance_id="example-g1",
        manifest=manifest,
        worker=manifest.workers[0],
        secret_environment={"EXAMPLE_TOKEN": "secret"},
    )

    with pytest.raises(WorkerLifecycleError, match="launch cleanup failure"):
        await service.start(lease=lease, request=request)

    orphan_id = next(iter(host.runtime))
    result = await service.start(lease=lease, request=request)

    assert f"oci:{orphan_id}" in host.stopped
    assert await service.healthy(result.handle_id) is True


@pytest.mark.asyncio
async def test_stale_start_waiting_behind_lease_reconciliation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _Host(tmp_path)
    release = _release()
    old_lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=6)
    new_lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=7)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
    )
    await service.reconcile_lease(old_lease)
    reconcile_started = asyncio.Event()
    allow_reconcile = asyncio.Event()
    original_reconcile = host.reconcile_managed_workers

    async def blocked_reconcile(**kwargs):  # type: ignore[no-untyped-def]
        reconcile_started.set()
        await allow_reconcile.wait()
        return await original_reconcile(**kwargs)

    monkeypatch.setattr(host, "reconcile_managed_workers", blocked_reconcile)
    reconcile = asyncio.create_task(service.reconcile_lease(new_lease))
    await reconcile_started.wait()
    manifest = _manifest()
    stale_start = asyncio.create_task(
        service.start(
            lease=old_lease,
            request=CapabilityWorkerStartRequest(
                tenant_id="tenant-a",
                instance_id="example-g1",
                manifest=manifest,
                worker=manifest.workers[0],
                secret_environment={"EXAMPLE_TOKEN": "secret"},
            ),
        )
    )
    stale_fence = asyncio.create_task(
        service.fence(
            lease=old_lease,
            tenant_id="tenant-a",
            capability_name="example",
        )
    )
    await asyncio.sleep(0)
    allow_reconcile.set()
    await reconcile

    with pytest.raises(CapabilityWorkerAPIError, match="lease changed"):
        await stale_start
    with pytest.raises(CapabilityWorkerAPIError, match="lease changed"):
        await stale_fence
    assert host.launches == []
    assert host.fenced == []

    current = await service.start(
        lease=new_lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g2",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "secret"},
        ),
    )
    with pytest.raises(CapabilityWorkerAPIError, match="lease changed"):
        await service.fence(
            lease=old_lease,
            tenant_id="tenant-a",
            capability_name="example",
        )
    assert await service.healthy(current.handle_id) is True
    assert host.fenced == []


@pytest.mark.asyncio
async def test_stable_service_rejects_unreviewed_command_and_secret(tmp_path: Path) -> None:
    release = _release()
    service = StableCapabilityWorkerService(
        host=_Host(tmp_path),  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
    )
    manifest = _manifest()
    unsafe_worker = manifest.workers[0].model_copy(
        update={"command": ("python", "-m", "untrusted.worker")}
    )
    unsafe_manifest = manifest.model_copy(update={"workers": (unsafe_worker,)})

    with pytest.raises(CapabilityWorkerAPIError, match="reviewed package"):
        await service.start(
            lease=CapabilityWorkerLease(release_id=release.id, lease_epoch=1),
            request=CapabilityWorkerStartRequest(
                tenant_id="tenant-a",
                instance_id="unsafe-g1",
                manifest=unsafe_manifest,
                worker=unsafe_worker,
            ),
        )
    with pytest.raises(CapabilityWorkerAPIError, match="undeclared secret"):
        await service.start(
            lease=CapabilityWorkerLease(release_id=release.id, lease_epoch=1),
            request=CapabilityWorkerStartRequest(
                tenant_id="tenant-a",
                instance_id="unsafe-g2",
                manifest=manifest,
                worker=manifest.workers[0],
                secret_environment={"HOST_DATABASE_PASSWORD": "no"},
            ),
        )
    await service.aclose()


@pytest.mark.asyncio
async def test_stable_service_fences_orphaned_prior_interface_generation(
    tmp_path: Path,
) -> None:
    release = _release()
    host = _Host(tmp_path)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
    )
    manifest = _manifest()
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=2)
    await service.reconcile_lease(lease)
    first = await service.start(
        lease=lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g1",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "first"},
        ),
    )
    second = await service.start(
        lease=lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g2",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "second"},
        ),
    )

    assert first.handle_id != second.handle_id
    assert host.stopped == [f"oci:{1:064x}"]
    assert "example-g1" not in (tmp_path / "workers.json").read_text(encoding="utf-8")
    assert "example-g2" in (tmp_path / "workers.json").read_text(encoding="utf-8")
    await service.aclose()


@pytest.mark.asyncio
async def test_lease_reconciliation_waits_for_admitted_proxy_and_rejects_stale_proxy(
    tmp_path: Path,
) -> None:
    release = _release()
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"complete"))
    )
    host = _Host(tmp_path)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
        http_client=upstream_client,
    )
    old_lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=8)
    new_lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=9)
    await service.reconcile_lease(old_lease)
    manifest = _manifest(mcp=True)
    result = await service.start(
        lease=old_lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g1",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "secret"},
        ),
    )

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    def request() -> Request:
        return Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/proxy",
                "raw_path": b"/proxy",
                "query_string": b"",
                "headers": (),
                "client": ("127.0.0.1", 1),
                "server": ("bootstrap", 80),
            },
            receive,
        )

    admitted = await service.proxy(
        result.handle_id,
        request(),
        lease=old_lease,
        suffix="",
    )
    reconcile = asyncio.create_task(service.reconcile_lease(new_lease))
    await asyncio.sleep(0)
    assert not reconcile.done()
    with pytest.raises(CapabilityWorkerAPIError, match="lease changed"):
        await service.proxy(
            result.handle_id,
            request(),
            lease=old_lease,
            suffix="",
        )

    assert b"".join([chunk async for chunk in admitted.body_iterator]) == b"complete"
    await reconcile
    assert host.runtime == {}
    await service.aclose()
    await upstream_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_kind", ["start", "stop", "fence"])
async def test_same_lease_lifecycle_waits_for_admitted_proxy(
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    release = _release()
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"complete"))
    )
    host = _Host(tmp_path)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
        http_client=upstream_client,
    )
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=8)
    await service.reconcile_lease(lease)
    manifest = _manifest(mcp=True)
    start_request = CapabilityWorkerStartRequest(
        tenant_id="tenant-a",
        instance_id="example-g1",
        manifest=manifest,
        worker=manifest.workers[0],
        secret_environment={"EXAMPLE_TOKEN": "secret"},
    )
    result = await service.start(lease=lease, request=start_request)
    old_container_id = next(iter(host.runtime))

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/proxy",
            "raw_path": b"/proxy",
            "query_string": b"",
            "headers": (),
            "client": ("127.0.0.1", 1),
            "server": ("bootstrap", 80),
        },
        receive,
    )
    admitted = await service.proxy(
        result.handle_id,
        request,
        lease=lease,
        suffix="",
    )
    if mutation_kind == "start":
        mutation = asyncio.create_task(
            service.start(
                lease=lease,
                request=start_request.model_copy(update={"instance_id": "example-g2"}),
            )
        )
    elif mutation_kind == "stop":
        mutation = asyncio.create_task(service.stop(result.handle_id, lease=lease))
    else:
        mutation = asyncio.create_task(
            service.fence(
                lease=lease,
                tenant_id="tenant-a",
                capability_name="example",
            )
        )
    await asyncio.sleep(0)

    assert not mutation.done()
    assert old_container_id in host.runtime
    assert b"".join([chunk async for chunk in admitted.body_iterator]) == b"complete"
    await mutation
    if mutation_kind == "start":
        assert f"oci:{old_container_id}" in host.stopped
    else:
        assert old_container_id not in host.runtime
    await service.aclose()
    await upstream_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_kind", ["stop", "reconcile"])
async def test_nonterminating_proxy_cannot_pin_worker_removal(
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    release = _release()
    stream = _BlockingStream()
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    )
    host = _Host(tmp_path)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
        proxy_timeout_seconds=1,
        http_client=upstream_client,
    )
    old_lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=8)
    await service.reconcile_lease(old_lease)
    manifest = _manifest(mcp=True)
    result = await service.start(
        lease=old_lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g1",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "secret"},
        ),
    )

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    admitted = await service.proxy(
        result.handle_id,
        Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/proxy",
                "raw_path": b"/proxy",
                "query_string": b"",
                "headers": (),
                "client": ("127.0.0.1", 1),
                "server": ("bootstrap", 80),
            },
            receive,
        ),
        lease=old_lease,
        suffix="",
    )

    async def consume() -> None:
        async for _ in admitted.body_iterator:
            pass

    consumer = asyncio.create_task(consume())
    await stream.entered.wait()
    if mutation_kind == "stop":
        mutation = asyncio.create_task(service.stop(result.handle_id, lease=old_lease))
    else:
        mutation = asyncio.create_task(
            service.reconcile_lease(
                CapabilityWorkerLease(release_id=release.id, lease_epoch=9)
            )
        )

    await asyncio.wait_for(mutation, timeout=2.5)
    assert host.runtime == {}
    assert stream.closed.is_set()
    assert service._inflight_proxy_operations == 0
    assert consumer.done()
    with suppress(asyncio.CancelledError, CapabilityWorkerAPIError):
        await consumer
    await service.aclose()
    await upstream_client.aclose()


@pytest.mark.asyncio
async def test_cancelled_proxy_stream_releases_lifecycle_admission(tmp_path: Path) -> None:
    release = _release()
    stream = _BlockingStream()
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    )
    host = _Host(tmp_path)
    service = StableCapabilityWorkerService(
        host=host,  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
        proxy_timeout_seconds=30,
        http_client=upstream_client,
    )
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=8)
    await service.reconcile_lease(lease)
    manifest = _manifest(mcp=True)
    result = await service.start(
        lease=lease,
        request=CapabilityWorkerStartRequest(
            tenant_id="tenant-a",
            instance_id="example-g1",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "secret"},
        ),
    )

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    admitted = await service.proxy(
        result.handle_id,
        Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/proxy",
                "raw_path": b"/proxy",
                "query_string": b"",
                "headers": (),
                "client": ("127.0.0.1", 1),
                "server": ("bootstrap", 80),
            },
            receive,
        ),
        lease=lease,
        suffix="",
    )

    async def consume() -> None:
        async for _ in admitted.body_iterator:
            pass

    consumer = asyncio.create_task(consume())
    await stream.entered.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert stream.closed.is_set()
    assert service._inflight_proxy_operations == 0
    await asyncio.wait_for(service.stop(result.handle_id, lease=lease), timeout=0.5)
    assert host.runtime == {}
    await service.aclose()
    await upstream_client.aclose()


@pytest.mark.asyncio
async def test_lease_authenticated_client_receives_private_proxy_endpoint(
    tmp_path: Path,
) -> None:
    release = _release()
    proxied: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        proxied.append(request)
        return httpx.Response(
            200,
            content=b"event: message\ndata: {}\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    service = StableCapabilityWorkerService(
        host=_Host(tmp_path),  # type: ignore[arg-type]
        release_loader=lambda _: release,
        state_path=tmp_path / "workers.json",
        http_client=upstream_client,
    )
    lease = CapabilityWorkerLease(release_id=release.id, lease_epoch=9)
    await service.reconcile_lease(lease)
    observed: list[tuple[str, int, str]] = []

    async def authorize(release_id: str, lease_epoch: int, control_token: str) -> None:
        observed.append((release_id, lease_epoch, control_token))
        if (release_id, lease_epoch, control_token) != (release.id, 9, "c" * 48):
            raise RuntimeError("stale lease")

    app = FastAPI()
    register_capability_worker_api(
        app,
        service=service,
        token="w" * 48,
        authorize_lease=authorize,
    )
    client = CapabilityWorkerClient(
        base_url="http://bootstrap/bootstrap/internal/v1/capability-workers",
        token="w" * 48,
        release_id=release.id,
        lease_epoch=9,
        control_token="c" * 48,
        transport=httpx.ASGITransport(app=app),
    )
    manifest = _manifest(mcp=True)
    handle = await client.start(
        WorkerLaunch(
            tenant_id="tenant-a",
            instance_id="example-g1",
            manifest=manifest,
            worker=manifest.workers[0],
            secret_environment={"EXAMPLE_TOKEN": "secret"},
        )
    )

    assert handle.endpoint is not None and handle.endpoint.endswith(f"/{handle.id}/proxy")
    assert handle.endpoint_headers["X-OpenTulpa-Release-ID"] == release.id
    assert await client.healthy(handle) is True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
    ) as proxy_client:
        response = await proxy_client.get(
            handle.endpoint,
            headers={**handle.endpoint_headers, "Accept": "text/event-stream"},
        )
    assert response.status_code == 200
    assert response.content == b"event: message\ndata: {}\n\n"
    assert proxied[0].url == "http://127.0.0.1:49152/mcp"
    assert "x-opentulpa-control-token" not in proxied[0].headers
    await client.stop(handle)
    assert observed == [(release.id, 9, "c" * 48)] * 4
    await client.aclose()
    await service.aclose()
    await upstream_client.aclose()
