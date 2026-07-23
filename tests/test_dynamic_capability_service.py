from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from opentulpa.capabilities import (
    CapabilityControlService,
    CapabilityManifest,
    CapabilityRevisionStore,
    CapabilityRuntimeUnavailableError,
    CapabilitySecretBinding,
    CapabilityTestCheck,
    CapabilityTestRequiredError,
    CapabilityTestStatus,
    CapabilityWorkerManager,
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


class _Evaluator:
    def __init__(self) -> None:
        self.failed: set[int] = set()
        self.calls: list[tuple[str, int]] = []

    async def evaluate(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> Sequence[CapabilityTestCheck]:
        self.calls.append((tenant_id, manifest.revision))
        status = (
            CapabilityTestStatus.FAILED
            if manifest.revision in self.failed
            else CapabilityTestStatus.PASSED
        )
        return (CapabilityTestCheck(name="pytest", status=status, message="safe"),)


class _Host:
    def __init__(
        self,
        *,
        healthy: bool = True,
        fail_revisions: set[int] | None = None,
    ) -> None:
        self.launches: list[WorkerLaunch] = []
        self.stopped: list[str] = []
        self.events: list[tuple[str, str]] = []
        self.is_healthy = healthy
        self.fail_revisions = fail_revisions or set()
        self.closed = False

    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        self.launches.append(launch)
        self.events.append(("start", launch.instance_id))
        if launch.manifest.revision in self.fail_revisions:
            raise WorkerLifecycleError("injected worker start failure")
        return WorkerHandle(
            id=f"handle-{len(self.launches)}",
            instance_id=launch.instance_id,
            capability_name=launch.manifest.name,
            capability_revision=launch.manifest.revision,
            manifest_digest=launch.manifest.content_digest,
            worker_name=launch.worker.name,
        )

    async def healthy(self, handle: WorkerHandle) -> bool:
        return self.is_healthy

    async def stop(self, handle: WorkerHandle) -> None:
        self.stopped.append(handle.instance_id)
        self.events.append(("stop", handle.instance_id))

    async def aclose(self) -> None:
        self.closed = True


class _Secrets:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.bind_calls: list[dict[str, Any]] = []
        self.revoked: list[tuple[str, str]] = []
        self.revision = 1

    async def bind(self, **kwargs: Any) -> Mapping[str, CapabilitySecretBinding]:
        self.bind_calls.append(kwargs)
        return {
            name: CapabilitySecretBinding(
                handle_id=handle_id,
                revision=self.revision,
                scopes=("capability.invoke",),
            )
            for name, handle_id in kwargs["secret_handles"].items()
        }

    async def resolve(self, **kwargs: Any) -> Mapping[str, str]:
        self.calls.append(kwargs)
        return {"CAPABILITY_TOKEN": "private-value"}

    async def revoke(self, *, tenant_id: str, instance_id: str) -> None:
        self.revoked.append((tenant_id, instance_id))


class _FencingHost(_Host):
    def __init__(self) -> None:
        super().__init__()
        self.fenced: list[tuple[str, str]] = []

    async def fence(self, *, tenant_id: str, capability_name: str) -> None:
        self.fenced.append((tenant_id, capability_name))


class _ToolHost:
    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.stopped: list[str] = []

    async def start(self, **kwargs: Any) -> None:
        self.started.append(kwargs)

    async def stop(self, instance_id: str) -> None:
        self.stopped.append(instance_id)


def _manifest(revision: int, *, worker: bool = False) -> CapabilityManifest:
    workers = (
        WorkerSpec(
            name="example_interface",
            kind=WorkerKind.INTERFACE,
            protocol="agent-interface-v1",
            command=("example-worker",),
            runtime=WorkerRuntime.OCI,
            image=f"example@sha256:{revision:064x}",
            secrets=(
                SecretRequirement(
                    name="CAPABILITY_TOKEN",
                    scopes=("capability.invoke",),
                ),
            )
            if worker
            else (),
        ),
    )
    return CapabilityManifest(
        name="example",
        version=f"1.{revision - 1}.0",
        revision=revision,
        artifact_digest=f"sha256:{revision:064x}",
        workers=workers,
        secrets=(
            SecretRequirement(
                name="CAPABILITY_TOKEN",
                scopes=("capability.invoke",),
            ),
        )
        if worker
        else (),
        config_schema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["safe"]},
                "agent_api_url": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )


@pytest.mark.asyncio
async def test_activation_requires_passing_digest_bound_test_and_is_tenant_scoped(
    tmp_path: Path,
) -> None:
    evaluator = _Evaluator()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        bundled=(),
    )
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1),
        expected_latest_revision=None,
    )

    with pytest.raises(CapabilityTestRequiredError):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=1,
            expected_generation=None,
            config={"mode": "safe"},
        )
    result = await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )
    with pytest.raises(ValueError, match="secret handles"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=1,
            expected_generation=None,
            config={"api_token": "must-not-be-persisted"},
        )
    active = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
        config={"mode": "safe"},
    )

    assert result.status is CapabilityTestStatus.PASSED
    assert active.config == {"mode": "safe"}
    assert service.list(tenant_id="tenant-b") == []
    assert service.get(tenant_id="tenant-a", capability_name="example") is not None


def test_dynamic_save_requires_out_of_process_content_addressed_artifact(
    tmp_path: Path,
) -> None:
    service = CapabilityControlService(
        revisions=CapabilityRevisionStore(tmp_path / "capabilities.sqlite3"),
        bundled=(),
    )
    in_process = CapabilityManifest(
        name="unsafe",
        version="1.0.0",
        artifact_digest=f"sha256:{'1' * 64}",
        module="opentulpa.api.app",
        entrypoint="create_app",
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    missing_artifact = _manifest(1).model_copy(update={"artifact_digest": None})
    host_subprocess = _manifest(1).model_copy(
        update={
            "workers": tuple(
                worker.model_copy(
                    update={
                        "runtime": "subprocess",
                        "image": None,
                    }
                )
                for worker in _manifest(1).workers
            )
        }
    )

    with pytest.raises(ValueError, match="out of process"):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            manifest=in_process,
            expected_latest_revision=None,
        )
    with pytest.raises(ValueError, match="artifact digest"):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            manifest=missing_artifact,
            expected_latest_revision=None,
        )
    with pytest.raises(ValueError, match="OCI workers"):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            manifest=host_subprocess,
            expected_latest_revision=None,
        )


@pytest.mark.asyncio
async def test_revision_activation_rollback_and_deactivate_use_cas(tmp_path: Path) -> None:
    evaluator = _Evaluator()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        bundled=(),
    )
    for revision in (1, 2):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            manifest=_manifest(revision),
            expected_latest_revision=revision - 1 or None,
        )
        await service.test(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=revision,
        )
    first = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
    )
    second = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=2,
        expected_generation=first.generation,
    )
    rollback = await service.rollback(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        expected_generation=second.generation,
    )
    removed = await service.deactivate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        expected_generation=rollback.generation,
    )

    assert (first.revision, second.revision, rollback.revision) == (1, 2, 1)
    assert (first.generation, second.generation, rollback.generation) == (1, 2, 3)
    assert removed == rollback
    assert store.active(namespace="tenant-a", capability_name="example") is None


@pytest.mark.asyncio
async def test_generation_handover_stops_old_interface_before_starting_new(
    tmp_path: Path,
) -> None:
    evaluator = _Evaluator()
    host = _Host()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(host),
        bundled=(),
    )
    for revision in (1, 2):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            manifest=_manifest(revision),
            expected_latest_revision=revision - 1 or None,
        )
        await service.test(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=revision,
        )
    first = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
    )
    await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=2,
        expected_generation=first.generation,
    )

    assert [event[0] for event in host.events] == ["start", "stop", "start"]
    assert host.events[1][1] == host.events[0][1]
    assert host.events[2][1] != host.events[0][1]


@pytest.mark.asyncio
async def test_failed_generation_restarts_previous_interface_and_keeps_pointer(
    tmp_path: Path,
) -> None:
    evaluator = _Evaluator()
    host = _Host(fail_revisions={2})
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(host),
        bundled=(),
    )
    for revision in (1, 2):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            manifest=_manifest(revision),
            expected_latest_revision=revision - 1 or None,
        )
        await service.test(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=revision,
        )
    first = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
    )

    with pytest.raises(WorkerLifecycleError, match="injected"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=2,
            expected_generation=first.generation,
        )

    active = store.active(namespace="tenant-a", capability_name="example")
    assert active == first
    assert [event[0] for event in host.events] == ["start", "stop", "start", "start"]
    assert host.events[-1][1] == host.events[0][1]
    assert await service.healthy() is False  # Service has not entered its started lifecycle.


@pytest.mark.asyncio
async def test_start_restores_persisted_workers_and_shutdown_stops_them(
    tmp_path: Path,
) -> None:
    evaluator = _Evaluator()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    initial_host = _Host()
    initial = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(initial_host),
        secret_resolver=_Secrets(),
        bundled=(),
    )
    initial.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1, worker=True),
        expected_latest_revision=None,
    )
    await initial.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )
    await initial.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
        secret_handles={"CAPABILITY_TOKEN": "token-handle"},
    )

    restored_host = _Host()
    restored_secrets = _Secrets()
    restored = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(restored_host),
        secret_resolver=restored_secrets,
        bundled=(),
    )
    await restored.start()

    assert restored.started is True
    assert await restored.healthy() is True
    assert len(restored_host.launches) == 1
    restored_host.is_healthy = False
    assert await restored.healthy() is False
    await restored.shutdown()
    assert restored.started is False
    assert len(restored_host.stopped) == 1
    assert restored_host.closed is True
    assert restored_secrets.revoked == [
        ("tenant-a", restored_secrets.calls[0]["instance_id"])
    ]


@pytest.mark.asyncio
async def test_start_overrides_stale_persisted_runtime_binding(tmp_path: Path) -> None:
    evaluator = _Evaluator()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    initial = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        secret_resolver=_Secrets(),
        bundled=(),
        config_defaults=lambda _tenant, _manifest: {
            "agent_api_url": "http://127.0.0.1:31001"
        },
    )
    initial.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1, worker=True),
        expected_latest_revision=None,
    )
    await initial.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )
    active = await initial.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
        secret_handles={"CAPABILITY_TOKEN": "token-handle"},
    )
    assert active.config["agent_api_url"] == "http://127.0.0.1:31001"

    restored_host = _Host()
    restored = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(restored_host),
        secret_resolver=_Secrets(),
        bundled=(),
        config_defaults=lambda _tenant, _manifest: {
            "agent_api_url": "http://127.0.0.1:32002"
        },
    )
    await restored.start()

    assert dict(restored_host.launches[0].config)["agent_api_url"] == (
        "http://127.0.0.1:32002"
    )
    assert store.active(
        namespace="tenant-a",
        capability_name="example",
    ).config["agent_api_url"] == "http://127.0.0.1:31001"
    await restored.shutdown()


@pytest.mark.asyncio
async def test_worker_activation_resolves_handles_without_persisting_secret_values(
    tmp_path: Path,
) -> None:
    evaluator = _Evaluator()
    host = _Host()
    secrets = _Secrets()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(host),
        secret_resolver=secrets,
        bundled=(),
    )
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1, worker=True),
        expected_latest_revision=None,
    )
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )
    activation = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
        secret_handles={"CAPABILITY_TOKEN": "secret-handle-1"},
    )

    assert activation.secret_handles == {"CAPABILITY_TOKEN": "secret-handle-1"}
    assert dict(host.launches[0].secret_environment) == {"CAPABILITY_TOKEN": "private-value"}
    assert "private-value" not in repr(activation)
    assert secrets.calls[0]["tenant_id"] == "tenant-a"

    await service.deactivate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        expected_generation=activation.generation,
    )
    assert secrets.revoked == [("tenant-a", secrets.calls[0]["instance_id"])]


@pytest.mark.asyncio
async def test_secret_revision_change_restarts_and_persists_new_worker_generation(
    tmp_path: Path,
) -> None:
    evaluator = _Evaluator()
    host = _Host()
    secrets = _Secrets()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(host),
        secret_resolver=secrets,
        bundled=(),
    )
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1, worker=True),
        expected_latest_revision=None,
    )
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )
    first = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
        secret_handles={"CAPABILITY_TOKEN": "secret-handle-1"},
    )

    secrets.revision = 2
    service.notify_secret_changed(
        tenant_id="tenant-a",
        actor_id="owner-a",
        secret_id="secret-handle-1",
    )
    await service.wait_for_secret_refresh()

    active = store.active(namespace="tenant-a", capability_name="example")
    assert active is not None
    assert active.generation == first.generation + 1
    assert active.secret_bindings["CAPABILITY_TOKEN"].revision == 2
    assert len(host.launches) == 2
    assert host.stopped == [secrets.calls[0]["instance_id"]]


@pytest.mark.asyncio
async def test_host_blocked_capability_cannot_activate(tmp_path: Path) -> None:
    evaluator = _Evaluator()
    service = CapabilityControlService(
        revisions=CapabilityRevisionStore(tmp_path / "capabilities.sqlite3"),
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        blocked_capabilities=("example",),
        bundled=(),
    )
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1),
        expected_latest_revision=None,
    )
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )

    with pytest.raises(CapabilityRuntimeUnavailableError, match="disabled"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=1,
            expected_generation=None,
        )


@pytest.mark.asyncio
async def test_start_fences_orphaned_host_disabled_capability_generation(
    tmp_path: Path,
) -> None:
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    evaluator = _Evaluator()
    initial = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        bundled=(),
    )
    initial.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1),
        expected_latest_revision=None,
    )
    await initial.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )
    await initial.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
    )
    fencing_host = _FencingHost()
    restored = CapabilityControlService(
        revisions=store,
        workers=CapabilityWorkerManager(fencing_host),
        blocked_capabilities=("example",),
        bundled=(),
    )

    await restored.start()

    assert fencing_host.fenced == [("tenant-a", "example")]
    assert store.active(namespace="tenant-a", capability_name="example") is None
    assert store.inactive(namespace="tenant-a", capability_name="example") is not None
    assert await restored.healthy() is True


@pytest.mark.asyncio
async def test_tool_host_follows_exact_capability_generation_lifecycle(tmp_path: Path) -> None:
    evaluator = _Evaluator()
    host = _Host()
    tools = _ToolHost()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(host),
        tool_host=tools,
        bundled=(),
    )
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1),
        expected_latest_revision=None,
    )
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )
    active = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
    )

    assert len(tools.started) == 1
    assert tools.started[0]["tenant_id"] == "tenant-a"
    assert tools.started[0]["manifest"].revision == 1
    await service.deactivate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        expected_generation=active.generation,
    )
    assert tools.stopped == [tools.started[0]["instance_id"]]


@pytest.mark.asyncio
async def test_mcp_capability_fails_closed_without_tool_host(tmp_path: Path) -> None:
    digest = f"sha256:{1:064x}"
    manifest = CapabilityManifest(
        name="weather",
        version="1.0.0",
        artifact_digest=digest,
        workers=(
            WorkerSpec(
                name="weather_mcp",
                kind=WorkerKind.MCP,
                protocol="mcp-v1",
                runtime=WorkerRuntime.OCI,
                transport=WorkerTransport.STREAMABLE_HTTP,
                command=("weather-server",),
                endpoint="http://127.0.0.1:8080/mcp",
                image=f"weather@{digest}",
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    evaluator = _Evaluator()
    service = CapabilityControlService(
        revisions=CapabilityRevisionStore(tmp_path / "capabilities.sqlite3"),
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        bundled=(),
    )
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=manifest,
        expected_latest_revision=None,
    )
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="weather",
        revision=1,
    )

    with pytest.raises(CapabilityRuntimeUnavailableError, match="MCP capability tool host"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="weather",
            revision=1,
            expected_generation=None,
        )


@pytest.mark.asyncio
async def test_failed_worker_start_revokes_issued_generation_credentials(tmp_path: Path) -> None:
    evaluator = _Evaluator()
    secrets = _Secrets()
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host(healthy=False)),
        secret_resolver=secrets,
        bundled=(),
    )
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(1, worker=True),
        expected_latest_revision=None,
    )
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
    )

    with pytest.raises(WorkerLifecycleError, match="health check"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=1,
            expected_generation=None,
            secret_handles={"CAPABILITY_TOKEN": "secret-handle-1"},
        )

    assert secrets.revoked == [("tenant-a", secrets.calls[0]["instance_id"])]


def test_bundled_seeding_is_idempotent_and_does_not_cross_tenants(tmp_path: Path) -> None:
    template = _manifest(1).model_copy(update={"seed": True})
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    service = CapabilityControlService(revisions=store, bundled=(template,))

    first = service.seed_bundled(tenant_id="tenant-a", actor_id="owner-a")
    second = service.seed_bundled(tenant_id="tenant-a", actor_id="owner-a")

    assert first == second
    assert len(store.list(namespace="tenant-a", capability_name="example")) == 1
    assert store.list(namespace="tenant-b", capability_name="example") == []


@pytest.mark.asyncio
async def test_start_reconciles_active_seed_to_exact_current_release_manifest(
    tmp_path: Path,
) -> None:
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    evaluator = _Evaluator()
    original_template = _manifest(1, worker=True).model_copy(update={"seed": True})
    candidate_template = _manifest(2, worker=True).model_copy(
        update={"revision": 1, "seed": True}
    )

    original = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        secret_resolver=_Secrets(),
        bundled=(original_template,),
    )
    original_revision = original.seed_bundled(
        tenant_id="tenant-a",
        actor_id="bootstrap",
    )[0]
    await original.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=original_revision.revision,
    )
    original_active = await original.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=original_revision.revision,
        expected_generation=None,
        secret_handles={"CAPABILITY_TOKEN": "secret-handle-1"},
    )
    await original.shutdown()

    candidate = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        secret_resolver=_Secrets(),
        bundled=(candidate_template,),
    )
    candidate_revision = candidate.seed_bundled(
        tenant_id="tenant-a",
        actor_id="bootstrap",
    )[0]
    await candidate.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=candidate_revision.revision,
    )
    candidate_active = await candidate.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=candidate_revision.revision,
        expected_generation=original_active.generation,
        secret_handles={"CAPABILITY_TOKEN": "secret-handle-1"},
    )
    await candidate.shutdown()

    restored_host = _Host()
    restored = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(restored_host),
        secret_resolver=_Secrets(),
        bundled=(original_template,),
    )
    restored_revision = restored.seed_bundled(
        tenant_id="tenant-a",
        actor_id="bootstrap",
    )[0]
    await restored.start()

    active = store.active(namespace="tenant-a", capability_name="example")
    assert active is not None
    assert active.revision == restored_revision.revision
    assert active.generation == candidate_active.generation + 1
    assert len(restored_host.launches) == 1
    assert restored_host.launches[0].manifest.version == original_template.version
    assert await restored.healthy() is True
    await restored.shutdown()


@pytest.mark.asyncio
async def test_release_state_restores_seed_config_and_secret_binding_across_schema_change(
    tmp_path: Path,
) -> None:
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    release_state_path = tmp_path / "capability_state" / "seed_activations.json"
    old_schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["old"]}},
        "required": ["mode"],
        "additionalProperties": False,
    }
    candidate_schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["old", "new"]}},
        "required": ["mode"],
        "additionalProperties": False,
    }
    old_template = _manifest(1, worker=True).model_copy(
        update={"seed": True, "config_schema": old_schema}
    )
    candidate_template = _manifest(2, worker=True).model_copy(
        update={"revision": 1, "seed": True, "config_schema": candidate_schema}
    )

    old_secrets = _Secrets()
    original = CapabilityControlService(
        revisions=store,
        evaluator=_Evaluator(),
        workers=CapabilityWorkerManager(_Host()),
        secret_resolver=old_secrets,
        bundled=(old_template,),
        release_state_path=release_state_path,
    )
    original_revision = original.seed_bundled(
        tenant_id="tenant-a",
        actor_id="bootstrap",
    )[0]
    await original.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=original_revision.revision,
    )
    await original.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=original_revision.revision,
        expected_generation=None,
        config={"mode": "old"},
        secret_handles={"CAPABILITY_TOKEN": "secret-handle-1"},
    )
    before_candidate = release_state_path.read_bytes()
    await original.shutdown()

    candidate_secrets = _Secrets()
    candidate_secrets.revision = 2
    candidate = CapabilityControlService(
        revisions=store,
        evaluator=_Evaluator(),
        workers=CapabilityWorkerManager(_Host()),
        secret_resolver=candidate_secrets,
        bundled=(candidate_template,),
        release_state_path=release_state_path,
    )
    candidate_revision = candidate.seed_bundled(
        tenant_id="tenant-a",
        actor_id="bootstrap",
    )[0]
    await candidate.start()
    candidate_current = store.active(namespace="tenant-a", capability_name="example")
    assert candidate_current is not None
    await candidate.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=candidate_revision.revision,
    )
    candidate_active = await candidate.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=candidate_revision.revision,
        expected_generation=candidate_current.generation,
        config={"mode": "new"},
        secret_handles={"CAPABILITY_TOKEN": "secret-handle-1"},
    )
    assert candidate_active.config == {"mode": "new"}
    assert candidate_active.secret_bindings["CAPABILITY_TOKEN"].revision == 2
    await candidate.shutdown()

    # Bootstrap restores only this scoped file; the candidate revision stays in product history.
    release_state_path.write_bytes(before_candidate)
    restored_host = _Host()
    restored = CapabilityControlService(
        revisions=store,
        evaluator=_Evaluator(),
        workers=CapabilityWorkerManager(restored_host),
        secret_resolver=_Secrets(),
        bundled=(old_template,),
        release_state_path=release_state_path,
    )
    restored_revision = restored.seed_bundled(
        tenant_id="tenant-a",
        actor_id="bootstrap",
    )[0]
    await restored.start()

    active = store.active(namespace="tenant-a", capability_name="example")
    assert active is not None
    assert active.revision == restored_revision.revision
    assert active.generation == candidate_active.generation + 1
    assert active.config == {"mode": "old"}
    assert active.secret_bindings["CAPABILITY_TOKEN"].revision == 1
    assert candidate_revision in store.list(namespace="tenant-a", capability_name="example")
    assert len(restored_host.launches) == 1
    assert dict(restored_host.launches[0].config) == {"mode": "old"}
    assert await restored.healthy() is True
    await restored.shutdown()


@pytest.mark.asyncio
async def test_start_deactivates_candidate_only_seed_but_retains_revision_history(
    tmp_path: Path,
) -> None:
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    evaluator = _Evaluator()
    candidate_template = _manifest(1).model_copy(update={"seed": True})
    candidate = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        bundled=(candidate_template,),
    )
    revision = candidate.seed_bundled(tenant_id="tenant-a", actor_id="bootstrap")[0]
    await candidate.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=revision.revision,
    )
    await candidate.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=revision.revision,
        expected_generation=None,
    )
    await candidate.shutdown()

    restored_host = _Host()
    restored = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(restored_host),
        bundled=(),
    )
    await restored.start()

    assert store.active(namespace="tenant-a", capability_name="example") is None
    assert store.list(namespace="tenant-a", capability_name="example") == [revision]
    assert restored_host.launches == []
    assert await restored.healthy() is True
    await restored.shutdown()


@pytest.mark.asyncio
async def test_activation_rejects_seed_revision_from_another_release(tmp_path: Path) -> None:
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    evaluator = _Evaluator()
    original_template = _manifest(1).model_copy(update={"seed": True})
    candidate_template = _manifest(2).model_copy(update={"revision": 1, "seed": True})
    service = CapabilityControlService(
        revisions=store,
        evaluator=evaluator,
        workers=CapabilityWorkerManager(_Host()),
        bundled=(original_template,),
    )
    original_revision = service.seed_bundled(
        tenant_id="tenant-a",
        actor_id="bootstrap",
    )[0]
    foreign_revision = store.append(
        namespace="tenant-a",
        manifest=candidate_template.model_copy(update={"revision": 2}),
        expected_latest_revision=original_revision.revision,
    )
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=foreign_revision.revision,
    )

    with pytest.raises(CapabilityRuntimeUnavailableError, match="current release"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            revision=foreign_revision.revision,
            expected_generation=None,
        )
