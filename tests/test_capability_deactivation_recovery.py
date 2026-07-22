from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from opentulpa.capabilities import (
    CapabilityControlService,
    CapabilityManifest,
    CapabilityRevisionConflictError,
    CapabilityRevisionStore,
    CapabilitySecretBinding,
    CapabilityTestCheck,
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
)


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
                name="isolated",
                status=CapabilityTestStatus.PASSED,
            ),
        )


class _WorkerHost:
    def __init__(self) -> None:
        self.active: dict[str, tuple[str, str]] = {}
        self.fail_stops = 0
        self.start_count = 0
        self.stop_count = 0
        self.max_active = 0

    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        if launch.instance_id in self.active:
            raise AssertionError("duplicate worker generation exposure")
        self.start_count += 1
        self.active[launch.instance_id] = (
            launch.tenant_id,
            launch.manifest.name,
        )
        self.max_active = max(self.max_active, len(self.active))
        return WorkerHandle(
            id=f"worker-{self.start_count}",
            instance_id=launch.instance_id,
            capability_name=launch.manifest.name,
            capability_revision=launch.manifest.revision,
            manifest_digest=launch.manifest.content_digest,
            worker_name=launch.worker.name,
        )

    async def healthy(self, handle: WorkerHandle) -> bool:
        return handle.instance_id in self.active

    async def stop(self, handle: WorkerHandle) -> None:
        self.stop_count += 1
        if self.fail_stops:
            self.fail_stops -= 1
            raise RuntimeError("injected worker stop failure")
        self.active.pop(handle.instance_id, None)

    async def fence(self, *, tenant_id: str, capability_name: str) -> None:
        self.active = {
            instance_id: owner
            for instance_id, owner in self.active.items()
            if owner != (tenant_id, capability_name)
        }


class _ToolHost:
    def __init__(self) -> None:
        self.active: set[str] = set()
        self.fail_stops = 0
        self.start_count = 0
        self.stop_count = 0
        self.max_active = 0

    async def start(self, **kwargs: Any) -> None:
        instance_id = str(kwargs["instance_id"])
        if instance_id in self.active:
            raise AssertionError("duplicate tool generation exposure")
        self.start_count += 1
        self.active.add(instance_id)
        self.max_active = max(self.max_active, len(self.active))

    async def stop(self, instance_id: str) -> None:
        self.stop_count += 1
        if self.fail_stops:
            self.fail_stops -= 1
            raise RuntimeError("injected tool stop failure")
        self.active.discard(instance_id)


class _Secrets:
    def __init__(self) -> None:
        self.active: set[str] = set()
        self.fail_revokes = 0
        self.resolve_count = 0
        self.revoke_count = 0

    async def bind(self, **kwargs: Any) -> Mapping[str, CapabilitySecretBinding]:
        return {
            name: CapabilitySecretBinding(
                handle_id=handle_id,
                revision=1,
                scopes=("capability.invoke",),
            )
            for name, handle_id in kwargs["secret_handles"].items()
        }

    async def resolve(self, **kwargs: Any) -> Mapping[str, str]:
        self.resolve_count += 1
        self.active.add(str(kwargs["instance_id"]))
        return {"CAPABILITY_TOKEN": "secret"}

    async def revoke(self, *, tenant_id: str, instance_id: str) -> None:
        del tenant_id
        self.revoke_count += 1
        if self.fail_revokes:
            self.fail_revokes -= 1
            raise RuntimeError("injected credential revoke failure")
        self.active.discard(instance_id)


def _manifest() -> CapabilityManifest:
    requirement = SecretRequirement(
        name="CAPABILITY_TOKEN",
        scopes=("capability.invoke",),
    )
    return CapabilityManifest(
        name="example",
        version="1.0.0",
        revision=1,
        artifact_digest=f"sha256:{1:064x}",
        workers=(
            WorkerSpec(
                name="example_interface",
                kind=WorkerKind.INTERFACE,
                protocol="agent-interface-v1",
                runtime=WorkerRuntime.OCI,
                command=("example-worker",),
                image=f"example@sha256:{1:064x}",
                secrets=(requirement,),
            ),
        ),
        secrets=(requirement,),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )


async def _activated_service(
    tmp_path: Path,
    *,
    store: CapabilityRevisionStore | None = None,
    worker_host: _WorkerHost | None = None,
    tool_host: _ToolHost | None = None,
    secrets: _Secrets | None = None,
) -> tuple[
    CapabilityControlService,
    CapabilityRevisionStore,
    _WorkerHost,
    _ToolHost,
    _Secrets,
    int,
]:
    revisions = store or CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    workers = worker_host or _WorkerHost()
    tools = tool_host or _ToolHost()
    resolver = secrets or _Secrets()
    service = CapabilityControlService(
        revisions=revisions,
        evaluator=_Evaluator(),
        workers=CapabilityWorkerManager(workers),
        tool_host=tools,
        secret_resolver=resolver,
        bundled=(),
    )
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=_manifest(),
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
        secret_handles={"CAPABILITY_TOKEN": "secret-handle"},
    )
    return service, revisions, workers, tools, resolver, active.generation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_boundary",
    ("begin_store", "tools", "workers", "credentials", "release_store", "final_store"),
)
async def test_failed_deactivation_restores_exact_generation_without_duplicate_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
) -> None:
    service, store, workers, tools, secrets, generation = await _activated_service(tmp_path)
    instance_id = next(iter(workers.active))

    if failure_boundary == "tools":
        tools.fail_stops = 1
    elif failure_boundary == "workers":
        workers.fail_stops = 1
    elif failure_boundary == "credentials":
        secrets.fail_revokes = 1
    elif failure_boundary == "begin_store":
        original_begin = store.begin_deactivation
        failed = False

        def fail_begin_once(**kwargs: Any):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("injected begin store failure")
            return original_begin(**kwargs)

        monkeypatch.setattr(store, "begin_deactivation", fail_begin_once)
    elif failure_boundary == "final_store":
        original_deactivate = store.deactivate
        failed = False

        def fail_deactivate_once(**kwargs: Any):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("injected final store failure")
            return original_deactivate(**kwargs)

        monkeypatch.setattr(store, "deactivate", fail_deactivate_once)
    else:
        original_persist = service._persist_release_seed_activations
        failed = False

        def fail_persist_once(*, restoring=()):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("injected release store failure")
            return original_persist(restoring=restoring)

        monkeypatch.setattr(service, "_persist_release_seed_activations", fail_persist_once)

    with pytest.raises((RuntimeError, WorkerLifecycleError)):
        await service.deactivate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            expected_generation=generation,
        )

    active = store.active(namespace="tenant-a", capability_name="example")
    assert active is not None
    assert active.generation == generation
    assert store.deactivating(namespace="tenant-a", capability_name="example") is None
    assert store.inactive(namespace="tenant-a", capability_name="example") is None
    assert workers.active == {instance_id: ("tenant-a", "example")}
    assert tools.active == {instance_id}
    assert secrets.active == {instance_id}
    assert workers.max_active == 1
    assert tools.max_active == 1


@pytest.mark.asyncio
async def test_deactivation_retry_is_idempotent_and_next_activation_advances_generation(
    tmp_path: Path,
) -> None:
    service, store, workers, tools, secrets, generation = await _activated_service(tmp_path)

    first = await service.deactivate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        expected_generation=generation,
    )
    counts = (workers.stop_count, tools.stop_count, secrets.revoke_count)
    second = await service.deactivate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        expected_generation=generation,
    )

    assert second == first
    assert (workers.stop_count, tools.stop_count, secrets.revoke_count) == counts
    assert store.active(namespace="tenant-a", capability_name="example") is None
    assert store.inactive(namespace="tenant-a", capability_name="example") == first

    next_active = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
        secret_handles={"CAPABILITY_TOKEN": "secret-handle"},
    )
    assert next_active.generation == generation + 1
    with pytest.raises(CapabilityRevisionConflictError):
        await service.deactivate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            capability_name="example",
            expected_generation=generation,
        )


@pytest.mark.asyncio
async def test_restart_reconciliation_finishes_hidden_generation_after_transient_failure(
    tmp_path: Path,
) -> None:
    service, store, workers, _tools, secrets, generation = await _activated_service(tmp_path)
    instance_id = next(iter(workers.active))
    store.begin_deactivation(
        namespace="tenant-a",
        capability_name="example",
        expected_generation=generation,
    )
    secrets.fail_revokes = 1

    failed_restart = CapabilityControlService(
        revisions=store,
        workers=CapabilityWorkerManager(workers),
        tool_host=_ToolHost(),
        secret_resolver=secrets,
        bundled=(),
    )
    await failed_restart.start()

    assert store.active(namespace="tenant-a", capability_name="example") is None
    pending = store.deactivating(namespace="tenant-a", capability_name="example")
    assert pending is not None and pending.generation == generation
    assert await failed_restart.healthy() is False
    assert instance_id not in workers.active

    recovered = CapabilityControlService(
        revisions=store,
        workers=CapabilityWorkerManager(workers),
        tool_host=_ToolHost(),
        secret_resolver=secrets,
        bundled=(),
    )
    await recovered.start()

    assert await recovered.healthy() is True
    assert store.deactivating(namespace="tenant-a", capability_name="example") is None
    tombstone = store.inactive(namespace="tenant-a", capability_name="example")
    assert tombstone is not None and tombstone.generation == generation
    assert workers.active == {}
    assert secrets.active == set()
    del service


def test_revision_store_transition_is_hidden_cas_and_reversible(tmp_path: Path) -> None:
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    store.append(
        namespace="tenant-a",
        manifest=_manifest(),
        expected_latest_revision=None,
    )
    active = store.activate(
        namespace="tenant-a",
        capability_name="example",
        revision=1,
        expected_generation=None,
    )

    transition = store.begin_deactivation(
        namespace="tenant-a",
        capability_name="example",
        expected_generation=active.generation,
    )

    assert transition == active
    assert store.active(namespace="tenant-a", capability_name="example") is None
    assert store.list_all_active() == []
    assert store.list_all_deactivating() == [active]
    assert (
        store.begin_deactivation(
            namespace="tenant-a",
            capability_name="example",
            expected_generation=active.generation,
        )
        == active
    )
    with pytest.raises(CapabilityRevisionConflictError, match="deactivation"):
        store.activate(
            namespace="tenant-a",
            capability_name="example",
            revision=1,
            expected_generation=None,
        )

    assert (
        store.cancel_deactivation(
            namespace="tenant-a",
            capability_name="example",
            expected_generation=active.generation,
        )
        == active
    )
    assert store.active(namespace="tenant-a", capability_name="example") == active
