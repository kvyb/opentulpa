import asyncio
import sys

import pytest

from opentulpa.capabilities import (
    CapabilityManifest,
    CapabilityWorkerManager,
    EvalCommand,
    HealthCheck,
    ResourceLimits,
    SecretRequirement,
    SubprocessWorkerHost,
    WorkerHandle,
    WorkerKind,
    WorkerLaunch,
    WorkerLifecycleError,
    WorkerSpec,
)
from opentulpa.capabilities.oci_workers import (
    OciCapabilityPolicy,
    OciCapabilityWorkerHost,
)


class FakeHost:
    def __init__(self, *, unhealthy: set[str] | None = None) -> None:
        self.unhealthy = unhealthy or set()
        self.launches: list[WorkerLaunch] = []
        self.stopped: list[str] = []
        self.closed = False

    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        self.launches.append(launch)
        return WorkerHandle(
            id=f"handle-{launch.worker.name}",
            instance_id=launch.instance_id,
            capability_name=launch.manifest.name,
            capability_revision=launch.manifest.revision,
            manifest_digest=launch.manifest.content_digest,
            worker_name=launch.worker.name,
        )

    async def healthy(self, handle: WorkerHandle) -> bool:
        return handle.worker_name not in self.unhealthy

    async def stop(self, handle: WorkerHandle) -> None:
        self.stopped.append(handle.worker_name)

    async def aclose(self) -> None:
        self.closed = True


class FenceHost(FakeHost):
    def __init__(
        self,
        *,
        fail_start: set[str] | None = None,
        fail_stop: set[str] | None = None,
        fail_fence: bool = False,
    ) -> None:
        super().__init__()
        self.fail_start = fail_start or set()
        self.fail_stop = fail_stop or set()
        self.fail_fence = fail_fence
        self.events: list[tuple[str, str]] = []

    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        self.launches.append(launch)
        self.events.append(("start", launch.worker.name))
        if launch.worker.name in self.fail_start:
            raise RuntimeError("worker start acknowledgement was lost")
        return WorkerHandle(
            id=f"handle-{launch.worker.name}",
            instance_id=launch.instance_id,
            capability_name=launch.manifest.name,
            capability_revision=launch.manifest.revision,
            manifest_digest=launch.manifest.content_digest,
            worker_name=launch.worker.name,
        )

    async def healthy(self, handle: WorkerHandle) -> bool:
        self.events.append(("healthy", handle.worker_name))
        return True

    async def stop(self, handle: WorkerHandle) -> None:
        self.events.append(("stop", handle.worker_name))
        if handle.worker_name in self.fail_stop:
            raise RuntimeError("worker stop acknowledgement was lost")

    async def fence(self, *, tenant_id: str, capability_name: str) -> None:
        self.events.append(("fence", f"{tenant_id}:{capability_name}"))
        if self.fail_fence:
            raise RuntimeError("remote fence acknowledgement was lost")


class FailingOptionalLocalHost(FakeHost):
    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        if launch.worker.name == "optional":
            raise RuntimeError("deterministic local optional failure")
        return await super().start(launch)


def _manifest() -> CapabilityManifest:
    return CapabilityManifest(
        name="telegram",
        version="1.0.0",
        workers=(
            WorkerSpec(
                name="required",
                kind=WorkerKind.INTERFACE,
                protocol="agent-interface-v1",
                command=("worker",),
                secrets=(
                    SecretRequirement(
                        name="TELEGRAM_BOT_TOKEN",
                        scopes=("telegram.receive", "telegram.send"),
                    ),
                ),
            ),
            WorkerSpec(
                name="optional",
                kind=WorkerKind.TRIGGER,
                protocol="agent-trigger-v1",
                command=("optional",),
                required=False,
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )


@pytest.mark.asyncio
async def test_manager_scopes_secrets_and_skips_unhealthy_optional_worker() -> None:
    host = FakeHost(unhealthy={"optional"})
    manager = CapabilityWorkerManager(host)
    active = await manager.start(
        instance_id="telegram-main",
        manifest=_manifest(),
        config={"pairing_required": True},
        secrets={"TELEGRAM_BOT_TOKEN": "secret", "UNDECLARED": "not-forwarded"},
    )

    assert [handle.worker_name for handle in active.handles] == ["required"]
    assert dict(host.launches[0].secret_environment) == {"TELEGRAM_BOT_TOKEN": "secret"}
    assert dict(host.launches[1].secret_environment) == {}
    assert "'secret'" not in repr(host.launches[0])
    assert await manager.healthy("telegram-main") is True

    await manager.stop("telegram-main")
    assert host.stopped == ["optional", "required"]

    await manager.aclose()
    assert host.closed is True


@pytest.mark.asyncio
async def test_manager_rolls_back_started_workers_on_required_failure() -> None:
    host = FakeHost(unhealthy={"required"})
    manager = CapabilityWorkerManager(host)

    with pytest.raises(WorkerLifecycleError, match="health check"):
        await manager.start(
            instance_id="telegram-main",
            manifest=_manifest(),
            secrets={"TELEGRAM_BOT_TOKEN": "secret"},
        )

    assert manager.active("telegram-main") is None
    assert host.stopped == ["required"]


@pytest.mark.asyncio
async def test_remote_optional_start_ack_loss_fails_and_fences_after_known_cleanup() -> None:
    host = FenceHost(fail_start={"optional"})
    manager = CapabilityWorkerManager(host)

    with pytest.raises(WorkerLifecycleError, match="failed to start"):
        await manager.start(
            instance_id="telegram-main",
            tenant_id="tenant-a",
            manifest=_manifest(),
            secrets={"TELEGRAM_BOT_TOKEN": "secret"},
        )

    assert manager.active("telegram-main") is None
    assert host.events == [
        ("start", "required"),
        ("healthy", "required"),
        ("start", "optional"),
        ("stop", "required"),
        ("fence", "tenant-a:telegram"),
    ]


@pytest.mark.asyncio
async def test_remote_required_start_ack_loss_fences_without_a_known_handle() -> None:
    host = FenceHost(fail_start={"required"})
    manager = CapabilityWorkerManager(host)

    with pytest.raises(WorkerLifecycleError, match="failed to start"):
        await manager.start(
            instance_id="telegram-main",
            tenant_id="tenant-a",
            manifest=_manifest(),
            secrets={"TELEGRAM_BOT_TOKEN": "secret"},
        )

    assert host.events == [
        ("start", "required"),
        ("fence", "tenant-a:telegram"),
    ]


@pytest.mark.asyncio
async def test_remote_fence_is_authoritative_after_known_stop_failure() -> None:
    host = FenceHost(fail_start={"optional"}, fail_stop={"required"})
    manager = CapabilityWorkerManager(host)

    with pytest.raises(WorkerLifecycleError, match="failed to start"):
        await manager.start(
            instance_id="telegram-main",
            tenant_id="tenant-a",
            manifest=_manifest(),
            secrets={"TELEGRAM_BOT_TOKEN": "secret"},
        )

    assert host.events[-2:] == [
        ("stop", "required"),
        ("fence", "tenant-a:telegram"),
    ]


@pytest.mark.asyncio
async def test_remote_fence_failure_reports_unconfirmed_cleanup_after_stop_attempt() -> None:
    host = FenceHost(
        fail_start={"optional"},
        fail_stop={"required"},
        fail_fence=True,
    )
    manager = CapabilityWorkerManager(host)

    with pytest.raises(WorkerLifecycleError, match="cleanup could not be confirmed"):
        await manager.start(
            instance_id="telegram-main",
            tenant_id="tenant-a",
            manifest=_manifest(),
            secrets={"TELEGRAM_BOT_TOKEN": "secret"},
        )

    assert host.events[-2:] == [
        ("stop", "required"),
        ("fence", "tenant-a:telegram"),
    ]


@pytest.mark.asyncio
async def test_local_optional_start_failure_remains_deterministically_optional() -> None:
    manager = CapabilityWorkerManager(FailingOptionalLocalHost())

    active = await manager.start(
        instance_id="telegram-main",
        manifest=_manifest(),
        secrets={"TELEGRAM_BOT_TOKEN": "secret"},
    )

    assert [handle.worker_name for handle in active.handles] == ["required"]


@pytest.mark.asyncio
async def test_manager_fails_before_launch_when_secret_grant_is_missing() -> None:
    host = FakeHost()
    manager = CapabilityWorkerManager(host)

    with pytest.raises(WorkerLifecycleError, match="TELEGRAM_BOT_TOKEN"):
        await manager.start(instance_id="telegram-main", manifest=_manifest())

    assert host.launches == []


@pytest.mark.asyncio
async def test_worker_hosts_reject_runtime_environment_secret_overrides(tmp_path) -> None:
    host = FakeHost()
    manager = CapabilityWorkerManager(host)
    with pytest.raises(WorkerLifecycleError, match="runtime environment"):
        await manager.start(
            instance_id="telegram-main",
            manifest=_manifest(),
            secrets={"PATH": "/untrusted/bin"},
        )
    assert host.launches == []

    launch = WorkerLaunch(
        instance_id="telegram-main",
        manifest=_manifest(),
        worker=_manifest().workers[0],
        secret_environment={"PYTHONPATH": "/untrusted/code"},
    )
    with pytest.raises(WorkerLifecycleError, match="runtime environment"):
        await SubprocessWorkerHost(base_environment={}).start(launch)

    oci = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path / "oci"),
    )
    try:
        with pytest.raises(WorkerLifecycleError, match="runtime environment"):
            await oci.start(launch)
    finally:
        await oci.aclose()


@pytest.mark.asyncio
async def test_manager_leaves_stdio_mcp_process_lifecycle_to_tool_runtime() -> None:
    manifest = CapabilityManifest(
        name="weather",
        version="1.0.0",
        workers=(
            WorkerSpec(
                name="weather_mcp",
                kind=WorkerKind.MCP,
                protocol="mcp-v1",
                command=("weather-server",),
                secrets=(
                    SecretRequirement(
                        name="WEATHER_TOKEN",
                        scopes=("weather.read",),
                    ),
                ),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    host = FakeHost()
    manager = CapabilityWorkerManager(host)

    active = await manager.start(
        instance_id="weather-main",
        manifest=manifest,
        config={"units": "metric"},
        secrets={"WEATHER_TOKEN": "secret"},
    )

    assert active.handles == ()
    assert host.launches == []
    assert await manager.healthy("weather-main") is True


@pytest.mark.asyncio
async def test_subprocess_host_starts_without_a_shell_and_stops_cleanly() -> None:
    manifest = CapabilityManifest(
        name="timer",
        version="1.0.0",
        workers=(
            WorkerSpec(
                name="timer_trigger",
                kind=WorkerKind.TRIGGER,
                protocol="agent-trigger-v1",
                command=(sys.executable, "-c", "import time; time.sleep(60)"),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    manager = CapabilityWorkerManager(SubprocessWorkerHost(base_environment={}))

    active = await manager.start(instance_id="timer-main", manifest=manifest)
    assert len(active.handles) == 1
    assert await manager.healthy("timer-main") is True

    await manager.stop("timer-main")
    assert manager.active("timer-main") is None


@pytest.mark.asyncio
async def test_ready_file_worker_must_signal_readiness_and_liveness_tracks_exit() -> None:
    worker = WorkerSpec(
        name="ready_interface",
        kind=WorkerKind.INTERFACE,
        protocol="agent-interface-v1",
        command=(
            sys.executable,
            "-c",
            (
                "import os,time; from pathlib import Path; "
                "Path(os.environ['OPENTULPA_WORKER_READY_FILE']).write_text("
                "str(os.getpid()), encoding='ascii'); time.sleep(0.2)"
            ),
        ),
        healthcheck=HealthCheck(kind="ready_file"),
        resources=ResourceLimits(startup_timeout_seconds=1),
    )
    manifest = CapabilityManifest(
        name="ready",
        version="1.0.0",
        workers=(worker,),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    manager = CapabilityWorkerManager(SubprocessWorkerHost(base_environment={}))

    await manager.start(instance_id="ready-main", manifest=manifest)
    assert await manager.healthy("ready-main") is True
    await asyncio.sleep(0.3)
    assert await manager.healthy("ready-main") is False
    await manager.stop("ready-main")


@pytest.mark.asyncio
async def test_ready_file_worker_fails_activation_without_protocol_signal() -> None:
    manifest = CapabilityManifest(
        name="not_ready",
        version="1.0.0",
        workers=(
            WorkerSpec(
                name="not_ready_interface",
                kind=WorkerKind.INTERFACE,
                protocol="agent-interface-v1",
                command=(sys.executable, "-c", "import time; time.sleep(60)"),
                healthcheck=HealthCheck(kind="ready_file"),
                resources=ResourceLimits(startup_timeout_seconds=0.1),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    manager = CapabilityWorkerManager(SubprocessWorkerHost(base_environment={}))

    with pytest.raises(WorkerLifecycleError, match="readiness timed out"):
        await manager.start(instance_id="not-ready-main", manifest=manifest)
    assert manager.active("not-ready-main") is None
