from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from opentulpa.capabilities import (
    CapabilityManifest,
    EvalCommand,
    NetworkPolicy,
    OciCapabilityPolicy,
    OciCapabilityWorkerHost,
    WorkerKind,
    WorkerLaunch,
    WorkerRuntime,
    WorkerSpec,
)
from opentulpa.capabilities.oci_workers import OciCommandResult
from opentulpa.capabilities.workers import WorkerLifecycleError

DIGEST = f"sha256:{'1' * 64}"
CONTAINER_ID = "2" * 64


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.environment_payload = ""
        self.containers: dict[str, tuple[str, dict[str, str]]] = {}
        self._container_count = 0
        self.remove_failures = 0
        self.run_failure: BaseException | None = None

    @property
    def container_present(self) -> bool:
        return bool(self.containers)

    async def run(
        self,
        argv,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> OciCommandResult:
        del timeout_seconds, max_output_bytes
        values = tuple(argv)
        self.calls.append(values)
        if values[1:3] == ("info", "--format"):
            return OciCommandResult(returncode=0, output=b'["name=rootless"]')
        if values[1:4] == ("image", "inspect", "--format"):
            return OciCommandResult(returncode=0, output=DIGEST.encode())
        if len(values) > 1 and values[1] == "run":
            env_file = Path(values[values.index("--env-file") + 1])
            self.environment_payload = env_file.read_text(encoding="utf-8")
            container_name = values[values.index("--name") + 1]
            labels: dict[str, str] = {}
            for index, value in enumerate(values):
                if value != "--label":
                    continue
                name, _, label_value = values[index + 1].partition("=")
                labels[name] = label_value
            self._container_count += 1
            container_id = (
                CONTAINER_ID
                if self._container_count == 1
                else str(self._container_count + 1) * 64
            )
            self.containers[container_id] = (container_name, labels)
            if self.run_failure is not None:
                failure = self.run_failure
                self.run_failure = None
                raise failure
            return OciCommandResult(returncode=0, output=container_id.encode())
        if len(values) > 1 and values[1] == "ps":
            filters = [
                values[index + 1]
                for index, value in enumerate(values)
                if value == "--filter"
            ]
            matches: list[str] = []
            for container_id, (container_name, labels) in self.containers.items():
                matched = True
                for value in filters:
                    name, _, expected = value.partition("=")
                    if name == "id":
                        matched = matched and container_id.startswith(expected)
                    elif name == "name":
                        matched = matched and expected in container_name
                    elif name == "label":
                        label_name, separator, label_value = expected.partition("=")
                        matched = matched and label_name in labels
                        if separator:
                            matched = matched and labels.get(label_name) == label_value
                if matched:
                    matches.append(container_id)
            return OciCommandResult(
                returncode=0,
                output="\n".join(matches).encode(),
            )
        if values[1:3] == ("rm", "--force"):
            if self.remove_failures:
                self.remove_failures -= 1
                return OciCommandResult(returncode=1, output=b"transient runtime failure")
            identifier = values[3]
            removed = next(
                (
                    container_id
                    for container_id, (container_name, _) in self.containers.items()
                    if identifier in {container_id, container_name}
                ),
                None,
            )
            if removed is None:
                return OciCommandResult(returncode=1, output=b"container not found")
            self.containers.pop(removed)
            return OciCommandResult(returncode=0, output=removed.encode())
        if values[1:3] == ("inspect", "--format"):
            present = values[-1] in self.containers
            return OciCommandResult(
                returncode=0 if present else 1,
                output=b"true" if present else b"",
            )
        return OciCommandResult(returncode=0)


def _launch(*, outbound: bool = False) -> WorkerLaunch:
    network = NetworkPolicy(
        outbound="allowlist" if outbound else "deny",
        allowed_hosts=("api.example.com:443",) if outbound else (),
    )
    worker = WorkerSpec(
        name="example_interface",
        kind=WorkerKind.INTERFACE,
        protocol="agent-interface-v1",
        runtime=WorkerRuntime.OCI,
        command=("python", "-m", "example"),
        image=f"example@{DIGEST}",
        network=network,
    )
    manifest = CapabilityManifest(
        name="example",
        version="1.0.0",
        artifact_digest=DIGEST,
        workers=(worker,),
        network=network,
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    return WorkerLaunch(
        instance_id="cap_tenant_example_g1",
        tenant_id="tenant-a",
        manifest=manifest,
        worker=worker,
        config={"mode": "safe"},
        secret_environment={"CAPABILITY_TOKEN": "private-token"},
    )


@pytest.mark.asyncio
async def test_oci_worker_is_rootless_bounded_and_does_not_put_secrets_in_argv(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path),
        runner=runner,
    )

    handle = await host.start(_launch())

    run = next(call for call in runner.calls if len(call) > 1 and call[1] == "run")
    assert "--read-only" in run
    assert run[run.index("--network") + 1] == "none"
    assert "--cap-drop" in run
    assert "--mount" not in run
    assert "private-token" not in " ".join(run)
    assert "CAPABILITY_TOKEN=private-token" in runner.environment_payload
    assert not list(tmp_path.glob(".env-*.tmp"))
    assert await host.healthy(handle)
    await host.stop(handle)


@pytest.mark.asyncio
async def test_oci_worker_stop_retries_after_transient_remove_failure(tmp_path: Path) -> None:
    runner = _Runner()
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path),
        runner=runner,
    )
    handle = await host.start(_launch())
    runner.remove_failures = 1

    with pytest.raises(WorkerLifecycleError, match="removal failed"):
        await host.stop(handle)

    assert runner.container_present is True
    assert await host.healthy(handle) is True

    await host.stop(handle)

    assert runner.container_present is False
    assert await host.healthy(handle) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run_failure",
    [RuntimeError("lost docker run response"), asyncio.CancelledError()],
)
async def test_oci_worker_ambiguous_launch_is_removed_and_confirmed_absent(
    tmp_path: Path,
    run_failure: BaseException,
) -> None:
    runner = _Runner()
    runner.run_failure = run_failure
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path),
        runner=runner,
    )

    with pytest.raises(type(run_failure)):
        await host.start(_launch(), release_id="release-a", lease_epoch=1)

    run_index = next(index for index, call in enumerate(runner.calls) if call[1] == "run")
    remove_index = next(
        index for index, call in enumerate(runner.calls) if call[1:3] == ("rm", "--force")
    )
    assert run_index < remove_index
    assert runner.calls[-1][1] == "ps"
    assert runner.containers == {}


@pytest.mark.asyncio
async def test_oci_worker_ambiguous_launch_cleanup_failure_is_fail_closed(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    runner.run_failure = RuntimeError("lost docker run response")
    runner.remove_failures = 1
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path),
        runner=runner,
    )

    with pytest.raises(WorkerLifecycleError, match="cleanup could not be confirmed"):
        await host.start(_launch(), release_id="release-a", lease_epoch=1)

    assert runner.container_present is True
    await host.fence(tenant_id="tenant-a", capability_name="example")
    assert runner.container_present is False


@pytest.mark.asyncio
async def test_oci_fence_discovers_unrecorded_worker_after_host_restart(tmp_path: Path) -> None:
    runner = _Runner()
    first = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path / "first"),
        runner=runner,
    )

    await first.start(_launch(), release_id="release-old", lease_epoch=4)

    run = next(call for call in runner.calls if len(call) > 1 and call[1] == "run")
    labels = {
        run[index + 1]
        for index, value in enumerate(run)
        if value == "--label"
    }
    assert "org.opentulpa.capability.managed=true" in labels
    assert all("tenant-a" not in label for label in labels)
    assert all("release-old" not in label for label in labels)

    restarted = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path / "first"),
        runner=runner,
    )
    await restarted.fence(tenant_id="tenant-a", capability_name="example")

    assert runner.container_present is False


@pytest.mark.asyncio
async def test_oci_fence_attempts_every_worker_after_one_removal_failure(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path),
        runner=runner,
    )
    first = await host.start(_launch(), release_id="release-a", lease_epoch=1)
    second = await host.start(
        replace(_launch(), instance_id="cap_tenant_example_g2"),
        release_id="release-a",
        lease_epoch=1,
    )
    runner.remove_failures = 1

    with pytest.raises(WorkerLifecycleError, match="could not be removed"):
        await host.fence(tenant_id="tenant-a", capability_name="example")

    assert len(runner.containers) == 1
    assert first.id.removeprefix("oci:") in runner.containers
    assert second.id.removeprefix("oci:") not in runner.containers
    assert await host.healthy(first) is True
    assert await host.healthy(second) is False

    await host.fence(tenant_id="tenant-a", capability_name="example")
    assert runner.containers == {}


@pytest.mark.asyncio
async def test_oci_hosts_do_not_remove_another_installation_same_instance(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    first = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path / "installation-a"),
        runner=runner,
    )
    second = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path / "installation-b"),
        runner=runner,
    )

    first_handle = await first.start(_launch(), release_id="release-a", lease_epoch=1)
    second_handle = await second.start(_launch(), release_id="release-b", lease_epoch=2)

    run_calls = [call for call in runner.calls if len(call) > 1 and call[1] == "run"]
    names = [call[call.index("--name") + 1] for call in run_calls]
    assert len(set(names)) == 2
    assert first_handle.id.removeprefix("oci:") in runner.containers
    assert second_handle.id.removeprefix("oci:") in runner.containers

    await second.fence(tenant_id="tenant-a", capability_name="example")

    assert first_handle.id.removeprefix("oci:") in runner.containers
    assert second_handle.id.removeprefix("oci:") not in runner.containers

    await first.fence(tenant_id="tenant-a", capability_name="example")
    assert runner.containers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier_kind", ["id", "name"])
async def test_oci_remove_does_not_hide_known_container_behind_scope_label(
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    runner = _Runner()
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path),
        runner=runner,
    )
    handle = await host.start(_launch(), release_id="release-a", lease_epoch=1)
    container_id = handle.id.removeprefix("oci:")
    name, labels = runner.containers[container_id]
    labels["org.opentulpa.capability.installation"] = "mismatched"
    runner.containers[container_id] = (name, labels)

    identifier = container_id if identifier_kind == "id" else name
    await host._remove(identifier)

    assert container_id not in runner.containers


@pytest.mark.asyncio
async def test_oci_worker_fails_closed_without_restricted_egress_network(
    tmp_path: Path,
) -> None:
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(state_root=tmp_path),
        runner=_Runner(),
    )

    with pytest.raises(WorkerLifecycleError, match="egress network"):
        await host.start(_launch(outbound=True))


@pytest.mark.asyncio
async def test_oci_worker_mounts_only_capability_state_with_host_limits(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    state_root = tmp_path / "tenant-state"
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(
            state_root=tmp_path / "runtime",
            persistent_state_root=state_root,
            restricted_egress_network="capability-egress",
            restricted_allowed_hosts=("api.example.com:443",),
            runtime_user="1234:5678",
        ),
        runner=runner,
    )

    await host.start(_launch(outbound=True))

    run = next(call for call in runner.calls if len(call) > 1 and call[1] == "run")
    mount = run[run.index("--mount") + 1]
    assert mount.startswith(f"type=bind,src={state_root.resolve()}/")
    assert mount.endswith(",dst=/state")
    assert run[run.index("--network") + 1] == "capability-egress"
    assert run[run.index("--user") + 1] == "1234:5678"
    assert run[run.index("--pids-limit") + 1] == "128"
    command = " ".join(run)
    assert "/workspace" not in command
    assert "docker.sock" not in command
    assert ".env" not in mount


@pytest.mark.asyncio
async def test_oci_worker_rejects_state_path_outside_private_mount(tmp_path: Path) -> None:
    host = OciCapabilityWorkerHost(
        policy=OciCapabilityPolicy(
            state_root=tmp_path / "runtime",
            persistent_state_root=tmp_path / "tenant-state",
        ),
        runner=_Runner(),
    )
    launch = _launch()
    launch = WorkerLaunch(
        instance_id=launch.instance_id,
        tenant_id=launch.tenant_id,
        manifest=launch.manifest,
        worker=launch.worker,
        config={"state_path": "/workspace/product.db"},
    )

    with pytest.raises(WorkerLifecycleError, match="below /state"):
        await host.start(launch)
