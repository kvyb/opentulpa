from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from opentulpa.bootstrap.host import ReleaseHostError
from opentulpa.bootstrap.models import ReleaseLaunchContext, ReleaseRecord
from opentulpa.bootstrap.oci_host import (
    LocalOciCommandRunner,
    OciCommandResult,
    OciMount,
    OciReleasePolicy,
    RootlessOciReleaseHost,
)


def _release() -> ReleaseRecord:
    return ReleaseRecord(
        id="release_green",
        candidate_id="candidate_green",
        source_commit="a" * 40,
        artifact_digest=f"sha256:{'b' * 64}",
        manifest_digest=f"sha256:{'c' * 64}",
        entrypoint=("python", "-m", "opentulpa"),
    )


class FakeOciRunner:
    def __init__(
        self,
        release: ReleaseRecord,
        *,
        rootless: bool = True,
        valid_image_labels: bool = True,
        trusted_source_layout: bool = True,
        source_layout: str = "full-source-v1",
        image_volumes: bool = False,
        log_output: bytes = b"",
    ) -> None:
        self.release = release
        self.rootless = rootless
        self.valid_image_labels = valid_image_labels
        self.trusted_source_layout = trusted_source_layout
        self.source_layout = source_layout
        self.image_volumes = image_volumes
        self.log_output = log_output
        self.commands: list[tuple[str, ...]] = []
        self.networks: set[str] = set()
        self.containers: dict[str, dict[str, object]] = {}
        self.fail_container_removal = False
        self.fail_network_removal = False
        self._counter = 0

    async def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> OciCommandResult:
        del timeout_seconds, max_output_bytes
        command = tuple(argv)
        self.commands.append(command)
        if command[1] == "info":
            if Path(command[0]).name == "podman":
                output = b"true" if self.rootless else b"false"
            else:
                output = b'["name=rootless"]' if self.rootless else b"[]"
            return OciCommandResult(returncode=0, output=output)
        if command[1:3] == ("image", "inspect"):
            image_labels = {
                "org.opentulpa.release.manifest-digest": self.release.manifest_digest,
                "org.opentulpa.release.source-commit": self.release.source_commit,
                "org.opentulpa.release.protocol-version": "1",
            }
            if self.trusted_source_layout:
                image_labels["org.opentulpa.release.source-layout"] = self.source_layout
            if not self.valid_image_labels:
                image_labels["org.opentulpa.release.source-commit"] = "0" * 40
            image_config: dict[str, object] = {
                "Labels": image_labels,
                "Volumes": {"/data": {}} if self.image_volumes else None,
            }
            payload = {
                "Id": self.release.artifact_digest,
                "Config": image_config,
            }
            return OciCommandResult(returncode=0, output=json.dumps(payload).encode())
        if command[1:4] == ("network", "create", "--internal"):
            self.networks.add(command[4])
            return OciCommandResult(returncode=0, output=b"d" * 64)
        if command[1:3] == ("network", "inspect"):
            if command[3] == "release-egress":
                return OciCommandResult(returncode=0, output=b"false")
            exists = command[3] in self.networks
            return OciCommandResult(returncode=0 if exists else 1, output=b"true" if exists else b"")
        if command[1:3] == ("network", "ls"):
            return OciCommandResult(returncode=0, output="\n".join(sorted(self.networks)).encode())
        if command[1:3] == ("network", "rm"):
            if self.fail_network_removal:
                return OciCommandResult(returncode=1)
            existed = command[3] in self.networks
            self.networks.discard(command[3])
            return OciCommandResult(returncode=0 if existed else 1)
        if command[1] == "run":
            self._counter += 1
            container_id = f"{self._counter:064x}"
            container_labels: dict[str, str] = {}
            for index, item in enumerate(command):
                if item == "--label":
                    key, value = command[index + 1].split("=", 1)
                    container_labels[key] = value
            environment: list[str] = []
            if "--env-file" in command:
                environment.extend(
                    Path(command[command.index("--env-file") + 1])
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            self.containers[container_id] = {
                "Image": self.release.artifact_digest,
                "Config": {"Labels": container_labels, "Env": environment},
                "network": command[command.index("--network") + 1],
                "port": 49_152 + self._counter,
            }
            return OciCommandResult(returncode=0, output=container_id.encode())
        if command[1] == "port":
            container = self.containers.get(command[2])
            if container is None:
                return OciCommandResult(returncode=1)
            return OciCommandResult(
                returncode=0,
                output=f"127.0.0.1:{container['port']}".encode(),
            )
        if command[1] == "ps":
            filters = [command[index + 1] for index, item in enumerate(command) if item == "--filter"]
            identifiers = []
            for identifier, container in self.containers.items():
                config = container["Config"]
                assert isinstance(config, dict)
                labels = config["Labels"]
                assert isinstance(labels, dict)
                matches = True
                for value in filters:
                    if value.startswith("label="):
                        key, expected = value.removeprefix("label=").split("=", 1)
                        matches = matches and labels.get(key) == expected
                    elif value.startswith("id="):
                        matches = matches and identifier.startswith(value.removeprefix("id="))
                    else:
                        matches = False
                if matches:
                    identifiers.append(identifier)
            return OciCommandResult(returncode=0, output="\n".join(identifiers).encode())
        if command[1] == "inspect":
            container = self.containers.get(command[2])
            if container is None:
                return OciCommandResult(returncode=1)
            return OciCommandResult(returncode=0, output=json.dumps(container).encode())
        if command[1] == "stop":
            return OciCommandResult(returncode=0)
        if command[1] == "logs":
            return OciCommandResult(returncode=0, output=self.log_output)
        if command[1] == "rm":
            identifier = command[-1]
            if self.fail_container_removal:
                return OciCommandResult(returncode=1)
            existed = identifier in self.containers
            self.containers.pop(identifier, None)
            return OciCommandResult(returncode=0 if existed else 1)
        raise AssertionError(f"unexpected OCI command: {command}")


def _control_app(release: ReleaseRecord) -> FastAPI:
    app = FastAPI()

    @app.get(release.health_path)
    async def health(request: Request) -> dict[str, object]:
        assert request.headers["authorization"].startswith("Bearer ")
        assert request.headers["x-opentulpa-release-id"] == release.id
        return {
            "healthy": True,
            "release_id": release.id,
            "protocol_version": 1,
            "components": {"runtime": True, "agent_api": True},
        }

    @app.post(release.drain_path)
    async def drain(request: Request) -> dict[str, object]:
        assert request.headers["authorization"].startswith("Bearer ")
        return {"drained": True, "in_flight": 0}

    return app


@pytest.mark.asyncio
async def test_rootless_oci_host_enforces_immutable_isolated_release_contract(
    tmp_path: Path,
) -> None:
    release = _release()
    workspace_root = tmp_path / "tenants"
    workspace = workspace_root / "tenant_1"
    workspace.mkdir(parents=True)
    runner = FakeOciRunner(release)
    policy = OciReleasePolicy(
        state_root=tmp_path / "state",
        production_network_name="release-egress",
        require_persistent_data_mount=True,
        production_environment=(("OPENAI_COMPATIBLE_API_KEY", "not-on-command-line"),),
        mounts=(
            OciMount(
                source=workspace,
                target="/workspace",
                read_only=False,
                production_only=True,
            ),
        ),
        allowed_mount_roots=(workspace_root,),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_control_app(release)),
        base_url="http://127.0.0.1:49153",
    ) as client:
        host = RootlessOciReleaseHost(policy=policy, runner=runner, http_client=client)
        prepared = await host.prepare(release)
        running = await host.start(
            prepared,
            ReleaseLaunchContext(
                mode="production",
                lease_epoch=7,
                secrets_enabled=True,
                ingress_enabled=False,
            ),
        )

        assert running.endpoint == "http://127.0.0.1:49153"
        assert running.control_token is not None
        assert "control_token" not in running.model_dump()
        assert running.control_token not in repr(running)
        assert (await host.probe(running)).healthy is True
        assert (await host.drain(running, timeout_seconds=1)).drained is True
        run = next(command for command in runner.commands if command[1] == "run")
        for required in (
            "--read-only",
            "--workdir",
            "/app",
            "--init",
            "--cap-drop",
            "--security-opt",
            "no-new-privileges:true",
            "--cpus",
            "--memory",
            "--pids-limit",
            "--user",
            "65532:65532",
        ):
            assert required in run
        assert run[run.index("--workdir") + 1] == "/app"
        network_name = run[run.index("--network") + 1]
        assert network_name == "release-egress"
        assert not any(command[1:4] == ("network", "create", "--internal") for command in runner.commands)
        assert "--add-host" in run
        assert "--env-file" in run
        assert f"127.0.0.1::{release.control_port}/tcp" in run
        assert release.artifact_digest in run
        assert f"type=bind,src={workspace.resolve()},dst=/workspace" in run
        assert all("not-on-command-line" not in argument for argument in run)
        assert all(running.control_token not in argument for argument in run)
        container = runner.containers[running.instance_id]
        config = container["Config"]
        assert isinstance(config, dict)
        assert "OPENTULPA_DATA_ROOT=/workspace" in config["Env"]
        assert "OPENAI_COMPATIBLE_API_KEY=not-on-command-line" in config["Env"]
        assert "PYTHONNOUSERSITE=1" in config["Env"]
        assert "PYTHONPATH=/app/src" in config["Env"]
        assert "PYTHONSAFEPATH=1" in config["Env"]

        await host.stop(running)
        assert network_name == "release-egress"


@pytest.mark.asyncio
async def test_staging_release_has_no_egress_secrets_or_persistent_mount(
    tmp_path: Path,
) -> None:
    release = _release()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = FakeOciRunner(release)
    policy = OciReleasePolicy(
        state_root=tmp_path / "state",
        production_network_name="release-egress",
        runtime_user="1234:5678",
        production_environment=(
            ("OPENAI_COMPATIBLE_API_KEY", "production-secret"),
            ("TELEGRAM_BOT_TOKEN", "telegram-secret"),
        ),
        mounts=(OciMount(source=workspace, target="/workspace", read_only=False),),
        allowed_mount_roots=(workspace,),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_control_app(release)),
        base_url="http://127.0.0.1:49153",
    ) as client:
        host = RootlessOciReleaseHost(policy=policy, runner=runner, http_client=client)
        running = await host.start(
            await host.prepare(release),
            ReleaseLaunchContext(mode="staging"),
        )

        run = next(command for command in runner.commands if command[1] == "run")
        network_name = run[run.index("--network") + 1]
        assert network_name.startswith(f"{policy.network_name}-")
        assert any(
            command[1:4] == ("network", "create", "--internal")
            and command[4] == network_name
            for command in runner.commands
        )
        assert "--add-host" not in run
        assert not any(argument.startswith("type=bind,") for argument in run)
        assert (
            "/workspace:rw,nosuid,nodev,size=512m,mode=700,uid=1234,gid=5678"
            in run
        )
        config = runner.containers[running.instance_id]["Config"]
        assert isinstance(config, dict)
        environment = config["Env"]
        assert isinstance(environment, list)
        assert "OPENTULPA_DISABLE_CONSUMERS=true" in environment
        assert "EVOLUTION_ENABLED=false" in environment
        assert "OPENAI_COMPATIBLE_API_KEY=staging-disabled" in environment
        assert "OPENAI_COMPATIBLE_API_KEY=production-secret" not in environment
        assert "TELEGRAM_BOT_TOKEN=telegram-secret" not in environment

        await host.stop(running)
        assert network_name not in runner.networks


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_resource", ["container", "network"])
async def test_containment_fails_closed_while_oci_resources_survive(
    tmp_path: Path,
    failed_resource: str,
) -> None:
    release = _release()
    runner = FakeOciRunner(release)
    policy = OciReleasePolicy(
        state_root=tmp_path / "state",
        production_network_name="release-egress",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_control_app(release)),
        base_url="http://127.0.0.1:49153",
    ) as client:
        host = RootlessOciReleaseHost(policy=policy, runner=runner, http_client=client)
        running = await host.start(
            await host.prepare(release),
            ReleaseLaunchContext(mode="staging"),
        )
        network_name = next(iter(runner.networks))
        if failed_resource == "container":
            runner.fail_container_removal = True
        else:
            runner.fail_network_removal = True

        assert await host.contain(running, attempts=2) is False
        if failed_resource == "container":
            assert running.instance_id in runner.containers
        else:
            assert network_name in runner.networks

        runner.fail_container_removal = False
        runner.fail_network_removal = False
        assert await host.contain(running, attempts=1) is True
        assert running.instance_id not in runner.containers
        assert network_name not in runner.networks


@pytest.mark.asyncio
async def test_oci_log_collection_is_bounded_and_redacted(tmp_path: Path) -> None:
    release = _release()
    runner = FakeOciRunner(release)
    policy = OciReleasePolicy(
        state_root=tmp_path / "state",
        production_network_name="release-egress",
        production_environment=(("OPENAI_COMPATIBLE_API_KEY", "production-secret"),),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_control_app(release)),
        base_url="http://127.0.0.1:49153",
    ) as client:
        host = RootlessOciReleaseHost(policy=policy, runner=runner, http_client=client)
        running = await host.start(
            await host.prepare(release),
            ReleaseLaunchContext(
                mode="production",
                lease_epoch=1,
                secrets_enabled=True,
                ingress_enabled=False,
            ),
        )
        runner.log_output = (
            b"production-secret Authorization: Bearer "
            + (running.control_token or "").encode()
            + b" Authorization: Basic dXNlcjpwYXNz Cookie: session=dynamic-secret "
            + b" password=hunter2 "
            + b"x" * 100
        )

        logs = await host.collect_logs(running, max_bytes=64)

        assert len(logs.encode()) <= 64
        assert "production-secret" not in logs
        assert running.control_token not in logs
        assert "hunter2" not in logs
        assert "dXNlcjpwYXNz" not in logs
        assert "dynamic-secret" not in logs
        assert "[redacted]" in logs
        await host.stop(running)


@pytest.mark.asyncio
async def test_oci_host_restart_discovers_only_matching_persisted_release(tmp_path: Path) -> None:
    release = _release()
    runner = FakeOciRunner(release)
    policy = OciReleasePolicy(
        state_root=tmp_path / "state",
        production_network_name="release-egress",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_control_app(release)),
        base_url="http://127.0.0.1:49153",
    ) as client:
        original = RootlessOciReleaseHost(policy=policy, runner=runner, http_client=client)
        running = await original.start(
            await original.prepare(release),
            ReleaseLaunchContext(
                mode="production",
                lease_epoch=3,
                secrets_enabled=True,
                ingress_enabled=False,
            ),
        )
        restarted = RootlessOciReleaseHost(
            policy=policy,
            runner=runner,
            http_client=client,
            release_loader=lambda release_id: release if release_id == release.id else None,
        )

        discovered = await restarted.discover(release.id)

        assert discovered == running
        await restarted.stop(discovered)


@pytest.mark.asyncio
async def test_oci_host_rejects_rootful_engine_and_mismatched_image(tmp_path: Path) -> None:
    release = _release()
    rootful = RootlessOciReleaseHost(
        policy=OciReleasePolicy(state_root=tmp_path / "rootful"),
        runner=FakeOciRunner(release, rootless=False),
    )
    with pytest.raises(ReleaseHostError, match="rootless"):
        await rootful.prepare(release)
    await rootful.aclose()

    mismatched = RootlessOciReleaseHost(
        policy=OciReleasePolicy(state_root=tmp_path / "mismatch"),
        runner=FakeOciRunner(release, valid_image_labels=False),
    )
    with pytest.raises(ReleaseHostError, match="labels"):
        await mismatched.prepare(release)
    await mismatched.aclose()

    untrusted_layout = RootlessOciReleaseHost(
        policy=OciReleasePolicy(state_root=tmp_path / "untrusted-layout"),
        runner=FakeOciRunner(release, trusted_source_layout=False),
    )
    with pytest.raises(ReleaseHostError, match="labels"):
        await untrusted_layout.prepare(release)
    await untrusted_layout.aclose()

    implicit_volume = RootlessOciReleaseHost(
        policy=OciReleasePolicy(state_root=tmp_path / "volume"),
        runner=FakeOciRunner(release, image_volumes=True),
    )
    with pytest.raises(ReleaseHostError, match="writable volumes"):
        await implicit_volume.prepare(release)
    await implicit_volume.aclose()


@pytest.mark.asyncio
async def test_oci_host_accepts_immediately_previous_trusted_layout_for_rollback(
    tmp_path: Path,
) -> None:
    release = _release()
    host = RootlessOciReleaseHost(
        policy=OciReleasePolicy(state_root=tmp_path / "previous-layout"),
        runner=FakeOciRunner(
            release,
            source_layout="capability-workers-manifests-web-assets-v1",
        ),
    )

    prepared = await host.prepare(release)

    assert prepared.release_id == release.id
    await host.aclose()


@pytest.mark.asyncio
async def test_oci_host_rejects_mounts_outside_explicit_allowlist(tmp_path: Path) -> None:
    release = _release()
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    host = RootlessOciReleaseHost(
        policy=OciReleasePolicy(
            state_root=tmp_path / "state",
            mounts=(
                OciMount(
                    source=outside,
                    target="/workspace",
                    read_only=False,
                    production_only=True,
                ),
            ),
            allowed_mount_roots=(allowed,),
        ),
        runner=FakeOciRunner(release),
    )

    with pytest.raises(ReleaseHostError, match="allowlist"):
        await host.prepare(release)
    await host.aclose()


@pytest.mark.asyncio
async def test_capability_state_snapshot_restores_only_release_coupled_state(
    tmp_path: Path,
) -> None:
    release = _release()
    workspace = tmp_path / "workspace"
    capability_state = workspace / ".opentulpa" / "deepagents" / "capability_state"
    capability_state.mkdir(parents=True)
    cursor = capability_state / "telegram.json"
    cursor.write_text('{"cursor":10}\n')
    (capability_state / "current").symlink_to("telegram.json")
    product = workspace / ".opentulpa" / "deepagents" / "checkpoints.db"
    product.write_text("product-before\n")
    host = RootlessOciReleaseHost(
        policy=OciReleasePolicy(
            state_root=tmp_path / "state",
            mounts=(OciMount(source=workspace, target="/workspace", read_only=False),),
            allowed_mount_roots=(workspace,),
        ),
        runner=FakeOciRunner(release),
    )

    await host.snapshot_state("activation_1")
    cursor.write_text('{"cursor":99}\n')
    (capability_state / "candidate-only.txt").write_text("candidate\n")
    product.write_text("product-during-probation\n")

    assert await host.restore_state("activation_1") is True
    assert cursor.read_text() == '{"cursor":10}\n'
    assert not (capability_state / "candidate-only.txt").exists()
    assert (capability_state / "current").is_symlink()
    assert os.readlink(capability_state / "current") == "telegram.json"
    assert product.read_text() == "product-during-probation\n"

    await host.discard_state_snapshot("activation_1")
    assert await host.restore_state("activation_1") is False
    await host.aclose()


@pytest.mark.asyncio
async def test_capability_state_snapshot_fails_closed_after_tampering(tmp_path: Path) -> None:
    release = _release()
    workspace = tmp_path / "workspace"
    capability_state = workspace / ".opentulpa" / "deepagents" / "capability_state"
    capability_state.mkdir(parents=True)
    (capability_state / "state.json").write_text('{"version":1}\n')
    state_root = tmp_path / "state"
    host = RootlessOciReleaseHost(
        policy=OciReleasePolicy(
            state_root=state_root,
            mounts=(OciMount(source=workspace, target="/workspace", read_only=False),),
            allowed_mount_roots=(workspace,),
        ),
        runner=FakeOciRunner(release),
    )
    await host.snapshot_state("activation_tampered")
    snapshot = next((state_root / "workspace-snapshots").iterdir())
    (snapshot / "data" / "state.json").write_text('{"version":2}\n')

    with pytest.raises(ReleaseHostError, match="integrity"):
        await host.restore_state("activation_tampered")

    await host.aclose()


@pytest.mark.asyncio
async def test_rootless_podman_is_supported(tmp_path: Path) -> None:
    release = _release()
    runner = FakeOciRunner(release)
    host = RootlessOciReleaseHost(
        policy=OciReleasePolicy(
            container_cli="podman",
            state_root=tmp_path / "state",
        ),
        runner=runner,
    )

    prepared = await host.prepare(release)

    assert prepared.release_id == release.id
    assert any(command[:2] == ("podman", "info") for command in runner.commands)
    await host.aclose()


@pytest.mark.asyncio
async def test_local_oci_runner_does_not_inherit_application_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-bootstrap-boundary")
    runner = LocalOciCommandRunner(cwd=tmp_path)

    result = await runner.run(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ.get('OPENAI_API_KEY', 'missing'))",
        ),
        timeout_seconds=2,
        max_output_bytes=1_024,
    )

    assert result.returncode == 0
    assert result.output.strip() == b"missing"
    assert os.environ["OPENAI_API_KEY"] == "must-not-cross-bootstrap-boundary"


@pytest.mark.parametrize(
    "name",
    (
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONPLATLIBDIR",
        "PYTHONSAFEPATH",
        "PYTHONUSERBASE",
    ),
)
def test_release_policy_rejects_python_import_environment_overrides(name: str) -> None:
    with pytest.raises(ValueError, match="allowlist"):
        OciReleasePolicy(production_environment=((name, "/workspace"),))
