from __future__ import annotations

import asyncio
import json
import signal
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr

from opentulpa.evolution.models import EvolutionEvent
from opentulpa.host import runtime as runtime_module
from opentulpa.host.models import HostConfig
from opentulpa.host.runtime import (
    RuntimeLiveSourceSpec,
    RuntimeProcessIdentity,
    RuntimeSupervisor,
    RuntimeUnavailableError,
)
from opentulpa.host.runtime_environment import RuntimeEnvFileManager


def _config() -> HostConfig:
    return HostConfig(
        revision=1,
        status="active",
        api_key=SecretStr("provider-secret-value"),
        base_url="https://models.example/v1",
        model="moonshotai/kimi-k3",
        telegram_bot_token=SecretStr("telegram-secret-value"),
        telegram_user_id=7,
        internal_runtime_token=SecretStr("internal-owner-secret-value"),
        telegram_pairing_code=SecretStr("pairing-secret-value"),
        created_at=datetime.now(UTC),
    )


def _source_checkout(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "src" / "opentulpa").mkdir(parents=True)
    (source / "src" / "opentulpa" / "__init__.py").write_text("", encoding="utf-8")
    return source


class _FakeProcess:
    _next_pid = 50_000

    def __init__(self) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode: int | None = None
        self.stdout = None
        self.stderr = None
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def exit(self, returncode: int) -> None:
        self.returncode = returncode
        self._exited.set()

    def terminate(self) -> None:
        self.exit(-signal.SIGTERM)

    def kill(self) -> None:
        self.exit(-signal.SIGKILL)


def _fake_live_child(
    spec: RuntimeLiveSourceSpec,
    *,
    process: _FakeProcess | None = None,
    launch_nonce: str = "n" * 32,
) -> Any:
    child_process = process or _FakeProcess()
    executable = Path(sys.executable).resolve()
    return runtime_module._Child(
        process=cast(asyncio.subprocess.Process, child_process),
        endpoint="http://127.0.0.1:8123",
        config=_config(),
        live_source=spec,
        launch_nonce=launch_nonce,
        process_group=child_process.pid,
        process_birth="test-birth-identity",
        executable=executable,
        argv=(str(executable), "-P", "-m", "opentulpa"),
        readers=(),
    )


def _live_source_spec_with_environment(
    tmp_path: Path,
    *,
    commit: str = "d" * 40,
) -> RuntimeLiveSourceSpec:
    interpreter = tmp_path / "runtime-env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o700)
    return RuntimeLiveSourceSpec(
        source_commit=commit,
        runtime_environment_id="e" * 64,
        runtime_python_interpreter=str(interpreter),
        runtime_dependency_lock_hash="f" * 64,
        runtime_pyproject_sha256="1" * 64,
        runtime_install_profile="runtime-no-dev-extras-no-install-project-v1",
    )


async def _wait_until(predicate: Any, *, timeout: float = 1) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.001)


async def _terminate_fake_child(child: Any) -> None:
    if child.process.returncode is None:
        child.process.exit(0)


async def _no_op_async(value: Any) -> None:
    del value


def test_runtime_probation_constructor_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="probation must be finite and nonnegative"):
        RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data", probation_seconds=-1)
    with pytest.raises(ValueError, match="probe interval must be finite and positive"):
        RuntimeSupervisor(
            project_root=tmp_path,
            data_root=tmp_path / "data2",
            probation_probe_interval_seconds=0,
        )


@pytest.mark.asyncio
async def test_child_environment_hides_interface_secrets_and_logs_redact_exact_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_checkout(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "host-telegram-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "host-webhook-secret")
    monkeypatch.setenv("OPENTULPA_OWNER_CUSTOMER_ID", "opentulpa-gf")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv(
        "OPENTULPA_INTERNAL_AGENT_API_URL",
        "http://host.docker.internal:8000",
    )
    spec = _live_source_spec_with_environment(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=source,
        data_root=tmp_path / "data",
        live_source_spec=spec,
    )
    runtime.configure_evolution_control(
        base_url="http://127.0.0.1:8000/bootstrap/internal/v1/evolution",
        token="e" * 48,
    )
    runtime.configure_sandbox_worker(
        base_url="http://127.0.0.1:8787/internal/v1/sandbox",
        token="s" * 48,
    )
    config = _config()

    environment = runtime._child_environment(config, port=8123, live_source=spec)
    runtime._redaction_values = {
        config.api_key.get_secret_value(),
        config.internal_runtime_token.get_secret_value(),
        "s" * 48,
    }
    runtime._append_log(
        "stderr",
        "provider-secret-value Authorization=internal-owner-secret-value "
        f"password=hunter2 sandbox={'s' * 48}",
    )

    assert environment["OPENAI_COMPATIBLE_API_KEY"] == "provider-secret-value"
    assert environment["OPENTULPA_OWNER_TOKEN"] == "internal-owner-secret-value"
    assert environment["OPENTULPA_OWNER_CUSTOMER_ID"] == "opentulpa-gf"
    assert environment["OPENTULPA_INTERNAL_AGENT_API_URL"] == "http://host.docker.internal:8000"
    assert environment["OPENTULPA_BOOTSTRAP_EVOLUTION_TOKEN"] == "e" * 48
    assert environment["OPENTULPA_BOOTSTRAP_EVOLUTION_URL"].endswith(
        "/bootstrap/internal/v1/evolution"
    )
    assert environment["OPENTULPA_SANDBOX_RPC_URL"].endswith("/internal/v1/sandbox")
    assert environment["OPENTULPA_SANDBOX_RPC_TOKEN"] == "s" * 48
    assert environment["PYTHONPATH"] == str(source / "src")
    assert "TELEGRAM_BOT_TOKEN" not in environment
    assert "TELEGRAM_WEBHOOK_SECRET" not in environment
    line = runtime.logs()[0].text
    assert runtime.logs()[0].stream_id == runtime.log_stream_id
    assert "provider-secret-value" not in line
    assert "internal-owner-secret-value" not in line
    assert "hunter2" not in line
    assert "s" * 48 not in line
    assert line.count("[redacted]") == 4

    monkeypatch.delenv("OPENTULPA_OWNER_CUSTOMER_ID")
    monkeypatch.delenv("OPENTULPA_INTERNAL_AGENT_API_URL")
    fallback_environment = runtime._child_environment(config, port=8124, live_source=spec)
    assert fallback_environment["OPENTULPA_OWNER_CUSTOMER_ID"] == "owner"
    assert fallback_environment["OPENTULPA_INTERNAL_AGENT_API_URL"] == "http://127.0.0.1:9000"
    monkeypatch.delenv("PORT")
    assert (
        runtime._child_environment(config, port=8125, live_source=spec)[
            "OPENTULPA_INTERNAL_AGENT_API_URL"
        ]
        == "http://127.0.0.1:8125"
    )
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_live_source_environment_loads_dotenv_without_leaking_host_owned_keys(
    tmp_path: Path,
) -> None:
    source = _source_checkout(tmp_path)
    dotenv = source / ".env"
    dotenv.write_text(
        "OPENAI_COMPATIBLE_API_KEY=dotenv-provider-secret\n"
        "TELEGRAM_BOT_TOKEN=dotenv-telegram-secret\n"
        "PATH=/tmp/hostile-bin\n"
        "OPENTULPA_OWNER_TOKEN=dotenv-owner-token\n",
        encoding="utf-8",
    )
    dotenv.chmod(0o600)
    spec = _live_source_spec_with_environment(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=source,
        data_root=tmp_path / "data",
        live_source_spec=spec,
    )

    environment = runtime._child_environment(_config(), port=8123, live_source=spec)

    assert environment["OPENAI_COMPATIBLE_API_KEY"] == "dotenv-provider-secret"
    assert environment["TELEGRAM_BOT_TOKEN"] == "dotenv-telegram-secret"
    assert environment["OPENTULPA_OWNER_TOKEN"] == "internal-owner-secret-value"
    assert environment["PATH"].startswith(f"{spec.python_interpreter_path.parent}:")
    assert "/tmp/hostile-bin" not in environment["PATH"]
    assert environment["PYTHONPATH"] == str(source / "src")
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_live_source_retry_reloads_dotenv_and_uses_release_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_checkout(tmp_path)
    dotenv = source / ".env"
    dotenv.write_text("RUNTIME_API_KEY=first\n", encoding="utf-8")
    dotenv.chmod(0o600)
    spec = _live_source_spec_with_environment(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=source,
        data_root=tmp_path / "data",
        live_source_spec=spec,
        control_path=tmp_path / "control" / "runtime-child.json",
    )
    runtime._child_uid = None
    runtime._child_gid = None
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = lambda leader_pid, launch_nonce: ()
    launches: list[tuple[tuple[str, ...], dict[str, str]]] = []
    attempts = 0

    async def create_process(*argv: str, **options: Any) -> _FakeProcess:
        process = _FakeProcess()
        launches.append((argv, dict(options["env"])))
        return process

    async def wait_ready(child: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            child.process.exit(98)
            dotenv.write_text("RUNTIME_API_KEY=second\n", encoding="utf-8")
            dotenv.chmod(0o600)
            raise runtime_module._ChildExitedBeforeReadyError("first child exited")

    async def terminate(child: Any) -> None:
        if child.process.returncode is None:
            child.process.exit(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runtime, "_checkout_live_source", lambda commit: source)
    monkeypatch.setattr(runtime, "_live_source_cwd", lambda root: source)
    monkeypatch.setattr(runtime, "_wait_ready", wait_ready)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)
    monkeypatch.setattr(runtime, "_capture_process_birth", lambda pid: f"test:{pid}")

    child = await runtime._spawn_live_source(_config(), live_source=spec)

    assert attempts == 2
    assert [launch[0][0] for launch in launches] == [str(spec.python_interpreter_path)] * 2
    assert [launch[1]["RUNTIME_API_KEY"] for launch in launches] == ["first", "second"]
    await runtime._stop_child(child)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_env_file_manager_restores_dotenv_when_restart_fails(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dotenv = source / ".env"
    dotenv.write_text("OPENAI_COMPATIBLE_API_KEY=previous\n", encoding="utf-8")
    dotenv.chmod(0o600)

    class FailingRuntime:
        status = "ready"

        async def replace_current_environment(self, *, apply: Any, restore: Any) -> None:
            apply()
            restore()
            raise RuntimeUnavailableError("restart failed")

    manager = RuntimeEnvFileManager(source_root=source, runtime=FailingRuntime())

    result = await manager.set(
        name="OPENAI_COMPATIBLE_API_KEY",
        value="updated",
        idempotency_key="env-update-1",
    )

    assert result["status"] == "failed"
    assert result["failure_stage"] == "runtime_restart"
    assert result["rollback_restored"] is True
    assert result["file_rollback_restored"] is True
    assert result["runtime_restored"] is True
    assert result["value"] == "[redacted]"
    assert dotenv.read_text(encoding="utf-8") == "OPENAI_COMPATIBLE_API_KEY=previous\n"


@pytest.mark.asyncio
async def test_runtime_env_file_manager_lists_names_without_values(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dotenv = source / ".env"
    dotenv.write_text("COMPOSIO_API_KEY=provider-secret\n", encoding="utf-8")
    dotenv.chmod(0o600)

    result = await RuntimeEnvFileManager(source_root=source, runtime=object()).read()  # type: ignore[arg-type]

    assert result == {
        "available": True,
        "variables": [{"name": "COMPOSIO_API_KEY", "set": True}],
        "count": 1,
    }


@pytest.mark.asyncio
async def test_runtime_env_file_manager_rejects_world_readable_existing_dotenv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dotenv = source / ".env"
    dotenv.write_text("OPENAI_COMPATIBLE_API_KEY=previous\n", encoding="utf-8")
    dotenv.chmod(0o644)

    class Runtime:
        status = "ready"

        async def replace_current_environment(self, *, apply: Any, restore: Any) -> None:
            raise AssertionError("unsafe .env files must not restart the runtime")

    result = await RuntimeEnvFileManager(source_root=source, runtime=Runtime()).set(
        name="OPENAI_COMPATIBLE_API_KEY",
        value="updated",
        idempotency_key="env-update-unsafe",
    )

    assert result["status"] == "failed"
    assert result["failure_stage"] == "env_read"
    assert result["error"]["code"] == "runtime_env_file_invalid"
    assert dotenv.read_text(encoding="utf-8") == "OPENAI_COMPATIBLE_API_KEY=previous\n"


@pytest.mark.asyncio
async def test_replace_current_environment_restores_env_when_child_stop_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = RuntimeLiveSourceSpec(source_commit="a" * 40)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        live_source_spec=spec,
    )
    previous = _fake_live_child(spec)
    runtime._child = previous
    runtime._status = "ready"
    applied = False
    restored = False

    def apply() -> None:
        nonlocal applied
        applied = True

    def restore() -> None:
        nonlocal restored
        restored = True

    def inspect_descendants(child: Any) -> tuple[RuntimeProcessIdentity, ...]:
        assert child is previous
        raise RuntimeUnavailableError("process table unavailable")

    async def terminate(child: Any) -> None:
        del child
        raise AssertionError("containment preflight failures must not signal the child")

    monkeypatch.setattr(runtime, "_ensure_controller_ownership", lambda: asyncio.sleep(0))
    monkeypatch.setattr(runtime, "_preflight_target", _no_op_async)
    monkeypatch.setattr(runtime, "_owned_descendants", inspect_descendants)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)

    with pytest.raises(RuntimeUnavailableError, match="containment preflight failed"):
        await runtime.replace_current_environment(apply=apply, restore=restore)

    assert applied is True
    assert restored is True
    assert runtime._child is previous
    assert previous.requested_stop is False
    assert previous.process.returncode is None
    assert runtime.status == "ready"
    runtime._child = None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_env_file_manager_does_not_claim_runtime_rollback_when_degraded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    dotenv = source / ".env"
    dotenv.write_text("COMPOSIO_API_KEY=previous\n", encoding="utf-8")
    dotenv.chmod(0o600)

    class DegradedRuntime:
        status = "ready"

        async def replace_current_environment(self, *, apply: Any, restore: Any) -> None:
            apply()
            restore()
            self.status = "recovery_required"
            raise RuntimeUnavailableError("runtime descendants could not be enumerated")

    result = await RuntimeEnvFileManager(source_root=source, runtime=DegradedRuntime()).set(
        name="COMPOSIO_API_KEY",
        value="updated",
        idempotency_key="env-update-degraded",
    )

    assert result["status"] == "failed"
    assert result["rollback_restored"] is False
    assert result["file_rollback_restored"] is True
    assert result["runtime_restored"] is False
    assert "runtime is recovery_required" in result["error"]["message"]
    assert dotenv.read_text(encoding="utf-8") == "COMPOSIO_API_KEY=previous\n"


@pytest.mark.asyncio
async def test_runtime_env_file_manager_rejects_host_owned_env_key(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    class Runtime:
        status = "ready"

        async def replace_current_environment(self, *, apply: Any, restore: Any) -> None:
            raise AssertionError("protected keys must not restart the runtime")

    result = await RuntimeEnvFileManager(source_root=source, runtime=Runtime()).set(
        name="PATH",
        value="/tmp/hostile",
        idempotency_key="env-update-2",
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "runtime_env_key_protected"
    assert not (source / ".env").exists()


@pytest.mark.asyncio
async def test_live_source_uses_stable_host_railway_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_checkout(tmp_path)
    bridge = source / "railway_sandbox_bridge" / "bridge.mjs"
    bridge.parent.mkdir()
    bridge.write_text("", encoding="utf-8")
    monkeypatch.setenv(
        "OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH",
        "/untrusted/inherited/bridge.mjs",
    )
    spec = _live_source_spec_with_environment(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=source,
        data_root=tmp_path / "data",
        live_source_spec=spec,
    )

    environment = runtime._child_environment(_config(), port=8123, live_source=spec)

    assert environment["PYTHONPATH"] == str(source / "src")
    assert environment["OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH"] == str(bridge)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_host_passes_telegram_owner_identity_without_writing_child_state(
    tmp_path: Path,
) -> None:
    source = _source_checkout(tmp_path)
    data_root = tmp_path / "runtime-data"
    spec = _live_source_spec_with_environment(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=source,
        data_root=data_root,
        live_source_spec=spec,
    )

    environment = runtime._child_environment(_config(), port=8123, live_source=spec)

    assert environment["OPENTULPA_TELEGRAM_OWNER_ID"] == "7"
    assert not data_root.exists()

    token_only = _config().model_copy(update={"telegram_user_id": None})
    pairing_environment = runtime._child_environment(token_only, port=8123, live_source=spec)
    assert pairing_environment["OPENTULPA_TELEGRAM_OWNER_ID"] == ""

    no_telegram = _config().model_copy(
        update={"telegram_bot_token": None, "telegram_user_id": None}
    )
    disabled_environment = runtime._child_environment(no_telegram, port=8123, live_source=spec)
    assert "OPENTULPA_TELEGRAM_OWNER_ID" not in disabled_environment
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_signal_accepts_unreadable_launch_nonce_for_same_process(
    tmp_path: Path,
) -> None:
    executable = Path(sys.executable).resolve()
    expected = RuntimeProcessIdentity(
        pid=12_345,
        process_group=12_345,
        executable=executable,
        argv=(str(executable), "-m", "opentulpa"),
        process_birth="process-birth-1",
        launch_nonce="launch-nonce-1",
    )
    observed = RuntimeProcessIdentity(
        pid=expected.pid,
        process_group=expected.process_group,
        executable=expected.executable,
        argv=expected.argv,
        process_birth=expected.process_birth,
        launch_nonce=None,
    )
    signals: list[tuple[RuntimeProcessIdentity, signal.Signals]] = []

    def inspect_process(pid: int) -> RuntimeProcessIdentity | None:
        assert pid == expected.pid
        return observed

    def fence_process(identity: RuntimeProcessIdentity, selected: signal.Signals) -> None:
        signals.append((identity, selected))

    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        process_inspector=inspect_process,
        process_fencer=fence_process,
    )

    assert await runtime._signal_verified_identity(expected, signal.SIGTERM) is True
    assert signals == [(expected, signal.SIGTERM)]
    await runtime._client.aclose()


def test_descendant_scan_ignores_unrelated_foreign_uid_host_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    runtime._child_uid = 65_532
    runtime._child_gid = 65_532
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime_module.os, "getpid", lambda: 100)
    monkeypatch.setattr(
        runtime,
        "_linux_process_table",
        lambda: {
            201: runtime_module._LinuxProcessMetadata(
                parent_pid=100,
                process_group=1,
                process_birth="linux:1",
                proc_uid=0,
            )
        },
    )
    uid_reads: list[Path] = []

    def read_uids(path: Path) -> tuple[int, tuple[int, int, int, int]]:
        uid_reads.append(path)
        raise PermissionError(13, "unrelated status is hidden")

    monkeypatch.setattr(runtime, "_linux_process_uids", read_uids)

    assert runtime._inspect_linux_descendants(200, "launch-nonce") == ()
    assert uid_reads == []


def test_descendant_scan_includes_runtime_uid_adopted_orphan_with_unreadable_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_nonce = "launch-nonce"
    identity = RuntimeProcessIdentity(
        pid=201,
        parent_pid=100,
        process_group=777,
        process_birth="linux:1",
        executable=Path("/runtime-worker"),
        argv=("/runtime-worker",),
        launch_nonce=None,
    )
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        process_inspector=lambda pid: identity if pid == identity.pid else None,
    )
    runtime._child_uid = 65_532
    runtime._child_gid = 65_532
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime_module.os, "getpid", lambda: 100)
    monkeypatch.setattr(
        runtime,
        "_linux_process_table",
        lambda: {
            identity.pid: runtime_module._LinuxProcessMetadata(
                parent_pid=identity.parent_pid,
                process_group=identity.process_group,
                process_birth=identity.process_birth,
                proc_uid=65_532,
            )
        },
    )
    monkeypatch.setattr(
        runtime,
        "_linux_process_uids",
        lambda path: (65_532, (65_532, 65_532, 65_532, 65_532)),
    )

    descendants = runtime._inspect_linux_descendants(200, launch_nonce)

    assert len(descendants) == 1
    assert descendants[0].pid == identity.pid
    assert descendants[0].process_birth == identity.process_birth
    assert descendants[0].launch_nonce == launch_nonce


def test_descendant_scan_accepts_unreadable_nonce_for_structural_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_nonce = "launch-nonce"
    observed = RuntimeProcessIdentity(
        pid=201,
        parent_pid=200,
        process_group=200,
        process_birth="linux:1",
        executable=Path("/runtime-worker"),
        argv=("/runtime-worker",),
        launch_nonce=None,
    )
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        process_inspector=lambda pid: observed if pid == observed.pid else None,
    )
    runtime._child_uid = 65_532
    runtime._child_gid = 65_532
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        runtime,
        "_linux_process_table",
        lambda: {
            observed.pid: runtime_module._LinuxProcessMetadata(
                parent_pid=200,
                process_group=observed.process_group,
                process_birth=observed.process_birth,
                proc_uid=65_532,
            )
        },
    )
    monkeypatch.setattr(
        runtime,
        "_linux_process_uids",
        lambda path: (65_532, (65_532, 65_532, 65_532, 65_532)),
    )

    descendants = runtime._inspect_linux_descendants(200, launch_nonce)

    assert len(descendants) == 1
    assert descendants[0].pid == observed.pid
    assert descendants[0].process_birth == observed.process_birth
    assert descendants[0].launch_nonce == launch_nonce


def test_non_root_descendant_scan_does_not_inspect_unrelated_same_uid_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    runtime._child_uid = None
    runtime._child_gid = None
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 501)
    monkeypatch.setattr(
        runtime,
        "_linux_process_table",
        lambda: {
            201: runtime_module._LinuxProcessMetadata(
                parent_pid=1,
                process_group=1,
                process_birth="linux:1",
                proc_uid=501,
            )
        },
    )
    uid_reads: list[Path] = []

    def read_uids(path: Path) -> tuple[int, tuple[int, int, int, int]]:
        uid_reads.append(path)
        raise PermissionError(13, "unrelated status is hidden")

    monkeypatch.setattr(runtime, "_linux_process_uids", read_uids)

    assert runtime._inspect_linux_descendants(200, "launch-nonce") == ()
    assert uid_reads == []


def test_descendant_scan_fails_closed_when_runtime_uid_status_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    runtime._child_uid = 65_532
    runtime._child_gid = 65_532
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        runtime,
        "_linux_process_table",
        lambda: {
            201: runtime_module._LinuxProcessMetadata(
                parent_pid=1,
                process_group=1,
                process_birth="linux:1",
                proc_uid=65_532,
            )
        },
    )
    monkeypatch.setattr(
        runtime,
        "_linux_process_uids",
        lambda path: (_ for _ in ()).throw(PermissionError(13, "hidden")),
    )
    monkeypatch.setattr(runtime, "_pid_alive", lambda pid: True)

    with pytest.raises(
        RuntimeUnavailableError,
        match=r"UID inspection failed for pid 201 \(PermissionError, errno=13\)",
    ):
        runtime._inspect_linux_descendants(200, "launch-nonce")


def test_process_table_error_preserves_safe_pid_and_errno(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    process_root = proc_root / "42"
    process_root.mkdir(parents=True)

    def fail_metadata(path: Path) -> tuple[int, int, str]:
        assert path == process_root
        raise PermissionError(13, "hidden")

    monkeypatch.setattr(
        RuntimeSupervisor,
        "_linux_process_metadata",
        staticmethod(fail_metadata),
    )

    with pytest.raises(
        RuntimeUnavailableError,
        match=r"pid 42 \(PermissionError, errno=13\)",
    ):
        RuntimeSupervisor._linux_process_table(proc_root)


@pytest.mark.asyncio
async def test_explicit_stop_fails_closed_when_containment_preflight_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = RuntimeLiveSourceSpec(source_commit="a" * 40)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        live_source_spec=spec,
    )
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = lambda leader_pid, launch_nonce: ()
    child = _fake_live_child(spec)
    runtime._child = child
    runtime._status = "ready"
    monkeypatch.setattr(
        runtime,
        "_owned_descendants",
        lambda selected: (_ for _ in ()).throw(RuntimeUnavailableError("proc unavailable")),
    )

    with pytest.raises(RuntimeUnavailableError, match="containment preflight failed"):
        await runtime.stop()

    assert runtime.status == "recovery_required"
    assert runtime.endpoint is None
    assert child.requested_stop is False
    assert child.process.returncode is None
    runtime._child = None
    runtime._status = "stopped"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_refuses_to_start_without_a_runtime_target(tmp_path: Path) -> None:
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")

    with pytest.raises(RuntimeUnavailableError, match="no runtime live source"):
        await runtime.start(_config())

    assert runtime.status == "stopped"
    assert runtime.endpoint is None
    await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("record_kind", ["ownership", "intent"])
async def test_generation_runtime_control_records_fail_closed(
    tmp_path: Path,
    record_kind: str,
) -> None:
    control_path = tmp_path / "control" / "runtime-child.json"
    path = (
        control_path
        if record_kind == "ownership"
        else control_path.with_name(f".{control_path.name}.intent")
    )
    path.parent.mkdir(mode=0o700)
    payload: dict[str, object] = {
        "format_version": 1,
        "host_pid": 1,
        "mode": "generation",
        "generation_id": "a" * 64,
        "source_commit": None,
        "launch_nonce": "generation-launch-nonce-000000",
        "executable": "/usr/bin/python3",
        "argv": ["/usr/bin/python3", "-m", "opentulpa"],
    }
    if record_kind == "ownership":
        payload.update(
            {
                "pid": 2,
                "process_group": 2,
                "host_birth": "generation-host-birth",
                "process_birth": "generation-process-birth",
            }
        )
    path.write_text(json.dumps(payload), encoding="ascii")
    path.chmod(0o600)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        control_path=control_path,
    )

    message = (
        "runtime ownership record is invalid"
        if record_kind == "ownership"
        else "runtime launch intent is invalid"
    )
    with pytest.raises(RuntimeUnavailableError, match=message):
        await runtime._ensure_controller_ownership()

    assert runtime.status == "recovery_required"
    runtime._release_owner_lock()
    await runtime._client.aclose()


@pytest.mark.asyncio
async def test_failed_live_source_candidate_restores_exact_previous_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_spec = RuntimeLiveSourceSpec(source_commit="a" * 40)
    candidate_spec = RuntimeLiveSourceSpec(source_commit="b" * 40)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        live_source_spec=previous_spec,
    )
    previous = _fake_live_child(previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    launches: list[RuntimeLiveSourceSpec] = []

    async def stop_child(child: Any) -> None:
        child.requested_stop = True

    async def preflight_target(target: Any) -> None:
        del target

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        del config
        launches.append(target.live_source)
        if target.live_source == candidate_spec:
            raise RuntimeUnavailableError("candidate source is unhealthy")
        return _fake_live_child(target.live_source)

    def adopt(child: Any) -> None:
        runtime._child = child
        runtime._status = "ready"

    monkeypatch.setattr(runtime, "_stop_child", stop_child)
    monkeypatch.setattr(runtime, "_preflight_target", preflight_target)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)
    monkeypatch.setattr(runtime, "_adopt_child", adopt)

    with pytest.raises(RuntimeUnavailableError, match="candidate source is unhealthy"):
        await runtime.replace_live_source(candidate_spec, rollback=previous_spec)

    assert launches == [candidate_spec, previous_spec]
    assert runtime.live_source == previous_spec
    assert runtime._child.live_source == previous_spec
    assert runtime.status == "ready"
    runtime._child = None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_requested_stop_does_not_restart_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = RuntimeLiveSourceSpec(source_commit="a" * 40)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        live_source_spec=spec,
        restart_backoff_seconds=0,
        max_restart_backoff_seconds=0,
    )
    child = _fake_live_child(spec)
    runtime._begin_selection(child.config, child.target)
    runtime._status = "ready"
    runtime._adopt_child(child)
    restart_calls = 0

    async def terminate(selected: Any) -> None:
        selected.process.exit(0)

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        nonlocal restart_calls
        del config, target
        restart_calls += 1
        raise AssertionError("requested stop restarted the child")

    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)

    await runtime.stop()
    await asyncio.sleep(0)

    assert restart_calls == 0
    assert runtime.status == "stopped"
    assert runtime.endpoint is None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_unexpected_exit_restarts_exact_live_source_with_bounded_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = RuntimeLiveSourceSpec(source_commit="a" * 40)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        live_source_spec=spec,
        max_unexpected_restarts=1,
        restart_backoff_seconds=0,
        max_restart_backoff_seconds=0,
    )
    first = _fake_live_child(spec)
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = lambda leader_pid, launch_nonce: ()
    runtime._begin_selection(first.config, first.target)
    runtime._status = "ready"
    runtime._adopt_child(first)
    launches: list[tuple[int, RuntimeLiveSourceSpec]] = []

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        launches.append((config.revision, target.live_source))
        replacement = _fake_live_child(target.live_source)
        runtime._status = "ready"
        return replacement

    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)

    first.process.exit(17)
    await _wait_until(lambda: runtime._child is not None and runtime._child is not first)
    replacement = runtime._child
    assert replacement is not None
    cast(_FakeProcess, replacement.process).exit(18)
    await _wait_until(lambda: runtime.status == "failed" and runtime._child is None)

    assert launches == [(first.config.revision, spec)]
    assert runtime.live_source == spec
    assert runtime.endpoint is None
    assert "restart budget" in str(runtime.error)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_ownership_record_fences_only_proven_process_identity(tmp_path: Path) -> None:
    spec = RuntimeLiveSourceSpec(source_commit="a" * 40)
    control_path = tmp_path / "controller" / "runtime-child.json"
    owner = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "owner-data",
        control_path=control_path,
    )
    child = _fake_live_child(spec, launch_nonce="launch-nonce-verified-000000000")
    owner._write_ownership_record(child)
    fenced: list[tuple[int, signal.Signals]] = []
    process_alive = True

    def inspect_process(pid: int) -> RuntimeProcessIdentity | None:
        if not process_alive:
            return None
        return RuntimeProcessIdentity(
            pid=pid,
            process_group=child.process_group,
            executable=child.executable,
            argv=child.argv,
            process_birth=child.process_birth,
            launch_nonce=child.launch_nonce,
        )

    def fence_process(
        identity: RuntimeProcessIdentity,
        selected_signal: signal.Signals,
    ) -> None:
        nonlocal process_alive
        fenced.append((identity.pid, selected_signal))
        process_alive = False

    recovering = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "recovering-data",
        control_path=control_path,
        process_inspector=inspect_process,
        process_fencer=fence_process,
    )

    await recovering._ensure_orphan_fenced()

    assert fenced == [(child.process.pid, signal.SIGTERM)]
    assert not control_path.exists()

    owner._write_ownership_record(child)
    ambiguous = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "ambiguous-data",
        control_path=control_path,
        process_inspector=lambda pid: RuntimeProcessIdentity(
            pid=pid,
            process_group=child.process_group,
            executable=tmp_path / "different-executable",
            argv=child.argv,
            process_birth=child.process_birth,
            launch_nonce=child.launch_nonce,
        ),
        process_fencer=fence_process,
    )
    with pytest.raises(RuntimeUnavailableError, match="identity is ambiguous"):
        await ambiguous._ensure_orphan_fenced()
    assert control_path.exists()
    assert len(fenced) == 1
    owner._remove_ownership_record(child.launch_nonce)
    assert stat.S_IMODE(control_path.parent.stat().st_mode) == 0o700
    await owner.shutdown()
    await recovering.shutdown()
    await ambiguous.shutdown()


@pytest.mark.asyncio
async def test_launch_intent_and_missing_nonce_both_fail_closed(tmp_path: Path) -> None:
    spec = RuntimeLiveSourceSpec(source_commit="a" * 40)
    control_path = tmp_path / "controller" / "runtime-child.json"
    owner = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "owner-data",
        control_path=control_path,
    )
    executable = Path(sys.executable).resolve()
    owner._write_launch_intent(
        live_source=spec,
        launch_nonce="launch-intent-incomplete-00000000",
        executable=executable,
        argv=(str(executable), "-P", "-m", "opentulpa"),
    )
    recovering_intent = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "intent-data",
        control_path=control_path,
    )

    with pytest.raises(RuntimeUnavailableError, match="launch intent requires manual recovery"):
        await recovering_intent._ensure_controller_ownership()
    assert recovering_intent.status == "recovery_required"
    owner._remove_launch_intent("launch-intent-incomplete-00000000")
    await recovering_intent.shutdown()

    child = _fake_live_child(spec, launch_nonce="launch-nonce-must-be-observed-0000")
    owner._write_ownership_record(child)
    fenced = False

    def inspect_without_nonce(pid: int) -> RuntimeProcessIdentity:
        return RuntimeProcessIdentity(
            pid=pid,
            process_group=child.process_group,
            executable=child.executable,
            argv=child.argv,
            process_birth=child.process_birth,
            launch_nonce=None,
        )

    def fence_process(
        identity: RuntimeProcessIdentity,
        selected_signal: signal.Signals,
    ) -> None:
        nonlocal fenced
        del identity, selected_signal
        fenced = True

    recovering_nonce = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "nonce-data",
        control_path=control_path,
        process_inspector=inspect_without_nonce,
        process_fencer=fence_process,
    )

    with pytest.raises(RuntimeUnavailableError, match="identity is ambiguous"):
        await recovering_nonce._ensure_controller_ownership()
    assert fenced is False
    assert control_path.exists()

    owner._remove_ownership_record(child.launch_nonce)
    await owner.shutdown()
    await recovering_nonce.shutdown()


@pytest.mark.asyncio
async def test_runtime_control_configuration_requires_fully_stopped_state(
    tmp_path: Path,
) -> None:
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    runtime._status = "starting"

    with pytest.raises(RuntimeUnavailableError, match="runtime transition"):
        runtime.configure_evolution_control(base_url="http://127.0.0.1:8000", token="e" * 48)
    with pytest.raises(RuntimeUnavailableError, match="runtime transition"):
        runtime.configure_sandbox_worker(
            base_url="http://127.0.0.1:8787",
            token="s" * 48,
        )
    runtime._status = "stopped"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_candidate_is_routed_during_probation_and_commits_after_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_spec = RuntimeLiveSourceSpec(source_commit="a" * 40)
    candidate_spec = RuntimeLiveSourceSpec(source_commit="b" * 40)
    requests: list[httpx.Request] = []

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(204)
        )
    )
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "probation",
        live_source_spec=previous_spec,
        probation_seconds=1,
        client=client,
    )
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = lambda leader_pid, launch_nonce: ()
    previous = _fake_live_child(previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    entered = asyncio.Event()
    release = asyncio.Event()

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        child = _fake_live_child(target.live_source)
        child.config = config
        child.endpoint = "http://candidate.runtime"
        return child

    async def probe(child: Any) -> None:
        assert child.live_source == candidate_spec
        entered.set()
        await release.wait()

    monkeypatch.setattr(runtime, "_ensure_controller_ownership", lambda: asyncio.sleep(0))
    monkeypatch.setattr(runtime, "_preflight_target", _no_op_async)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)
    monkeypatch.setattr(runtime, "_probe_child_readiness", probe)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", _terminate_fake_child)
    replacement = asyncio.create_task(runtime.replace_live_source(candidate_spec))
    await entered.wait()
    assert runtime.status == "probation"
    assert runtime.endpoint == "http://candidate.runtime"
    event = EvolutionEvent(
        event_key="candidate:probation:event",
        event_type="candidate.passed",
        release_id="release",
        payload={},
    )
    await runtime.deliver_evolution_event(event)
    assert requests[0].url.host == "candidate.runtime"
    release.set()
    await replacement
    assert runtime.status == "ready"
    assert runtime.live_source == candidate_spec
    await runtime.shutdown()
    await client.aclose()
