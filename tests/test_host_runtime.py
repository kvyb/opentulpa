from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import signal
import stat
import sys
import sysconfig
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from opentulpa.evolution.generation import (
    GenerationDescriptor,
    GenerationIdentity,
    GenerationManifest,
    StateContract,
    canonical_json_bytes,
)
from opentulpa.evolution.generation_store import (
    GenerationStore,
    InstalledGeneration,
    runtime_tree_sha256,
)
from opentulpa.evolution.models import EvolutionEvent
from opentulpa.host import runtime as runtime_module
from opentulpa.host.models import HostConfig
from opentulpa.host.runtime import (
    RuntimeGenerationSpec,
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


def _contract() -> StateContract:
    return StateContract(
        runtime_protocol=1,
        controller_min=1,
        controller_max=2,
        product_state_schema=1,
        workspace_api=1,
    )


def _fake_generation(
    tmp_path: Path,
    character: str = "a",
) -> tuple[InstalledGeneration, RuntimeGenerationSpec]:
    for root_name in ("data", "application"):
        (tmp_path / root_name).mkdir(mode=0o700, exist_ok=True)
    contract = _contract()
    wheel = f"wheel-{character}".encode()
    lock = f"lock-{character}".encode()
    identity = GenerationIdentity(
        source_commit=character * 40,
        source_tree_sha256=character * 64,
        wheel_sha256=hashlib.sha256(wheel).hexdigest(),
        uv_lock_sha256=hashlib.sha256(lock).hexdigest(),
        evaluator_fingerprint=f"sha256:{character * 64}",
        evaluation_input_sha256=character * 64,
        python_runtime_sha256=character * 64,
        cpython_version=platform.python_version(),
        cpython_cache_tag=str(sys.implementation.cache_tag),
        cpython_abi_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
        os_name="posix",
        platform=sysconfig.get_platform(),
        machine=platform.machine(),
        build_recipe_version="runtime-test-v1",
        runtime_protocol=1,
        controller_min=1,
        controller_max=2,
        state_contract_sha256=contract.sha256(),
        install_profile="runtime",
        entrypoint=("venv/bin/opentulpa", "--serve"),
    )
    path = tmp_path / "generations" / identity.generation_id
    path.parent.mkdir(mode=0o711, exist_ok=True)
    path.parent.chmod(0o711)
    path.mkdir(mode=0o555, exist_ok=True)
    path.chmod(0o555)
    manifest = GenerationManifest(
        identity=identity,
        state_contract=contract,
        descriptor=GenerationDescriptor(
            wheel_path="artifacts/opentulpa.whl",
            wheel_size_bytes=len(wheel),
            uv_lock_path="artifacts/uv.lock",
            uv_lock_size_bytes=len(lock),
            venv_path="venv",
        ),
        runtime_tree_sha256=character * 64,
    )
    manifest_digest = f"sha256:{hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()}"
    installed = InstalledGeneration(
        generation_id=identity.generation_id,
        path=path,
        manifest=manifest,
        manifest_digest=manifest_digest,
        interpreter_path=path / "venv/bin/python",
        entrypoint_path=path / "venv/bin/opentulpa",
    )
    spec = RuntimeGenerationSpec(
        generation_id=identity.generation_id,
        expected_manifest_digest=manifest_digest,
        expected_state_contract_digest=contract.sha256(),
        expected_evaluator_fingerprint=identity.evaluator_fingerprint,
        expected_install_profile=identity.install_profile,
        controller_protocol=1,
    )
    return installed, spec


def _published_generation(
    tmp_path: Path,
) -> tuple[GenerationStore, RuntimeGenerationSpec, Path]:
    for root_name in ("data", "application"):
        (tmp_path / root_name).mkdir(mode=0o700, exist_ok=True)
    generations_root = tmp_path / "published-generations"
    generations_root.mkdir(mode=0o711)
    contract = _contract()
    wheel = b"runtime test wheel"
    lock = b"version = 1\n"
    identity = GenerationIdentity(
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        wheel_sha256=hashlib.sha256(wheel).hexdigest(),
        uv_lock_sha256=hashlib.sha256(lock).hexdigest(),
        evaluator_fingerprint=f"sha256:{'c' * 64}",
        evaluation_input_sha256="d" * 64,
        python_runtime_sha256="e" * 64,
        cpython_version=platform.python_version(),
        cpython_cache_tag=str(sys.implementation.cache_tag),
        cpython_abi_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
        os_name="posix",
        platform=sysconfig.get_platform(),
        machine=platform.machine(),
        build_recipe_version="runtime-test-v1",
        runtime_protocol=1,
        controller_min=1,
        controller_max=2,
        state_contract_sha256=contract.sha256(),
        install_profile="runtime",
        entrypoint=("venv/bin/python", "-I", "-m", "opentulpa"),
    )
    generation = generations_root / identity.generation_id
    artifacts = generation / "artifacts"
    bin_path = generation / "venv" / "bin"
    artifacts.mkdir(parents=True)
    bin_path.mkdir(parents=True)
    wheel_path = artifacts / "opentulpa.whl"
    lock_path = artifacts / "uv.lock"
    interpreter = bin_path / "python"
    wheel_path.write_bytes(wheel)
    lock_path.write_bytes(lock)
    interpreter.write_bytes(b"trusted interpreter placeholder\n")
    wheel_path.chmod(0o444)
    lock_path.chmod(0o444)
    interpreter.chmod(0o555)
    for directory in (artifacts, bin_path, generation / "venv"):
        directory.chmod(0o555)
    manifest = GenerationManifest(
        identity=identity,
        state_contract=contract,
        descriptor=GenerationDescriptor(
            wheel_path="artifacts/opentulpa.whl",
            wheel_size_bytes=len(wheel),
            uv_lock_path="artifacts/uv.lock",
            uv_lock_size_bytes=len(lock),
            venv_path="venv",
        ),
        runtime_tree_sha256=runtime_tree_sha256(generation),
    )
    manifest_bytes = canonical_json_bytes(manifest)
    (generation / "manifest.json").write_bytes(manifest_bytes)
    (generation / "COMPLETE").write_bytes(b"")
    (generation / "manifest.json").chmod(0o444)
    (generation / "COMPLETE").chmod(0o444)
    generation.chmod(0o555)
    spec = RuntimeGenerationSpec(
        generation_id=identity.generation_id,
        expected_manifest_digest=f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}",
        expected_state_contract_digest=contract.sha256(),
        expected_evaluator_fingerprint=identity.evaluator_fingerprint,
        expected_install_profile=identity.install_profile,
        controller_protocol=1,
    )
    return GenerationStore(generations_root), spec, interpreter


class _RecordingGenerationStore:
    def __init__(self, *installed: InstalledGeneration) -> None:
        self.installed = {item.generation_id: item for item in installed}
        self.root = installed[0].path.parent
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.lock_held = False
        self.blocked_replacement_attempts = 0

    def open(self, generation_id: str, **provenance: object) -> InstalledGeneration:
        self.calls.append((generation_id, provenance))
        if generation_id not in self.installed:
            raise RuntimeError("unexpected generation")
        return self.installed[generation_id]

    @contextmanager
    def locked(self) -> Any:
        assert not self.lock_held
        self.lock_held = True
        try:
            yield
        finally:
            self.lock_held = False

    def attempt_replacement(self) -> bool:
        if self.lock_held:
            self.blocked_replacement_attempts += 1
            return False
        return True


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


def _fake_child(
    installed: InstalledGeneration,
    spec: RuntimeGenerationSpec,
    *,
    process: _FakeProcess | None = None,
    launch_nonce: str = "n" * 32,
) -> Any:
    child_process = process or _FakeProcess()
    return runtime_module._Child(
        process=cast(asyncio.subprocess.Process, child_process),
        endpoint="http://127.0.0.1:8123",
        config=_config(),
        generation=spec,
        installed_generation=installed,
        live_source=None,
        launch_nonce=launch_nonce,
        process_group=child_process.pid,
        process_birth="test-birth-identity",
        executable=installed.interpreter_path,
        argv=installed.entrypoint_argv,
        readers=(),
    )


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
        generation=None,
        installed_generation=None,
        live_source=spec,
        launch_nonce=launch_nonce,
        process_group=child_process.pid,
        process_birth="test-birth-identity",
        executable=executable,
        argv=(str(executable), "-P", "-m", "opentulpa"),
        readers=(),
    )


def _live_source_spec_with_environment(tmp_path: Path, *, commit: str = "d" * 40) -> RuntimeLiveSourceSpec:
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
        runtime_install_profile="runtime-no-dev-no-install-project-v1",
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
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "host-telegram-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "host-webhook-secret")
    monkeypatch.setenv("OPENTULPA_OWNER_CUSTOMER_ID", "opentulpa-gf")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv(
        "OPENTULPA_INTERNAL_AGENT_API_URL",
        "http://host.docker.internal:8000",
    )
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
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

    environment = runtime._child_environment(
        config,
        port=8123,
        installed_generation=installed,
    )
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
    assert environment["OPENTULPA_INTERNAL_AGENT_API_URL"] == (
        "http://host.docker.internal:8000"
    )
    assert environment["OPENTULPA_BOOTSTRAP_EVOLUTION_TOKEN"] == "e" * 48
    assert environment["OPENTULPA_BOOTSTRAP_EVOLUTION_URL"].endswith(
        "/bootstrap/internal/v1/evolution"
    )
    assert environment["OPENTULPA_SANDBOX_RPC_URL"].endswith("/internal/v1/sandbox")
    assert environment["OPENTULPA_SANDBOX_RPC_TOKEN"] == "s" * 48
    assert "PYTHONPATH" not in environment
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
    fallback_environment = runtime._child_environment(
        config,
        port=8124,
        installed_generation=installed,
    )
    assert fallback_environment["OPENTULPA_OWNER_CUSTOMER_ID"] == "owner"
    assert fallback_environment["OPENTULPA_INTERNAL_AGENT_API_URL"] == (
        "http://127.0.0.1:9000"
    )
    monkeypatch.delenv("PORT")
    assert runtime._child_environment(
        config,
        port=8125,
        installed_generation=installed,
    )[
        "OPENTULPA_INTERNAL_AGENT_API_URL"
    ] == (
        "http://127.0.0.1:8125"
    )
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_live_source_environment_loads_dotenv_without_leaking_host_owned_keys(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "src" / "opentulpa").mkdir(parents=True)
    (source / "src" / "opentulpa" / "__init__.py").write_text("", encoding="utf-8")
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

    environment = runtime._child_environment(
        _config(),
        port=8123,
        live_source=spec,
    )

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
    source = tmp_path / "source"
    (source / "src" / "opentulpa").mkdir(parents=True)
    (source / "src" / "opentulpa" / "__init__.py").write_text("", encoding="utf-8")
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
    assert result["value"] == "[redacted]"
    assert dotenv.read_text(encoding="utf-8") == "OPENAI_COMPATIBLE_API_KEY=previous\n"


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
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
    )
    previous = _fake_child(installed, spec)
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

    async def stop_child(child: Any) -> None:
        assert child is previous
        runtime._status = "recovery_required"
        raise RuntimeUnavailableError("stop failed")

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        del config, target
        raise AssertionError("stop failures must not spawn a replacement child")

    monkeypatch.setattr(runtime, "_ensure_controller_ownership", lambda: asyncio.sleep(0))
    monkeypatch.setattr(runtime, "_preflight_target", _no_op_async)
    monkeypatch.setattr(runtime, "_stop_child", stop_child)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)

    with pytest.raises(RuntimeUnavailableError, match="stop failed"):
        await runtime.replace_current_environment(apply=apply, restore=restore)

    assert applied is True
    assert restored is True
    assert runtime._child is previous
    assert runtime.status == "recovery_required"
    runtime._child = None
    await runtime.shutdown()


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
async def test_generation_runtime_uses_stable_host_railway_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = tmp_path / "railway_sandbox_bridge" / "bridge.mjs"
    bridge.parent.mkdir()
    bridge.write_text("", encoding="utf-8")
    installed, spec = _fake_generation(tmp_path)
    monkeypatch.setenv(
        "OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH",
        "/untrusted/inherited/bridge.mjs",
    )
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
    )

    environment = runtime._child_environment(
        _config(),
        port=8123,
        installed_generation=installed,
    )

    assert "PYTHONPATH" not in environment
    assert environment["OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH"] == str(bridge)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_host_passes_telegram_owner_identity_without_writing_child_state(
    tmp_path: Path,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    data_root = tmp_path / "runtime-data"
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=data_root,
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
    )

    environment = runtime._child_environment(
        _config(),
        port=8123,
        installed_generation=installed,
    )

    assert environment["OPENTULPA_TELEGRAM_OWNER_ID"] == "7"
    assert not data_root.exists()
    await runtime.shutdown()


def test_generation_spec_is_strict_persistable_and_constructible_from_release_metadata(
    tmp_path: Path,
) -> None:
    _, spec = _fake_generation(tmp_path)
    persisted = spec.model_dump(mode="json")

    assert RuntimeGenerationSpec.model_validate(persisted) == spec
    assert (
        RuntimeGenerationSpec.from_release_metadata(
            {
                "artifact_kind": "python_generation",
                "image_reference": f"python-generation:{spec.generation_id}",
                "manifest_digest": spec.manifest_digest,
                "state_contract_sha256": spec.state_contract_digest,
                "evaluator_fingerprint": spec.evaluator_fingerprint,
                "install_profile": spec.install_profile,
                "controller_protocol": spec.controller_protocol,
                "untrusted_extra": "ignored",
            }
        )
        == spec
    )
    with pytest.raises(ValidationError):
        RuntimeGenerationSpec.model_validate({**persisted, "controller_protocol": "1"})
    with pytest.raises(ValidationError):
        RuntimeGenerationSpec.model_validate({**persisted, "unexpected": True})


@pytest.mark.asyncio
async def test_runtime_refuses_to_start_without_a_runtime_target(tmp_path: Path) -> None:
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")

    with pytest.raises(RuntimeUnavailableError, match="no runtime generation or live source"):
        await runtime.start(_config())

    assert runtime.status == "stopped"
    assert runtime.endpoint is None
    await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("record_kind", ["ownership", "intent"])
async def test_legacy_runtime_control_records_fail_closed(
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
        "mode": "legacy",
        "generation_id": None,
        "legacy_source_root": "/tmp/legacy-source",
        "launch_nonce": "legacy-launch-nonce-00000000",
        "executable": "/usr/bin/python3",
        "argv": ["/usr/bin/python3", "-m", "opentulpa"],
    }
    if record_kind == "ownership":
        payload.update(
            {
                "pid": 2,
                "process_group": 2,
                "host_birth": "legacy-host-birth",
                "process_birth": "legacy-process-birth",
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
async def test_generation_launch_uses_exact_entrypoint_external_cwd_and_clean_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    store = _RecordingGenerationStore(installed)
    application_root = tmp_path / "application"
    data_root = tmp_path / "data"
    for key, value in {
        "PYTHONPATH": "/checkout/src",
        "PYTHONHOME": "/untrusted/python",
        "VIRTUAL_ENV": "/checkout/.venv",
        "CONDA_PREFIX": "/untrusted/conda",
        "PIP_EDITABLE": "/checkout",
        "UV_PROJECT_ENVIRONMENT": "/checkout/.venv",
        "OPENTULPA_SOURCE_ROOT": "/checkout",
        "LD_PRELOAD": "/tmp/inject.so",
        "DYLD_INSERT_LIBRARIES": "/tmp/inject.dylib",
        "GIT_CONFIG_SYSTEM": "/tmp/hostile-gitconfig",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
        "AWS_SECRET_ACCESS_KEY": "host-aws-secret",
        "DATABASE_URL": "postgres://host-secret",
        "GITHUB_TOKEN": "host-github-secret",
        "RAILWAY_TOKEN": "host-railway-secret",
        "OPENTULPA_PRIVATE_TOKEN": "host-private-secret",
        "SSL_CERT_FILE": "/tmp/hostile-ca.pem",
        "HOME": "/tmp/hostile-home",
        "PATH": "/tmp/hostile-bin",
    }.items():
        monkeypatch.setenv(key, value)
    runtime = RuntimeSupervisor(
        project_root=tmp_path / "host-source",
        data_root=data_root,
        application_root=application_root,
        generation_store=store,  # type: ignore[arg-type]
        generation_spec=spec,
        control_path=tmp_path / "control" / "runtime-child.json",
    )
    runtime._child_uid = None
    runtime._child_gid = None
    process = _FakeProcess()
    launch: dict[str, Any] = {}

    async def create_process(*argv: str, **options: Any) -> _FakeProcess:
        assert store.lock_held is True
        assert store.attempt_replacement() is False
        launch["argv"] = argv
        launch["options"] = options
        return process

    async def ready(child: Any) -> None:
        assert child.generation_id == spec.generation_id

    async def terminate(child: Any) -> None:
        child.process.exit(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runtime, "_wait_ready", ready)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)
    monkeypatch.setattr(runtime, "_capture_process_birth", lambda pid: f"test:{pid}")

    await runtime.start(_config())

    assert launch["argv"] == installed.entrypoint_argv
    assert launch["argv"][0] == str(installed.entrypoint_path)
    assert launch["argv"][0] != sys.executable
    options = launch["options"]
    assert options["cwd"] == application_root
    assert options["start_new_session"] is True
    assert "user" not in options
    assert "group" not in options
    environment = options["env"]
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PIP_EDITABLE",
        "UV_PROJECT_ENVIRONMENT",
        "OPENTULPA_SOURCE_ROOT",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "GIT_CONFIG_SYSTEM",
        "SSH_AUTH_SOCK",
        "AWS_SECRET_ACCESS_KEY",
        "DATABASE_URL",
        "GITHUB_TOKEN",
        "RAILWAY_TOKEN",
        "OPENTULPA_PRIVATE_TOKEN",
        "SSL_CERT_FILE",
    ):
        assert key not in environment
    assert environment["HOME"] == str(application_root)
    assert environment["PATH"].startswith(f"{installed.interpreter_path.parent}:")
    assert "/tmp/hostile-bin" not in environment["PATH"]
    assert environment["OPENTULPA_APPLICATION_ROOT"] == str(application_root)
    assert environment["OPENTULPA_DATA_ROOT"] == str(data_root)
    assert environment["OPENTULPA_GENERATION_ID"] == spec.generation_id
    assert environment["OPENTULPA_GENERATION_MANIFEST_DIGEST"] == spec.manifest_digest
    assert environment["OPENTULPA_GENERATION_SOURCE_COMMIT"] == (
        installed.manifest.identity.source_commit
    )
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    active_child = runtime._child
    assert active_child is not None
    assert environment["OPENTULPA_LAUNCH_NONCE"] == active_child.launch_nonce
    expected_open = (
        spec.generation_id,
        {
            "expected_manifest_digest": spec.manifest_digest,
            "expected_state_contract_digest": spec.state_contract_digest,
            "expected_evaluator_fingerprint": spec.evaluator_fingerprint,
            "expected_install_profile": spec.install_profile,
            "controller_protocol": spec.controller_protocol,
        },
    )
    assert store.calls == [expected_open, expected_open, expected_open]
    assert store.blocked_replacement_attempts == 1
    assert runtime.generation_id == spec.generation_id
    assert runtime.endpoint is not None
    assert any("integrity-only same-UID mode" in entry.text for entry in runtime.logs())
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_generation_readiness_rejects_identity_mismatch_unless_explicitly_disabled(
    tmp_path: Path,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    child = _fake_child(installed, spec)
    identity_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/_runtime/identity":
            identity_requests.append(request)
            return httpx.Response(
                200,
                json={"generation_id": "0" * 64, "launch_nonce": child.launch_nonce},
            )
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"generation_id": spec.generation_id})
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    strict_runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "strict-data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
        startup_timeout_seconds=0.01,
        client=client,
    )

    with pytest.raises(RuntimeUnavailableError, match="readiness timed out"):
        await strict_runtime._wait_ready(child)
    assert identity_requests
    assert all(
        request.headers["X-OpenTulpa-Launch-Nonce"] == child.launch_nonce
        for request in identity_requests
    )

    compatibility_runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "compat-data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
        strict_generation_readiness=False,
        client=client,
    )
    await compatibility_runtime._wait_ready(child)
    await strict_runtime.shutdown()
    await compatibility_runtime.shutdown()
    await client.aclose()


@pytest.mark.asyncio
async def test_evolution_event_delivery_targets_exact_serving_child_identity(
    tmp_path: Path,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    child = _fake_child(installed, spec, launch_nonce="event-launch-nonce-000000000000")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
        client=client,
    )
    runtime._child = child
    runtime._status = "ready"
    event = EvolutionEvent(
        event_key="candidate:candidate-1:failed",
        event_type="candidate.failed",
        candidate_id="candidate-1",
        payload={"status": "failed"},
    )

    await runtime.deliver_evolution_event(event)

    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/_runtime/evolution-events"
    assert request.headers["Authorization"] == "Bearer internal-owner-secret-value"
    assert request.headers["X-OpenTulpa-Launch-Nonce"] == child.launch_nonce
    assert EvolutionEvent.model_validate_json(request.content) == event
    runtime._child = None
    runtime._status = "stopped"
    await runtime.shutdown()
    await client.aclose()


@pytest.mark.asyncio
async def test_failed_generation_candidate_restores_exact_previous_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(  # type: ignore[arg-type]
            previous_installed,
            candidate_installed,
        ),
        generation_spec=previous_spec,
    )
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    launches: list[RuntimeGenerationSpec] = []

    async def stop_child(child: Any) -> None:
        child.requested_stop = True

    async def preflight_target(target: Any) -> None:
        del target

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        del config
        assert target.generation is not None
        launches.append(target.generation)
        if target.generation == candidate_spec:
            raise RuntimeUnavailableError("candidate is unhealthy")
        return _fake_child(previous_installed, target.generation)

    def adopt(child: Any) -> None:
        runtime._child = child
        runtime._status = "ready"

    monkeypatch.setattr(runtime, "_stop_child", stop_child)
    monkeypatch.setattr(runtime, "_preflight_target", preflight_target)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)
    monkeypatch.setattr(runtime, "_adopt_child", adopt)

    with pytest.raises(RuntimeUnavailableError, match="candidate is unhealthy"):
        await runtime.replace_generation(candidate_spec, rollback=previous_spec)

    assert launches == [candidate_spec, previous_spec]
    assert runtime.generation == previous_spec
    assert runtime._child.generation == previous_spec
    assert runtime.status == "ready"
    runtime._child = None
    await runtime.shutdown()


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
        assert target.live_source is not None
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
async def test_restart_current_uses_running_exact_generation_not_selected_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, running_spec = _fake_generation(tmp_path, "a")
    _, other_spec = _fake_generation(tmp_path, "b")
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=running_spec,
    )
    running = _fake_child(installed, running_spec)
    runtime._child = running
    runtime._selected_generation = other_spec
    targets: list[Any] = []

    async def replace_config(
        config: HostConfig,
        *,
        rollback: HostConfig | None,
        target: Any,
    ) -> None:
        assert config is running.config
        assert rollback is running.config
        targets.append(target)

    monkeypatch.setattr(runtime, "_replace_config_locked", replace_config)

    await runtime.restart_current()

    assert len(targets) == 1
    assert targets[0].generation == running_spec
    runtime._child = None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_concurrent_generation_replacements_are_serialized_and_hide_draining_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_installed, base_spec = _fake_generation(tmp_path, "a")
    first_installed, first_spec = _fake_generation(tmp_path, "b")
    second_installed, second_spec = _fake_generation(tmp_path, "c")
    installed_by_id = {
        base_spec.generation_id: base_installed,
        first_spec.generation_id: first_installed,
        second_spec.generation_id: second_installed,
    }
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(  # type: ignore[arg-type]
            base_installed,
            first_installed,
            second_installed,
        ),
        generation_spec=base_spec,
    )
    runtime._status = "ready"
    runtime._child = _fake_child(base_installed, base_spec)
    active_spawns = 0
    maximum_active_spawns = 0
    launches: list[str] = []
    endpoints_during_stop: list[str | None] = []

    async def stop_child(child: Any) -> None:
        del child
        endpoints_during_stop.append(runtime.endpoint)
        await asyncio.sleep(0.001)

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        nonlocal active_spawns, maximum_active_spawns
        del config
        assert target.generation is not None
        active_spawns += 1
        maximum_active_spawns = max(maximum_active_spawns, active_spawns)
        launches.append(target.generation.generation_id)
        await asyncio.sleep(0.01)
        active_spawns -= 1
        return _fake_child(installed_by_id[target.generation.generation_id], target.generation)

    def adopt(child: Any) -> None:
        runtime._child = child
        runtime._status = "ready"

    monkeypatch.setattr(runtime, "_stop_child", stop_child)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)
    monkeypatch.setattr(runtime, "_adopt_child", adopt)

    await asyncio.gather(
        runtime.replace_generation(first_spec),
        runtime.replace_generation(second_spec),
    )

    assert launches == [first_spec.generation_id, second_spec.generation_id]
    assert maximum_active_spawns == 1
    assert endpoints_during_stop == [None, None]
    assert runtime.generation == second_spec
    runtime._child = None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_process_group_shutdown_uses_term_then_kill_for_owned_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    process = _FakeProcess()
    child = _fake_child(installed, spec, process=process)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        shutdown_timeout_seconds=0.01,
    )
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = lambda leader_pid, launch_nonce: ()
    signals: list[tuple[int, signal.Signals]] = []

    def kill_group(process_group: int, selected_signal: int) -> None:
        if selected_signal == 0:
            if process.returncode is not None:
                raise ProcessLookupError
            return
        selected = signal.Signals(selected_signal)
        signals.append((process_group, selected))
        if selected == signal.SIGKILL:
            process.exit(-signal.SIGKILL)

    monkeypatch.setattr(runtime_module.sys, "platform", "darwin")
    monkeypatch.setattr(runtime_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(runtime_module.os, "killpg", kill_group)
    monkeypatch.setattr(runtime, "_capture_process_birth", lambda pid: child.process_birth)

    await runtime._terminate_child_process_group(child)

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_linux_leader_pid_reuse_fails_closed_without_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    child = _fake_child(installed, spec)
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = lambda leader_pid, launch_nonce: ()
    runtime._child = child
    runtime._status = "ready"
    inspections = 0
    pid_signals: list[tuple[int, int]] = []

    def inspect_process(pid: int) -> RuntimeProcessIdentity:
        nonlocal inspections
        inspections += 1
        return RuntimeProcessIdentity(
            pid=pid,
            process_group=child.process_group,
            executable=child.executable,
            argv=child.argv,
            process_birth=(child.process_birth if inspections == 1 else "reused-leader-birth"),
            launch_nonce=child.launch_nonce,
        )

    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(runtime_module.signal, "pidfd_send_signal", None, raising=False)
    monkeypatch.setattr(
        runtime_module.os,
        "killpg",
        lambda process_group, selected_signal: pytest.fail("Linux must not signal PGIDs"),
    )
    runtime._process_inspector = inspect_process
    runtime._process_signaler = lambda pid, selected_signal: pid_signals.append(
        (pid, selected_signal)
    )

    with pytest.raises(RuntimeUnavailableError, match="changed before PID signal"):
        await runtime._stop_child(child)

    assert pid_signals == []
    assert child.process.returncode is None
    assert runtime._child is child
    assert runtime.status == "recovery_required"

    async def complete_termination(selected: Any) -> None:
        selected.process.exit(0)

    monkeypatch.setattr(runtime, "_terminate_child_process_group", complete_termination)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_linux_descendant_pgid_reuse_fails_closed_without_any_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    child = _fake_child(installed, spec)
    cast(_FakeProcess, child.process).exit(21)
    expected = RuntimeProcessIdentity(
        pid=child.process.pid + 77,
        process_group=child.process_group,
        executable=child.executable,
        argv=(str(child.executable), "descendant"),
        process_birth="original-descendant-birth",
        launch_nonce=child.launch_nonce,
    )
    reused = RuntimeProcessIdentity(
        pid=expected.pid,
        process_group=expected.process_group,
        executable=expected.executable,
        argv=expected.argv,
        process_birth="reused-descendant-birth",
        launch_nonce=expected.launch_nonce,
    )
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = lambda leader_pid, launch_nonce: (expected,)
    runtime._process_inspector = lambda pid: reused
    runtime._child = child
    runtime._status = "ready"
    pid_signals: list[tuple[int, int]] = []
    runtime._process_signaler = lambda pid, selected_signal: pid_signals.append(
        (pid, selected_signal)
    )
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(
        runtime_module.os,
        "killpg",
        lambda process_group, selected_signal: pytest.fail("Linux must not signal PGIDs"),
    )

    with pytest.raises(RuntimeUnavailableError, match="changed before fencing"):
        await runtime._stop_child(child)

    assert pid_signals == []
    assert runtime._child is child
    assert runtime.status == "recovery_required"

    runtime._descendant_inspector = lambda leader_pid, launch_nonce: ()
    await runtime.shutdown()


def test_linux_process_executable_uses_argv_when_proc_link_is_permission_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "123"
    proc_root.mkdir()
    executable_link = proc_root / "exe"
    executable = tmp_path / "generation" / "venv" / "bin" / "python"
    original_resolve = Path.resolve

    def deny_proc_executable(path: Path, *, strict: bool = False) -> Path:
        if path == executable_link:
            raise PermissionError("cross-UID proc executable is hidden")
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", deny_proc_executable)

    observed = RuntimeSupervisor._linux_process_executable(
        proc_root,
        (str(executable), "-I", "-m", "opentulpa"),
    )

    assert observed == executable


@pytest.mark.asyncio
async def test_linux_verified_signal_prefers_pidfd_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    child = _fake_child(installed, spec)
    expected = runtime_module.RuntimeSupervisor._child_process_identity(child)
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    runtime._process_inspector = lambda pid: expected
    fallback_signals: list[int] = []
    runtime._process_signaler = lambda pid, selected_signal: fallback_signals.append(pid)
    pidfd_calls: list[tuple[int, signal.Signals]] = []
    descriptor = runtime_module.os.open("/dev/null", runtime_module.os.O_RDONLY)

    monkeypatch.setattr(runtime_module.os, "pidfd_open", lambda pid, flags: descriptor, raising=False)
    monkeypatch.setattr(
        runtime_module.signal,
        "pidfd_send_signal",
        lambda pidfd, selected_signal, siginfo, flags: pidfd_calls.append(
            (pidfd, signal.Signals(selected_signal))
        ),
        raising=False,
    )

    signaled = await runtime._signal_verified_identity(expected, signal.SIGTERM)

    assert signaled is True
    assert pidfd_calls == [(descriptor, signal.SIGTERM)]
    assert fallback_signals == []
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_requested_stop_does_not_restart_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
        restart_backoff_seconds=0,
        max_restart_backoff_seconds=0,
    )
    child = _fake_child(installed, spec)
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
async def test_unexpected_exit_restarts_exact_generation_with_bounded_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
        max_unexpected_restarts=1,
        restart_backoff_seconds=0,
        max_restart_backoff_seconds=0,
    )
    first = _fake_child(installed, spec)
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = lambda leader_pid, launch_nonce: ()
    runtime._begin_selection(first.config, first.target)
    runtime._status = "ready"
    runtime._adopt_child(first)
    launches: list[tuple[int, RuntimeGenerationSpec]] = []

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        assert target.generation is not None
        launches.append((config.revision, target.generation))
        replacement = _fake_child(installed, target.generation)
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
    assert runtime.generation == spec
    assert runtime.endpoint is None
    assert "restart budget" in str(runtime.error)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_ownership_record_fences_only_proven_process_identity(
    tmp_path: Path,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    control_path = tmp_path / "controller" / "runtime-child.json"
    owner = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "owner-data",
        control_path=control_path,
    )
    child = _fake_child(installed, spec, launch_nonce="launch-nonce-verified-000000000")
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
async def test_failed_termination_retains_ownership_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    control_path = tmp_path / "controller" / "runtime-child.json"
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        control_path=control_path,
    )
    child = _fake_child(installed, spec, launch_nonce="launch-nonce-retained-000000000")
    runtime._write_ownership_record(child)

    async def refuse_ambiguous_termination(selected: Any) -> None:
        del selected
        raise RuntimeUnavailableError("process group ownership is ambiguous")

    monkeypatch.setattr(
        runtime,
        "_terminate_child_process_group",
        refuse_ambiguous_termination,
    )

    with pytest.raises(RuntimeUnavailableError, match="ownership is ambiguous"):
        await runtime._stop_child(child)

    assert control_path.exists()
    runtime._remove_ownership_record(child.launch_nonce)

    async def complete_termination(selected: Any) -> None:
        selected.process.exit(0)

    monkeypatch.setattr(runtime, "_terminate_child_process_group", complete_termination)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_generation_child_identity_defaults_and_configuration_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="configured together"):
        RuntimeSupervisor(
            project_root=tmp_path,
            data_root=tmp_path / "missing-pair",
            child_uid=65532,
        )
    with pytest.raises(ValueError, match="differ from controller and root"):
        RuntimeSupervisor(
            project_root=tmp_path,
            data_root=tmp_path / "controller-identity",
            child_uid=runtime_module.os.geteuid(),
            child_gid=runtime_module.os.getegid(),
        )

    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime_module.os, "getegid", lambda: 0)
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "root-data")
    monkeypatch.setattr(
        runtime,
        "_identity_switch_executable",
        lambda: Path("/usr/bin/setpriv"),
    )

    spawn_argv = runtime._child_spawn_argv(("/runtime/python", "-m", "opentulpa"))

    assert spawn_argv == (
        "/usr/bin/setpriv",
        "--reuid=65532",
        "--regid=65532",
        "--clear-groups",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--bounding-set=-all",
        "--no-new-privs",
        "--",
        "/runtime/python",
        "-m",
        "opentulpa",
    )
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_generation_root_preflight_fails_before_stopping_old_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    data_root = tmp_path / "private-root"
    data_root.mkdir(mode=0o700)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=data_root,
        generation_store=_RecordingGenerationStore(  # type: ignore[arg-type]
            previous_installed,
            candidate_installed,
        ),
        generation_spec=previous_spec,
    )
    runtime._child_uid = 65532
    runtime._child_gid = 65532
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    stopped = False

    async def stop_child(child: Any) -> None:
        nonlocal stopped
        del child
        stopped = True

    monkeypatch.setattr(runtime, "_stop_child", stop_child)

    with pytest.raises(RuntimeUnavailableError, match="not usable by the runtime child"):
        await runtime.replace_generation(candidate_spec)

    assert stopped is False
    assert runtime._child is previous
    assert runtime.status == "ready"
    runtime._child = None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_generation_preflight_rejects_child_writable_store_before_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    store = _RecordingGenerationStore(previous_installed, candidate_installed)
    store.root.chmod(0o733)
    data_root = tmp_path / "child-data"
    application_root = tmp_path / "child-application"
    data_root.mkdir(mode=0o777)
    application_root.mkdir(mode=0o777)
    data_root.chmod(0o777)
    application_root.chmod(0o777)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=data_root,
        application_root=application_root,
        generation_store=store,  # type: ignore[arg-type]
        generation_spec=previous_spec,
    )
    runtime._child_uid = 65532
    runtime._child_gid = 65532
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    stopped = False

    async def stop_child(child: Any) -> None:
        nonlocal stopped
        del child
        stopped = True

    monkeypatch.setattr(runtime, "_stop_child", stop_child)
    monkeypatch.setattr(runtime, "_require_root_usable", lambda root, label: None)

    with pytest.raises(RuntimeUnavailableError, match="write a protected"):
        await runtime.replace_generation(candidate_spec)

    assert stopped is False
    assert runtime._child is previous
    assert runtime.status == "ready"
    runtime._child = None
    store.root.chmod(0o711)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_public_stop_retains_unfenced_child_and_blocks_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    child = _fake_child(installed, spec)
    runtime._child = child
    runtime._status = "ready"

    async def fail_termination(selected: Any) -> None:
        del selected
        raise RuntimeUnavailableError("owned process group could not be fenced")

    monkeypatch.setattr(runtime, "_terminate_child_process_group", fail_termination)

    with pytest.raises(RuntimeUnavailableError, match="could not be fenced"):
        await runtime.stop()

    assert runtime._child is child
    assert runtime.status == "recovery_required"
    with pytest.raises(RuntimeUnavailableError, match="recovery is required"):
        await runtime.start(_config())
    runtime._child = None
    runtime._status = "stopped"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_cancelled_generation_replacement_cleans_candidate_and_restores_exact_previous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    store = _RecordingGenerationStore(previous_installed, candidate_installed)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        application_root=tmp_path / "application",
        generation_store=store,  # type: ignore[arg-type]
        generation_spec=previous_spec,
        control_path=tmp_path / "control" / "runtime-child.json",
        restart_backoff_seconds=0,
    )
    runtime._child_uid = None
    runtime._child_gid = None
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    candidate_waiting = asyncio.Event()
    never_ready = asyncio.Event()
    spawned: list[_FakeProcess] = []

    async def create_process(*argv: str, **options: Any) -> _FakeProcess:
        del argv, options
        process = _FakeProcess()
        spawned.append(process)
        return process

    async def wait_ready(child: Any) -> None:
        if child.generation == candidate_spec:
            candidate_waiting.set()
            await never_ready.wait()

    async def terminate(child: Any) -> None:
        child.process.exit(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runtime, "_wait_ready", wait_ready)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)
    monkeypatch.setattr(runtime, "_capture_process_birth", lambda pid: f"test:{pid}")

    replacement = asyncio.create_task(runtime.replace_generation(candidate_spec))
    await candidate_waiting.wait()
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement

    assert len(spawned) == 2
    assert spawned[0].returncode == 0
    assert runtime._child is not None
    assert runtime._child.generation == previous_spec
    assert runtime.generation == previous_spec
    assert runtime._operation_task is None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_stop_interrupts_cancellation_rollback_and_cleans_its_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        application_root=tmp_path / "application",
        generation_store=_RecordingGenerationStore(  # type: ignore[arg-type]
            previous_installed,
            candidate_installed,
        ),
        generation_spec=previous_spec,
        control_path=tmp_path / "control" / "runtime-child.json",
    )
    runtime._child_uid = None
    runtime._child_gid = None
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    candidate_waiting = asyncio.Event()
    rollback_waiting = asyncio.Event()
    never_ready = asyncio.Event()
    spawned: list[_FakeProcess] = []

    async def create_process(*argv: str, **options: Any) -> _FakeProcess:
        del argv, options
        process = _FakeProcess()
        spawned.append(process)
        return process

    async def wait_ready(child: Any) -> None:
        if child.generation == candidate_spec:
            candidate_waiting.set()
        else:
            rollback_waiting.set()
        await never_ready.wait()

    async def terminate(child: Any) -> None:
        child.process.exit(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runtime, "_wait_ready", wait_ready)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)
    monkeypatch.setattr(runtime, "_capture_process_birth", lambda pid: f"test:{pid}")

    replacement = asyncio.create_task(runtime.replace_generation(candidate_spec))
    await candidate_waiting.wait()
    replacement.cancel()
    await rollback_waiting.wait()
    await runtime.stop()

    assert replacement.cancelled()
    assert len(spawned) == 2
    assert all(process.returncode == 0 for process in spawned)
    assert runtime._child is None
    assert runtime._operation_task is None
    assert runtime._watcher_task is None
    assert runtime.status == "stopped"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_failed_candidate_cleanup_blocks_rollback_and_shutdown_retains_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    control_path = tmp_path / "control" / "runtime-child.json"
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        application_root=tmp_path / "application",
        generation_store=_RecordingGenerationStore(  # type: ignore[arg-type]
            previous_installed,
            candidate_installed,
        ),
        generation_spec=previous_spec,
        control_path=control_path,
    )
    runtime._child_uid = None
    runtime._child_gid = None
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    candidate_waiting = asyncio.Event()
    never_ready = asyncio.Event()
    spawned: list[_FakeProcess] = []
    fail_candidate_cleanup = True

    async def create_process(*argv: str, **options: Any) -> _FakeProcess:
        del argv, options
        process = _FakeProcess()
        spawned.append(process)
        return process

    async def wait_ready(child: Any) -> None:
        del child
        candidate_waiting.set()
        await never_ready.wait()

    async def terminate(child: Any) -> None:
        if child.generation == candidate_spec and fail_candidate_cleanup:
            raise RuntimeUnavailableError("candidate containment is unproven")
        child.process.exit(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runtime, "_wait_ready", wait_ready)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)
    monkeypatch.setattr(runtime, "_capture_process_birth", lambda pid: f"test:{pid}")

    replacement = asyncio.create_task(runtime.replace_generation(candidate_spec))
    await candidate_waiting.wait()
    replacement.cancel()
    with pytest.raises(RuntimeUnavailableError, match="candidate containment is unproven"):
        await replacement

    assert len(spawned) == 1
    assert runtime.status == "recovery_required"
    assert runtime._child is not None
    assert runtime._child.generation == candidate_spec
    assert control_path.exists()
    assert runtime._owner_lock_descriptor is not None

    with pytest.raises(RuntimeUnavailableError, match="candidate containment is unproven"):
        await runtime.shutdown()
    assert runtime._owner_lock_descriptor is not None
    assert control_path.exists()
    assert runtime._child is not None

    fail_candidate_cleanup = False
    await runtime.shutdown()
    assert runtime._owner_lock_descriptor is None
    assert not control_path.exists()


@pytest.mark.asyncio
async def test_stop_cancels_startup_cleanup_without_detached_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        application_root=tmp_path / "application",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
        control_path=tmp_path / "control" / "runtime-child.json",
    )
    runtime._child_uid = None
    runtime._child_gid = None
    process = _FakeProcess()
    candidate_waiting = asyncio.Event()
    never_ready = asyncio.Event()

    async def create_process(*argv: str, **options: Any) -> _FakeProcess:
        del argv, options
        return process

    async def wait_ready(child: Any) -> None:
        del child
        candidate_waiting.set()
        await never_ready.wait()

    async def terminate(child: Any) -> None:
        child.process.exit(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runtime, "_wait_ready", wait_ready)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)
    monkeypatch.setattr(runtime, "_capture_process_birth", lambda pid: f"test:{pid}")

    startup = asyncio.create_task(runtime.start(_config()))
    await candidate_waiting.wait()
    await runtime.stop()

    assert startup.cancelled()
    assert process.returncode == 0
    assert runtime._child is None
    assert runtime._watcher_task is None
    assert runtime._operation_task is None
    assert runtime._desired_config is None
    assert runtime._desired_target is None
    assert runtime.status == "stopped"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_unexpected_leader_exit_fences_descendants_before_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
        shutdown_timeout_seconds=0.2,
        max_unexpected_restarts=1,
        restart_backoff_seconds=0,
        max_restart_backoff_seconds=0,
    )
    leader = _fake_child(installed, spec)
    runtime._begin_selection(leader.config, leader.target)
    runtime._status = "ready"
    runtime._adopt_child(leader)
    group_alive = True
    escaped_alive = True
    group_signals: list[signal.Signals] = []
    descendant_signals: list[signal.Signals] = []
    events: list[str] = []
    restarted_after_fence = False
    escaped = RuntimeProcessIdentity(
        pid=leader.process.pid + 100,
        process_group=leader.process.pid + 100,
        executable=installed.interpreter_path,
        argv=(str(installed.interpreter_path), "escaped"),
        parent_pid=runtime_module.os.getpid(),
        process_birth="escaped-birth",
        launch_nonce=leader.launch_nonce,
    )

    def kill_group(process_group: int, selected_signal: int) -> None:
        nonlocal group_alive
        assert process_group == leader.process_group
        if selected_signal == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        selected = signal.Signals(selected_signal)
        group_signals.append(selected)
        if selected == signal.SIGKILL:
            group_alive = False

    def inspect_descendants(
        leader_pid: int,
        launch_nonce: str,
    ) -> tuple[RuntimeProcessIdentity, ...]:
        assert leader_pid == leader.process.pid
        assert launch_nonce == leader.launch_nonce
        events.append("enumerate")
        return (escaped,) if escaped_alive else ()

    def inspect_process(pid: int) -> RuntimeProcessIdentity | None:
        events.append("revalidate")
        return escaped if pid == escaped.pid and escaped_alive else None

    def signal_process(pid: int, selected_signal: int) -> None:
        nonlocal escaped_alive
        assert pid == escaped.pid
        selected = signal.Signals(selected_signal)
        descendant_signals.append(selected)
        events.append(f"signal-{selected.name}")
        if selected == signal.SIGKILL:
            escaped_alive = False

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        nonlocal restarted_after_fence
        del config
        assert target.generation == spec
        events.append("restart")
        restarted_after_fence = not escaped_alive and leader.process.returncode is not None
        runtime._status = "ready"
        return _fake_child(installed, spec)

    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "killpg", kill_group)
    runtime._subreaper_enabled = True
    runtime._descendant_inspector = inspect_descendants
    runtime._uses_default_descendant_inspector = False
    runtime._process_inspector = inspect_process
    runtime._process_fencer = lambda identity, selected: signal_process(identity.pid, selected)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)

    cast(_FakeProcess, leader.process).exit(19)
    await _wait_until(lambda: runtime._child is not None and runtime._child is not leader)

    assert group_signals == []
    assert descendant_signals == [signal.SIGTERM, signal.SIGKILL]
    assert events.index("enumerate") < events.index("revalidate")
    assert events.index("revalidate") < events.index("signal-SIGTERM")
    assert events.index("signal-SIGKILL") < events.index("restart")
    assert restarted_after_fence is True

    async def terminate_replacement(child: Any) -> None:
        child.process.exit(0)

    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate_replacement)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_owner_flock_excludes_second_controller_until_shutdown(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "controller" / "runtime-child.json"
    first = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "first-data",
        control_path=control_path,
    )
    second = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "second-data",
        control_path=control_path,
    )

    first._acquire_owner_lock()
    with pytest.raises(RuntimeUnavailableError, match="another runtime controller"):
        second._acquire_owner_lock()

    await first.shutdown()
    second._acquire_owner_lock()
    await second.shutdown()


@pytest.mark.asyncio
async def test_orphan_fencing_requires_exact_nonce_birth_and_second_inspection(
    tmp_path: Path,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    control_path = tmp_path / "controller" / "runtime-child.json"
    owner = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "owner-data",
        control_path=control_path,
    )
    child = _fake_child(installed, spec, launch_nonce="launch-nonce-proof-000000000000")
    owner._write_ownership_record(child)
    inspections = 0
    fenced = False

    def inspect_process(pid: int) -> RuntimeProcessIdentity:
        nonlocal inspections
        inspections += 1
        return RuntimeProcessIdentity(
            pid=pid,
            process_group=child.process_group,
            executable=child.executable,
            argv=child.argv,
            process_birth=child.process_birth if inspections == 1 else "reused-process-birth",
            launch_nonce=child.launch_nonce,
        )

    def fence_process(
        identity: RuntimeProcessIdentity,
        selected_signal: signal.Signals,
    ) -> None:
        nonlocal fenced
        del identity, selected_signal
        fenced = True

    recovering = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "recover-data",
        control_path=control_path,
        process_inspector=inspect_process,
        process_fencer=fence_process,
    )

    with pytest.raises(RuntimeUnavailableError, match="changed before fencing"):
        await recovering._ensure_orphan_fenced()
    assert inspections == 2
    assert fenced is False
    assert control_path.exists()

    owner._remove_ownership_record(child.launch_nonce)
    await owner.shutdown()
    await recovering.shutdown()


@pytest.mark.asyncio
async def test_reused_controller_pid_does_not_block_birth_verified_orphan_fencing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    control_path = tmp_path / "controller" / "runtime-child.json"
    owner = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "owner-data",
        control_path=control_path,
    )
    child = _fake_child(installed, spec, launch_nonce="launch-nonce-controller-birth-000")
    owner._write_ownership_record(child)
    record = owner._read_ownership_record()
    assert record is not None
    reused_pid_record = record.model_copy(
        update={"host_pid": 4242, "host_birth": "old-controller-birth"}
    )
    owner._atomic_write_control_file(
        control_path,
        reused_pid_record.model_dump_json().encode("utf-8"),
        label="ownership record",
    )
    fenced: list[int] = []

    process_alive = True

    def inspect_process(pid: int) -> RuntimeProcessIdentity | None:
        assert pid == child.process.pid
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
        fenced.append(identity.pid)
        assert selected_signal == signal.SIGTERM
        process_alive = False

    recovering = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "recover-data",
        control_path=control_path,
        process_inspector=inspect_process,
        process_fencer=fence_process,
        descendant_inspector=lambda _pid, _nonce: (),
    )
    monkeypatch.setattr(
        recovering,
        "_capture_process_birth",
        lambda pid: "reused-controller-birth" if pid == 4242 else f"birth:{pid}",
    )

    await recovering._ensure_controller_ownership()

    assert fenced == [child.process.pid]
    assert not control_path.exists()
    await owner.shutdown()
    await recovering.shutdown()


@pytest.mark.asyncio
async def test_generation_retry_uses_new_port_and_nonce_only_after_exact_child_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        application_root=tmp_path / "application",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
        control_path=tmp_path / "control" / "runtime-child.json",
    )
    runtime._child_uid = None
    runtime._child_gid = None
    processes: list[_FakeProcess] = []
    environments: list[dict[str, str]] = []
    original_wait_ready = runtime._wait_ready

    async def create_process(*argv: str, **options: Any) -> _FakeProcess:
        del argv
        process = _FakeProcess()
        processes.append(process)
        environments.append(options["env"])
        return process

    async def wait_ready(child: Any) -> None:
        if len(processes) == 1:
            child.process.exit(98)
            await original_wait_ready(child)

    async def terminate(child: Any) -> None:
        if child.process.returncode is None:
            child.process.exit(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runtime, "_wait_ready", wait_ready)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", terminate)
    monkeypatch.setattr(runtime, "_capture_process_birth", lambda pid: f"test:{pid}")

    await runtime.start(_config())

    assert len(processes) == 2
    assert environments[0]["PORT"] != environments[1]["PORT"]
    assert environments[0]["OPENTULPA_LAUNCH_NONCE"] != environments[1][
        "OPENTULPA_LAUNCH_NONCE"
    ]
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_mismatched_generation_rollback_is_rejected_before_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    _, wrong_rollback = _fake_generation(tmp_path, "c")
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_store=_RecordingGenerationStore(  # type: ignore[arg-type]
            previous_installed,
            candidate_installed,
        ),
        generation_spec=previous_spec,
    )
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    stopped = False

    async def stop_child(child: Any) -> None:
        nonlocal stopped
        del child
        stopped = True

    monkeypatch.setattr(runtime, "_stop_child", stop_child)

    with pytest.raises(RuntimeUnavailableError, match="does not match"):
        await runtime.replace_generation(candidate_spec, rollback=wrong_rollback)

    assert stopped is False
    assert runtime._child is previous
    runtime._child = None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_mismatched_configuration_rollback_is_rejected_before_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        application_root=tmp_path / "application",
        generation_store=_RecordingGenerationStore(installed),  # type: ignore[arg-type]
        generation_spec=spec,
    )
    runtime._child_uid = None
    runtime._child_gid = None
    previous = _fake_child(installed, spec)
    runtime._child = previous
    runtime._status = "ready"
    stopped = False

    async def stop_child(child: Any) -> None:
        nonlocal stopped
        del child
        stopped = True

    monkeypatch.setattr(runtime, "_stop_child", stop_child)
    mismatched = previous.config.model_copy(update={"model": "different-model"})
    candidate = previous.config.model_copy(update={"revision": previous.config.revision + 1})

    with pytest.raises(RuntimeUnavailableError, match="rollback configuration does not match"):
        await runtime.replace(candidate, rollback=mismatched)

    assert stopped is False
    assert runtime._child is previous
    runtime._child = None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_launch_intent_and_missing_nonce_both_fail_closed(
    tmp_path: Path,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    control_path = tmp_path / "controller" / "runtime-child.json"
    owner = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "owner-data",
        control_path=control_path,
    )
    owner._write_launch_intent(
        generation=spec,
        launch_nonce="launch-intent-incomplete-00000000",
        executable=installed.interpreter_path,
        argv=installed.entrypoint_argv,
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

    child = _fake_child(installed, spec, launch_nonce="launch-nonce-must-be-observed-0000")
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
async def test_corrupt_generation_is_rejected_before_process_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, spec, interpreter = _published_generation(tmp_path)
    interpreter.chmod(0o755)
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        application_root=tmp_path / "application",
        generation_store=store,
        generation_spec=spec,
        startup_timeout_seconds=0.01,
    )
    process_created = False

    async def create_process(*argv: str, **options: Any) -> _FakeProcess:
        nonlocal process_created
        del argv, options
        process_created = True
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    with pytest.raises(RuntimeUnavailableError, match="not published read-only"):
        await runtime.start(_config())

    assert process_created is False
    assert runtime.status == "failed"
    assert runtime.endpoint is None
    await runtime.shutdown()


@pytest.mark.parametrize("replacement_kind", ["target", "config"])
@pytest.mark.asyncio
async def test_startup_failure_rollback_waits_through_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / replacement_kind,
        generation_store=_RecordingGenerationStore(previous_installed, candidate_installed),  # type: ignore[arg-type]
        generation_spec=previous_spec,
    )
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    respawn_started = asyncio.Event()
    release_respawn = asyncio.Event()
    candidate_config = previous.config.model_copy(update={"revision": 2})
    launches: list[Any] = []

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        launches.append((config, target))
        if len(launches) == 1:
            raise RuntimeUnavailableError("candidate startup failed")
        respawn_started.set()
        await release_respawn.wait()
        restored = _fake_child(previous_installed, previous_spec)
        restored.config = config
        runtime._status = "ready"
        return restored

    monkeypatch.setattr(runtime, "_ensure_controller_ownership", lambda: asyncio.sleep(0))
    monkeypatch.setattr(runtime, "_preflight_target", _no_op_async)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", _terminate_fake_child)
    operation = (
        runtime.replace_generation(candidate_spec)
        if replacement_kind == "target"
        else runtime.replace(candidate_config, rollback=previous.config)
    )
    task = asyncio.create_task(operation)
    await respawn_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_respawn.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime._child is not None
    assert runtime._child.generation == previous_spec
    assert runtime._child.config == previous.config
    assert runtime.status == "ready"
    assert runtime._unexpected_restarts == 0
    await runtime.shutdown()


@pytest.mark.parametrize("retains_nonce", [True, False])
@pytest.mark.asyncio
async def test_linux_discovery_finds_pid1_new_group_runtime_uid_survivor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retains_nonce: bool,
) -> None:
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 0)
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    survivor_pid = 62001
    nonce = "survivor-runtime-nonce-00000000"
    table = {
        survivor_pid: runtime_module._LinuxProcessMetadata(
            parent_pid=1,
            process_group=survivor_pid,
            process_birth="linux:survivor",
            proc_uid=65532,
            status_uids=(65532, 65532, 65532, 65532),
        ),
        62002: runtime_module._LinuxProcessMetadata(
            parent_pid=1,
            process_group=62002,
            process_birth="linux:root-helper",
            proc_uid=0,
            status_uids=(0, 0, 0, 0),
        ),
    }
    identity = RuntimeProcessIdentity(
        pid=survivor_pid,
        process_group=survivor_pid,
        executable=tmp_path / "runtime",
        argv=(str(tmp_path / "runtime"),),
        parent_pid=1,
        process_birth="linux:survivor",
        launch_nonce=nonce if retains_nonce else None,
    )
    runtime._process_inspector = lambda pid: identity
    monkeypatch.setattr(runtime, "_linux_process_table", lambda: table)
    if retains_nonce:
        assert runtime._inspect_linux_descendants(61000, nonce) == (identity,)
    else:
        with pytest.raises(RuntimeUnavailableError, match="removed its launch identity"):
            runtime._inspect_linux_descendants(61000, nonce)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_nonroot_linux_discovery_finds_reparented_same_uid_nonce_survivor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module.os, "geteuid", lambda: 501)
    runtime = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "data")
    nonce = "nonroot-survivor-runtime-nonce-00"
    survivor_pid = 63001
    unrelated_pid = 63002
    table = {
        survivor_pid: runtime_module._LinuxProcessMetadata(
            parent_pid=1,
            process_group=survivor_pid,
            process_birth="linux:survivor",
            proc_uid=501,
            status_uids=(501, 501, 501, 501),
        ),
        unrelated_pid: runtime_module._LinuxProcessMetadata(
            parent_pid=1,
            process_group=unrelated_pid,
            process_birth="linux:unrelated",
            proc_uid=501,
            status_uids=(501, 501, 501, 501),
        ),
    }
    survivor = RuntimeProcessIdentity(
        pid=survivor_pid,
        process_group=survivor_pid,
        executable=tmp_path / "runtime",
        argv=(str(tmp_path / "runtime"),),
        parent_pid=1,
        process_birth="linux:survivor",
        launch_nonce=nonce,
    )
    unrelated = RuntimeProcessIdentity(
        pid=unrelated_pid,
        process_group=unrelated_pid,
        executable=tmp_path / "other",
        argv=(str(tmp_path / "other"),),
        parent_pid=1,
        process_birth="linux:unrelated",
        launch_nonce=None,
    )
    identities = {survivor_pid: survivor, unrelated_pid: unrelated}
    runtime._process_inspector = identities.get
    monkeypatch.setattr(runtime, "_linux_process_table", lambda: table)

    assert runtime._inspect_linux_descendants(62000, nonce) == (survivor,)
    identities[survivor_pid] = RuntimeProcessIdentity(
        pid=survivor_pid,
        process_group=survivor_pid,
        executable=survivor.executable,
        argv=survivor.argv,
        parent_pid=1,
        process_birth="linux:survivor",
        launch_nonce=None,
    )
    with pytest.raises(RuntimeUnavailableError, match="removed its launch identity"):
        runtime._inspect_linux_descendants(
            62000,
            nonce,
            expected_executable=survivor.executable,
            expected_argv=survivor.argv,
        )
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_absent_leader_without_descendants_clears_stale_ownership_and_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, spec = _fake_generation(tmp_path)
    control_path = tmp_path / "control" / "runtime-child.json"
    owner = RuntimeSupervisor(project_root=tmp_path, data_root=tmp_path / "owner-data", control_path=control_path)
    stale = _fake_child(installed, spec, launch_nonce="stale-runtime-nonce-000000000")
    owner._write_ownership_record(stale)
    owner._write_launch_intent(
        generation=spec,
        launch_nonce=stale.launch_nonce,
        executable=stale.executable,
        argv=stale.argv,
    )
    recovering = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "recovering-data",
        control_path=control_path,
        process_inspector=lambda pid: None,
        descendant_inspector=lambda pid, nonce: (),
    )
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(
        recovering,
        "_enable_child_subreaper",
        lambda: setattr(recovering, "_subreaper_enabled", True),
    )
    await recovering._ensure_orphan_fenced()
    assert recovering._ownership_checked is True
    assert not control_path.exists()
    assert not recovering._intent_path.exists()
    await recovering.shutdown()
    await owner.shutdown()


@pytest.mark.asyncio
async def test_candidate_is_routed_during_probation_and_commits_after_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_installed, previous_spec = _fake_generation(tmp_path, "a")
    candidate_installed, candidate_spec = _fake_generation(tmp_path, "b")
    requests: list[httpx.Request] = []

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(204)
        )
    )
    runtime = RuntimeSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "probation",
        generation_store=_RecordingGenerationStore(previous_installed, candidate_installed),  # type: ignore[arg-type]
        generation_spec=previous_spec,
        probation_seconds=1,
        client=client,
    )
    previous = _fake_child(previous_installed, previous_spec)
    runtime._child = previous
    runtime._status = "ready"
    entered = asyncio.Event()
    release = asyncio.Event()

    async def spawn_target(config: HostConfig, target: Any) -> Any:
        child = _fake_child(candidate_installed, candidate_spec)
        child.config = config
        child.endpoint = "http://candidate.runtime"
        return child

    async def probe(child: Any) -> None:
        assert child.generation == candidate_spec
        entered.set()
        await release.wait()

    monkeypatch.setattr(runtime, "_ensure_controller_ownership", lambda: asyncio.sleep(0))
    monkeypatch.setattr(runtime, "_preflight_target", _no_op_async)
    monkeypatch.setattr(runtime, "_spawn_target", spawn_target)
    monkeypatch.setattr(runtime, "_probe_child_readiness", probe)
    monkeypatch.setattr(runtime, "_terminate_child_process_group", _terminate_fake_child)
    replacement = asyncio.create_task(runtime.replace_generation(candidate_spec))
    await entered.wait()
    assert runtime.status == "probation"
    assert runtime.endpoint == "http://candidate.runtime"
    event = EvolutionEvent(
        event_key="candidate:probation:event",
        event_type="candidate.passed",
        candidate_id="candidate",
        payload={},
    )
    await runtime.deliver_evolution_event(event)
    assert requests[0].url.host == "candidate.runtime"
    release.set()
    await replacement
    assert runtime.status == "ready"
    assert runtime.generation == candidate_spec
    await runtime.shutdown()
    await client.aclose()
