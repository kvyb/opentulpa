from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.deep_agent import sandbox
from opentulpa.deep_agent.sandbox import (
    TenantContainerBackend,
    TenantContainerPolicy,
    TenantSandboxBackend,
)


def _successful_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    del kwargs
    return subprocess.CompletedProcess(argv, 0, stdout=b"ok")


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _mounted_workspace(argv: list[str]) -> Path:
    mount = _option(argv, "--mount")
    source = next(part.removeprefix("src=") for part in mount.split(",") if part.startswith("src="))
    return Path(source)


def test_policy_rejects_unsafe_image_and_accepts_explicit_network() -> None:
    with pytest.raises(ValueError, match="OCI image"):
        TenantContainerPolicy(image="--privileged")
    assert TenantContainerPolicy(network_enabled=True).network_enabled is True


def test_persistent_container_launch_is_least_privilege_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = list(argv)
        captured["kwargs"] = dict(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=b"x" * 512)

    monkeypatch.setattr(sandbox.subprocess, "run", _run)
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
        policy=TenantContainerPolicy(
            timeout_seconds=7,
            max_output_bytes=1_024,
            pid_limit=32,
            network_enabled=True,
        ),
    )

    response = backend.execute("printf hello", timeout=999)

    argv = captured["argv"]
    kwargs = captured["kwargs"]
    assert response.exit_code == 0
    assert response.truncated is False
    assert len(response.output.encode()) == 512
    assert kwargs["timeout"] == 7
    assert kwargs["env"] == {"PATH": sandbox.os.environ.get("PATH", "")}
    assert "--rm" in argv
    container_name = _option(argv, "--name")
    assert container_name.startswith(f"opentulpa-sbx-{backend.workspace.name}-")
    assert all(character.islower() or character.isdigit() or character in "_.-" for character in container_name)
    assert "--init" in argv
    assert "--read-only" in argv
    assert _option(argv, "--pull") == "never"
    assert _option(argv, "--security-opt") == "no-new-privileges:true"
    assert _option(argv, "--cap-drop") == "ALL"
    assert _option(argv, "--network") == "bridge"
    assert _option(argv, "--ipc") == "none"
    assert _option(argv, "--cpus") == "1"
    assert _option(argv, "--memory") == "512m"
    assert _option(argv, "--memory-swap") == "512m"
    assert _option(argv, "--pids-limit") == "32"
    assert _option(argv, "--workdir") == "/workspace"
    assert not _option(argv, "--user").startswith("0:")
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    mounted_workspace = _mounted_workspace(argv)
    assert mounted_workspace != backend.workspace
    assert mounted_workspace.parent.name.startswith(f".{backend.workspace.name}.transaction-")
    assert not mounted_workspace.exists()
    assert mounts == [
        f"type=bind,src={mounted_workspace},dst=/workspace,bind-propagation=rprivate"
    ]
    assert not mounted_workspace.is_relative_to(sandbox._REPOSITORY_ROOT)  # noqa: SLF001
    assert ".env" not in mounts[0]
    assert "docker.sock" not in mounts[0]


def test_scratch_execution_has_no_host_mount_and_leaves_no_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok")

    monkeypatch.setattr(sandbox.subprocess, "run", _run)
    monkeypatch.setattr(
        sandbox,
        "get_runtime",
        lambda: SimpleNamespace(context=SimpleNamespace(tenant_id="tenant-a")),
    )
    root = tmp_path / "workspaces"
    backend = TenantSandboxBackend(workspaces_root=root, persistent_files=False)

    assert backend.execute("touch scratch.txt").exit_code == 0
    assert backend.execute("test ! -e prior-run.txt").exit_code == 0

    assert len(captured) == 2
    for argv in captured:
        assert "--rm" in argv
        assert "--mount" not in argv
        tmpfs_values = [argv[index + 1] for index, value in enumerate(argv) if value == "--tmpfs"]
        assert any(value.startswith("/workspace:rw,nosuid,nodev,") for value in tmpfs_values)
    assert not root.exists()


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/safe/../../outside.txt",
        "safe\\..\\outside.txt",
    ],
)
def test_workspace_file_operations_reject_traversal(
    tmp_path: Path,
    path: str,
) -> None:
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
    )

    with pytest.raises(ValueError, match="workspace path"):
        backend.write(path, "secret")

    assert not (tmp_path / "outside.txt").exists()


def test_symlink_and_special_file_make_workspace_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        del kwargs
        calls += 1
        return subprocess.CompletedProcess(argv, 0, stdout=b"unsafe")

    monkeypatch.setattr(sandbox.subprocess, "run", _run)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_backend = TenantContainerBackend(
        tenant_id="tenant-symlink",
        workspaces_root=tmp_path / "workspaces",
    )
    (symlink_backend.workspace / "escape").symlink_to(outside, target_is_directory=True)

    symlink_result = symlink_backend.execute("cat escape/secret")

    assert symlink_result.exit_code == 126
    assert symlink_result.output == "workspace failed sandbox security validation"
    assert calls == 0

    special_file_backend = TenantContainerBackend(
        tenant_id="tenant-socket",
        workspaces_root=tmp_path / "workspaces",
    )
    os.mkfifo(special_file_backend.workspace / "agent.pipe")
    special_file_result = special_file_backend.execute("true")

    assert special_file_result.exit_code == 126
    assert calls == 0


@pytest.mark.parametrize("poison", ["symlink", "hardlink", "oversized", "fifo"])
def test_invalid_command_workspace_is_discarded_and_next_command_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    calls = 0
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        del kwargs
        calls += 1
        staged = _mounted_workspace(argv)
        if calls == 2:
            (staged / "recovered.txt").write_text("safe", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout=b"recovered")
        if poison == "symlink":
            (staged / "escape").symlink_to(outside)
        elif poison == "hardlink":
            os.link(staged / "notes.txt", staged / "notes-alias.txt")
        elif poison == "oversized":
            (staged / "too-large.bin").write_bytes(b"x" * 65)
        else:
            os.mkfifo(staged / "agent.pipe")
        return subprocess.CompletedProcess(argv, 0, stdout=b"unsafe")

    monkeypatch.setattr(sandbox.subprocess, "run", _run)
    root = tmp_path / "workspaces"
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=root,
        policy=TenantContainerPolicy(max_file_bytes=64),
    )
    backend.write("/notes.txt", "original")

    rejected = backend.execute("create invalid workspace entry")

    assert rejected == sandbox.ExecuteResponse(
        output="workspace failed sandbox security validation",
        exit_code=126,
        truncated=False,
    )
    assert (backend.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert sorted(path.name for path in backend.workspace.iterdir()) == ["notes.txt"]
    assert outside.read_text(encoding="utf-8") == "outside"

    recovered = backend.execute("create safe file")

    assert recovered.exit_code == 0
    assert (backend.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert (backend.workspace / "recovered.txt").read_text(encoding="utf-8") == "safe"
    assert not any(".transaction-" in path.name for path in root.iterdir())


def test_commit_validation_failure_atomically_restores_previous_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        del kwargs
        calls += 1
        staged = _mounted_workspace(argv)
        name = "candidate.txt" if calls == 1 else "recovered.txt"
        (staged / name).write_text("new", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok")

    monkeypatch.setattr(sandbox.subprocess, "run", _run)
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
    )
    backend.write("/notes.txt", "original")
    validate = backend._validate_workspace_tree  # noqa: SLF001
    rejected_once = False

    def reject_promoted_candidate(workspace: Path | None = None) -> None:
        nonlocal rejected_once
        validate(workspace)
        if (
            workspace is None
            and not rejected_once
            and (backend.workspace / "candidate.txt").exists()
        ):
            rejected_once = True
            raise sandbox._WorkspaceSecurityError("injected commit validation failure")  # noqa: SLF001

    monkeypatch.setattr(backend, "_validate_workspace_tree", reject_promoted_candidate)

    rejected = backend.execute("candidate")

    assert rejected.exit_code == 126
    assert (backend.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not (backend.workspace / "candidate.txt").exists()

    recovered = backend.execute("recover")

    assert recovered.exit_code == 0
    assert (backend.workspace / "recovered.txt").read_text(encoding="utf-8") == "new"


def test_commit_authority_rejection_never_promotes_staged_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        (_mounted_workspace(argv) / "candidate.txt").write_text("unsafe", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok")

    checks = 0

    @contextmanager
    def reject_commit() -> Iterator[None]:
        nonlocal checks
        checks += 1
        raise sandbox._WorkspaceSecurityError("lease revoked")  # noqa: SLF001
        yield

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
        commit_authority=reject_commit,
    )
    backend.write("/notes.txt", "original")

    rejected = backend.execute("mutate")

    assert rejected.exit_code == 126
    assert checks == 1
    assert (backend.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not (backend.workspace / "candidate.txt").exists()


def test_workspace_root_and_preexisting_tenant_symlinks_are_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        TenantContainerBackend(tenant_id="tenant-a", workspaces_root=linked_root)

    digest = hashlib.sha256(b"tenant-a").hexdigest()[:24]
    outside = tmp_path / "outside"
    outside.mkdir()
    (real_root / digest).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        TenantContainerBackend(tenant_id="tenant-a", workspaces_root=real_root)


def test_tenant_workspaces_are_isolated_and_persist_across_backend_instances(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspaces"
    tenant_a = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    tenant_b = TenantContainerBackend(tenant_id="tenant-b", workspaces_root=root)
    assert tenant_a.write("/notes.txt", "a").path == "/notes.txt"
    assert tenant_b.write("/notes.txt", "b").path == "/notes.txt"

    tenant_a_again = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)

    assert tenant_a.workspace != tenant_b.workspace
    assert (tenant_a_again.workspace / "notes.txt").read_text() == "a"
    assert (tenant_b.workspace / "notes.txt").read_text() == "b"
    with pytest.raises(ValueError, match="traversal"):
        tenant_a.read(f"/../{tenant_b.workspace.name}/notes.txt")


def test_same_tenant_execution_is_serialized_across_backend_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        nonlocal active, maximum_active
        del kwargs
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok")

    monkeypatch.setattr(sandbox.subprocess, "run", _run)
    root = tmp_path / "workspaces"
    first = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    second = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda backend: backend.execute("true"), [first, second]))

    assert [result.exit_code for result in results] == [0, 0]
    assert maximum_active == 1


def test_failed_transaction_does_not_block_or_mutate_another_tenant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendezvous = threading.Barrier(2)

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        staged = _mounted_workspace(argv)
        rendezvous.wait(timeout=2)
        if argv[-1] == "poison":
            os.mkfifo(staged / "agent.pipe")
        else:
            (staged / "safe.txt").write_text("tenant-b", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok")

    monkeypatch.setattr(sandbox.subprocess, "run", _run)
    root = tmp_path / "workspaces"
    tenant_a = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    tenant_b = TenantContainerBackend(tenant_id="tenant-b", workspaces_root=root)
    tenant_a.write("/notes.txt", "a")
    tenant_b.write("/notes.txt", "b")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(tenant_a.execute, "poison")
        future_b = executor.submit(tenant_b.execute, "safe")
        result_a = future_a.result()
        result_b = future_b.result()

    assert result_a.exit_code == 126
    assert result_b.exit_code == 0
    assert (tenant_a.workspace / "notes.txt").read_text(encoding="utf-8") == "a"
    assert not (tenant_a.workspace / "agent.pipe").exists()
    assert (tenant_b.workspace / "notes.txt").read_text(encoding="utf-8") == "b"
    assert (tenant_b.workspace / "safe.txt").read_text(encoding="utf-8") == "tenant-b"


def test_timeout_output_is_capped_and_timeout_cannot_exceed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_timeout = 0

    def _timeout(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        nonlocal captured_timeout
        if len(argv) > 1 and argv[1] == "rm":
            return subprocess.CompletedProcess(argv, 0, stdout=b"")
        if len(argv) > 1 and argv[1] == "ps":
            return subprocess.CompletedProcess(argv, 0, stdout=b"")
        captured_timeout = int(kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, captured_timeout, output=b"x" * 500)

    monkeypatch.setattr(sandbox.subprocess, "run", _timeout)
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
        policy=TenantContainerPolicy(timeout_seconds=3, max_output_bytes=1_024),
    )

    response = backend.execute("sleep 10", timeout=120)

    assert captured_timeout == 3
    assert response.exit_code == 124
    assert response.truncated is False
    assert len(response.output.encode()) <= 1_024
    assert response.output.endswith("command timed out after 3s")


def test_runtime_env_workspace_content_is_mounted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        payload = (_mounted_workspace(argv) / ".env").read_bytes()
        return subprocess.CompletedProcess(argv, 0, stdout=payload)

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
    )
    (backend.workspace / ".env").write_text("TOKEN=test", encoding="utf-8")

    response = backend.execute("env")

    assert response.exit_code == 0
    assert response.output == "TOKEN=test"


def test_real_subprocess_timeout_removes_orphan_before_discarding_stage(tmp_path: Path) -> None:
    state_root = tmp_path / "fake-runtime-state"
    state_root.mkdir()
    runtime = tmp_path / "fake-docker"
    runtime.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            state_root = Path({str(state_root)!r})
            args = sys.argv[1:]

            if args[0] == "_worker":
                name, workspace = args[1], Path(args[2])
                stop = state_root / f"{{name}}.stop"
                while not stop.exists():
                    time.sleep(0.01)
                (workspace / "late-from-orphan.txt").write_text("late", encoding="utf-8")
                (state_root / name).unlink(missing_ok=True)
                raise SystemExit(0)

            if args[0] == "run":
                name = args[args.index("--name") + 1]
                mount = args[args.index("--mount") + 1]
                workspace = next(
                    Path(part.removeprefix("src="))
                    for part in mount.split(",")
                    if part.startswith("src=")
                )
                (state_root / name).write_text(str(workspace), encoding="utf-8")
                subprocess.Popen(
                    [sys.executable, __file__, "_worker", name, str(workspace)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(60)
                raise SystemExit(0)

            if args[0] == "rm":
                name = args[-1]
                (state_root / f"{{name}}.stop").touch()
                deadline = time.monotonic() + 4
                while (state_root / name).exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                raise SystemExit(0 if not (state_root / name).exists() else 1)

            if args[0] == "ps":
                value = args[args.index("--filter") + 1]
                name = value.removeprefix("name=^/").removesuffix("$")
                if (state_root / name).exists():
                    print(name)
                raise SystemExit(0)

            raise SystemExit(2)
            """
        ),
        encoding="utf-8",
    )
    runtime.chmod(0o700)
    root = tmp_path / "workspaces"
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=root,
        container_cli=str(runtime),
        policy=TenantContainerPolicy(timeout_seconds=1, cleanup_timeout_seconds=5),
    )
    backend.write("/notes.txt", "original")

    response = backend.execute("sleep forever")

    assert response.exit_code == 124
    assert response.output.endswith("command timed out after 1s")
    assert (backend.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not (backend.workspace / "late-from-orphan.txt").exists()
    assert not any(".transaction-" in path.name for path in root.iterdir())
    assert not [path for path in state_root.iterdir() if not path.name.endswith(".stop")]


def test_timeout_cleanup_failure_retains_isolated_transaction_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls = 0

    def fail_cleanup(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        nonlocal run_calls
        if argv[1] == "run":
            run_calls += 1
            staged = _mounted_workspace(argv)
            (staged / "late.txt").write_text("unsafe", encoding="utf-8")
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"partial")
        return subprocess.CompletedProcess(argv, 1, stdout=b"runtime unavailable")

    monkeypatch.setattr(sandbox.subprocess, "run", fail_cleanup)
    root = tmp_path / "workspaces"
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=root,
        policy=TenantContainerPolicy(timeout_seconds=1, cleanup_timeout_seconds=1),
    )
    backend.write("/notes.txt", "original")

    response = backend.execute("timeout")

    assert response.exit_code == 124
    assert response.output.endswith("sandbox cleanup is pending")
    assert (backend.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not (backend.workspace / "late.txt").exists()
    assert any(".transaction-" in path.name for path in root.iterdir())
    assert backend.execute("must not run").exit_code == 126
    assert run_calls == 1
    with pytest.raises(sandbox._WorkspaceSecurityError, match="cleanup is unconfirmed"):  # noqa: SLF001
        TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)


@pytest.mark.asyncio
async def test_cancellation_removes_container_and_never_promotes_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    stopped = threading.Event()
    running = threading.Event()

    def controlled_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        if argv[1] == "run":
            running.set()
            started.set()
            assert stopped.wait(timeout=3)
            (_mounted_workspace(argv) / "late.txt").write_text("unsafe", encoding="utf-8")
            running.clear()
            return subprocess.CompletedProcess(argv, 137, stdout=b"cancelled")
        if argv[1] == "rm":
            stopped.set()
            return subprocess.CompletedProcess(argv, 0, stdout=b"")
        name = argv[argv.index("--filter") + 1].removeprefix("name=^/").removesuffix("$")
        return subprocess.CompletedProcess(argv, 0, stdout=name.encode() if running.is_set() else b"")

    monkeypatch.setattr(sandbox.subprocess, "run", controlled_run)
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
        policy=TenantContainerPolicy(timeout_seconds=5, cleanup_timeout_seconds=2),
    )
    backend.write("/notes.txt", "original")
    task = asyncio.create_task(backend.aexecute("wait"))
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stopped.is_set()
    assert (backend.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not (backend.workspace / "late.txt").exists()
    assert not any(".transaction-" in path.name for path in backend.workspace.parent.iterdir())


def test_output_overflow_is_drained_bounded_and_never_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = threading.Event()
    running = threading.Event()

    def noisy_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if argv[1] == "run":
            running.set()
            descriptor = int(kwargs["stdout"])
            for _ in range(128):
                os.write(descriptor, b"x" * 4_096)
                if stopped.is_set():
                    break
            stopped.wait(timeout=2)
            (_mounted_workspace(argv) / "overflow.txt").write_text("unsafe", encoding="utf-8")
            running.clear()
            return subprocess.CompletedProcess(argv, 137, stdout=None)
        if argv[1] == "rm":
            stopped.set()
            return subprocess.CompletedProcess(argv, 0, stdout=b"")
        name = argv[argv.index("--filter") + 1].removeprefix("name=^/").removesuffix("$")
        return subprocess.CompletedProcess(argv, 0, stdout=name.encode() if running.is_set() else b"")

    monkeypatch.setattr(sandbox.subprocess, "run", noisy_run)
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
        policy=TenantContainerPolicy(max_output_bytes=1_024, cleanup_timeout_seconds=2),
    )
    backend.write("/notes.txt", "original")

    response = backend.execute("yes")

    assert response.exit_code == 125
    assert response.truncated is True
    assert len(response.output.encode()) <= 1_024
    assert response.output.endswith("sandbox output exceeded its limit")
    assert stopped.is_set()
    assert (backend.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not (backend.workspace / "overflow.txt").exists()


class _SimulatedProcessCrash(BaseException):
    pass


@pytest.mark.parametrize("crash_point", ["after_previous", "after_promoted"])
def test_crash_during_workspace_rename_restores_previous_on_next_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    def successful_mutation(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        (_mounted_workspace(argv) / "candidate.txt").write_text("candidate", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok")

    monkeypatch.setattr(sandbox.subprocess, "run", successful_mutation)
    root = tmp_path / "workspaces"
    backend = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    backend.write("/notes.txt", "original")
    original_replace = sandbox.os.replace

    def crashing_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        should_crash = (
            crash_point == "after_previous"
            and source_path == backend.workspace
            and destination_path.name == "previous"
        ) or (
            crash_point == "after_promoted"
            and source_path.name == "workspace"
            and destination_path == backend.workspace
        )
        original_replace(source, destination)
        if should_crash:
            raise _SimulatedProcessCrash

    monkeypatch.setattr(sandbox.os, "replace", crashing_replace)
    with pytest.raises(_SimulatedProcessCrash):
        backend.execute("mutate")
    monkeypatch.setattr(sandbox.os, "replace", original_replace)

    recovered = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)

    assert (recovered.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not (recovered.workspace / "candidate.txt").exists()
    assert not any(".transaction-" in path.name for path in root.iterdir())


def test_crash_after_committed_journal_keeps_validated_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def successful_mutation(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        (_mounted_workspace(argv) / "candidate.txt").write_text("candidate", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok")

    monkeypatch.setattr(sandbox.subprocess, "run", successful_mutation)
    root = tmp_path / "workspaces"
    backend = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    backend.write("/notes.txt", "original")
    write_journal = backend._write_transaction_journal  # noqa: SLF001

    def crash_after_journal(
        transaction: Path,
        *,
        phase: str,
        container_name: str | None,
    ) -> None:
        write_journal(transaction, phase=phase, container_name=container_name)
        if phase == "committed":
            raise _SimulatedProcessCrash

    monkeypatch.setattr(backend, "_write_transaction_journal", crash_after_journal)
    with pytest.raises(_SimulatedProcessCrash):
        backend.execute("mutate")

    recovered = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)

    assert (recovered.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert (recovered.workspace / "candidate.txt").read_text(encoding="utf-8") == "candidate"
    assert not any(".transaction-" in path.name for path in root.iterdir())


def test_crash_during_transaction_deletion_recovers_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox.subprocess, "run", _successful_run)
    root = tmp_path / "workspaces"
    backend = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    backend.write("/notes.txt", "original")
    remove = sandbox.shutil.rmtree
    crashed = False

    def crash_once(path: str | Path, *args: Any, **kwargs: Any) -> None:
        nonlocal crashed
        if ".garbage-" in Path(path).name and not crashed:
            crashed = True
            raise _SimulatedProcessCrash
        remove(path, *args, **kwargs)

    monkeypatch.setattr(sandbox.shutil, "rmtree", crash_once)
    with pytest.raises(_SimulatedProcessCrash):
        backend.execute("true")
    monkeypatch.setattr(sandbox.shutil, "rmtree", remove)

    recovered = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)

    assert (recovered.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not any(".garbage-" in path.name for path in root.iterdir())


def test_crash_before_transaction_journal_discards_partial_stage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspaces"
    backend = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    backend.write("/notes.txt", "original")
    transaction = root / f".{backend.workspace.name}.transaction-partial"
    transaction.mkdir(mode=0o700)
    (transaction / "partial-copy.txt").write_text("partial", encoding="utf-8")

    recovered = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)

    assert (recovered.workspace / "notes.txt").read_text(encoding="utf-8") == "original"
    assert not transaction.exists()


def test_ambiguous_previous_transactions_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspaces"
    backend = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    for suffix in ("one", "two"):
        transaction = root / f".{backend.workspace.name}.transaction-{suffix}"
        transaction.mkdir(mode=0o700)
        sandbox.shutil.copytree(backend.workspace, transaction / "previous")
        backend._write_transaction_journal(  # noqa: SLF001
            transaction,
            phase="previous",
            container_name=None,
        )

    with pytest.raises(sandbox._WorkspaceSecurityError, match="ambiguous"):  # noqa: SLF001
        TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
