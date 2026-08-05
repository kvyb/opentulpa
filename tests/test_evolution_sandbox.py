from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from opentulpa.evolution import sandbox as sandbox_module
from opentulpa.evolution.process import BoundedProcessResult
from opentulpa.evolution.sandbox import (
    CandidateContainerBackend,
    CandidateProcessBackend,
    CandidateSandboxPolicy,
    resolve_local_oci_image,
)


def test_candidate_sandbox_requires_workspace_under_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    candidate = allowed / "candidate"
    outside = tmp_path / "outside"
    candidate.mkdir(parents=True)
    outside.mkdir()

    backend = CandidateContainerBackend(workspace=candidate, allowed_root=allowed)

    assert backend.id.startswith("local-")
    with pytest.raises(ValueError, match="escaped"):
        CandidateContainerBackend(workspace=outside, allowed_root=allowed)


def test_candidate_sandbox_fails_closed_after_a_symlink_appears(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    candidate = allowed / "candidate"
    candidate.mkdir(parents=True)
    backend = CandidateContainerBackend(workspace=candidate, allowed_root=allowed)
    (candidate / "escape").symlink_to(tmp_path)

    with pytest.raises(RuntimeError, match="symbolic link"):
        backend.read("/escape/file.txt")
    with pytest.raises(RuntimeError, match="security validation"):
        backend.ls("/")


def test_candidate_sandbox_policy_rejects_unbounded_configuration() -> None:
    with pytest.raises(ValueError, match="image"):
        CandidateSandboxPolicy(image="python; unsafe")
    with pytest.raises(ValueError, match="timeout"):
        CandidateSandboxPolicy(timeout_seconds=0)


def test_strong_process_command_has_only_explicit_mounts_and_private_namespaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "candidate"
    controller = tmp_path / "controller-tools"
    wheelhouse = tmp_path / "wheelhouse"
    workspace.mkdir()
    controller.mkdir()
    wheelhouse.mkdir()
    monkeypatch.setattr(
        sandbox_module,
        "_system_mount_arguments",
        lambda: ["--ro-bind", "/usr", "/usr", "--dir", "/etc"],
    )

    argv = sandbox_module._build_bubblewrap_argv(  # noqa: SLF001
        setpriv=Path("/usr/bin/setpriv"),
        bwrap=Path("/usr/bin/bwrap"),
        prlimit=Path("/usr/bin/prlimit"),
        uid=65_533,
        gid=65_533,
        workspace=workspace,
        workspace_read_only=False,
        read_only_mounts=((controller, "/opt/controller-tools"), (wheelhouse, "/wheelhouse")),
        environment={"PATH": "/controller/bin:/usr/bin:/bin", "UV_OFFLINE": "1"},
        command=("/bin/sh", "-c", "true"),
        memory_bytes=1024 * 1024 * 1024,
        pid_limit=64,
        file_bytes=1024 * 1024,
        cpu_seconds=30,
    )

    assert argv[0] == "/usr/bin/bwrap"
    assert {
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
    } <= set(argv)
    shm_index = argv.index("/dev/shm")
    assert argv[shm_index - 1 : shm_index + 5] == [
        "--tmpfs",
        "/dev/shm",
        "--chmod",
        "1777",
        "/dev/shm",
        "--tmpfs",
    ]
    command_separator = argv.index("--")
    assert argv[command_separator + 1 : command_separator + 10] == [
        "/usr/bin/setpriv",
        "--reuid=65533",
        "--regid=65533",
        "--clear-groups",
        "--no-new-privs",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "/usr/bin/prlimit",
    ]
    binds = [
        tuple(argv[index : index + 3])
        for index, value in enumerate(argv)
        if value in {"--bind", "--ro-bind"}
    ]
    assert ("--bind", str(workspace), "/workspace") in binds
    assert ("--ro-bind", str(controller), "/opt/controller-tools") in binds
    assert ("--ro-bind", str(wheelhouse), "/wheelhouse") in binds
    parent_index = argv.index("/opt") - 1
    assert argv[parent_index : parent_index + 5] == [
        "--dir",
        "/opt",
        "--chmod",
        "0755",
        "/opt",
    ]
    assert all(source != "/" and destination != "/" for _, source, destination in binds)
    assert not any("product" in source or "bootstrap" in source for _, source, _ in binds)
    assert "--share-net" not in argv
    assert argv[argv.index("--setenv") + 1 : argv.index("--setenv") + 3] == [
        "PATH",
        "/controller/bin:/usr/bin:/bin",
    ]


@pytest.mark.parametrize("destination", ("//opt/tools", "/opt/./tools", "/opt/../tools", "/"))
def test_bubblewrap_rejects_noncanonical_mount_destinations(
    tmp_path: Path,
    destination: str,
) -> None:
    with pytest.raises(ValueError, match="destination"):
        sandbox_module._build_bubblewrap_argv(  # noqa: SLF001
            setpriv=Path("/usr/bin/setpriv"),
            bwrap=Path("/usr/bin/bwrap"),
            prlimit=Path("/usr/bin/prlimit"),
            uid=65_533,
            gid=65_533,
            workspace=tmp_path,
            workspace_read_only=False,
            read_only_mounts=((tmp_path, destination),),
            environment={"PATH": "/usr/bin:/bin"},
            command=("/bin/sh", "-c", "true"),
            memory_bytes=1024 * 1024 * 1024,
            pid_limit=64,
            file_bytes=1024 * 1024,
            cpu_seconds=30,
        )


def test_bubblewrap_does_not_recreate_system_mount_parents(tmp_path: Path) -> None:
    argv = sandbox_module._build_bubblewrap_argv(  # noqa: SLF001
        setpriv=Path("/usr/bin/setpriv"),
        bwrap=Path("/usr/bin/bwrap"),
        prlimit=Path("/usr/bin/prlimit"),
        uid=65_533,
        gid=65_533,
        workspace=tmp_path,
        workspace_read_only=False,
        read_only_mounts=((tmp_path, "/usr/local/opentulpa"),),
        environment={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        command=("/bin/sh", "-c", "true"),
        memory_bytes=1024 * 1024 * 1024,
        pid_limit=64,
        file_bytes=1024 * 1024,
        cpu_seconds=30,
    )

    setup = argv[: argv.index("--")]
    assert ("--dir", "/usr/local") not in zip(setup, setup[1:], strict=False)


def test_process_backend_fails_closed_without_linux_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox_module.platform, "system", lambda: "Darwin")

    assert CandidateProcessBackend.is_supported() is False
    assert (
        CandidateProcessBackend.unavailable_reason() == "strong sandbox requires Linux namespaces"
    )


@pytest.mark.parametrize(
    ("engine", "rootless_output"),
    [("docker", b'["name=rootless"]'), ("podman", b"true")],
)
def test_local_image_resolution_requires_rootless_engine_and_returns_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    rootless_output: bytes,
) -> None:
    calls: list[tuple[str, ...]] = []

    def run(argv: tuple[str, ...], **_: Any) -> BoundedProcessResult:
        calls.append(argv)
        output = rootless_output if argv[1] == "info" else f"sha256:{'a' * 64}".encode()
        return BoundedProcessResult(
            returncode=0,
            output=output,
            truncated=False,
            timed_out=False,
        )

    monkeypatch.setattr(sandbox_module, "run_bounded_process", run)

    resolved = resolve_local_oci_image(
        container_cli=engine,
        image="opentulpa:test",
        cwd=tmp_path,
    )

    assert resolved == f"sha256:{'a' * 64}"
    assert [call[1] for call in calls] == ["info", "image"]


def test_local_image_resolution_rejects_rootful_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sandbox_module,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            returncode=0,
            output=b"[]",
            truncated=False,
            timed_out=False,
        ),
    )

    with pytest.raises(RuntimeError, match="rootless"):
        resolve_local_oci_image(
            container_cli="docker",
            image="opentulpa:test",
            cwd=tmp_path,
        )


def test_direct_local_resolution_accepts_recognized_macos_desktop_vm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(argv: tuple[str, ...], **_: Any) -> BoundedProcessResult:
        if argv[1:3] == ("info", "--format") and "SecurityOptions" in argv[-1]:
            output = b"[]"
        elif argv[1] == "info":
            output = b"OrbStack|orbstack"
        else:
            output = f"sha256:{'a' * 64}".encode()
        return BoundedProcessResult(
            returncode=0,
            output=output,
            truncated=False,
            timed_out=False,
        )

    monkeypatch.setattr(sandbox_module, "run_bounded_process", run)
    monkeypatch.setattr(sandbox_module.platform, "system", lambda: "Darwin")

    resolved = resolve_local_oci_image(
        container_cli="docker",
        image="opentulpa:test",
        cwd=tmp_path,
        allow_desktop_vm=True,
    )

    assert resolved == f"sha256:{'a' * 64}"


def test_candidate_sandbox_enforces_total_workspace_size(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    candidate = allowed / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "one.bin").write_bytes(b"a" * 800)
    (candidate / "two.bin").write_bytes(b"b" * 800)

    with pytest.raises(RuntimeError, match="total size"):
        CandidateContainerBackend(
            workspace=candidate,
            allowed_root=allowed,
            policy=CandidateSandboxPolicy(
                max_file_bytes=1_024,
                max_total_bytes=1_500,
            ),
        )


def test_candidate_shell_mount_is_writable_and_still_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    candidate = allowed / "candidate"
    candidate.mkdir(parents=True)
    captured: list[str] = []

    def run(argv: list[str], **_: Any) -> BoundedProcessResult:
        captured.extend(argv)
        (candidate / "created-by-shell.py").write_text("VALUE = 1\n", encoding="utf-8")
        return BoundedProcessResult(
            returncode=0,
            output=b"ok",
            truncated=False,
            timed_out=False,
        )

    monkeypatch.setattr(sandbox_module, "run_bounded_process", run)
    backend = CandidateContainerBackend(
        workspace=candidate,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(network_enabled=True),
    )

    result = backend.execute("true")

    mount = captured[captured.index("--mount") + 1]
    assert mount.endswith("dst=/workspace")
    assert "readonly" not in mount
    assert captured[captured.index("--network") + 1] == "bridge"
    assert captured[-1].startswith("ulimit -S -f ")
    assert "ulimit -H -f" in captured[-1]
    assert (candidate / "created-by-shell.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert result.exit_code == 0


def test_candidate_network_can_be_disabled_for_sensitive_rehearsals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    candidate = allowed / "candidate"
    candidate.mkdir(parents=True)
    captured: list[str] = []

    def run(argv: list[str], **_: Any) -> BoundedProcessResult:
        captured.extend(argv)
        return BoundedProcessResult(returncode=0, output=b"", truncated=False, timed_out=False)

    monkeypatch.setattr(sandbox_module, "run_bounded_process", run)
    backend = CandidateContainerBackend(
        workspace=candidate,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(network_enabled=False),
    )

    backend.execute("true")

    assert captured[captured.index("--network") + 1] == "none"


def test_candidate_shell_reverts_an_invalid_tree_instead_of_bricking_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    candidate = allowed / "candidate"
    candidate.mkdir(parents=True)
    original = candidate / "original.py"
    original.write_text("VALUE = 1\n", encoding="utf-8")

    def run(*_: Any, **__: Any) -> BoundedProcessResult:
        original.write_text("VALUE = 2\n", encoding="utf-8")
        (candidate / "escape").symlink_to(tmp_path)
        return BoundedProcessResult(returncode=0, output=b"", truncated=False, timed_out=False)

    monkeypatch.setattr(sandbox_module, "run_bounded_process", run)
    backend = CandidateContainerBackend(workspace=candidate, allowed_root=allowed)

    result = backend.execute("make invalid tree")

    assert result.exit_code == 126
    assert original.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (candidate / "escape").exists()
    assert backend.ls("/")


def test_candidate_shell_monitor_kills_growth_and_restores_prior_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    candidate = allowed / "candidate"
    candidate.mkdir(parents=True)
    original = candidate / "original.py"
    original.write_text("VALUE = 1\n", encoding="utf-8")
    killed = threading.Event()

    def remove(*_: Any, **__: Any) -> None:
        killed.set()

    def run(*_: Any, **__: Any) -> BoundedProcessResult:
        (candidate / "growing.bin").write_bytes(b"x" * 2_048)
        deadline = time.monotonic() + 2
        while not killed.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        return BoundedProcessResult(returncode=137, output=b"", truncated=False, timed_out=False)

    monkeypatch.setattr(sandbox_module, "_force_remove_container", remove)
    monkeypatch.setattr(sandbox_module, "run_bounded_process", run)
    backend = CandidateContainerBackend(
        workspace=candidate,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(max_file_bytes=1_024, max_total_bytes=4_096),
    )

    result = backend.execute("grow forever")

    assert killed.is_set()
    assert result.exit_code == 126
    assert original.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (candidate / "growing.bin").exists()
