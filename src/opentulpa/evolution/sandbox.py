"""Isolated Deep Agents backend for editing a disposable source candidate."""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from opentulpa.evolution.process import run_bounded_process

_IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}\Z")
_IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RESOURCE_RE = re.compile(r"[1-9][0-9]*(?:\.[0-9]+)?(?:[kmgt]i?|b)?\Z", re.I)


def _force_remove_container(container_cli: str, container_name: str) -> None:
    try:
        subprocess.run(
            [container_cli, "rm", "--force", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env={"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def resolve_local_oci_image(
    *,
    container_cli: str,
    image: str,
    cwd: Path,
    allow_desktop_vm: bool = False,
) -> str:
    """Resolve a reviewed local tag to the immutable image ID used for this process."""

    if not container_cli or "\x00" in container_cli or not _IMAGE_RE.fullmatch(image):
        raise ValueError("OCI image configuration is invalid")
    _require_rootless_oci_engine(
        container_cli=container_cli,
        cwd=cwd,
        allow_desktop_vm=allow_desktop_vm,
    )
    result = run_bounded_process(
        (container_cli, "image", "inspect", "--format", "{{.Id}}", image),
        cwd=cwd,
        env={"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"},
        timeout_seconds=30,
        max_output_bytes=1_024,
    )
    resolved = result.output.decode("ascii", errors="ignore").strip().lower()
    if result.returncode != 0 or result.truncated or not _IMAGE_ID_RE.fullmatch(resolved):
        raise RuntimeError("configured OCI image is not available as an immutable local image")
    return resolved


def _require_rootless_oci_engine(
    *,
    container_cli: str,
    cwd: Path,
    allow_desktop_vm: bool = False,
) -> None:
    engine = Path(container_cli).name
    if engine == "docker":
        arguments = (container_cli, "info", "--format", "{{json .SecurityOptions}}")
    elif engine == "podman":
        arguments = (container_cli, "info", "--format", "{{.Host.Security.Rootless}}")
    else:
        raise ValueError("container_cli must be Docker or Podman")
    result = run_bounded_process(
        arguments,
        cwd=cwd,
        env={"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"},
        timeout_seconds=30,
        max_output_bytes=16_384,
    )
    output = result.output.decode("utf-8", errors="ignore").strip().casefold()
    rootless = "rootless" in output if engine == "docker" else output == "true"
    if result.returncode == 0 and not result.truncated and rootless:
        return
    if allow_desktop_vm and engine == "docker" and platform.system() == "Darwin":
        identity = run_bounded_process(
            (container_cli, "info", "--format", "{{.OperatingSystem}}|{{.Name}}"),
            cwd=cwd,
            env={"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"},
            timeout_seconds=30,
            max_output_bytes=1_024,
        )
        recognized = identity.output.decode("utf-8", errors="ignore").strip().casefold()
        if (
            identity.returncode == 0
            and not identity.truncated
            and recognized in {"docker desktop|docker-desktop", "orbstack|orbstack"}
        ):
            return
    raise RuntimeError("configured OCI engine must operate in rootless mode")


@dataclass(frozen=True, slots=True)
class CandidateSandboxPolicy:
    """Hard resource and filesystem limits for generated code."""

    image: str = "python:3.12-slim"
    cpu_limit: str = "2"
    memory_limit: str = "2g"
    pid_limit: int = 256
    timeout_seconds: int = 300
    max_output_bytes: int = 1_000_000
    max_file_bytes: int = 20 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_entries: int = 50_000
    network_enabled: bool = False

    def __post_init__(self) -> None:
        if not _IMAGE_RE.fullmatch(self.image):
            raise ValueError("candidate sandbox image is invalid")
        if not _RESOURCE_RE.fullmatch(self.cpu_limit):
            raise ValueError("candidate sandbox cpu_limit is invalid")
        if not _RESOURCE_RE.fullmatch(self.memory_limit):
            raise ValueError("candidate sandbox memory_limit is invalid")
        if not 16 <= self.pid_limit <= 4_096:
            raise ValueError("candidate sandbox pid_limit is invalid")
        if not 1 <= self.timeout_seconds <= 3_600:
            raise ValueError("candidate sandbox timeout_seconds is invalid")
        if self.max_output_bytes < 1_024:
            raise ValueError("candidate sandbox max_output_bytes is invalid")
        if self.max_file_bytes < 1_024:
            raise ValueError("candidate sandbox max_file_bytes is invalid")
        if self.max_total_bytes < self.max_file_bytes:
            raise ValueError("candidate sandbox max_total_bytes is invalid")
        if self.max_entries < 100:
            raise ValueError("candidate sandbox max_entries is invalid")


class CandidateContainerBackend(LocalShellBackend):
    """Guard host edits and execute commands as a non-root OCI process."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        allowed_root: str | Path,
        policy: CandidateSandboxPolicy | None = None,
        container_cli: str = "docker",
    ) -> None:
        self._policy = policy or CandidateSandboxPolicy()
        self._container_cli = str(container_cli or "").strip()
        if not self._container_cli or "\x00" in self._container_cli:
            raise ValueError("container_cli is invalid")
        root = Path(allowed_root).expanduser()
        candidate = Path(workspace).expanduser()
        if root.is_symlink() or candidate.is_symlink():
            raise ValueError("candidate sandbox paths cannot be symlinks")
        self._allowed_root = root.resolve(strict=True)
        self._workspace = candidate.resolve(strict=True)
        if not self._workspace.is_dir() or not self._is_relative_to(
            self._workspace, self._allowed_root
        ):
            raise ValueError("candidate workspace escaped the configured root")
        if "," in str(self._workspace):
            raise ValueError("candidate workspace cannot contain a comma")
        self._lock = threading.RLock()
        self._containers_lock = threading.Lock()
        self._active_containers: set[str] = set()
        self._compromised = False
        super().__init__(
            root_dir=self._workspace,
            virtual_mode=True,
            timeout=self._policy.timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
            env={},
            inherit_env=False,
        )
        self._validate_tree()

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def _validate_tree(self) -> None:
        if self._compromised:
            raise RuntimeError("candidate workspace failed security validation")
        try:
            self._scan_tree()
        except (OSError, RuntimeError):
            self._compromised = True
            raise

    def _scan_tree(self) -> None:
        entries = 0
        total_bytes = 0
        for directory, directory_names, file_names in os.walk(
            self._workspace,
            topdown=True,
            followlinks=False,
        ):
            for name in [*directory_names, *file_names]:
                entries += 1
                if entries > self._policy.max_entries:
                    raise RuntimeError("candidate workspace entry limit exceeded")
                path = Path(directory) / name
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise RuntimeError("candidate workspace contains a symbolic link")
                if stat.S_ISREG(metadata.st_mode):
                    if metadata.st_nlink != 1:
                        raise RuntimeError("candidate workspace contains a hard link")
                    if metadata.st_size > self._policy.max_file_bytes:
                        raise RuntimeError("candidate workspace file limit exceeded")
                    total_bytes += metadata.st_size
                    if total_bytes > self._policy.max_total_bytes:
                        raise RuntimeError("candidate workspace total size limit exceeded")
                elif not stat.S_ISDIR(metadata.st_mode):
                    raise RuntimeError("candidate workspace contains a special file")

    def _recovery_copy(self) -> tuple[Path, int]:
        backup = self._workspace.parent / f".{self._workspace.name}.recovery-{uuid4().hex}"
        root_mode = stat.S_IMODE(self._workspace.stat().st_mode)
        try:
            shutil.copytree(self._workspace, backup, copy_function=shutil.copy2)
            backup.chmod(0o700)
        except OSError:
            shutil.rmtree(backup, ignore_errors=True)
            raise RuntimeError("candidate workspace recovery snapshot failed") from None
        return backup, root_mode

    def _restore_recovery_copy(self, backup: Path, root_mode: int) -> None:
        def retry_with_owner_access(operation, path: str, _: BaseException) -> None:  # type: ignore[no-untyped-def]
            os.chmod(path, stat.S_IRWXU)
            operation(path)

        try:
            self._workspace.chmod(0o700)
            for entry in os.scandir(self._workspace):
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(path, onexc=retry_with_owner_access)
                else:
                    path.unlink()
            shutil.copytree(
                backup,
                self._workspace,
                dirs_exist_ok=True,
                copy_function=shutil.copy2,
            )
            self._workspace.chmod(root_mode)
            self._compromised = False
            self._scan_tree()
        except (OSError, RuntimeError):
            self._compromised = True
            raise RuntimeError("candidate workspace recovery failed") from None

    def _guarded(self, operation: object, *args: object, **kwargs: object) -> object:
        with self._lock:
            self._validate_tree()
            result = operation(*args, **kwargs)  # type: ignore[operator]
            self._validate_tree()
            return result

    def ls(self, path: str):  # type: ignore[no-untyped-def]
        return self._guarded(super().ls, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):  # type: ignore[no-untyped-def]
        return self._guarded(super().read, file_path, offset, limit)

    def write(self, file_path: str, content: str):  # type: ignore[no-untyped-def]
        if len(content.encode("utf-8")) > self._policy.max_file_bytes:
            raise ValueError("candidate file exceeds size limit")
        return self._guarded(super().write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        return self._guarded(
            super().edit,
            file_path,
            old_string,
            new_string,
            replace_all,
        )

    def glob(self, pattern: str, path: str | None = None):  # type: ignore[no-untyped-def]
        return self._guarded(super().glob, pattern, path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ):
        return self._guarded(super().grep, pattern, path, glob)

    def upload_files(self, files: list[tuple[str, bytes]]):  # type: ignore[no-untyped-def]
        if any(len(content) > self._policy.max_file_bytes for _, content in files):
            raise ValueError("candidate upload exceeds size limit")
        return self._guarded(super().upload_files, files)

    def download_files(self, paths: list[str]):  # type: ignore[no-untyped-def]
        return self._guarded(super().download_files, paths)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        safe_command = str(command or "").strip()
        if not safe_command or "\x00" in safe_command:
            return ExecuteResponse(output="command is invalid", exit_code=2, truncated=False)
        effective_timeout = min(
            self._policy.timeout_seconds,
            max(1, int(timeout or self._policy.timeout_seconds)),
        )
        mount_source = str(self._workspace)
        container_name = f"opentulpa-candidate-{uuid4().hex}"
        file_blocks = max(2, (self._policy.max_file_bytes + 511) // 512)
        bounded_command = (
            f"ulimit -S -f {file_blocks} && ulimit -H -f {file_blocks} && {safe_command}"
        )
        uid = os.getuid() if hasattr(os, "getuid") else 65_532
        gid = os.getgid() if hasattr(os, "getgid") else 65_532
        argv = [
            self._container_cli,
            "run",
            "--rm",
            "--name",
            container_name,
            "--init",
            "--pull",
            "never",
            "--read-only",
            "--network",
            "bridge" if self._policy.network_enabled else "none",
            "--ipc",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--cpus",
            self._policy.cpu_limit,
            "--memory",
            self._policy.memory_limit,
            "--memory-swap",
            self._policy.memory_limit,
            "--pids-limit",
            str(self._policy.pid_limit),
            "--user",
            f"{uid}:{gid}",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
            "--mount",
            f"type=bind,src={mount_source},dst=/workspace",
            "--workdir",
            "/workspace",
            self._policy.image,
            "/bin/sh",
            "-lc",
            bounded_command,
        ]
        with self._lock:
            with self._containers_lock:
                self._active_containers.add(container_name)
            backup: Path | None = None
            backup_mode = 0o700
            monitor_stop = threading.Event()
            workspace_invalid = threading.Event()
            monitor: threading.Thread | None = None

            def monitor_workspace() -> None:
                while not monitor_stop.wait(0.1):
                    try:
                        self._scan_tree()
                    except (OSError, RuntimeError):
                        workspace_invalid.set()
                        _force_remove_container(self._container_cli, container_name)
                        return

            try:
                self._validate_tree()
                backup, backup_mode = self._recovery_copy()
                monitor = threading.Thread(
                    target=monitor_workspace,
                    name="opentulpa-candidate-quota",
                    daemon=True,
                )
                monitor.start()
                completed = run_bounded_process(
                    argv,
                    cwd=self._workspace,
                    env={"PATH": os.environ.get("PATH", os.defpath)},
                    timeout_seconds=effective_timeout,
                    max_output_bytes=self._policy.max_output_bytes,
                    timeout_cleanup=lambda: _force_remove_container(
                        self._container_cli, container_name
                    ),
                )
                monitor_stop.set()
                monitor.join(timeout=5)
                try:
                    self._scan_tree()
                except (OSError, RuntimeError):
                    workspace_invalid.set()
                if workspace_invalid.is_set():
                    self._restore_recovery_copy(backup, backup_mode)
                    return ExecuteResponse(
                        output="command exceeded workspace safety limits; changes were reverted",
                        exit_code=126,
                        truncated=False,
                    )
                if completed.timed_out:
                    return ExecuteResponse(
                        output=(
                            completed.output.decode("utf-8", errors="replace")
                            + "\ncommand timed out"
                        ).strip(),
                        exit_code=124,
                        truncated=completed.truncated,
                    )
            except FileNotFoundError:
                return ExecuteResponse(
                    output="candidate sandbox container runtime is unavailable",
                    exit_code=127,
                    truncated=False,
                )
            except (OSError, RuntimeError):
                if backup is not None:
                    with suppress(RuntimeError):
                        self._restore_recovery_copy(backup, backup_mode)
                return ExecuteResponse(
                    output="candidate workspace failed security validation",
                    exit_code=126,
                    truncated=False,
                )
            finally:
                monitor_stop.set()
                if monitor is not None:
                    monitor.join(timeout=5)
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
                with self._containers_lock:
                    self._active_containers.discard(container_name)
        output = completed.output.decode("utf-8", errors="replace")
        return ExecuteResponse(
            output=output,
            exit_code=completed.returncode,
            truncated=completed.truncated,
        )

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        task = asyncio.create_task(asyncio.to_thread(self.execute, command, timeout=timeout))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with self._containers_lock:
                active = tuple(self._active_containers)
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        _force_remove_container,
                        self._container_cli,
                        container_name,
                    )
                    for container_name in active
                ),
                return_exceptions=True,
            )
            await asyncio.gather(task, return_exceptions=True)
            raise


class CandidateProcessBackend(CandidateContainerBackend):
    """Railway source sandbox using a credential-less, unprivileged host process."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        allowed_root: str | Path,
        policy: CandidateSandboxPolicy,
        uid: int = 65_532,
        gid: int = 65_532,
    ) -> None:
        if not policy.network_enabled:
            raise ValueError("process source sandbox requires explicit network enablement")
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise RuntimeError("process source sandbox requires a root host supervisor")
        if uid < 1 or gid < 1:
            raise ValueError("process source sandbox identity is invalid")
        if shutil.which("setpriv") is None or shutil.which("prlimit") is None:
            raise RuntimeError("process source sandbox requires setpriv and prlimit")
        self._process_uid = uid
        self._process_gid = gid
        super().__init__(
            workspace=workspace,
            allowed_root=allowed_root,
            policy=policy,
            container_cli="process",
        )

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        safe_command = str(command or "").strip()
        if not safe_command or "\x00" in safe_command:
            return ExecuteResponse(output="command is invalid", exit_code=2, truncated=False)
        effective_timeout = min(
            self._policy.timeout_seconds,
            max(1, int(timeout or self._policy.timeout_seconds)),
        )
        memory_bytes = _resource_bytes(self._policy.memory_limit)
        file_bytes = self._policy.max_file_bytes
        argv = (
            "setpriv",
            f"--reuid={self._process_uid}",
            f"--regid={self._process_gid}",
            "--clear-groups",
            "--no-new-privs",
            "prlimit",
            f"--as={memory_bytes}:{memory_bytes}",
            f"--nproc={self._policy.pid_limit}:{self._policy.pid_limit}",
            f"--fsize={file_bytes}:{file_bytes}",
            f"--cpu={effective_timeout + 5}:{effective_timeout + 5}",
            "--",
            "/bin/sh",
            "-c",
            _shell_with_group_cleanup(safe_command),
        )
        with self._lock:
            backup: Path | None = None
            backup_mode = 0o700
            monitor_stop = threading.Event()
            workspace_invalid = threading.Event()
            monitor: threading.Thread | None = None

            def monitor_workspace() -> None:
                while not monitor_stop.wait(0.1):
                    try:
                        self._scan_tree()
                    except (OSError, RuntimeError):
                        workspace_invalid.set()
                        return

            try:
                self._validate_tree()
                self._chown_workspace(self._process_uid, self._process_gid)
                backup, backup_mode = self._recovery_copy()
                monitor = threading.Thread(
                    target=monitor_workspace,
                    name="opentulpa-process-candidate-quota",
                    daemon=True,
                )
                monitor.start()
                completed = run_bounded_process(
                    argv,
                    cwd=self._workspace,
                    env={
                        "HOME": "/tmp",
                        "PATH": os.environ.get("PATH", os.defpath),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONHASHSEED": "0",
                    },
                    timeout_seconds=effective_timeout,
                    max_output_bytes=self._policy.max_output_bytes,
                    abort_event=workspace_invalid,
                )
                monitor_stop.set()
                monitor.join(timeout=5)
                try:
                    self._scan_tree()
                except (OSError, RuntimeError):
                    workspace_invalid.set()
                if workspace_invalid.is_set():
                    self._restore_recovery_copy(backup, backup_mode)
                    return ExecuteResponse(
                        output="command exceeded workspace safety limits; changes were reverted",
                        exit_code=126,
                        truncated=False,
                    )
                if completed.timed_out:
                    return ExecuteResponse(
                        output=(
                            completed.output.decode("utf-8", errors="replace")
                            + "\ncommand timed out"
                        ).strip(),
                        exit_code=124,
                        truncated=completed.truncated,
                    )
            except (OSError, RuntimeError):
                if backup is not None:
                    with suppress(RuntimeError):
                        self._restore_recovery_copy(backup, backup_mode)
                return ExecuteResponse(
                    output="candidate workspace failed security validation",
                    exit_code=126,
                    truncated=False,
                )
            finally:
                monitor_stop.set()
                if monitor is not None:
                    monitor.join(timeout=5)
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
                with suppress(OSError):
                    self._chown_workspace(0, 0)
        return ExecuteResponse(
            output=completed.output.decode("utf-8", errors="replace"),
            exit_code=completed.returncode,
            truncated=completed.truncated,
        )

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return await asyncio.to_thread(self.execute, command, timeout=timeout)

    def _chown_workspace(self, uid: int, gid: int) -> None:
        os.chown(self._workspace, uid, gid)
        for directory, directory_names, file_names in os.walk(
            self._workspace,
            topdown=True,
            followlinks=False,
        ):
            for name in [*directory_names, *file_names]:
                os.chown(
                    Path(directory) / name,
                    uid,
                    gid,
                    follow_symlinks=False,
                )


def _resource_bytes(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)(?:\.0+)?([kmgt]i?|b)?", value.strip(), re.I)
    if match is None:
        raise ValueError("candidate memory limit is invalid")
    amount = int(match.group(1))
    suffix = str(match.group(2) or "b").casefold()
    multipliers: dict[str, int] = {
        "b": 1,
        "k": 1024,
        "ki": 1024,
        "m": 1024 * 1024,
        "mi": 1024 * 1024,
        "g": 1024 * 1024 * 1024,
        "gi": 1024 * 1024 * 1024,
        "t": 1024 * 1024 * 1024 * 1024,
        "ti": 1024 * 1024 * 1024 * 1024,
    }
    return amount * multipliers[suffix]


def _shell_with_group_cleanup(command: str) -> str:
    cleanup = (
        "status=$?; trap '' TERM HUP INT; kill -TERM -$$ >/dev/null 2>&1 || true; exit $status"
    )
    return f"trap {shlex.quote(cleanup)} EXIT; {command}"


__all__ = [
    "CandidateContainerBackend",
    "CandidateProcessBackend",
    "CandidateSandboxPolicy",
    "resolve_local_oci_image",
]
