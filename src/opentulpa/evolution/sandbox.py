"""Isolated Deep Agents backend for editing a disposable source candidate."""

from __future__ import annotations

import asyncio
import os
import platform
import pwd
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
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
_PROCESS_SANDBOX_EXECUTION_LOCK = threading.RLock()
_RUNTIME_UID = 65_532
_RUNTIME_GID = 65_532
_CANDIDATE_UID = 65_533
_CANDIDATE_GID = 65_533
_SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/usr/local"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
)
_SAFE_ETC_PATHS = (Path("/etc/passwd"), Path("/etc/group"), Path("/etc/nsswitch.conf"))


def _default_process_identity() -> tuple[int, int]:
    """Use the dedicated identity that is distinct from the served runtime."""

    for name in ("opentulpa-candidate",):
        try:
            entry = pwd.getpwnam(name)
        except KeyError:
            continue
        if entry.pw_uid > 0 and entry.pw_gid > 0:
            return int(entry.pw_uid), int(entry.pw_gid)
    for uid in (_CANDIDATE_UID,):
        try:
            entry = pwd.getpwuid(uid)
        except KeyError:
            continue
        if entry.pw_uid > 0 and entry.pw_gid > 0:
            return int(entry.pw_uid), int(entry.pw_gid)
    return _CANDIDATE_UID, _CANDIDATE_GID


def _trusted_root_executable(name: str) -> tuple[Path | None, str | None]:
    raw = shutil.which(name)
    if raw is None:
        return None, f"strong sandbox requires {name}"
    path = Path(raw)
    try:
        metadata = path.lstat()
    except OSError:
        return None, f"strong sandbox could not inspect {name}"
    if (
        not path.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(path, os.X_OK)
    ):
        return None, f"strong sandbox requires a root-owned immutable {name} executable"
    return path, None


def _system_mount_arguments() -> list[str]:
    arguments: list[str] = []
    for source in _SYSTEM_ROOTS:
        if source.is_symlink():
            arguments.extend(("--symlink", os.readlink(source), str(source)))
        elif source.is_dir():
            arguments.extend(("--ro-bind", str(source), str(source)))
    arguments.extend(("--dir", "/etc"))
    for source in _SAFE_ETC_PATHS:
        if source.is_file() and not source.is_symlink():
            arguments.extend(("--ro-bind", str(source), str(source)))
    return arguments


def _mount_parent_arguments(destinations: Iterable[str]) -> list[str]:
    parents: set[str] = set()
    for destination in destinations:
        parts = destination.strip("/").split("/")
        for index in range(1, len(parts)):
            parent = "/" + "/".join(parts[:index])
            if any(
                parent == str(system_root) or parent.startswith(f"{system_root}/")
                for system_root in _SYSTEM_ROOTS
            ):
                continue
            parents.add(parent)
    arguments: list[str] = []
    for parent in sorted(parents, key=lambda value: (value.count("/"), value)):
        arguments.extend(("--dir", parent, "--chmod", "0755", parent))
    return arguments


def _strong_sandbox_tools() -> tuple[Path, Path, Path, Path] | str:
    if platform.system() != "Linux":
        return "strong sandbox requires Linux namespaces"
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return "strong sandbox requires a root host supervisor"
    reaper = Path(sys.executable).resolve()
    try:
        metadata = reaper.lstat()
    except OSError:
        return "strong sandbox could not inspect its Python executable"
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(reaper, os.X_OK)
    ):
        return "strong sandbox requires a root-owned immutable Python executable"
    resolved = [reaper]
    for name in ("setpriv", "bwrap", "prlimit"):
        path, reason = _trusted_root_executable(name)
        if reason is not None or path is None:
            return reason or f"strong sandbox requires {name}"
        resolved.append(path)
    return resolved[0], resolved[1], resolved[2], resolved[3]


def strong_sandbox_unavailable_reason(*, probe: bool = True) -> str | None:
    """Return why the required root-Linux namespace sandbox cannot run."""

    tools = _strong_sandbox_tools()
    if isinstance(tools, str):
        return tools
    reaper, setpriv, bwrap, prlimit = tools
    if not probe:
        return None
    probe_argv = _build_bubblewrap_argv(
        reaper=reaper,
        setpriv=setpriv,
        bwrap=bwrap,
        prlimit=prlimit,
        uid=_CANDIDATE_UID,
        gid=_CANDIDATE_GID,
        workspace=None,
        workspace_read_only=True,
        read_only_mounts=(),
        environment={"PATH": "/usr/bin:/bin"},
        command=("/usr/bin/true",),
        memory_bytes=64 * 1024 * 1024,
        pid_limit=16,
        file_bytes=1024,
        cpu_seconds=5,
    )
    try:
        completed = subprocess.run(
            probe_argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            env={},
        )
    except (OSError, subprocess.TimeoutExpired):
        return "strong sandbox namespace probe could not run"
    if completed.returncode != 0:
        return (
            "strong sandbox requires permitted bubblewrap mount, PID, and network namespaces"
        )
    return None


def _build_bubblewrap_argv(
    *,
    reaper: Path,
    setpriv: Path,
    bwrap: Path,
    prlimit: Path,
    uid: int,
    gid: int,
    workspace: Path | None,
    workspace_read_only: bool,
    read_only_mounts: Sequence[tuple[Path, str]],
    environment: Mapping[str, str],
    command: Sequence[str],
    memory_bytes: int,
    pid_limit: int,
    file_bytes: int,
    cpu_seconds: int,
) -> list[str]:
    if not command or any(not value or "\x00" in value for value in command):
        raise ValueError("sandbox command is invalid")
    argv = [
        str(reaper),
        "-P",
        "-m",
        "opentulpa.evolution.process",
        "--",
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--chmod",
        "1777",
        "/dev/shm",
        "--tmpfs",
        "/tmp",
        "--chmod",
        "1777",
        "/tmp",
        *_system_mount_arguments(),
    ]
    destinations: set[str] = set()
    validated_mounts: list[tuple[Path, str]] = []
    for source, destination in read_only_mounts:
        if (
            destination == "/"
            or not destination.startswith("/")
            or destination.startswith("//")
            or any(component in {"", ".", ".."} for component in destination[1:].split("/"))
            or "\x00" in destination
            or "," in destination
            or any(ord(character) < 0x20 or ord(character) == 0x7f for character in destination)
            or destination in destinations
            or destination == "/workspace"
            or destination.startswith("/workspace/")
        ):
            raise ValueError("sandbox read-only mount destination is invalid")
        if source == Path("/"):
            raise ValueError("sandbox cannot bind the host root")
        source_text = str(source)
        if "," in source_text or any(
            ord(character) < 0x20 or ord(character) == 0x7f for character in source_text
        ):
            raise ValueError("sandbox read-only mount source has invalid mount grammar")
        destinations.add(destination)
        validated_mounts.append((source, destination))
    argv.extend(_mount_parent_arguments(destinations))
    for source, destination in validated_mounts:
        argv.extend(("--ro-bind", str(source), destination))
    if workspace is not None:
        argv.extend(
            (
                "--ro-bind" if workspace_read_only else "--bind",
                str(workspace),
                "/workspace",
                "--chdir",
                "/workspace",
            )
        )
    else:
        argv.extend(("--chdir", "/tmp"))
    for name, value in sorted(environment.items()):
        if not name or "\x00" in name or "=" in name or "\x00" in value:
            raise ValueError("sandbox environment is invalid")
        argv.extend(("--setenv", name, value))
    argv.extend(
        (
            "--",
            str(setpriv),
            f"--reuid={uid}",
            f"--regid={gid}",
            "--clear-groups",
            "--no-new-privs",
            "--bounding-set=-all",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            str(prlimit),
            f"--as={memory_bytes}:{memory_bytes}",
            f"--nproc={pid_limit}:{pid_limit}",
            f"--fsize={file_bytes}:{file_bytes}",
            f"--cpu={cpu_seconds}:{cpu_seconds}",
            "--",
            *command,
        )
    )
    return argv


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
    allow_internal_symlinks: bool = False

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
        if (
            root.is_symlink()
            or candidate.is_symlink()
            or any(part.is_symlink() for part in (root, candidate))
        ):
            raise ValueError("candidate sandbox paths cannot be symlinks")
        self._allowed_root = root.resolve(strict=True)
        self._workspace = candidate.resolve(strict=True)
        if not self._workspace.is_dir() or not self._is_relative_to(
            self._workspace, self._allowed_root
        ):
            raise ValueError("candidate workspace escaped the configured root")
        if "," in str(self._workspace) or any(
            ord(character) < 0x20 or ord(character) == 0x7f for character in str(self._workspace)
        ):
            raise ValueError("candidate workspace has invalid mount grammar")
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
                    if not self._policy.allow_internal_symlinks:
                        raise RuntimeError("candidate workspace contains a symbolic link")
                    try:
                        target = path.resolve(strict=False)
                    except (OSError, RuntimeError):
                        raise RuntimeError(
                            "candidate workspace contains an invalid symbolic link"
                        ) from None
                    if not self._is_relative_to(target, self._workspace):
                        raise RuntimeError(
                            "candidate workspace symbolic link escaped the workspace"
                        )
                    continue
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
            shutil.copytree(
                self._workspace,
                backup,
                copy_function=shutil.copy2,
                symlinks=self._policy.allow_internal_symlinks,
            )
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
                symlinks=self._policy.allow_internal_symlinks,
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


class TrustedLocalCandidateBackend(CandidateContainerBackend):
    """Trusted host shell for high-fidelity source evolution candidates."""

    @staticmethod
    def unavailable_reason() -> str | None:
        return None

    @staticmethod
    def is_supported() -> bool:
        return True

    @staticmethod
    def supported() -> bool:
        return True

    def __init__(
        self,
        *,
        workspace: str | Path,
        allowed_root: str | Path,
        policy: CandidateSandboxPolicy | None = None,
    ) -> None:
        super().__init__(
            workspace=workspace,
            allowed_root=allowed_root,
            policy=policy,
            container_cli="trusted-local",
        )
        self._execution_env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ExecuteResponse:
        safe_command = str(command or "").strip()
        if not safe_command or "\x00" in safe_command:
            return ExecuteResponse(output="command is invalid", exit_code=2, truncated=False)
        effective_timeout = min(
            self._policy.timeout_seconds,
            max(1, int(timeout or self._policy.timeout_seconds)),
        )
        shell = "/bin/sh" if Path("/bin/sh").is_file() else "sh"
        with self._lock:
            backup: Path | None = None
            backup_mode = 0o700
            monitor_stop = threading.Event()
            workspace_invalid = threading.Event()
            abort_requested = threading.Event()
            monitor: threading.Thread | None = None

            def monitor_workspace() -> None:
                while not monitor_stop.wait(0.1):
                    if cancel_event is not None and cancel_event.is_set():
                        abort_requested.set()
                        return
                    try:
                        self._scan_tree()
                    except (OSError, RuntimeError):
                        workspace_invalid.set()
                        abort_requested.set()
                        return

            try:
                self._validate_tree()
                backup, backup_mode = self._recovery_copy()
                monitor = threading.Thread(
                    target=monitor_workspace,
                    name="opentulpa-trusted-local-candidate-quota",
                    daemon=True,
                )
                monitor.start()
                completed = run_bounded_process(
                    (shell, "-c", safe_command),
                    cwd=self._workspace,
                    env=self._execution_env,
                    timeout_seconds=effective_timeout,
                    max_output_bytes=self._policy.max_output_bytes,
                    abort_event=abort_requested,
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
                if cancel_event is not None and cancel_event.is_set():
                    return ExecuteResponse(
                        output="trusted local execution was cancelled",
                        exit_code=130,
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
        return ExecuteResponse(
            output=completed.output.decode("utf-8", errors="replace"),
            exit_code=completed.returncode,
            truncated=completed.truncated,
        )

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        cancel_event = threading.Event()
        task = asyncio.create_task(
            asyncio.to_thread(
                self.execute,
                command,
                timeout=timeout,
                cancel_event=cancel_event,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancel_event.set()
            await asyncio.gather(task, return_exceptions=True)
            raise


class CandidateProcessBackend(CandidateContainerBackend):
    """Root-supervised bubblewrap sandbox for an untrusted source workspace."""

    @staticmethod
    def unavailable_reason() -> str | None:
        return strong_sandbox_unavailable_reason()

    @staticmethod
    def is_supported() -> bool:
        return CandidateProcessBackend.unavailable_reason() is None

    @staticmethod
    def supported() -> bool:
        return CandidateProcessBackend.is_supported()

    def __init__(
        self,
        *,
        workspace: str | Path,
        allowed_root: str | Path,
        policy: CandidateSandboxPolicy,
        uid: int | None = None,
        gid: int | None = None,
        read_only_mounts: Sequence[tuple[str | Path, str]] = (),
    ) -> None:
        unavailable = self.unavailable_reason()
        if unavailable is not None:
            raise RuntimeError(unavailable)
        if (uid is None) != (gid is None):
            raise ValueError("process source sandbox identity must include both uid and gid")
        if uid is None or gid is None:
            uid, gid = _default_process_identity()
        if uid < 1 or gid < 1 or uid == _RUNTIME_UID or gid == _RUNTIME_GID:
            raise ValueError("process source sandbox identity is invalid")
        tools = _strong_sandbox_tools()
        if isinstance(tools, str):
            raise RuntimeError(tools)
        self._reaper, self._setpriv, self._bwrap, self._prlimit = tools
        self._process_uid = uid
        self._process_gid = gid
        self._read_only_mounts = tuple(
            (self._validated_read_only_source(Path(source)), destination)
            for source, destination in read_only_mounts
        )
        mounted_bins = [
            f"{destination.rstrip('/')}/bin"
            for source, destination in self._read_only_mounts
            if (source / "bin").is_dir()
        ]
        self._sandbox_path = ":".join((*mounted_bins, "/usr/local/bin", "/usr/bin", "/bin"))
        self._sandbox_wheelhouse = next(
            (
                destination
                for source, destination in self._read_only_mounts
                if source.name == "wheelhouse"
            ),
            "/wheelhouse",
        )
        super().__init__(
            workspace=workspace,
            allowed_root=allowed_root,
            policy=policy,
            container_cli="process",
        )
        self._lock = _PROCESS_SANDBOX_EXECUTION_LOCK

    @staticmethod
    def _validated_read_only_source(path: Path) -> Path:
        candidate = path.expanduser()
        if not candidate.is_absolute() or candidate == Path("/"):
            raise ValueError("sandbox read-only mount source is invalid")
        current = Path(candidate.anchor)
        for component in candidate.parts[1:]:
            current /= component
            try:
                metadata = current.lstat()
            except OSError as exc:
                raise ValueError("sandbox read-only mount source is unavailable") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("sandbox read-only mount source cannot contain symbolic links")
            if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
                raise ValueError("sandbox read-only mount source must be root-owned and immutable")
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_IMODE(metadata.st_mode) & stat.S_IXOTH:
                raise ValueError("sandbox read-only mount source is not candidate-traversable")
        if not candidate.is_dir() and not candidate.is_file():
            raise ValueError("sandbox read-only mount source is invalid")
        return candidate

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ExecuteResponse:
        safe_command = str(command or "").strip()
        if not safe_command or "\x00" in safe_command:
            return ExecuteResponse(output="command is invalid", exit_code=2, truncated=False)
        effective_timeout = min(
            self._policy.timeout_seconds,
            max(1, int(timeout or self._policy.timeout_seconds)),
        )
        memory_bytes = _resource_bytes(self._policy.memory_limit)
        file_bytes = self._policy.max_file_bytes
        argv = _build_bubblewrap_argv(
            reaper=self._reaper,
            setpriv=self._setpriv,
            bwrap=self._bwrap,
            prlimit=self._prlimit,
            uid=self._process_uid,
            gid=self._process_gid,
            workspace=self._workspace,
            workspace_read_only=False,
            read_only_mounts=self._read_only_mounts,
            environment={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "HOME": "/tmp",
                "PATH": self._sandbox_path,
                "PIP_CONFIG_FILE": "/dev/null",
                "PIP_FIND_LINKS": self._sandbox_wheelhouse,
                "PIP_NO_INDEX": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "TMPDIR": "/tmp",
                "UV_FIND_LINKS": self._sandbox_wheelhouse,
                "UV_NO_INDEX": "1",
                "UV_OFFLINE": "1",
            },
            command=("/bin/sh", "-c", _shell_with_group_cleanup(safe_command)),
            memory_bytes=memory_bytes,
            pid_limit=self._policy.pid_limit,
            file_bytes=file_bytes,
            cpu_seconds=effective_timeout + 5,
        )
        with self._lock:
            backup: Path | None = None
            backup_mode = 0o700
            traversal_modes: list[tuple[Path, int]] = []
            monitor_stop = threading.Event()
            workspace_invalid = threading.Event()
            abort_requested = threading.Event()
            monitor: threading.Thread | None = None

            def monitor_workspace() -> None:
                while not monitor_stop.wait(0.1):
                    if cancel_event is not None and cancel_event.is_set():
                        abort_requested.set()
                        return
                    try:
                        self._scan_tree()
                    except (OSError, RuntimeError):
                        workspace_invalid.set()
                        abort_requested.set()
                        return

            try:
                self._validate_tree()
                backup, backup_mode = self._recovery_copy()
                traversal_modes = self._grant_workspace_traversal()
                self._assign_workspace(self._process_uid, self._process_gid)
                monitor = threading.Thread(
                    target=monitor_workspace,
                    name="opentulpa-process-candidate-quota",
                    daemon=True,
                )
                monitor.start()
                completed = run_bounded_process(
                    argv,
                    cwd=self._workspace,
                    env={},
                    timeout_seconds=effective_timeout,
                    max_output_bytes=self._policy.max_output_bytes,
                    abort_event=abort_requested,
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
                if cancel_event is not None and cancel_event.is_set():
                    return ExecuteResponse(
                        output="sandbox execution was cancelled",
                        exit_code=130,
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
                    self._assign_workspace(0, 0)
                self._restore_workspace_traversal(traversal_modes)
        return ExecuteResponse(
            output=completed.output.decode("utf-8", errors="replace"),
            exit_code=completed.returncode,
            truncated=completed.truncated,
        )

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return await asyncio.to_thread(self.execute, command, timeout=timeout)

    def _assign_workspace(self, uid: int, gid: int) -> None:
        os.chown(self._workspace, uid, gid, follow_symlinks=False)
        self._workspace.chmod(0o700)
        for directory, directory_names, file_names in os.walk(
            self._workspace,
            topdown=True,
            followlinks=False,
        ):
            for name in directory_names:
                path = Path(directory) / name
                os.chown(path, uid, gid, follow_symlinks=False)
                if not path.is_symlink():
                    path.chmod(0o700)
            for name in file_names:
                path = Path(directory) / name
                metadata = path.lstat()
                os.chown(path, uid, gid, follow_symlinks=False)
                if not stat.S_ISLNK(metadata.st_mode):
                    path.chmod(0o700 if stat.S_IMODE(metadata.st_mode) & 0o111 else 0o600)

    def _grant_workspace_traversal(self) -> list[tuple[Path, int]]:
        changed: list[tuple[Path, int]] = []
        current = self._workspace.parent
        try:
            while current != current.parent:
                mode = stat.S_IMODE(current.stat().st_mode)
                if not mode & stat.S_IXOTH:
                    changed.append((current, mode))
                    current.chmod(mode | stat.S_IXOTH)
                current = current.parent
        except OSError:
            self._restore_workspace_traversal(changed)
            raise
        return changed

    @staticmethod
    def _restore_workspace_traversal(changed: list[tuple[Path, int]]) -> None:
        for path, mode in reversed(changed):
            with suppress(OSError):
                path.chmod(mode)


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
    "TrustedLocalCandidateBackend",
    "resolve_local_oci_image",
    "strong_sandbox_unavailable_reason",
]
