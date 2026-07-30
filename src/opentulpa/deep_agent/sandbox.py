"""Tenant-scoped OCI execution backend for Deep Agents."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol

from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from langgraph.runtime import get_runtime

_CONTAINER_IMAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}\Z")
_CONTAINER_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_RESOURCE_VALUE_PATTERN = re.compile(r"[1-9][0-9]*(?:\.[0-9]+)?(?:[kmgt]i?|b)?\Z", re.I)
_TRANSACTION_JOURNAL = "journal.json"
_TRANSACTION_PHASES = frozenset(
    {"staged", "executing", "quiescent", "previous", "promoted", "committed"}
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SENSITIVE_COMPONENTS = frozenset(
    {
        ".aws",
        ".docker",
        ".git",
        ".gnupg",
        ".kube",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".ssh",
        "containerd.sock",
        "credentials",
        "credentials.json",
        "docker.sock",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "podman.sock",
    }
)
_FORBIDDEN_ROOTS = (
    Path("/"),
    Path.home(),
    Path.home() / ".aws",
    Path.home() / ".docker",
    Path.home() / ".ssh",
    Path("/etc"),
    Path("/run"),
    Path("/var/run"),
    Path("/var/lib/docker"),
)


class _WorkspaceSecurityError(RuntimeError):
    """Internal fail-closed workspace validation error."""


@dataclass(slots=True)
class _ExecutionControl:
    """Coordinate cancellation with a blocking container CLI invocation."""

    cancelled: threading.Event = field(default_factory=threading.Event)
    name_ready: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    container_name: str | None = None

    def publish_name(self, container_name: str) -> None:
        self.container_name = container_name
        self.name_ready.set()


@dataclass(slots=True)
class _BoundedOutputCapture:
    """Drain a child pipe continuously while retaining only a fixed byte prefix."""

    limit: int
    data: bytearray = field(default_factory=bytearray)
    overflowed: threading.Event = field(default_factory=threading.Event)
    failed: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            remaining = max(0, self.limit - len(self.data))
            self.data.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self.overflowed.set()

    def drain(self, descriptor: int) -> None:
        try:
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    return
                self.append(chunk)
        except OSError:
            self.failed.set()
        finally:
            with suppress(OSError):
                os.close(descriptor)

    def output(self) -> bytes:
        with self._lock:
            return bytes(self.data)


class TenantExecutionProvider(Protocol):
    """Execute a command for one trusted tenant without exposing runtime authority."""

    def execute(
        self,
        *,
        tenant_id: str,
        command: str,
        timeout: int,
        workspace: Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ExecuteResponse: ...


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_sensitive_component(component: str) -> bool:
    lowered = component.casefold()
    return (
        lowered in _SENSITIVE_COMPONENTS
        or lowered == ".env"
        or lowered.startswith(".env.")
        or lowered.endswith((".key", ".pem"))
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class TenantContainerPolicy:
    image: str = "opentulpa-tenant-sandbox:0.1.0"
    cpu_limit: str = "1"
    memory_limit: str = "512m"
    pid_limit: int = 128
    timeout_seconds: int = 60
    cleanup_timeout_seconds: int = 10
    max_output_bytes: int = 512_000
    max_file_bytes: int = 10 * 1024 * 1024
    max_upload_files: int = 20
    max_workspace_entries: int = 20_000
    max_command_characters: int = 100_000
    scratch_size_mb: int = 128
    network_enabled: bool = False

    def __post_init__(self) -> None:
        if not _CONTAINER_IMAGE_PATTERN.fullmatch(self.image):
            raise ValueError("sandbox image must be a pinned OCI image reference")
        if not _RESOURCE_VALUE_PATTERN.fullmatch(self.cpu_limit):
            raise ValueError("sandbox cpu_limit is invalid")
        if not _RESOURCE_VALUE_PATTERN.fullmatch(self.memory_limit):
            raise ValueError("sandbox memory_limit is invalid")
        if self.pid_limit < 1:
            raise ValueError("sandbox pid_limit must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("sandbox timeout_seconds must be positive")
        if not 1 <= self.cleanup_timeout_seconds <= 60:
            raise ValueError("sandbox cleanup_timeout_seconds must be between 1 and 60")
        if self.max_output_bytes < 1_024:
            raise ValueError("sandbox max_output_bytes must be at least 1024")
        if self.max_file_bytes < 1:
            raise ValueError("sandbox max_file_bytes must be positive")
        if self.max_upload_files < 1:
            raise ValueError("sandbox max_upload_files must be positive")
        if self.max_workspace_entries < 1:
            raise ValueError("sandbox max_workspace_entries must be positive")
        if self.max_command_characters < 1:
            raise ValueError("sandbox max_command_characters must be positive")
        if self.scratch_size_mb < 1:
            raise ValueError("sandbox scratch_size_mb must be positive")


class TenantContainerBackend(SandboxBackendProtocol):
    """Run commands in disposable containers with an optional tenant workspace mount."""

    _tenant_locks: ClassVar[dict[tuple[str, str], threading.Lock]] = {}
    _tenant_locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        tenant_id: str,
        workspaces_root: str | Path,
        policy: TenantContainerPolicy | None = None,
        container_cli: str = "docker",
        persistent_workspace: bool = True,
        execution_provider: TenantExecutionProvider | None = None,
        commit_authority: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        safe_tenant = str(tenant_id or "").strip()
        if not safe_tenant or len(safe_tenant) > 512 or _contains_control_characters(safe_tenant):
            raise ValueError("tenant_id is invalid")
        safe_cli = str(container_cli or "docker").strip() or "docker"
        if "\x00" in safe_cli or _contains_control_characters(safe_cli):
            raise ValueError("container_cli is invalid")

        self._tenant_id = safe_tenant
        self._policy = policy or TenantContainerPolicy()
        self._container_cli = safe_cli
        self._persistent_workspace = bool(persistent_workspace)
        self._execution_provider = execution_provider
        self._commit_authority = commit_authority
        digest = hashlib.sha256(safe_tenant.encode("utf-8")).hexdigest()[:24]
        self._workspace_digest = digest
        self._workspaces_root = self._prepare_root(workspaces_root)
        self._workspace = self._workspaces_root / digest
        lock_key = (str(self._workspaces_root), digest)
        with self._tenant_locks_guard:
            self._lock = self._tenant_locks.setdefault(lock_key, threading.Lock())
        self._filesystem: FilesystemBackend | None = None
        if self._persistent_workspace:
            self._prepare_workspace_root()
            if self._workspace.is_symlink():
                raise ValueError("tenant workspace cannot be a symlink")
            with self._serialized_workspace():
                pass
            self._filesystem = FilesystemBackend(
                root_dir=self._workspace,
                virtual_mode=True,
                max_file_size_mb=max(1, self._policy.max_file_bytes // (1024 * 1024)),
            )

    @property
    def id(self) -> str:
        suffix = "workspace" if self._persistent_workspace else "scratch"
        digest = hashlib.sha256(self._tenant_id.encode()).hexdigest()[:16]
        return f"tenant-container:{suffix}:{digest}"

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def persistent_workspace(self) -> bool:
        return self._persistent_workspace

    @staticmethod
    def _prepare_root(workspaces_root: str | Path) -> Path:
        raw_root = Path(workspaces_root).expanduser()
        raw_value = str(raw_root)
        if not raw_value or "," in raw_value or _contains_control_characters(raw_value):
            raise ValueError("workspaces_root is invalid")
        if raw_root.is_symlink():
            raise ValueError("workspaces_root cannot be a symlink")
        root = raw_root.resolve(strict=False)
        resolved_forbidden = {path.expanduser().resolve(strict=False) for path in _FORBIDDEN_ROOTS}
        if root in resolved_forbidden:
            raise ValueError("workspaces_root points at a protected host directory")
        return root

    def _prepare_workspace_root(self) -> None:
        self._workspaces_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self._workspaces_root.is_dir() or self._workspaces_root.is_symlink():
            raise ValueError("workspaces_root must be a regular directory")
        resolved_root = self._workspaces_root.resolve(strict=True)
        if resolved_root != self._workspaces_root:
            raise ValueError("workspaces_root failed canonical validation")
        with suppress(OSError):
            os.chmod(self._workspaces_root, 0o700)

    def _container_user(self, workspace: Path | None = None) -> str:
        uid = os.getuid() if hasattr(os, "getuid") else 65_532
        gid = os.getgid() if hasattr(os, "getgid") else 65_532
        if uid == 0:
            uid = gid = 65_532
            if workspace is not None:
                for directory, directory_names, file_names in os.walk(
                    workspace, topdown=True, followlinks=False
                ):
                    with suppress(OSError):
                        os.chown(directory, uid, gid, follow_symlinks=False)
                    for name in [*directory_names, *file_names]:
                        with suppress(OSError):
                            os.chown(
                                Path(directory) / name,
                                uid,
                                gid,
                                follow_symlinks=False,
                            )
        return f"{uid}:{gid}"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _fsync_tree(cls, root: Path) -> None:
        """Flush a validated tree before making it the durable live workspace."""

        directories: list[Path] = []
        for directory, _, file_names in os.walk(root, topdown=True, followlinks=False):
            directory_path = Path(directory)
            directories.append(directory_path)
            for name in file_names:
                path = directory_path / name
                descriptor = os.open(
                    path,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    metadata = os.fstat(descriptor)
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise _WorkspaceSecurityError("workspace changed while being flushed")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        for flushed_directory in reversed(directories):
            cls._fsync_directory(flushed_directory)

    @property
    def _workspace_marker(self) -> Path:
        return self._workspaces_root / f".{self._workspace_digest}.initialized"

    def _write_workspace_marker(self) -> None:
        marker = self._workspace_marker
        descriptor = os.open(
            marker,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.write(descriptor, b"opentulpa-tenant-workspace-v1\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._fsync_directory(self._workspaces_root)

    def _validate_workspace_marker(self) -> bool:
        marker = self._workspace_marker
        if not os.path.lexists(marker):
            return False
        metadata = marker.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise _WorkspaceSecurityError("invalid workspace initialization marker")
        return True

    def _transaction_prefix(self) -> str:
        return f".{self._workspace_digest}.transaction-"

    def _garbage_prefix(self) -> str:
        return f".{self._workspace_digest}.garbage-"

    def _validate_transaction_path(self, transaction: Path) -> None:
        if (
            transaction.parent != self._workspaces_root
            or not transaction.name.startswith(self._transaction_prefix())
            or transaction.is_symlink()
            or not transaction.is_dir()
            or transaction.resolve(strict=True) != transaction
        ):
            raise _WorkspaceSecurityError("invalid workspace transaction path")

    def _write_transaction_journal(
        self,
        transaction: Path,
        *,
        phase: str,
        container_name: str | None,
    ) -> None:
        self._validate_transaction_path(transaction)
        if phase not in _TRANSACTION_PHASES:
            raise _WorkspaceSecurityError("invalid workspace transaction phase")
        if container_name is not None and not _CONTAINER_NAME_PATTERN.fullmatch(container_name):
            raise _WorkspaceSecurityError("invalid sandbox container name")
        payload = json.dumps(
            {
                "container_name": container_name,
                "phase": phase,
                "version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        temporary = transaction / f".{_TRANSACTION_JOURNAL}.tmp"
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_TRUNC
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, transaction / _TRANSACTION_JOURNAL)
        self._fsync_directory(transaction)

    def _read_transaction_journal(self, transaction: Path) -> tuple[str, str | None] | None:
        self._validate_transaction_path(transaction)
        journal = transaction / _TRANSACTION_JOURNAL
        if not os.path.lexists(journal):
            return None
        descriptor = os.open(
            journal,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > 4_096
            ):
                raise _WorkspaceSecurityError("invalid workspace transaction journal")
            raw = os.read(descriptor, 4_097)
        finally:
            os.close(descriptor)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise _WorkspaceSecurityError("invalid workspace transaction journal") from exc
        if not isinstance(payload, dict) or set(payload) != {"container_name", "phase", "version"}:
            raise _WorkspaceSecurityError("invalid workspace transaction journal")
        phase = payload.get("phase")
        container_name = payload.get("container_name")
        if (
            payload.get("version") != 1
            or not isinstance(phase, str)
            or phase not in _TRANSACTION_PHASES
            or (
                container_name is not None
                and (
                    not isinstance(container_name, str)
                    or not _CONTAINER_NAME_PATTERN.fullmatch(container_name)
                )
            )
        ):
            raise _WorkspaceSecurityError("invalid workspace transaction journal")
        return phase, container_name

    def _force_remove_container(self, container_name: str) -> bool:
        """Force-remove a generated container and prove that its bind mount is quiescent."""

        if not _CONTAINER_NAME_PATTERN.fullmatch(container_name):
            raise _WorkspaceSecurityError("invalid sandbox container name")
        timeout = self._policy.cleanup_timeout_seconds
        deadline = time.monotonic() + timeout
        environment = {"PATH": os.environ.get("PATH", "")}
        absent_since: float | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                removed = subprocess.run(
                    [self._container_cli, "rm", "--force", container_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    timeout=max(0.1, min(2.0, remaining)),
                    env=environment,
                )
                inspected = subprocess.run(
                    [
                        self._container_cli,
                        "ps",
                        "--all",
                        "--filter",
                        f"name=^/{container_name}$",
                        "--format",
                        "{{.Names}}",
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=max(0.1, min(2.0, remaining)),
                    env=environment,
                )
            except (OSError, subprocess.TimeoutExpired):
                return False
            if inspected.returncode != 0:
                return False
            names = {
                line.strip()
                for line in _as_bytes(inspected.stdout).decode("utf-8", errors="replace").splitlines()
                if line.strip()
            }
            if container_name not in names:
                if removed.returncode == 0:
                    return True
                now = time.monotonic()
                if absent_since is None:
                    absent_since = now
                elif now - absent_since >= 0.2:
                    return True
            else:
                absent_since = None
            time.sleep(min(0.05, remaining))

    def _recover_workspace(self) -> None:
        """Resolve interrupted promotions conservatively while holding the tenant lock."""

        marker_exists = self._validate_workspace_marker()
        for entry in self._workspaces_root.iterdir():
            if not entry.name.startswith(self._garbage_prefix()):
                continue
            if (
                entry.parent != self._workspaces_root
                or entry.is_symlink()
                or not entry.is_dir()
                or entry.resolve(strict=True) != entry
            ):
                raise _WorkspaceSecurityError("invalid workspace garbage path")
            shutil.rmtree(entry, ignore_errors=False, onexc=self._make_removable)
            self._fsync_directory(self._workspaces_root)
        transactions: list[tuple[Path, str | None, str | None, bool]] = []
        prefix = self._transaction_prefix()
        for entry in self._workspaces_root.iterdir():
            if not entry.name.startswith(prefix):
                continue
            self._validate_transaction_path(entry)
            journal = self._read_transaction_journal(entry)
            phase, container_name = journal if journal is not None else (None, None)
            previous = entry / "previous"
            if os.path.lexists(previous):
                if previous.is_symlink() or not previous.is_dir():
                    raise _WorkspaceSecurityError("invalid previous workspace transaction")
                has_previous = True
            else:
                has_previous = False
            transactions.append((entry, phase, container_name, has_previous))

        previous_transactions = [record for record in transactions if record[3]]
        if len(previous_transactions) > 1 or (previous_transactions and len(transactions) > 1):
            raise _WorkspaceSecurityError("ambiguous workspace recovery state")

        if previous_transactions:
            transaction, phase, container_name, _ = previous_transactions[0]
            if (
                phase == "executing"
                and container_name is not None
                and not self._force_remove_container(container_name)
            ):
                raise _WorkspaceSecurityError("sandbox container cleanup is unconfirmed")
            if phase == "committed":
                if not self._workspace.exists() or self._workspace.is_symlink():
                    raise _WorkspaceSecurityError("committed workspace is unavailable")
                self._validate_workspace_tree()
                self._discard_transaction(transaction)
            else:
                rejected = transaction / "rejected-recovery"
                if self._workspace.exists():
                    if rejected.exists():
                        raise _WorkspaceSecurityError("ambiguous rejected workspace recovery state")
                    os.replace(self._workspace, rejected)
                    self._fsync_directory(self._workspaces_root)
                    self._fsync_directory(transaction)
                previous = transaction / "previous"
                if previous.exists():
                    os.replace(previous, self._workspace)
                    self._fsync_directory(transaction)
                    self._fsync_directory(self._workspaces_root)
                if not self._workspace.exists():
                    raise _WorkspaceSecurityError("previous workspace could not be restored")
                self._validate_workspace_tree()
                self._fsync_tree(self._workspace)
                self._discard_transaction(transaction)
            transactions = []

        for transaction, phase, container_name, _ in transactions:
            if phase is None:
                if (
                    not marker_exists
                    or not self._workspace.exists()
                    or self._workspace.is_symlink()
                ):
                    raise _WorkspaceSecurityError("untrusted legacy workspace transaction")
                self._validate_workspace_tree()
                self._discard_transaction(transaction)
                continue
            if phase == "committed":
                if not self._workspace.exists() or self._workspace.is_symlink():
                    raise _WorkspaceSecurityError("committed workspace is unavailable")
                self._validate_workspace_tree()
            elif (
                phase == "executing"
                and container_name is not None
                and not self._force_remove_container(container_name)
            ):
                raise _WorkspaceSecurityError("sandbox container cleanup is unconfirmed")
            self._discard_transaction(transaction)

        if os.path.lexists(self._workspace):
            if self._workspace.is_symlink() or not self._workspace.is_dir():
                raise _WorkspaceSecurityError("tenant workspace cannot be a symlink")
        else:
            if marker_exists:
                raise _WorkspaceSecurityError("initialized tenant workspace is missing")
            self._workspace.mkdir(mode=0o700)
            self._fsync_directory(self._workspaces_root)

        resolved_workspace = self._workspace.resolve(strict=True)
        if resolved_workspace != self._workspace or not _is_relative_to(
            resolved_workspace, self._workspaces_root
        ):
            raise _WorkspaceSecurityError("tenant workspace escaped workspaces_root")
        if resolved_workspace == _REPOSITORY_ROOT or _is_relative_to(
            _REPOSITORY_ROOT, resolved_workspace
        ):
            raise _WorkspaceSecurityError("tenant workspace cannot contain the repository")
        self._validate_workspace_tree()
        if not marker_exists:
            self._write_workspace_marker()
        with suppress(OSError):
            os.chmod(self._workspace, 0o700)

    @contextmanager
    def _serialized_workspace(self, *, recover: bool = True) -> Iterator[None]:
        """Serialize a tenant across local and stable-host backend instances."""

        with self._lock:
            if not self._persistent_workspace:
                yield
                return
            lock_root = self._workspaces_root / ".locks"
            lock_root.mkdir(mode=0o700, exist_ok=True)
            if lock_root.is_symlink() or not lock_root.is_dir():
                raise _WorkspaceSecurityError("invalid workspace lock root")
            if lock_root.resolve(strict=True).parent != self._workspaces_root:
                raise _WorkspaceSecurityError("workspace lock root escaped")
            lock_path = lock_root / f"{self._workspace_digest}.lock"
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise _WorkspaceSecurityError("invalid workspace lock file")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                if recover:
                    self._recover_workspace()
                yield
            finally:
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _require_filesystem(self) -> FilesystemBackend:
        if self._filesystem is None:
            raise RuntimeError("scratch files are provided by StateBackend")
        return self._filesystem

    def _validate_virtual_path(self, value: str, *, pattern: bool = False) -> Path:
        raw_path = str(value or "")
        if (
            not raw_path
            or len(raw_path) > 4_096
            or "\x00" in raw_path
            or "\\" in raw_path
            or _contains_control_characters(raw_path)
        ):
            raise ValueError("workspace path is invalid")
        components = [component for component in raw_path.removeprefix("/").split("/") if component]
        for component in components:
            if component in {".", ".."} or component.startswith("~"):
                raise ValueError("workspace path traversal is not allowed")
            if not pattern and _is_sensitive_component(component):
                raise ValueError("sensitive workspace paths are not allowed")
            if (
                pattern
                and not any(marker in component for marker in "*?[")
                and _is_sensitive_component(component)
            ):
                raise ValueError("sensitive workspace paths are not allowed")
        target = self._workspace.joinpath(*components)
        if pattern:
            return target
        current = self._workspace
        for component in components:
            current = current / component
            if os.path.lexists(current):
                metadata = current.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError("workspace symlinks are not allowed")
                if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                    raise ValueError("special workspace files are not allowed")
        resolved_target = target.resolve(strict=False)
        if not _is_relative_to(resolved_target, self._workspace):
            raise ValueError("workspace path traversal is not allowed")
        return target

    def _validate_workspace_tree(self, workspace: Path | None = None) -> None:
        if not self._persistent_workspace:
            return
        root = self._workspace if workspace is None else workspace
        if root.is_symlink() or not root.is_dir():
            raise _WorkspaceSecurityError("invalid workspace root")
        resolved_workspace = root.resolve(strict=True)
        if resolved_workspace != root or not _is_relative_to(
            resolved_workspace, self._workspaces_root
        ):
            raise _WorkspaceSecurityError("workspace root escaped")

        def fail_walk(error: OSError) -> None:
            raise _WorkspaceSecurityError("workspace tree cannot be inspected") from error

        entries = 0
        for directory, directory_names, file_names in os.walk(
            root,
            topdown=True,
            onerror=fail_walk,
            followlinks=False,
        ):
            for name in [*directory_names, *file_names]:
                entries += 1
                if entries > self._policy.max_workspace_entries:
                    raise _WorkspaceSecurityError("workspace entry limit exceeded")
                if _is_sensitive_component(name):
                    raise _WorkspaceSecurityError("sensitive workspace entry")
                path = Path(directory) / name
                try:
                    metadata = path.lstat()
                except OSError as exc:
                    raise _WorkspaceSecurityError(
                        "workspace entry changed during validation"
                    ) from exc
                if stat.S_ISLNK(metadata.st_mode):
                    raise _WorkspaceSecurityError("workspace symlink")
                if stat.S_ISREG(metadata.st_mode):
                    if metadata.st_nlink > 1:
                        raise _WorkspaceSecurityError("workspace hard link")
                    if metadata.st_size > self._policy.max_file_bytes:
                        raise _WorkspaceSecurityError("workspace file size limit exceeded")
                    continue
                if not stat.S_ISDIR(metadata.st_mode):
                    raise _WorkspaceSecurityError("workspace special file")

    def _stage_workspace(self) -> tuple[Path, Path]:
        transaction = Path(
            tempfile.mkdtemp(
                prefix=f".{self._workspace_digest}.transaction-",
                dir=self._workspaces_root,
            )
        )
        with suppress(OSError):
            os.chmod(transaction, 0o700)
        staged = transaction / "workspace"
        try:
            shutil.copytree(
                self._workspace,
                staged,
                symlinks=False,
                copy_function=shutil.copy2,
            )
            self._validate_workspace_tree(staged)
            self._write_transaction_journal(
                transaction,
                phase="staged",
                container_name=None,
            )
            self._fsync_directory(self._workspaces_root)
        except Exception:
            self._discard_transaction(transaction)
            raise
        return transaction, staged

    @staticmethod
    def _make_removable(function: Callable[[str], object], path: str, error: BaseException) -> None:
        del error
        with suppress(OSError):
            os.chmod(path, stat.S_IRWXU, follow_symlinks=False)
        function(path)

    def _discard_transaction(self, transaction: Path) -> None:
        """Durably tombstone a quiescent transaction before best-effort deletion."""

        self._validate_transaction_path(transaction)
        garbage = self._workspaces_root / f"{self._garbage_prefix()}{secrets.token_hex(8)}"
        os.replace(transaction, garbage)
        self._fsync_directory(self._workspaces_root)
        shutil.rmtree(garbage, ignore_errors=False, onexc=self._make_removable)
        self._fsync_directory(self._workspaces_root)

    def _commit_workspace(self, transaction: Path, staged: Path) -> None:
        journal = self._read_transaction_journal(transaction)
        if journal is None:
            raise _WorkspaceSecurityError("workspace transaction journal is unavailable")
        _, container_name = journal
        self._validate_workspace_tree(staged)
        self._fsync_tree(staged)
        previous = transaction / "previous"
        os.replace(self._workspace, previous)
        self._fsync_directory(self._workspaces_root)
        self._fsync_directory(transaction)
        self._write_transaction_journal(
            transaction,
            phase="previous",
            container_name=container_name,
        )
        promoted = False
        try:
            os.replace(staged, self._workspace)
            promoted = True
            self._fsync_directory(transaction)
            self._fsync_directory(self._workspaces_root)
            self._write_transaction_journal(
                transaction,
                phase="promoted",
                container_name=container_name,
            )
            self._validate_workspace_tree()
            self._fsync_tree(self._workspace)
            self._write_transaction_journal(
                transaction,
                phase="committed",
                container_name=container_name,
            )
        except Exception:
            if promoted and self._workspace.exists():
                os.replace(self._workspace, transaction / "rejected")
                self._fsync_directory(self._workspaces_root)
                self._fsync_directory(transaction)
            if previous.exists():
                os.replace(previous, self._workspace)
                self._fsync_directory(transaction)
                self._fsync_directory(self._workspaces_root)
            raise

    def _new_container_name(self) -> str:
        name = f"opentulpa-sbx-{self._workspace_digest}-{secrets.token_hex(8)}"
        if not _CONTAINER_NAME_PATTERN.fullmatch(name):  # pragma: no cover - construction invariant
            raise _WorkspaceSecurityError("generated sandbox container name is invalid")
        return name

    def _container_argv(
        self,
        command: str,
        *,
        container_name: str,
        workspace: Path | None = None,
    ) -> list[str]:
        if not _CONTAINER_NAME_PATTERN.fullmatch(container_name):
            raise _WorkspaceSecurityError("invalid sandbox container name")
        policy = self._policy
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
            "--security-opt",
            "no-new-privileges:true",
            "--cap-drop",
            "ALL",
            "--network",
            "bridge" if policy.network_enabled else "none",
            "--ipc",
            "none",
            "--cpus",
            policy.cpu_limit,
            "--memory",
            policy.memory_limit,
            "--memory-swap",
            policy.memory_limit,
            "--pids-limit",
            str(policy.pid_limit),
            "--ulimit",
            "nofile=1024:1024",
            "--ulimit",
            f"nproc={policy.pid_limit}:{policy.pid_limit}",
            "--stop-timeout",
            "2",
            "--hostname",
            "opentulpa-sandbox",
            "--log-driver",
            "none",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        ]
        if self._persistent_workspace:
            if workspace is None:
                raise RuntimeError("persistent sandbox execution requires a staged workspace")
            argv.extend(
                [
                    "--mount",
                    (f"type=bind,src={workspace},dst=/workspace,bind-propagation=rprivate"),
                ]
            )
        else:
            argv.extend(
                [
                    "--tmpfs",
                    (f"/workspace:rw,nosuid,nodev,size={policy.scratch_size_mb}m,mode=1777"),
                ]
            )
        argv.extend(
            [
                "--workdir",
                "/workspace",
                "--user",
                self._container_user(workspace),
                "--env",
                "HOME=/tmp",
                "--env",
                "TMPDIR=/tmp",
                policy.image,
                "/bin/sh",
                "-lc",
                command,
            ]
        )
        return argv

    def _truncate_output(self, raw: bytes | str | None) -> tuple[str, bool]:
        output = _as_bytes(raw)
        truncated = len(output) > self._policy.max_output_bytes
        if truncated:
            output = output[: self._policy.max_output_bytes]
        return output.decode("utf-8", errors="replace"), truncated

    def _run_container_bounded(
        self,
        argv: list[str],
        *,
        container_name: str,
        timeout: int,
    ) -> tuple[
        subprocess.CompletedProcess[bytes] | None,
        OSError | subprocess.TimeoutExpired | None,
        bytes,
        bool,
        bool,
    ]:
        """Run the CLI with a draining, memory-bounded output pipe."""

        read_descriptor, write_descriptor = os.pipe()
        capture = _BoundedOutputCapture(self._policy.max_output_bytes)
        reader = threading.Thread(
            target=capture.drain,
            args=(read_descriptor,),
            name="opentulpa-sandbox-output",
            daemon=True,
        )
        invocation_finished = threading.Event()

        def stop_on_overflow() -> None:
            while not invocation_finished.wait(timeout=0.02):
                if not capture.overflowed.is_set():
                    continue
                self._force_remove_container(container_name)
                invocation_finished.wait(timeout=0.05)

        overflow_cleanup = threading.Thread(
            target=stop_on_overflow,
            name="opentulpa-sandbox-output-cleanup",
            daemon=True,
        )
        completed: subprocess.CompletedProcess[bytes] | None = None
        failure: OSError | subprocess.TimeoutExpired | None = None
        reader.start()
        overflow_cleanup.start()
        try:
            raw_completed = subprocess.run(
                argv,
                check=False,
                stdout=write_descriptor,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env={"PATH": os.environ.get("PATH", "")},
            )
            capture.append(_as_bytes(raw_completed.stdout))
            completed = subprocess.CompletedProcess(
                raw_completed.args,
                raw_completed.returncode,
                stdout=b"",
            )
        except subprocess.TimeoutExpired as exc:
            capture.append(_as_bytes(exc.stdout))
            failure = exc
        except OSError as exc:
            failure = exc
        finally:
            invocation_finished.set()
            with suppress(OSError):
                os.close(write_descriptor)
            reader.join(timeout=2.0)
            overflow_cleanup.join(timeout=self._policy.cleanup_timeout_seconds + 1.0)

        capture_failed = reader.is_alive() or capture.failed.is_set()
        if capture_failed:
            failure = OSError("sandbox output capture failed")
        output = capture.output()
        if completed is not None:
            completed = subprocess.CompletedProcess(
                completed.args,
                completed.returncode,
                stdout=output,
            )
        return completed, failure, output, capture.overflowed.is_set(), capture_failed

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._execute(command, timeout=timeout, control=None)

    def _execute(
        self,
        command: str,
        *,
        timeout: int | None,
        control: _ExecutionControl | None,
    ) -> ExecuteResponse:
        safe_command = str(command or "").strip()
        if not safe_command:
            return ExecuteResponse(output="command is required", exit_code=2, truncated=False)
        if len(safe_command) > self._policy.max_command_characters:
            return ExecuteResponse(
                output="command exceeds sandbox limit", exit_code=2, truncated=False
            )
        requested_timeout = (
            self._policy.timeout_seconds if timeout is None else max(1, int(timeout))
        )
        effective_timeout = min(requested_timeout, self._policy.timeout_seconds)
        if self._execution_provider is not None:
            try:
                return self._execute_with_provider(
                    safe_command,
                    timeout=effective_timeout,
                    control=control,
                )
            finally:
                if control is not None:
                    control.finished.set()

        completed: subprocess.CompletedProcess[bytes] | None = None
        failure: OSError | subprocess.TimeoutExpired | None = None
        transaction: Path | None = None
        transaction_committed = False
        container_quiescent = False
        cancelled = False
        container_name: str | None = None
        captured_output = b""
        output_overflow = False
        output_capture_failed = False
        try:
            with self._serialized_workspace():
                self._validate_workspace_tree()
                workspace: Path | None = None
                if self._persistent_workspace:
                    transaction, workspace = self._stage_workspace()
                container_name = self._new_container_name()
                if transaction is not None:
                    self._write_transaction_journal(
                        transaction,
                        phase="executing",
                        container_name=container_name,
                    )
                if control is not None:
                    control.publish_name(container_name)
                    cancelled = control.cancelled.is_set()
                argv = self._container_argv(
                    safe_command,
                    container_name=container_name,
                    workspace=workspace,
                )
                if not cancelled:
                    (
                        completed,
                        failure,
                        captured_output,
                        output_overflow,
                        output_capture_failed,
                    ) = self._run_container_bounded(
                        argv,
                        container_name=container_name,
                        timeout=effective_timeout,
                    )
                    if (
                        failure is None
                        and completed is not None
                        and not output_overflow
                        and not output_capture_failed
                    ):
                        container_quiescent = True
                        if transaction is not None:
                            self._write_transaction_journal(
                                transaction,
                                phase="quiescent",
                                container_name=container_name,
                            )
                    elif isinstance(failure, OSError) and not output_capture_failed:
                        container_quiescent = True
                if control is not None and control.cancelled.is_set():
                    cancelled = True
                if (
                    isinstance(failure, subprocess.TimeoutExpired)
                    or cancelled
                    or output_overflow
                    or output_capture_failed
                ):
                    container_quiescent = self._force_remove_container(container_name)
                elif workspace is not None and failure is None and completed is not None:
                    if transaction is None:
                        raise _WorkspaceSecurityError("workspace transaction is unavailable")
                    self._validate_workspace_tree(workspace)
                    authority = self._commit_authority
                    if authority is None:
                        self._commit_workspace(transaction, workspace)
                    else:
                        with authority():
                            self._commit_workspace(transaction, workspace)
                    transaction_committed = True
        except (_WorkspaceSecurityError, OSError):
            return ExecuteResponse(
                output="workspace failed sandbox security validation",
                exit_code=126,
                truncated=False,
            )
        finally:
            if transaction is not None and transaction.exists():
                prior_workspace_is_retained = (transaction / "previous").exists()
                if transaction_committed or (container_quiescent and not prior_workspace_is_retained):
                    with (
                        suppress(OSError, _WorkspaceSecurityError),
                        self._serialized_workspace(recover=False),
                    ):
                        if transaction.exists():
                            self._discard_transaction(transaction)
            if control is not None:
                control.finished.set()

        if isinstance(failure, FileNotFoundError):
            cli_name = Path(self._container_cli).name or "container runtime"
            return ExecuteResponse(
                output=f"sandbox unavailable: {cli_name} is not installed",
                exit_code=127,
                truncated=False,
            )
        if output_overflow:
            message = "sandbox output exceeded its limit"
            if not container_quiescent:
                message = f"{message}; sandbox cleanup is pending"
            reserved = len(message.encode("utf-8")) + 1
            keep = max(0, self._policy.max_output_bytes - reserved)
            prefix = captured_output[:keep].decode("utf-8", errors="replace").strip()
            output = f"{prefix}\n{message}".strip()
            return ExecuteResponse(output=output, exit_code=125, truncated=True)
        if isinstance(failure, subprocess.TimeoutExpired):
            message = f"command timed out after {effective_timeout}s"
            if not container_quiescent:
                message = f"{message}; sandbox cleanup is pending"
            raw = captured_output
            reserved = len(message.encode("utf-8")) + 1
            keep = max(0, self._policy.max_output_bytes - reserved)
            truncated = len(raw) > keep
            prefix = raw[:keep].decode("utf-8", errors="replace").strip()
            output = f"{prefix}\n{message}".strip()
            return ExecuteResponse(output=output, exit_code=124, truncated=truncated)
        if failure is not None:
            return ExecuteResponse(
                output="sandbox runtime failed",
                exit_code=127,
                truncated=False,
            )
        if cancelled:
            return ExecuteResponse(
                output="sandbox execution was cancelled",
                exit_code=130,
                truncated=False,
            )
        if completed is None:
            return ExecuteResponse(output="sandbox runtime failed", exit_code=127, truncated=False)
        output, truncated = self._truncate_output(completed.stdout)
        return ExecuteResponse(
            output=output,
            exit_code=int(completed.returncode),
            truncated=truncated,
        )

    def _execute_with_provider(
        self,
        command: str,
        *,
        timeout: int,
        control: _ExecutionControl | None,
    ) -> ExecuteResponse:
        transaction: Path | None = None
        committed = False
        try:
            provider = self._execution_provider
            if provider is None:
                raise RuntimeError("sandbox execution provider is unavailable")
            with self._serialized_workspace():
                self._validate_workspace_tree()
                workspace: Path | None = None
                if self._persistent_workspace:
                    transaction, workspace = self._stage_workspace()
                response = provider.execute(
                    tenant_id=self._tenant_id,
                    command=command,
                    timeout=timeout,
                    workspace=workspace,
                    cancel_event=control.cancelled if control is not None else None,
                )
                if control is not None and control.cancelled.is_set():
                    return ExecuteResponse(
                        output="sandbox execution was cancelled",
                        exit_code=130,
                        truncated=False,
                    )
                if transaction is not None and workspace is not None:
                    self._validate_workspace_tree(workspace)
                    authority = self._commit_authority
                    if authority is None:
                        self._commit_workspace(transaction, workspace)
                    else:
                        with authority():
                            self._commit_workspace(transaction, workspace)
                    committed = True
                return response
        except Exception:
            return ExecuteResponse(
                output="sandbox execution service unavailable",
                exit_code=127,
                truncated=False,
            )
        finally:
            if transaction is not None and transaction.exists():
                with suppress(OSError, _WorkspaceSecurityError):
                    if committed or not (transaction / "previous").exists():
                        self._discard_transaction(transaction)

    def _cancel_controlled_execution(self, control: _ExecutionControl) -> None:
        if self._execution_provider is not None:
            control.finished.wait(timeout=self._policy.cleanup_timeout_seconds)
            return
        if not control.name_ready.wait(timeout=1.0):
            return
        container_name = control.container_name
        if container_name is None:
            return
        deadline = time.monotonic() + self._policy.cleanup_timeout_seconds
        while not control.finished.is_set() and time.monotonic() < deadline:
            self._force_remove_container(container_name)
            control.finished.wait(timeout=0.05)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        control = _ExecutionControl()
        provider_execution = self._execution_provider is not None
        worker = asyncio.create_task(
            asyncio.to_thread(self._execute, command, timeout=timeout, control=control)
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancellation:
            control.cancelled.set()
            cleanup = asyncio.create_task(
                asyncio.to_thread(self._cancel_controlled_execution, control)
            )
            deadline = time.monotonic() + self._policy.cleanup_timeout_seconds + 1.0
            while not worker.done() or not cleanup.done():
                if provider_execution and time.monotonic() >= deadline:
                    break
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    control.cancelled.set()
            if worker.done():
                worker.result()
            else:
                worker.add_done_callback(lambda task: task.exception())
            if cleanup.done():
                cleanup.result()
            else:
                cleanup.add_done_callback(lambda task: task.exception())
            raise cancellation

    def ls(self, path: str) -> LsResult:
        with self._serialized_workspace():
            self._validate_workspace_tree()
            self._validate_virtual_path(path)
            return self._require_filesystem().ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        with self._serialized_workspace():
            self._validate_workspace_tree()
            self._validate_virtual_path(file_path)
            return self._require_filesystem().read(file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        if len(content.encode("utf-8")) > self._policy.max_file_bytes:
            raise ValueError("workspace file exceeds size limit")
        with self._serialized_workspace():
            self._validate_workspace_tree()
            self._validate_virtual_path(file_path)
            result = self._require_filesystem().write(file_path, content)
            self._validate_workspace_tree()
            return result

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if len(new_string.encode("utf-8")) > self._policy.max_file_bytes:
            raise ValueError("workspace edit exceeds size limit")
        with self._serialized_workspace():
            self._validate_workspace_tree()
            target = self._validate_virtual_path(file_path)
            if target.exists():
                descriptor = os.open(
                    target,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                    current = stream.read()
                normalized_old = old_string.replace("\r\n", "\n").replace("\r", "\n")
                normalized_new = new_string.replace("\r\n", "\n").replace("\r", "\n")
                occurrences = current.count(normalized_old)
                if (replace_all and occurrences) or (not replace_all and occurrences == 1):
                    replacement_count = occurrences if replace_all else 1
                    projected_size = len(current.encode("utf-8")) + replacement_count * (
                        len(normalized_new.encode("utf-8")) - len(normalized_old.encode("utf-8"))
                    )
                    if projected_size > self._policy.max_file_bytes:
                        raise ValueError("workspace edit exceeds size limit")
            result = self._require_filesystem().edit(file_path, old_string, new_string, replace_all)
            if target.exists() and target.stat().st_size > self._policy.max_file_bytes:
                raise _WorkspaceSecurityError("workspace file exceeded size limit")
            self._validate_workspace_tree()
            return result

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        with self._serialized_workspace():
            self._validate_workspace_tree()
            self._validate_virtual_path(pattern, pattern=True)
            if path is not None:
                self._validate_virtual_path(path)
            return self._require_filesystem().glob(pattern, path)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        with self._serialized_workspace():
            self._validate_workspace_tree()
            if path is not None:
                self._validate_virtual_path(path)
            if glob is not None:
                self._validate_virtual_path(glob, pattern=True)
            return self._require_filesystem().grep(pattern, path, glob)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        if len(files) > self._policy.max_upload_files:
            raise ValueError("workspace upload count exceeds limit")
        if any(len(content) > self._policy.max_file_bytes for _, content in files):
            raise ValueError("workspace upload exceeds size limit")
        with self._serialized_workspace():
            self._validate_workspace_tree()
            for path, _ in files:
                self._validate_virtual_path(path)
            responses = self._require_filesystem().upload_files(files)
            self._validate_workspace_tree()
            return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if len(paths) > self._policy.max_upload_files:
            raise ValueError("workspace download count exceeds limit")
        with self._serialized_workspace():
            self._validate_workspace_tree()
            targets = [self._validate_virtual_path(path) for path in paths]
            if any(
                target.exists() and target.stat().st_size > self._policy.max_file_bytes
                for target in targets
            ):
                raise ValueError("workspace download exceeds size limit")
            return self._require_filesystem().download_files(paths)


class TenantSandboxBackend(SandboxBackendProtocol):
    """Resolve the trusted run tenant at execution time without a backend factory."""

    def __init__(
        self,
        *,
        workspaces_root: str | Path,
        policy: TenantContainerPolicy | None = None,
        container_cli: str = "docker",
        persistent_files: bool = False,
        persistent_execution_workspace: bool | None = None,
        execution_provider: TenantExecutionProvider | None = None,
    ) -> None:
        self._workspaces_root = Path(workspaces_root).expanduser()
        self._policy = policy or TenantContainerPolicy()
        self._container_cli = str(container_cli or "docker").strip() or "docker"
        self._persistent_files = persistent_files
        self._execution_provider = execution_provider
        self._persistent_execution_workspace = (
            persistent_files
            if persistent_execution_workspace is None
            else bool(persistent_execution_workspace)
        )
        if self._persistent_files and not self._persistent_execution_workspace:
            raise ValueError("persistent files require a persistent execution workspace")
        self._state = StateBackend()
        self._containers: dict[str, TenantContainerBackend] = {}
        self._containers_lock = threading.Lock()

    @property
    def id(self) -> str:
        files = "workspace-files" if self._persistent_files else "state-files"
        execution = "workspace-exec" if self._persistent_execution_workspace else "scratch-exec"
        suffix = f"{files}:{execution}"
        return f"tenant-sandbox:{suffix}"

    @property
    def persistent_files(self) -> bool:
        return self._persistent_files

    @property
    def persistent_execution_workspace(self) -> bool:
        return self._persistent_execution_workspace

    def _container(self) -> TenantContainerBackend:
        runtime = get_runtime()
        context = getattr(runtime, "context", None)
        tenant_id = str(getattr(context, "tenant_id", "") or "").strip()
        if not tenant_id:
            raise RuntimeError("trusted AgentRunContext tenant_id is unavailable")
        with self._containers_lock:
            backend = self._containers.get(tenant_id)
            if backend is None:
                backend = TenantContainerBackend(
                    tenant_id=tenant_id,
                    workspaces_root=self._workspaces_root,
                    policy=self._policy,
                    container_cli=self._container_cli,
                    persistent_workspace=self._persistent_execution_workspace,
                    execution_provider=self._execution_provider,
                )
                self._containers[tenant_id] = backend
            return backend

    def _files(self) -> TenantContainerBackend | StateBackend:
        return self._container() if self._persistent_files else self._state

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._container().execute(command, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return await self._container().aexecute(command, timeout=timeout)

    def ls(self, path: str) -> LsResult:
        return self._files().ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._files().read(file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._files().write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._files().edit(file_path, old_string, new_string, replace_all)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self._files().glob(pattern, path)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return self._files().grep(pattern, path, glob)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._files().upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._files().download_files(paths)
