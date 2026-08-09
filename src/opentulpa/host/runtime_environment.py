"""Host-owned runtime environments for live source releases."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
import sysconfig
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import JsonValue

from opentulpa.evolution.git_security import (
    discover_git_directories,
    repository_mutation_lock,
    run_hardened_git,
)
from opentulpa.evolution.process import BoundedProcessResult, run_bounded_process

_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_WRITABLE_ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]{0,127}\Z")
_RAW_ENV_VALUE_RE = re.compile(r"[A-Za-z0-9_./:@%+=,\-]*\Z")
_DOTENV_MAX_BYTES = 256 * 1024
_DOTENV_MAX_VALUE_BYTES = 64 * 1024
_DOTENV_MAX_KEYS = 512
_TRUSTED_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_INSTALL_PROFILE = "runtime-no-dev-no-install-project-v1"
_METADATA_FILENAME = "runtime-env.json"
_SECRET_ENV_RE = re.compile(r"(api[_-]?key|authorization|cookie|password|passwd|secret|token)", re.I)
_PROTECTED_ENVIRONMENT_KEYS = frozenset(
    {
        "CONDA_PREFIX",
        "DYLD_INSERT_LIBRARIES",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "HOME",
        "HOST",
        "LD_PRELOAD",
        "PATH",
        "PORT",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
        "OPENTULPA_APPLICATION_ROOT",
        "OPENTULPA_DATA_ROOT",
        "OPENTULPA_DYNAMIC_HOST",
        "OPENTULPA_INTERNAL_AGENT_API_URL",
        "OPENTULPA_LAUNCH_NONCE",
        "OPENTULPA_LIVE_SOURCE_ROOT",
        "OPENTULPA_OWNER_CUSTOMER_ID",
        "OPENTULPA_OWNER_TOKEN",
        "OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH",
        "OPENTULPA_SOURCE_COMMIT",
        "OPENTULPA_TELEGRAM_OWNER_ID",
        "OPENTULPA_TELEGRAM_PAIRING_CODE",
    }
)
_PROTECTED_ENVIRONMENT_PREFIXES = (
    "OPENTULPA_BOOTSTRAP_",
    "OPENTULPA_SANDBOX_RPC_",
)


class RuntimeEnvironmentError(RuntimeError):
    """Sanitized runtime-environment failure with a stable stage label."""

    def __init__(self, code: str, public_message: str, *, stage: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.stage = stage


class BoundedCommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> BoundedProcessResult: ...


class RuntimeEnvironmentRestartPort(Protocol):
    @property
    def status(self) -> str: ...

    async def replace_current_environment(
        self,
        *,
        apply: Callable[[], None],
        restore: Callable[[], None],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class LiveSourceRuntimeEnvironment:
    id: str
    source_commit: str
    python_interpreter: Path
    dependency_lock_hash: str
    pyproject_sha256: str
    install_profile: str

    def release_metadata(self) -> dict[str, JsonValue]:
        return {
            "runtime_environment_id": self.id,
            "runtime_python_interpreter": str(self.python_interpreter),
            "runtime_dependency_lock_hash": self.dependency_lock_hash,
            "runtime_pyproject_sha256": self.pyproject_sha256,
            "runtime_install_profile": self.install_profile,
        }


class LiveSourceRuntimeEnvironmentStore:
    """Prepare one immutable dependency venv per evaluated live-source commit."""

    def __init__(
        self,
        *,
        source_repository: Path,
        envs_root: Path,
        worktrees_root: Path,
        uv_cli: str,
        python_executable: str = sys.executable,
        install_profile: str = _INSTALL_PROFILE,
        timeout_seconds: int = 1_800,
        max_output_bytes: int = 1_000_000,
        runner: BoundedCommandRunner | None = None,
    ) -> None:
        self._repository = source_repository.expanduser().resolve(strict=True)
        self._envs_root = envs_root.expanduser().absolute()
        self._worktrees_root = worktrees_root.expanduser().absolute()
        self._uv_cli = self._trusted_executable(uv_cli, expected_name="uv")
        self._python_executable = self._trusted_executable(
            python_executable,
            expected_name=Path(python_executable).name,
        )
        self._install_profile = str(install_profile or "").strip()
        if not self._install_profile or "\x00" in self._install_profile:
            raise ValueError("runtime install profile is invalid")
        if timeout_seconds < 60 or max_output_bytes < 1_024:
            raise ValueError("runtime environment builder limits are invalid")
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._runner = runner or run_bounded_process
        self._prepare_root(self._envs_root, mode=0o711)
        self._prepare_root(self._worktrees_root, mode=0o700)
        self._lock_path = self._envs_root / ".runtime-envs.lock"

    def prepare(
        self,
        source_commit: str,
        *,
        workspace: Path | None = None,
    ) -> LiveSourceRuntimeEnvironment:
        safe_commit = self._validate_commit(source_commit)
        if workspace is not None:
            return self._prepare_from_workspace(safe_commit, workspace.expanduser().resolve(strict=True))
        workspace_path = self._create_worktree(safe_commit)
        try:
            return self._prepare_from_workspace(safe_commit, workspace_path)
        finally:
            self._remove_worktree(workspace_path)

    def _prepare_from_workspace(
        self,
        source_commit: str,
        workspace: Path,
    ) -> LiveSourceRuntimeEnvironment:
        self._verify_workspace(source_commit, workspace)
        pyproject_hash = self._regular_file_sha256(workspace / "pyproject.toml", label="pyproject")
        lock_hash = self._regular_file_sha256(workspace / "uv.lock", label="uv lock")
        env_id = self._environment_id(
            source_commit=source_commit,
            pyproject_sha256=pyproject_hash,
            dependency_lock_hash=lock_hash,
        )
        target = self._envs_root / env_id
        with self._locked():
            existing = self._open_complete_environment(
                target,
                source_commit=source_commit,
                dependency_lock_hash=lock_hash,
                pyproject_sha256=pyproject_hash,
            )
            if existing is not None:
                return existing
            if target.exists():
                self._remove_tree(target)
            staging = self._envs_root / f".{env_id}.build-{uuid4().hex}"
            try:
                self._run_uv_sync(workspace, staging)
                interpreter = self._python_interpreter(staging)
                assert interpreter is not None
                environment = LiveSourceRuntimeEnvironment(
                    id=env_id,
                    source_commit=source_commit,
                    python_interpreter=interpreter,
                    dependency_lock_hash=lock_hash,
                    pyproject_sha256=pyproject_hash,
                    install_profile=self._install_profile,
                )
                self._write_environment_metadata(staging, environment)
                self._make_tree_runtime_readable(staging)
                os.replace(staging, target)
                self._fsync_directory(self._envs_root)
                return environment
            except RuntimeEnvironmentError:
                raise
            except Exception as exc:
                raise RuntimeEnvironmentError(
                    "runtime_dependency_install_failed",
                    "Runtime dependency environment preparation failed.",
                    stage="dependency_install",
                ) from exc
            finally:
                if staging.exists():
                    self._remove_tree(staging)

    def _run_uv_sync(self, workspace: Path, target: Path) -> None:
        target.mkdir(mode=0o700)
        home = self._worktrees_root / ".uv-home"
        home.mkdir(mode=0o700, exist_ok=True)
        result = self._runner(
            (
                str(self._uv_cli),
                "sync",
                "--frozen",
                "--no-dev",
                "--no-install-project",
                "--project",
                str(workspace),
                "--python",
                str(self._python_executable),
            ),
            cwd=workspace,
            env={
                "HOME": str(home),
                "PATH": f"{self._uv_cli.parent}:{_TRUSTED_SYSTEM_PATH}",
                "PYTHONNOUSERSITE": "1",
                "UV_PROJECT_ENVIRONMENT": str(target),
            },
            timeout_seconds=self._timeout_seconds,
            max_output_bytes=self._max_output_bytes,
        )
        if result.returncode != 0 or result.timed_out or result.truncated:
            detail = result.output.decode("utf-8", errors="replace").strip()
            suffix = f" Output: {detail[:1_000]}" if detail else ""
            raise RuntimeEnvironmentError(
                "runtime_dependency_install_failed",
                f"Runtime dependency install failed before activation.{suffix}",
                stage="dependency_install",
            )
        self._python_interpreter(target)

    def _environment_id(
        self,
        *,
        source_commit: str,
        pyproject_sha256: str,
        dependency_lock_hash: str,
    ) -> str:
        payload = {
            "format_version": 1,
            "install_profile": self._install_profile,
            "python": self._python_runtime_identity(),
            "pyproject_sha256": pyproject_sha256,
            "source_commit": source_commit,
            "sync": ["--frozen", "--no-dev", "--no-install-project"],
            "uv_lock_sha256": dependency_lock_hash,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def _python_runtime_identity(self) -> dict[str, str]:
        return {
            "abi_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
            "cache_tag": str(sys.implementation.cache_tag or ""),
            "executable": str(self._python_executable),
            "machine": platform.machine(),
            "platform": sysconfig.get_platform(),
            "soabi": str(sysconfig.get_config_var("SOABI") or ""),
            "version": platform.python_version(),
        }

    def _create_worktree(self, source_commit: str) -> Path:
        workspace = self._worktrees_root / f"runtime-env-{source_commit[:12]}-{uuid4().hex}"
        _, common_directory = discover_git_directories(self._repository)
        with repository_mutation_lock(common_directory):
            self._git(
                self._repository,
                "worktree",
                "add",
                "--detach",
                str(workspace),
                source_commit,
            )
        return workspace

    def _remove_worktree(self, workspace: Path) -> None:
        _, common_directory = discover_git_directories(self._repository)
        with repository_mutation_lock(common_directory):
            self._git(
                self._repository,
                "worktree",
                "remove",
                "--force",
                str(workspace),
                check=False,
            )
        if workspace.exists():
            self._remove_tree(workspace)

    def _verify_workspace(self, source_commit: str, workspace: Path) -> None:
        if workspace.is_symlink() or not workspace.is_dir():
            raise RuntimeEnvironmentError(
                "runtime_source_unavailable",
                "Runtime source workspace is unavailable.",
                stage="source_verify",
            )
        resolved = self._git(workspace, "rev-parse", "--verify", "HEAD^{commit}").strip()
        if resolved != source_commit:
            raise RuntimeEnvironmentError(
                "runtime_source_unavailable",
                "Runtime source workspace does not match the release commit.",
                stage="source_verify",
            )
        if self._git(workspace, "status", "--porcelain=v1", "--untracked-files=all", "-z"):
            raise RuntimeEnvironmentError(
                "runtime_source_dirty",
                "Runtime source workspace is dirty.",
                stage="source_verify",
            )

    def _open_complete_environment(
        self,
        root: Path,
        *,
        source_commit: str,
        dependency_lock_hash: str,
        pyproject_sha256: str,
    ) -> LiveSourceRuntimeEnvironment | None:
        metadata_path = root / _METADATA_FILENAME
        interpreter = self._python_interpreter(root, required=False)
        if interpreter is None or not metadata_path.is_file() or metadata_path.is_symlink():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        environment = LiveSourceRuntimeEnvironment(
            id=str(payload.get("id") or ""),
            source_commit=str(payload.get("source_commit") or ""),
            python_interpreter=interpreter,
            dependency_lock_hash=str(payload.get("dependency_lock_hash") or ""),
            pyproject_sha256=str(payload.get("pyproject_sha256") or ""),
            install_profile=str(payload.get("install_profile") or ""),
        )
        if (
            environment.id != root.name
            or environment.source_commit != source_commit
            or environment.dependency_lock_hash != dependency_lock_hash
            or environment.pyproject_sha256 != pyproject_sha256
            or environment.install_profile != self._install_profile
        ):
            return None
        return environment

    def _write_environment_metadata(
        self,
        root: Path,
        environment: LiveSourceRuntimeEnvironment,
    ) -> None:
        payload = {
            "format_version": 1,
            "id": environment.id,
            "source_commit": environment.source_commit,
            "python_interpreter": str(environment.python_interpreter),
            "dependency_lock_hash": environment.dependency_lock_hash,
            "pyproject_sha256": environment.pyproject_sha256,
            "install_profile": environment.install_profile,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        path = root / _METADATA_FILENAME
        path.write_bytes(encoded + b"\n")
        path.chmod(0o444)

    @staticmethod
    def _python_interpreter(root: Path, *, required: bool = True) -> Path | None:
        interpreter = root / "bin" / "python"
        if interpreter.is_file() and os.access(interpreter, os.X_OK):
            return interpreter.absolute()
        if required:
            raise RuntimeEnvironmentError(
                "runtime_dependency_install_failed",
                "Runtime dependency install did not produce an executable Python.",
                stage="dependency_install",
            )
        return None

    @staticmethod
    def _regular_file_sha256(path: Path, *, label: str) -> str:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeEnvironmentError(
                "runtime_source_unavailable",
                f"Runtime {label} file is unavailable.",
                stage="source_verify",
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeEnvironmentError(
                "runtime_source_unavailable",
                f"Runtime {label} file is unsafe.",
                stage="source_verify",
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _validate_commit(source_commit: str) -> str:
        safe = str(source_commit or "").strip().lower()
        if _COMMIT_RE.fullmatch(safe) is None:
            raise RuntimeEnvironmentError(
                "runtime_source_unavailable",
                "Runtime source commit identity is invalid.",
                stage="source_verify",
            )
        return safe

    @staticmethod
    def _trusted_executable(raw_path: str, *, expected_name: str) -> Path:
        path = Path(str(raw_path or "")).expanduser()
        if not path.is_absolute() or not path.is_file() or "\x00" in str(path):
            raise ValueError(f"trusted executable {expected_name} is unavailable")
        if path.name != expected_name or not os.access(path, os.X_OK):
            raise ValueError(f"trusted executable {expected_name} is invalid")
        return path.absolute()

    @staticmethod
    def _prepare_root(path: Path, *, mode: int) -> None:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("runtime environment root is unsafe")
        path.chmod(mode)

    def _locked(self) -> Any:
        return _FileLock(self._lock_path)

    @staticmethod
    def _make_tree_runtime_readable(root: Path) -> None:
        for directory, directory_names, file_names in os.walk(root, topdown=False, followlinks=False):
            directory_path = Path(directory)
            for name in file_names:
                path = directory_path / name
                if path.is_symlink():
                    continue
                mode = 0o555 if os.access(path, os.X_OK) else 0o444
                path.chmod(mode)
            for name in directory_names:
                path = directory_path / name
                if path.is_symlink():
                    continue
                path.chmod(0o555)
        root.chmod(0o555)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not path.exists():
            return
        for directory, directory_names, file_names in os.walk(path, topdown=False, followlinks=False):
            directory_path = Path(directory)
            with os.scandir(directory_path) as entries:
                for entry in entries:
                    with suppress(OSError):
                        candidate = Path(entry.path)
                        if not candidate.is_symlink():
                            candidate.chmod(0o700)
            for name in directory_names:
                with suppress(OSError):
                    candidate = directory_path / name
                    if not candidate.is_symlink():
                        candidate.chmod(0o700)
            for name in file_names:
                with suppress(OSError):
                    candidate = directory_path / name
                    if not candidate.is_symlink():
                        candidate.chmod(0o600)
            with suppress(OSError):
                directory_path.chmod(0o700)
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _git(
        repository: Path,
        *arguments: str,
        check: bool = True,
    ) -> str:
        result = run_hardened_git(
            repository,
            tuple(arguments),
            env={
                "GIT_AUTHOR_NAME": "OpenTulpa Host",
                "GIT_AUTHOR_EMAIL": "host@opentulpa.local",
                "GIT_COMMITTER_NAME": "OpenTulpa Host",
                "GIT_COMMITTER_EMAIL": "host@opentulpa.local",
            },
            timeout_seconds=120,
            max_output_bytes=10 * 1024 * 1024,
        )
        if check and (result.returncode != 0 or result.truncated):
            raise RuntimeEnvironmentError(
                "runtime_source_unavailable",
                "Runtime source Git operation failed.",
                stage="source_verify",
            )
        return result.output.decode("utf-8", errors="replace")


class RuntimeEnvFileManager:
    """Atomically update the live-source .env and restart the child with rollback."""

    def __init__(self, *, source_root: Path, runtime: RuntimeEnvironmentRestartPort) -> None:
        self._source_root = source_root.expanduser().resolve(strict=True)
        if self._source_root.is_symlink() or not self._source_root.is_dir():
            raise ValueError("runtime .env source root is invalid")
        self._path = self._source_root / ".env"
        self._runtime = runtime
        self._lock = asyncio.Lock()

    async def read(self) -> dict[str, JsonValue]:
        try:
            async with self._lock:
                payload = await asyncio.to_thread(self._read_payload)
                values = parse_dotenv_payload(payload or b"")
        except RuntimeEnvironmentError as exc:
            return {
                "available": False,
                "variables": [],
                "count": 0,
                "error": {
                    "code": exc.code,
                    "message": exc.public_message,
                    "retryable": False,
                },
            }
        variables: list[JsonValue] = [
            {"name": name, "value": value, "set": True}
            for name, value in sorted(values.items())
        ]
        return {"available": True, "variables": variables, "count": len(variables)}

    async def set(
        self,
        *,
        name: str,
        value: str,
        idempotency_key: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        del audit_context
        try:
            safe_name = self._validate_writable_name(name)
            safe_value = self._validate_value(value)
            safe_key = str(idempotency_key or "").strip()
            if not safe_key or len(safe_key) > 200:
                raise RuntimeEnvironmentError(
                    "runtime_env_idempotency_key_invalid",
                    "Runtime .env idempotency key is invalid.",
                    stage="env_write",
                )
        except RuntimeEnvironmentError as exc:
            return self._failure_result(str(name or "")[:128], exc, rollback_restored=True)
        async with self._lock:
            try:
                previous = await asyncio.to_thread(self._read_payload)
            except RuntimeEnvironmentError as exc:
                return self._failure_result(safe_name, exc, rollback_restored=True)
            try:
                parsed = parse_dotenv_payload(previous or b"")
                new_payload = self._updated_payload(previous, parsed, name=safe_name, value=safe_value)
            except RuntimeEnvironmentError as exc:
                return self._failure_result(safe_name, exc, rollback_restored=True)

            state = {"applied": False}

            def apply_update() -> None:
                self._write_payload(new_payload)
                state["applied"] = True

            def restore_previous() -> None:
                self._write_payload(previous)

            try:
                await self._runtime.replace_current_environment(
                    apply=apply_update,
                    restore=restore_previous,
                )
            except Exception as exc:
                stage = "runtime_restart" if state["applied"] else "env_write"
                restored = await asyncio.to_thread(self._payload_matches, previous)
                error = RuntimeEnvironmentError(
                    "runtime_env_update_failed",
                    "Runtime .env update failed; previous environment was restored."
                    if restored
                    else "Runtime .env update failed and previous environment could not be restored.",
                    stage=stage,
                )
                return self._failure_result(
                    safe_name,
                    error,
                    rollback_restored=restored,
                    cause=exc,
                )
        return {
            "status": "updated",
            "name": safe_name,
            "changed": previous != new_payload,
            "restarted": True,
            "dotenv_sha256": hashlib.sha256(new_payload).hexdigest(),
            "value": "[set]",
        }

    @staticmethod
    def _failure_result(
        name: str,
        error: RuntimeEnvironmentError,
        *,
        rollback_restored: bool,
        cause: Exception | None = None,
    ) -> dict[str, JsonValue]:
        del cause
        return {
            "status": "failed",
            "name": name,
            "changed": False,
            "restarted": False,
            "rollback_restored": rollback_restored,
            "failure_stage": error.stage,
            "error": {
                "code": error.code,
                "message": error.public_message,
                "retryable": error.stage in {"runtime_restart", "dependency_install"},
            },
            "value": "[redacted]",
        }

    def _read_payload(self) -> bytes | None:
        return read_runtime_dotenv_payload(self._path)

    def _write_payload(self, payload: bytes | None) -> None:
        self._source_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self._path.is_symlink():
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                "Runtime .env file is unsafe.",
                stage="env_write",
            )
        if payload is None:
            try:
                self._path.unlink(missing_ok=True)
                self._fsync_directory(self._source_root)
            except OSError as exc:
                raise RuntimeEnvironmentError(
                    "runtime_env_write_failed",
                    "Runtime .env file could not be restored.",
                    stage="env_write",
                ) from exc
            return
        temporary = self._source_root / f".env.{uuid4().hex}.tmp"
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written < 1:
                    raise OSError("runtime .env write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, self._path)
            self._path.chmod(0o600)
            self._fsync_directory(self._source_root)
        except OSError as exc:
            raise RuntimeEnvironmentError(
                "runtime_env_write_failed",
                "Runtime .env file could not be written.",
                stage="env_write",
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(OSError):
                temporary.unlink()

    def _payload_matches(self, expected: bytes | None) -> bool:
        try:
            return self._read_payload() == expected
        except RuntimeEnvironmentError:
            return False

    def _updated_payload(
        self,
        previous: bytes | None,
        parsed: Mapping[str, str],
        *,
        name: str,
        value: str,
    ) -> bytes:
        updated = {**dict(parsed), name: value}
        lines = (previous or b"").decode("utf-8", errors="strict").splitlines()
        output: list[str] = []
        replaced = False
        for line in lines:
            key = dotenv_line_key(line)
            if key != name:
                output.append(line)
                continue
            if not replaced:
                output.append(f"{name}={format_dotenv_value(value)}")
                replaced = True
        if not replaced:
            if output and output[-1].strip():
                output.append("")
            output.append(f"{name}={format_dotenv_value(value)}")
        payload = ("\n".join(output).rstrip("\n") + "\n").encode("utf-8")
        if len(payload) > _DOTENV_MAX_BYTES:
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                "Runtime .env file would exceed its size limit.",
                stage="env_write",
            )
        verified = parse_dotenv_payload(payload)
        if verified.get(name) != updated[name]:
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                "Runtime .env update could not be verified.",
                stage="env_write",
            )
        return payload

    @staticmethod
    def _validate_writable_name(name: str) -> str:
        safe = str(name or "").strip()
        if _WRITABLE_ENV_NAME_RE.fullmatch(safe) is None:
            raise RuntimeEnvironmentError(
                "runtime_env_name_invalid",
                "Runtime .env variable name is invalid.",
                stage="env_write",
            )
        if runtime_env_key_is_protected(safe):
            raise RuntimeEnvironmentError(
                "runtime_env_key_protected",
                "Runtime .env variable is owned by the host.",
                stage="env_write",
            )
        return safe

    @staticmethod
    def _validate_value(value: str) -> str:
        text = str(value)
        encoded = text.encode("utf-8")
        if "\x00" in text or "\n" in text or "\r" in text or len(encoded) > _DOTENV_MAX_VALUE_BYTES:
            raise RuntimeEnvironmentError(
                "runtime_env_value_invalid",
                "Runtime .env variable value is invalid.",
                stage="env_write",
            )
        return text

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class _FileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def __enter__(self) -> _FileLock:
        descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        self._descriptor = descriptor
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def runtime_env_key_is_protected(key: str) -> bool:
    value = str(key or "").strip()
    return value in _PROTECTED_ENVIRONMENT_KEYS or any(
        value.startswith(prefix) for prefix in _PROTECTED_ENVIRONMENT_PREFIXES
    )


def load_runtime_dotenv(path: Path) -> dict[str, str]:
    payload = read_runtime_dotenv_payload(path)
    if payload is None:
        return {}
    return parse_dotenv_payload(payload)


def read_runtime_dotenv_payload(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeEnvironmentError(
            "runtime_env_file_invalid",
            "Runtime .env file is unsafe.",
            stage="env_read",
        ) from exc
    _validate_runtime_dotenv_metadata(metadata)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                "Runtime .env file is unsafe.",
                stage="env_read",
            )
        _validate_runtime_dotenv_metadata(opened)
        return _read_limited_dotenv_descriptor(descriptor)
    except RuntimeEnvironmentError:
        raise
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeEnvironmentError(
            "runtime_env_file_invalid",
            "Runtime .env file is unsafe.",
            stage="env_read",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_runtime_dotenv_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeEnvironmentError(
            "runtime_env_file_invalid",
            "Runtime .env file is unsafe.",
            stage="env_read",
        )
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise RuntimeEnvironmentError(
            "runtime_env_file_invalid",
            "Runtime .env file must be owned by the runtime user and readable only by that user.",
            stage="env_read",
        )
    if metadata.st_size > _DOTENV_MAX_BYTES:
        raise RuntimeEnvironmentError(
            "runtime_env_file_invalid",
            "Runtime .env file exceeds its size limit.",
            stage="env_read",
        )


def _read_limited_dotenv_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, 64 * 1024)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > _DOTENV_MAX_BYTES:
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                "Runtime .env file exceeds its size limit.",
                stage="env_read",
            )
        chunks.append(chunk)


def parse_dotenv_payload(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeEnvironmentError(
            "runtime_env_file_invalid",
            "Runtime .env file must be UTF-8.",
            stage="env_read",
        ) from exc
    values: dict[str, str] = {}
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                f"Runtime .env line {index} is invalid.",
                stage="env_read",
            )
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip()
        if _ENV_NAME_RE.fullmatch(key) is None:
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                f"Runtime .env line {index} has an invalid key.",
                stage="env_read",
            )
        value = parse_dotenv_value(raw_value.strip(), line_number=index)
        if "\x00" in value or len(value.encode("utf-8")) > _DOTENV_MAX_VALUE_BYTES:
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                f"Runtime .env line {index} value is invalid.",
                stage="env_read",
            )
        values[key] = value
        if len(values) > _DOTENV_MAX_KEYS:
            raise RuntimeEnvironmentError(
                "runtime_env_file_invalid",
                "Runtime .env file has too many keys.",
                stage="env_read",
            )
    return values


def parse_dotenv_value(raw_value: str, *, line_number: int) -> str:
    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] == "'":
        return raw_value[1:-1]
    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] == '"':
        return _decode_double_quoted_dotenv(raw_value[1:-1], line_number=line_number)
    if raw_value.startswith(("'", '"')):
        raise RuntimeEnvironmentError(
            "runtime_env_file_invalid",
            f"Runtime .env line {line_number} has an unterminated quoted value.",
            stage="env_read",
        )
    return _strip_unquoted_comment(raw_value).strip()


def _decode_double_quoted_dotenv(value: str, *, line_number: int) -> str:
    output: list[str] = []
    escaped = False
    for character in value:
        if escaped:
            output.append(
                {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "\\": "\\",
                    '"': '"',
                }.get(character, character)
            )
            escaped = False
        elif character == "\\":
            escaped = True
        else:
            output.append(character)
    if escaped:
        raise RuntimeEnvironmentError(
            "runtime_env_file_invalid",
            f"Runtime .env line {line_number} has an invalid escape.",
            stage="env_read",
        )
    return "".join(output)


def _strip_unquoted_comment(value: str) -> str:
    for index, character in enumerate(value):
        if character == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value


def dotenv_line_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").lstrip()
    if "=" not in stripped:
        return None
    key = stripped.split("=", 1)[0].strip()
    return key if _ENV_NAME_RE.fullmatch(key) is not None else None


def format_dotenv_value(value: str) -> str:
    if _RAW_ENV_VALUE_RE.fullmatch(value) is not None:
        return value
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def filtered_runtime_dotenv(values: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in values.items()
        if _ENV_NAME_RE.fullmatch(key) is not None and not runtime_env_key_is_protected(key)
    }


def runtime_dotenv_secret_values(values: Mapping[str, str]) -> set[str]:
    return {
        value
        for key, value in values.items()
        if value and _SECRET_ENV_RE.search(key) is not None
    }


__all__ = [
    "LiveSourceRuntimeEnvironment",
    "LiveSourceRuntimeEnvironmentStore",
    "RuntimeEnvFileManager",
    "RuntimeEnvironmentError",
    "filtered_runtime_dotenv",
    "format_dotenv_value",
    "load_runtime_dotenv",
    "parse_dotenv_payload",
    "runtime_dotenv_secret_values",
    "runtime_env_key_is_protected",
]
