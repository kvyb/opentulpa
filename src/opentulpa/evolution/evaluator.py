"""Trusted, reproducible evaluation runners for generated candidates."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from opentulpa.evolution.process import run_bounded_process
from opentulpa.evolution.sandbox import (
    _build_bubblewrap_argv,
    _strong_sandbox_tools,
    strong_sandbox_unavailable_reason,
)

EvaluationStage = Literal["build", "contract", "public", "security"]

_IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}\Z")
_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|authorization|password|secret|token)\s*[:=]\s*\S+")
_TRUSTED_OWNER_UID = 0


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


def _sanitize_output(value: str, *, limit: int = 8_000) -> str:
    compact = value.replace("\x00", "")
    redacted = _SECRET_RE.sub(lambda match: f"{match.group(1)}=[redacted]", compact)
    return redacted[:limit]


@dataclass(frozen=True, slots=True)
class EvaluationCommand:
    """One supervisor-owned evaluation command."""

    name: str
    argv: tuple[str, ...]
    stage: EvaluationStage = "public"
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,99}", self.name):
            raise ValueError("evaluation command name is invalid")
        if not self.argv or any(not item or "\x00" in item for item in self.argv):
            raise ValueError("evaluation argv is invalid")
        if self.timeout_seconds < 1 or self.timeout_seconds > 3_600:
            raise ValueError("evaluation timeout must be between 1 and 3600 seconds")


@dataclass(frozen=True, slots=True)
class EvaluationCommandResult:
    """Sanitized result safe to persist with candidate metadata."""

    name: str
    stage: EvaluationStage
    passed: bool
    exit_code: int
    duration_seconds: float
    output: str = ""


@dataclass(frozen=True, slots=True)
class EvaluationExecutables:
    """Executable names or exact paths used by the trusted evaluation gates."""

    python: str = "python"
    ruff: str = "ruff"
    mypy: str = "mypy"
    pytest: str = "pytest"

    def __post_init__(self) -> None:
        if any(not value or "\x00" in value for value in self.values()):
            raise ValueError("evaluation executable is invalid")

    def values(self) -> tuple[str, ...]:
        return self.python, self.ruff, self.mypy, self.pytest

    @classmethod
    def isolated(cls, controller_root: str | Path = "/controller") -> EvaluationExecutables:
        binary_root = str(controller_root).rstrip("/") + "/bin"
        return cls(
            python=f"{binary_root}/python",
            ruff=f"{binary_root}/ruff",
            mypy=f"{binary_root}/mypy",
            pytest=f"{binary_root}/pytest",
        )


class EvaluationRunner(Protocol):
    """Execute one fixed command against a disposable candidate checkout."""

    @property
    def fingerprint(self) -> str: ...

    async def run(
        self,
        *,
        workspace: Path,
        command: EvaluationCommand,
    ) -> EvaluationCommandResult: ...


class LocalEvaluationRunner:
    """Development-only runner; generated candidate code executes on the host."""

    def __init__(self, *, extra_env: dict[str, str] | None = None) -> None:
        self._env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            **(extra_env or {}),
        }

    @property
    def fingerprint(self) -> str:
        return "local-development-runner-v1"

    async def run(
        self,
        *,
        workspace: Path,
        command: EvaluationCommand,
    ) -> EvaluationCommandResult:
        root = _validated_workspace(workspace)
        return await asyncio.to_thread(
            _run_process,
            list(command.argv),
            root,
            command,
            self._env,
        )


class IsolatedProcessEvaluationRunner:
    """Run fixed gates in the same strong namespace sandbox as source commands."""

    def __init__(
        self,
        *,
        controller_root: Path,
        wheelhouse: Path,
        uid: int = 65_533,
        gid: int = 65_533,
        memory_bytes: int = 2 * 1024 * 1024 * 1024,
        pid_limit: int = 256,
        max_output_bytes: int = 1_000_000,
        temporary_root: Path = Path("/var/tmp/opentulpa-evaluation"),
        dependency_site: Path | None = None,
    ) -> None:
        unavailable = self.unavailable_reason(
            controller_root=controller_root,
            wheelhouse=wheelhouse,
            dependency_site=dependency_site,
        )
        if unavailable is not None:
            raise RuntimeError(unavailable)
        if uid < 1 or gid < 1 or uid == 65_532 or gid == 65_532:
            raise ValueError("isolated process evaluation identity is invalid")
        if memory_bytes < 64 * 1024 * 1024:
            raise ValueError("isolated process evaluation limits are invalid")
        if pid_limit < 16 or pid_limit > 4_096 or max_output_bytes < 1_024:
            raise ValueError("isolated process evaluation limits are invalid")
        safe_temporary_root = temporary_root.expanduser()
        if safe_temporary_root.is_symlink():
            raise ValueError("isolated process evaluation temporary root is invalid")
        safe_temporary_root.mkdir(parents=True, exist_ok=True, mode=0o711)
        os.chown(safe_temporary_root, 0, 0)
        safe_temporary_root.chmod(0o711)
        tools = _strong_sandbox_tools()
        if isinstance(tools, str):
            raise RuntimeError(tools)
        self._setpriv, self._bwrap, self._prlimit = tools
        self._uid = uid
        self._gid = gid
        self._memory_bytes = memory_bytes
        self._pid_limit = pid_limit
        self._max_output_bytes = max_output_bytes
        self._temporary_root = safe_temporary_root.resolve(strict=True)
        self._controller_root = controller_root.expanduser().absolute()
        self._wheelhouse = wheelhouse.expanduser().absolute()
        self._dependency_site = (
            dependency_site.expanduser().absolute() if dependency_site is not None else None
        )
        self._executables = EvaluationExecutables.isolated(self._controller_root)
        self._host_executables = {
            isolated: self._controller_root / "bin" / Path(isolated).name
            for isolated in self._executables.values()
        }
        self._fingerprint = self._input_fingerprint()

    @classmethod
    def unavailable_reason(
        cls,
        *,
        controller_root: Path,
        wheelhouse: Path,
        dependency_site: Path | None = None,
    ) -> str | None:
        unavailable = strong_sandbox_unavailable_reason()
        if unavailable is not None:
            return unavailable
        roots = [
            (controller_root, "controller generation"),
            (wheelhouse, "offline wheelhouse"),
        ]
        if dependency_site is not None:
            roots.append((dependency_site, "resolved dependency site"))
        for path, label in roots:
            reason = cls._read_only_root_reason(path, label=label)
            if reason is not None:
                return reason
        for name in EvaluationExecutables.isolated(controller_root).values():
            executable = controller_root.expanduser().absolute() / "bin" / Path(name).name
            reason = cls._executable_reason(executable)
            if reason is not None:
                return reason
        try:
            _evaluation_package_inputs(
                controller_root.expanduser().absolute(),
                wheelhouse.expanduser().absolute(),
            )
        except (OSError, RuntimeError, ValueError):
            return "isolated evaluation exact package or wheelhouse inputs are unavailable"
        return None

    @classmethod
    def is_supported(cls, *, controller_root: Path, wheelhouse: Path) -> bool:
        return (
            cls.unavailable_reason(
                controller_root=controller_root,
                wheelhouse=wheelhouse,
            )
            is None
        )

    @property
    def executables(self) -> EvaluationExecutables:
        return self._executables

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def run(
        self,
        *,
        workspace: Path,
        command: EvaluationCommand,
    ) -> EvaluationCommandResult:
        root = _validated_workspace(workspace)
        await asyncio.to_thread(self._require_immutable_source, root)
        evaluation_root, evaluation_workspace = await asyncio.to_thread(
            self._evaluation_copy,
            root,
        )
        try:
            if command.argv[0] not in self._host_executables:
                raise ValueError("isolated evaluator command is not an exact trusted executable")
            argv = _build_bubblewrap_argv(
                setpriv=self._setpriv,
                bwrap=self._bwrap,
                prlimit=self._prlimit,
                uid=self._uid,
                gid=self._gid,
                workspace=evaluation_workspace,
                workspace_read_only=False,
                read_only_mounts=(
                    (self._controller_root, str(self._controller_root)),
                    (self._wheelhouse, str(self._wheelhouse)),
                    *(
                        ((self._dependency_site, "/dependency-site"),)
                        if self._dependency_site is not None
                        else ()
                    ),
                ),
                environment={
                    "HOME": "/tmp",
                    "MYPY_CACHE_DIR": "/tmp/mypy-cache",
                    **(
                        {
                            "PYTHONPATH": "/dependency-site",
                        }
                        if self._dependency_site is not None
                        else {}
                    ),
                    "PATH": (f"{self._controller_root}/bin:/usr/local/bin:/usr/bin:/bin"),
                    "PIP_CONFIG_FILE": "/dev/null",
                    "PIP_FIND_LINKS": str(self._wheelhouse),
                    "PIP_NO_INDEX": "1",
                    "PYTEST_ADDOPTS": "-p no:cacheprovider",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "PYTHONPYCACHEPREFIX": "/tmp/pycache",
                    "RUFF_CACHE_DIR": "/tmp/ruff-cache",
                    "TMPDIR": "/tmp",
                    "UV_FIND_LINKS": str(self._wheelhouse),
                    "UV_NO_INDEX": "1",
                    "UV_OFFLINE": "1",
                },
                command=command.argv,
                memory_bytes=self._memory_bytes,
                pid_limit=self._pid_limit,
                file_bytes=20 * 1024 * 1024,
                cpu_seconds=command.timeout_seconds + 5,
            )
            return await asyncio.to_thread(
                _run_process,
                argv,
                evaluation_workspace,
                command,
                {},
                self._max_output_bytes,
            )
        finally:
            await asyncio.to_thread(shutil.rmtree, evaluation_root, True)

    @staticmethod
    def _require_immutable_source(root: Path) -> None:
        root_metadata = root.stat()
        if root_metadata.st_uid != 0 or root_metadata.st_mode & 0o022:
            raise RuntimeError("candidate source is not owned by the stable host")
        for directory, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            for name in (*directory_names, *file_names):
                path = Path(directory) / name
                metadata = path.lstat()
                if metadata.st_uid != 0 or metadata.st_mode & 0o022:
                    raise RuntimeError("candidate source is not owned by the stable host")

    def _evaluation_copy(self, root: Path) -> tuple[Path, Path]:
        evaluation_root = Path(tempfile.mkdtemp(prefix="run-", dir=str(self._temporary_root)))
        workspace = evaluation_root / "workspace"
        try:
            shutil.copytree(root, workspace, symlinks=True)
            self._make_owned_writable(evaluation_root)
        except Exception:
            shutil.rmtree(evaluation_root, ignore_errors=True)
            raise
        return evaluation_root, workspace

    def _make_owned_writable(self, root: Path) -> None:
        for directory, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            directory_path = Path(directory)
            os.chown(directory_path, self._uid, self._gid, follow_symlinks=False)
            directory_path.chmod(0o700)
            for name in directory_names:
                path = directory_path / name
                if path.is_symlink():
                    os.chown(path, self._uid, self._gid, follow_symlinks=False)
            for name in file_names:
                path = directory_path / name
                os.chown(path, self._uid, self._gid, follow_symlinks=False)
                if not path.is_symlink():
                    metadata = path.stat()
                    path.chmod(0o700 if stat.S_IMODE(metadata.st_mode) & 0o111 else 0o600)

    @staticmethod
    def _read_only_root_reason(path: Path, *, label: str) -> str | None:
        candidate = path.expanduser().absolute()
        current = Path(candidate.anchor)
        for component in candidate.parts[1:]:
            current /= component
            try:
                metadata = current.lstat()
            except OSError:
                return f"isolated evaluation {label} is unavailable"
            if stat.S_ISLNK(metadata.st_mode):
                return f"isolated evaluation {label} has a symbolic-link ancestor"
            if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
                return f"isolated evaluation {label} must be root-owned and immutable"
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_IMODE(metadata.st_mode) & stat.S_IXOTH:
                return f"isolated evaluation {label} is not candidate-traversable"
        if not candidate.is_dir():
            return f"isolated evaluation {label} is unavailable"
        return None

    @staticmethod
    def _executable_reason(path: Path) -> str | None:
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
            target = resolved.stat()
        except OSError:
            return f"isolated evaluation requires exact {path.name} from the controller generation"
        if (
            not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode))
            or not stat.S_ISREG(target.st_mode)
            or metadata.st_uid != 0
            or target.st_uid != 0
            or (
                stat.S_ISREG(metadata.st_mode)
                and stat.S_IMODE(metadata.st_mode) & 0o022
            )
            or stat.S_IMODE(target.st_mode) & 0o022
            or not os.access(path, os.X_OK)
        ):
            return (
                f"isolated evaluation requires immutable {path.name} from the controller generation"
            )
        for ancestor in (resolved, *resolved.parents):
            try:
                ancestor_metadata = ancestor.lstat()
            except OSError:
                return (
                    f"isolated evaluation requires immutable {path.name} from the controller generation"
                )
            if stat.S_ISLNK(ancestor_metadata.st_mode):
                return (
                    f"isolated evaluation requires immutable {path.name} from the controller generation"
                )
            if (
                ancestor_metadata.st_uid != _TRUSTED_OWNER_UID
                or stat.S_IMODE(ancestor_metadata.st_mode) & 0o022
                or (
                    stat.S_ISDIR(ancestor_metadata.st_mode)
                    and not stat.S_IMODE(ancestor_metadata.st_mode) & stat.S_IXOTH
                )
            ):
                return (
                    f"isolated evaluation requires immutable {path.name} from the controller generation"
                )
        return None

    def _input_fingerprint(self) -> str:
        dependency_site = getattr(self, "_dependency_site", None)
        tools = {
            Path(isolated).name: {
                "path": isolated,
                "sha256": _file_sha256(host.resolve(strict=True)),
            }
            for isolated, host in sorted(self._host_executables.items())
        }
        packages, wheels = _evaluation_package_inputs(
            self._controller_root,
            self._wheelhouse,
        )
        payload = {
            "version": "root-linux-bwrap-v1",
            "uid": self._uid,
            "gid": self._gid,
            "memory": self._memory_bytes,
            "pids": self._pid_limit,
            "sandbox_tools": {
                path.name: _file_sha256(path)
                for path in (self._setpriv, self._bwrap, self._prlimit)
            },
            "executables": tools,
            "packages": packages,
            "wheelhouse": wheels,
            "dependency_site": (
                _immutable_tree_sha256(dependency_site) if dependency_site is not None else None
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class OciEvaluationRunner:
    """Run candidate checks without host secrets, network, or writable host state."""

    def __init__(
        self,
        *,
        image: str,
        container_cli: str = "docker",
        cpu_limit: str = "2",
        memory_limit: str = "2g",
        pid_limit: int = 256,
        max_output_bytes: int = 1_000_000,
    ) -> None:
        if not _IMAGE_RE.fullmatch(image):
            raise ValueError("evaluation image is invalid")
        if not container_cli or "\x00" in container_cli:
            raise ValueError("container_cli is invalid")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", cpu_limit):
            raise ValueError("cpu_limit is invalid")
        if not re.fullmatch(r"[1-9][0-9]*(?:[kmgt]i?|b)?", memory_limit, re.I):
            raise ValueError("memory_limit is invalid")
        if pid_limit < 16 or pid_limit > 4_096:
            raise ValueError("pid_limit is invalid")
        if max_output_bytes < 1_024:
            raise ValueError("max_output_bytes is invalid")
        self._image = image
        self._container_cli = container_cli
        self._cpu_limit = cpu_limit
        self._memory_limit = memory_limit
        self._pid_limit = pid_limit
        self._max_output_bytes = max_output_bytes

    @property
    def fingerprint(self) -> str:
        return f"oci:{self._image}"

    async def run(
        self,
        *,
        workspace: Path,
        command: EvaluationCommand,
    ) -> EvaluationCommandResult:
        root = _validated_workspace(workspace)
        mount_source = str(root)
        if "," in mount_source:
            raise ValueError("candidate workspace cannot contain a comma")
        container_name = f"opentulpa-evaluator-{uuid4().hex}"
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
            "none",
            "--ipc",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--cpus",
            self._cpu_limit,
            "--memory",
            self._memory_limit,
            "--memory-swap",
            self._memory_limit,
            "--pids-limit",
            str(self._pid_limit),
            "--user",
            f"{uid}:{gid}",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONHASHSEED=0",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
            "--mount",
            f"type=bind,src={mount_source},dst=/workspace,readonly",
            "--workdir",
            "/workspace",
            self._image,
            *command.argv,
        ]
        task = asyncio.create_task(
            asyncio.to_thread(
                _run_process,
                argv,
                root,
                command,
                {"PATH": os.environ.get("PATH", os.defpath)},
                self._max_output_bytes,
                lambda: _force_remove_container(self._container_cli, container_name),
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.to_thread(
                _force_remove_container,
                self._container_cli,
                container_name,
            )
            await asyncio.gather(task, return_exceptions=True)
            raise


class CandidateEvaluator:
    """Run fixed gates in order and stop after the first failed stage."""

    def __init__(
        self,
        *,
        runner: EvaluationRunner,
        commands: Sequence[EvaluationCommand],
    ) -> None:
        if not commands:
            raise ValueError("at least one evaluation command is required")
        names = [command.name for command in commands]
        if len(names) != len(set(names)):
            raise ValueError("evaluation command names must be unique")
        self._runner = runner
        self._commands = tuple(commands)

    @property
    def fingerprint(self) -> str:
        payload = {
            "runner": self._runner.fingerprint,
            "commands": [
                {
                    "name": command.name,
                    "stage": command.stage,
                    "argv": command.argv,
                    "timeout_seconds": command.timeout_seconds,
                }
                for command in self._commands
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    async def evaluate(self, workspace: Path) -> tuple[EvaluationCommandResult, ...]:
        results: list[EvaluationCommandResult] = []
        failed_stage: EvaluationStage | None = None
        for command in self._commands:
            if failed_stage is not None and command.stage != failed_stage:
                break
            result = await self._runner.run(workspace=workspace, command=command)
            results.append(result)
            if not result.passed:
                failed_stage = command.stage
        return tuple(results)


def trusted_default_commands(
    *,
    timeout_seconds: int = 900,
    executables: EvaluationExecutables | None = None,
) -> tuple[EvaluationCommand, ...]:
    """Return the supervisor-owned release gates candidates cannot rewrite."""

    timeout = max(60, min(int(timeout_seconds), 3_600))
    tools = executables or EvaluationExecutables()
    return (
        EvaluationCommand(
            name="python.compile",
            stage="build",
            argv=(
                tools.python,
                "-I",
                "-c",
                (
                    "from pathlib import Path; "
                    "roots=(Path('src'),Path('tests')); "
                    "files=(p for root in roots if root.exists() for p in root.rglob('*.py')); "
                    "[compile(p.read_bytes(),str(p),'exec') for p in files]"
                ),
            ),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="ruff",
            stage="public",
            argv=(tools.ruff, "check", "src", "tests"),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="mypy",
            stage="public",
            argv=(tools.mypy, "src"),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="pytest",
            stage="public",
            argv=(tools.pytest, "-q"),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="legacy.runtime.absent",
            stage="security",
            argv=(
                tools.python,
                "-I",
                "-c",
                (
                    "from pathlib import Path; "
                    "needles=('OpenTulpa'+'LangGraphRuntime','build_'+'runtime_graph',"
                    "'tool_group_'+'exec'); "
                    "bad=[]; "
                    "roots=[Path('src'),Path('tests')]; "
                    "files=(p for r in roots if r.exists() for p in r.rglob('*.py')); "
                    "[(bad.append((str(p),n))) for p in files for n in needles "
                    "if n in p.read_text(encoding='utf-8',errors='ignore')]; "
                    "assert not bad,bad"
                ),
            ),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="source.secret.paths",
            stage="security",
            argv=(
                tools.python,
                "-I",
                "-c",
                (
                    "from pathlib import Path; "
                    "bad=[p for p in Path('.').rglob('*') if p.is_file() and "
                    "(p.name in {'.env','id_rsa','id_ed25519'} or "
                    "p.suffix.lower() in {'.key','.pem','.p12','.pfx'})]; "
                    "assert not bad,[str(p) for p in bad]"
                ),
            ),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="kernel.contract",
            stage="contract",
            argv=(
                tools.python,
                "-I",
                "-c",
                (
                    "from pathlib import Path; "
                    "p=Path('pyproject.toml').read_text(encoding='utf-8'); "
                    "s=Path('src/opentulpa/deep_agent/service.py').read_text(encoding='utf-8'); "
                    "assert 'deepagents==0.6.12' in p; "
                    "assert 'create_deep_agent' in s and 'class DeepAgentService' in s; "
                    "assert not any(n in s for n in "
                    "('OpenTulpa'+'LangGraphRuntime','build_'+'runtime_graph','tool_group_'+'exec'))"
                ),
            ),
            timeout_seconds=timeout,
        ),
    )


def _validated_workspace(workspace: Path) -> Path:
    root = workspace.expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate workspace must be a regular directory")
    return root.resolve(strict=True)


def _run_process(
    argv: list[str],
    cwd: Path,
    command: EvaluationCommand,
    env: dict[str, str],
    max_output_bytes: int = 1_000_000,
    timeout_cleanup: Callable[[], None] | None = None,
) -> EvaluationCommandResult:
    started = time.monotonic()
    completed = run_bounded_process(
        argv,
        cwd=cwd,
        env=env,
        timeout_seconds=command.timeout_seconds,
        max_output_bytes=max_output_bytes,
        timeout_cleanup=timeout_cleanup,
    )
    raw = completed.output
    exit_code = completed.returncode
    duration = time.monotonic() - started
    return EvaluationCommandResult(
        name=command.name,
        stage=command.stage,
        passed=exit_code == 0,
        exit_code=exit_code,
        duration_seconds=max(0.0, duration),
        output=_sanitize_output(raw.decode("utf-8", errors="replace")),
    )


def _file_sha256(path: Path) -> str:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or metadata.st_nlink != 1
    ):
        raise RuntimeError("isolated evaluation input is not root-owned and immutable")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_tree_sha256(root: Path) -> str:
    entries: list[tuple[str, str, int]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        for name in (*directory_names, *sorted(file_names)):
            path = Path(directory) / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != _TRUSTED_OWNER_UID
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise RuntimeError("isolated evaluation dependency site is unsafe")
            if stat.S_ISDIR(metadata.st_mode):
                entries.append((relative, "directory", 0))
            elif stat.S_ISREG(metadata.st_mode):
                entries.append((relative, _file_sha256(path), metadata.st_size))
            else:
                raise RuntimeError("isolated evaluation dependency site is unsafe")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_package_inputs(
    controller_root: Path,
    wheelhouse: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    packages: dict[str, dict[str, str]] = {}
    for distribution in ("mypy", "pytest", "ruff"):
        records = tuple(
            controller_root.glob(f"lib/python*/site-packages/{distribution}-*.dist-info/RECORD")
        )
        if len(records) != 1:
            raise RuntimeError(
                f"isolated evaluation requires exact {distribution} package metadata"
            )
        record = records[0]
        metadata_path = record.with_name("METADATA")
        packages[distribution] = {
            "metadata_sha256": _file_sha256(metadata_path),
            "record_sha256": _file_sha256(record),
            "recorded_files_sha256": _recorded_distribution_sha256(
                record,
                controller_root=controller_root,
            ),
        }
    wheels: dict[str, str] = {}
    for path in sorted(wheelhouse.rglob("*")):
        path_metadata = path.lstat()
        if stat.S_ISLNK(path_metadata.st_mode) or path_metadata.st_uid != _TRUSTED_OWNER_UID:
            raise RuntimeError("isolated evaluation wheelhouse is unsafe")
        if stat.S_ISDIR(path_metadata.st_mode):
            if stat.S_IMODE(path_metadata.st_mode) & 0o022:
                raise RuntimeError("isolated evaluation wheelhouse is mutable")
            continue
        if not stat.S_ISREG(path_metadata.st_mode) or path.suffix != ".whl":
            raise RuntimeError("isolated evaluation wheelhouse contains a non-wheel input")
        wheels[str(path.relative_to(wheelhouse))] = _file_sha256(path)
    return packages, wheels


def _recorded_distribution_sha256(record: Path, *, controller_root: Path) -> str:
    site_packages = record.parent.parent
    entries: list[tuple[str, str]] = []
    with record.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not row[0]:
                raise RuntimeError("isolated evaluation package RECORD is invalid")
            path = (site_packages / row[0]).resolve(strict=True)
            try:
                relative = path.relative_to(controller_root)
            except ValueError as exc:
                raise RuntimeError("isolated evaluation package escaped the controller") from exc
            entries.append((relative.as_posix(), _file_sha256(path)))
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CandidateEvaluator",
    "EvaluationCommand",
    "EvaluationCommandResult",
    "EvaluationExecutables",
    "EvaluationRunner",
    "EvaluationStage",
    "IsolatedProcessEvaluationRunner",
    "LocalEvaluationRunner",
    "OciEvaluationRunner",
    "trusted_default_commands",
]
