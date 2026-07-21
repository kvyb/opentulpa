"""Trusted, reproducible evaluation runners for generated candidates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from opentulpa.evolution.process import run_bounded_process

EvaluationStage = Literal["build", "contract", "public", "security"]

_IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}\Z")
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\s*[:=]\s*\S+"
)


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


def trusted_default_commands(*, timeout_seconds: int = 900) -> tuple[EvaluationCommand, ...]:
    """Return the supervisor-owned release gates candidates cannot rewrite."""

    timeout = max(60, min(int(timeout_seconds), 3_600))
    return (
        EvaluationCommand(
            name="python.compile",
            stage="build",
            argv=("python", "-I", "-m", "compileall", "-q", "src", "tests"),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="ruff",
            stage="public",
            argv=("ruff", "check", "src", "tests"),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="mypy",
            stage="public",
            argv=("mypy", "src"),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="pytest",
            stage="public",
            argv=("python", "-m", "pytest", "-q"),
            timeout_seconds=timeout,
        ),
        EvaluationCommand(
            name="legacy.runtime.absent",
            stage="security",
            argv=(
                "python",
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
                "python",
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
                "python",
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


__all__ = [
    "CandidateEvaluator",
    "EvaluationCommand",
    "EvaluationCommandResult",
    "EvaluationRunner",
    "EvaluationStage",
    "LocalEvaluationRunner",
    "OciEvaluationRunner",
    "trusted_default_commands",
]
