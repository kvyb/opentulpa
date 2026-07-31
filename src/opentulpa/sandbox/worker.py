"""Private mandatory sandbox worker service."""

from __future__ import annotations

import base64
import binascii
import io
import os
import re
import secrets
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import Annotated, Protocol

import uvicorn
from deepagents.backends.protocol import ExecuteResponse
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.deep_agent.process_sandbox import RestrictedProcessExecutionProvider
from opentulpa.deep_agent.sandbox import TenantContainerPolicy
from opentulpa.evolution.sandbox import CandidateProcessBackend

_WORKSPACE_ID_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}\Z")
_WORKSPACE_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SECRET_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}\Z")
_WORKSPACE_KIND_FIELD_PATTERN = r"[a-z][a-z0-9_]{0,63}$"
_SECRET_NAME_FIELD_PATTERN = r"[A-Za-z0-9_.-]{1,80}$"
_ENV_NAME_FIELD_PATTERN = r"[A-Za-z_][A-Za-z0-9_]{0,79}$"
_SENSITIVE_COMPONENTS = frozenset(
    {
        ".aws",
        ".docker",
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


class SandboxWorkerError(RuntimeError):
    """Sanitized sandbox worker failure."""


class _SandboxModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SandboxHealthResult(_SandboxModel):
    ok: bool
    tier: str
    checks: dict[str, bool]
    error: str | None = None


class SandboxWorkspaceCreateRequest(_SandboxModel):
    kind: str = Field(default="scratch", pattern=_WORKSPACE_KIND_FIELD_PATTERN)
    tenant_id: str = Field(default="owner", min_length=1, max_length=512)


class SandboxWorkspaceResult(_SandboxModel):
    workspace_id: str
    kind: str


class SandboxArchiveRequest(_SandboxModel):
    archive: str = Field(min_length=1)


class SandboxArchiveResult(_SandboxModel):
    archive: str


class SandboxSecretFile(_SandboxModel):
    name: str = Field(pattern=_SECRET_NAME_FIELD_PATTERN)
    content: str = Field(min_length=1, max_length=200_000)
    env: str | None = Field(default=None, pattern=_ENV_NAME_FIELD_PATTERN)


class SandboxExecuteRequest(_SandboxModel):
    command: str = Field(min_length=1, max_length=100_000)
    timeout: int = Field(ge=1, le=3_600)
    secret_files: list[SandboxSecretFile] = Field(default_factory=list, max_length=8)


class SandboxExecuteResult(_SandboxModel):
    output: str
    exit_code: int
    truncated: bool


class SandboxExecutionEngine(Protocol):
    """Backend used by the private worker to run inside one workspace."""

    @property
    def tier(self) -> str: ...

    def health_checks(self) -> dict[str, bool]: ...

    def execute(
        self,
        *,
        workspace: Path,
        command: str,
        timeout: int,
        cancel_event: threading.Event | None = None,
    ) -> ExecuteResponse: ...


class RestrictedProcessEngine:
    """Production Linux worker backend using the existing unprivileged process sandbox."""

    def __init__(
        self,
        *,
        policy: TenantContainerPolicy,
        max_workspace_bytes: int,
    ) -> None:
        self._policy = policy
        self._provider = (
            RestrictedProcessExecutionProvider(
                policy=policy,
                max_workspace_bytes=max_workspace_bytes,
            )
            if CandidateProcessBackend.supported()
            else None
        )

    @property
    def tier(self) -> str:
        return "native-process"

    def health_checks(self) -> dict[str, bool]:
        return {
            "process_backend": self._provider is not None,
            "linux_root": bool(
                os.name == "posix"
                and hasattr(os, "geteuid")
                and os.geteuid() == 0
            ),
            "setpriv": shutil.which("setpriv") is not None,
            "prlimit": shutil.which("prlimit") is not None,
            "ssh": shutil.which("ssh") is not None,
        }

    def execute(
        self,
        *,
        workspace: Path,
        command: str,
        timeout: int,
        cancel_event: threading.Event | None = None,
    ) -> ExecuteResponse:
        if self._provider is None:
            return ExecuteResponse(
                output="sandbox worker native process backend is unavailable",
                exit_code=127,
                truncated=False,
            )
        return self._provider.execute(
            tenant_id="worker",
            command=command,
            timeout=timeout,
            workspace=workspace,
            cancel_event=cancel_event,
        )


class DevProcessEngine:
    """Explicit dev-only executor for tests and local iteration without root privileges."""

    def __init__(self, *, max_output_bytes: int) -> None:
        self._max_output_bytes = max_output_bytes

    @property
    def tier(self) -> str:
        return "dev-process"

    def health_checks(self) -> dict[str, bool]:
        return {"dev_mode": True, "ssh": shutil.which("ssh") is not None}

    def execute(
        self,
        *,
        workspace: Path,
        command: str,
        timeout: int,
        cancel_event: threading.Event | None = None,
    ) -> ExecuteResponse:
        process = subprocess.Popen(
            ["/bin/sh", "-lc", command],
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                "HOME": str(workspace),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
                "PATH": os.environ.get("PATH", os.defpath),
            },
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            with suppress(OSError, ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                stdout, _ = process.communicate(timeout=2)
            if process.poll() is None:
                with suppress(OSError, ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                stdout, _ = process.communicate()
            truncated = len(stdout) > self._max_output_bytes
            output = stdout[: self._max_output_bytes].decode("utf-8", errors="replace")
            return ExecuteResponse(
                output=(output + "\ncommand timed out").strip(),
                exit_code=124,
                truncated=truncated,
            )
        if cancel_event is not None and cancel_event.is_set():
            return ExecuteResponse(output="sandbox execution was cancelled", exit_code=130, truncated=False)
        truncated = len(stdout) > self._max_output_bytes
        return ExecuteResponse(
            output=stdout[: self._max_output_bytes].decode("utf-8", errors="replace"),
            exit_code=int(process.returncode or 0),
            truncated=truncated,
        )


class SandboxWorkerService:
    """Own tenant workspaces and execute commands without host application secrets."""

    def __init__(
        self,
        *,
        root: Path,
        engine: SandboxExecutionEngine,
        max_archive_bytes: int,
        max_entries: int,
        max_file_bytes: int,
    ) -> None:
        configured = root.expanduser()
        if configured.is_symlink():
            raise ValueError("sandbox root cannot be a symlink")
        configured.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._root = configured.resolve(strict=True)
        self._workspaces = self._root / "workspaces"
        self._workspaces.mkdir(mode=0o700, exist_ok=True)
        self._engine = engine
        self._max_archive_bytes = max_archive_bytes
        self._max_entries = max_entries
        self._max_file_bytes = max_file_bytes
        self._lock = threading.Lock()

    def health(self) -> SandboxHealthResult:
        checks = self._engine.health_checks()
        canary = False
        canary_error: str | None = None
        if checks and all(checks.values()):
            with tempfile.TemporaryDirectory(prefix="opentulpa-sandbox-health-", dir=self._root) as tmp:
                workspace = Path(tmp)
                result = self._engine.execute(
                    workspace=workspace,
                    command="printf opentulpa-sandbox-ready",
                    timeout=5,
                )
                canary = (
                    result.exit_code == 0
                    and result.output.strip() == "opentulpa-sandbox-ready"
                )
                if not canary:
                    output = str(result.output or "").strip()
                    canary_error = (
                        f"sandbox worker canary command failed with exit {result.exit_code}"
                    )
                    if output:
                        canary_error = f"{canary_error}: {output[:300]}"
        checks["execute"] = canary
        ok = all(checks.values())
        if ok:
            error = None
        elif canary_error:
            error = canary_error
        else:
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            error = f"sandbox worker failed checks: {failed}" if failed else "sandbox worker canary failed"
        return SandboxHealthResult(
            ok=ok,
            tier=self._engine.tier,
            checks=checks,
            error=error,
        )

    def create_workspace(self, *, kind: str, tenant_id: str) -> SandboxWorkspaceResult:
        del tenant_id
        if _WORKSPACE_KIND_PATTERN.fullmatch(kind) is None:
            raise SandboxWorkerError("workspace kind is invalid")
        workspace_id = f"{kind}-{secrets.token_urlsafe(18).replace('-', '_')}"
        path = self._workspace_path(workspace_id, must_exist=False)
        path.mkdir(mode=0o700)
        return SandboxWorkspaceResult(workspace_id=workspace_id, kind=kind)

    def delete_workspace(self, workspace_id: str) -> None:
        path = self._workspace_path(workspace_id)
        shutil.rmtree(path)

    def put_archive(self, workspace_id: str, encoded_archive: str) -> None:
        workspace = self._workspace_path(workspace_id)
        raw = self._decode_archive(encoded_archive)
        with self._lock:
            replacement = workspace.parent / f".{workspace.name}.replace-{secrets.token_hex(8)}"
            replacement.mkdir(mode=0o700)
            try:
                self._extract_archive(raw, replacement)
                self._validate_tree(replacement)
                backup = workspace.parent / f".{workspace.name}.previous-{secrets.token_hex(8)}"
                os.replace(workspace, backup)
                os.replace(replacement, workspace)
                shutil.rmtree(backup, ignore_errors=True)
            except Exception:
                shutil.rmtree(replacement, ignore_errors=True)
                raise

    def get_archive(self, workspace_id: str) -> str:
        workspace = self._workspace_path(workspace_id)
        self._validate_tree(workspace)
        with tempfile.SpooledTemporaryFile(max_size=1_000_000) as archive_file:
            with tarfile.open(fileobj=archive_file, mode="w:gz") as archive:
                archive.add(workspace, arcname=".", recursive=True)
            size = archive_file.tell()
            if size > self._max_archive_bytes:
                raise SandboxWorkerError("workspace archive exceeds its limit")
            archive_file.seek(0)
            return base64.b64encode(archive_file.read()).decode("ascii")

    def execute(
        self,
        *,
        workspace_id: str,
        command: str,
        timeout: int,
        secret_files: list[SandboxSecretFile],
    ) -> ExecuteResponse:
        workspace = self._workspace_path(workspace_id)
        self._validate_tree(workspace)
        with self._temporary_secret_files(workspace, secret_files) as secrets_context:
            result = self._engine.execute(
                workspace=workspace,
                command=self._with_secret_environment(command, secrets_context.environment),
                timeout=timeout,
            )
            output = result.output
            for value in secrets_context.values:
                if value:
                    output = output.replace(value, "[redacted]")
            response = ExecuteResponse(
                output=output,
                exit_code=result.exit_code,
                truncated=result.truncated,
            )
        self._validate_tree(workspace)
        return response

    def _workspace_path(self, workspace_id: str, *, must_exist: bool = True) -> Path:
        if _WORKSPACE_ID_PATTERN.fullmatch(str(workspace_id or "")) is None:
            raise SandboxWorkerError("workspace id is invalid")
        path = (self._workspaces / workspace_id).resolve()
        if not path.is_relative_to(self._workspaces):
            raise SandboxWorkerError("workspace escaped sandbox root")
        if must_exist and (not path.exists() or path.is_symlink() or not path.is_dir()):
            raise SandboxWorkerError("workspace is unavailable")
        return path

    def _decode_archive(self, encoded_archive: str) -> bytes:
        try:
            raw = base64.b64decode(encoded_archive, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SandboxWorkerError("workspace archive is invalid") from exc
        if len(raw) > self._max_archive_bytes:
            raise SandboxWorkerError("workspace archive exceeds its limit")
        return raw

    def _extract_archive(self, raw: bytes, target: Path) -> None:
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
                members = archive.getmembers()
                if len(members) > self._max_entries:
                    raise SandboxWorkerError("workspace archive entry limit exceeded")
                total_size = 0
                for member in members:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts:
                        raise SandboxWorkerError("workspace archive path is invalid")
                    if not (member.isdir() or member.isreg()):
                        raise SandboxWorkerError("workspace archive contains an unsupported entry")
                    if member.name not in {"", "."}:
                        for component in path.parts:
                            if component in _SENSITIVE_COMPONENTS:
                                raise SandboxWorkerError("workspace archive contains sensitive paths")
                    if member.isreg():
                        if member.size > self._max_file_bytes:
                            raise SandboxWorkerError("workspace archive file limit exceeded")
                        total_size += member.size
                if total_size > self._max_archive_bytes * 8:
                    raise SandboxWorkerError("workspace archive expanded beyond its limit")
                archive.extractall(target, filter="data")
        except tarfile.TarError as exc:
            raise SandboxWorkerError("workspace archive could not be extracted") from exc

    def _validate_tree(self, root: Path | None = None) -> None:
        target = (root or self._workspaces).resolve()
        if target.is_symlink() or not target.is_relative_to(self._root):
            raise SandboxWorkerError("workspace tree failed validation")
        for total_files, path in enumerate(target.rglob("*"), start=1):
            if path.is_symlink() or not path.is_relative_to(self._root):
                raise SandboxWorkerError("workspace tree contains an unsafe path")
            if path.name in _SENSITIVE_COMPONENTS:
                raise SandboxWorkerError("workspace tree contains sensitive paths")
            if total_files > self._max_entries:
                raise SandboxWorkerError("workspace tree entry limit exceeded")
            if path.is_file() and path.stat().st_size > self._max_file_bytes:
                raise SandboxWorkerError("workspace file limit exceeded")

    @contextmanager
    def _temporary_secret_files(
        self,
        workspace: Path,
        secret_files: list[SandboxSecretFile],
    ) -> Iterator[_SecretContext]:
        if not secret_files:
            yield _SecretContext(environment={}, values=())
            return
        root = workspace / ".opentulpa_secret_mounts" / secrets.token_hex(12)
        root.mkdir(mode=0o700, parents=True)
        environment: dict[str, str] = {}
        values: list[str] = []
        try:
            for item in secret_files:
                path = root / item.name
                if path.exists() or path.is_symlink():
                    raise SandboxWorkerError("secret mount path is invalid")
                path.write_text(item.content, encoding="utf-8")
                path.chmod(0o600)
                values.append(item.content)
                if item.env:
                    environment[item.env] = str(path)
            yield _SecretContext(environment=environment, values=tuple(values))
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)

    @staticmethod
    def _with_secret_environment(command: str, environment: dict[str, str]) -> str:
        if not environment:
            return command
        exports = " ".join(
            f"{name}={_shell_quote(value)}" for name, value in sorted(environment.items())
        )
        return f"export {exports}; {command}"


class _SecretContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    environment: dict[str, str]
    values: tuple[str, ...]


def create_sandbox_worker_app(*, service: SandboxWorkerService, token: str) -> FastAPI:
    expected_token = str(token or "").strip()
    if len(expected_token) < 32:
        raise ValueError("sandbox worker token must contain at least 32 characters")

    async def authorize(
        supplied: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Sandbox-Token", max_length=500),
        ] = None,
    ) -> None:
        if not secrets.compare_digest(str(supplied or ""), expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid sandbox worker credentials are required",
            )

    app = FastAPI(title="OpenTulpa Sandbox Worker", version="1")
    router = APIRouter(
        prefix="/internal/v1/sandbox",
        dependencies=[Depends(authorize)],
        include_in_schema=False,
    )

    @router.get("/health", response_model=SandboxHealthResult)
    async def health() -> SandboxHealthResult:
        return service.health()

    @router.post("/workspaces", response_model=SandboxWorkspaceResult, status_code=201)
    async def create_workspace(body: SandboxWorkspaceCreateRequest) -> SandboxWorkspaceResult:
        try:
            return service.create_workspace(kind=body.kind, tenant_id=body.tenant_id)
        except (OSError, SandboxWorkerError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="sandbox workspace is unavailable") from exc

    @router.post("/workspaces/{workspace_id}/archive", status_code=204)
    async def put_archive(workspace_id: str, body: SandboxArchiveRequest) -> None:
        try:
            service.put_archive(workspace_id, body.archive)
        except (OSError, SandboxWorkerError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="sandbox workspace archive is invalid") from exc

    @router.get("/workspaces/{workspace_id}/archive", response_model=SandboxArchiveResult)
    async def get_archive(workspace_id: str) -> SandboxArchiveResult:
        try:
            return SandboxArchiveResult(archive=service.get_archive(workspace_id))
        except (OSError, SandboxWorkerError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="sandbox workspace archive is unavailable") from exc

    @router.post("/workspaces/{workspace_id}/execute", response_model=SandboxExecuteResult)
    async def execute(
        workspace_id: str,
        body: SandboxExecuteRequest,
        request: Request,
    ) -> SandboxExecuteResult:
        del request
        try:
            result = service.execute(
                workspace_id=workspace_id,
                command=body.command,
                timeout=body.timeout,
                secret_files=body.secret_files,
            )
        except (OSError, SandboxWorkerError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="sandbox execution is unavailable") from exc
        return SandboxExecuteResult(
            output=result.output,
            exit_code=int(result.exit_code or 0),
            truncated=result.truncated,
        )

    @router.delete("/workspaces/{workspace_id}", status_code=204)
    async def delete_workspace(workspace_id: str) -> None:
        try:
            service.delete_workspace(workspace_id)
        except (OSError, SandboxWorkerError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="sandbox workspace is unavailable") from exc

    app.include_router(router)
    return app


def build_default_worker_service() -> SandboxWorkerService:
    root = Path(
        os.environ.get("OPENTULPA_SANDBOX_WORKER_ROOT")
        or Path(tempfile.gettempdir()) / "opentulpa-sandbox-worker"
    )
    max_output = int(os.environ.get("OPENTULPA_SANDBOX_MAX_OUTPUT_BYTES") or 512_000)
    max_archive = int(os.environ.get("OPENTULPA_SANDBOX_MAX_ARCHIVE_BYTES") or 32 * 1024 * 1024)
    max_entries = int(os.environ.get("OPENTULPA_SANDBOX_MAX_ENTRIES") or 20_000)
    max_file = int(os.environ.get("OPENTULPA_SANDBOX_MAX_FILE_BYTES") or 10 * 1024 * 1024)
    policy = TenantContainerPolicy(
        image="opentulpa-sandbox-worker",
        cpu_limit=os.environ.get("OPENTULPA_SANDBOX_CPU_LIMIT") or "1",
        memory_limit=os.environ.get("OPENTULPA_SANDBOX_MEMORY_LIMIT") or "512m",
        pid_limit=int(os.environ.get("OPENTULPA_SANDBOX_PID_LIMIT") or 128),
        timeout_seconds=int(os.environ.get("OPENTULPA_SANDBOX_TIMEOUT_SECONDS") or 60),
        max_output_bytes=max_output,
        max_file_bytes=max_file,
        max_workspace_entries=max_entries,
        network_enabled=True,
    )
    dev = str(os.environ.get("OPENTULPA_DEV_ALLOW_NO_SANDBOX") or "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    engine: SandboxExecutionEngine
    if dev and not CandidateProcessBackend.supported():
        engine = DevProcessEngine(max_output_bytes=max_output)
    else:
        engine = RestrictedProcessEngine(
            policy=policy,
            max_workspace_bytes=max_archive,
        )
    return SandboxWorkerService(
        root=root,
        engine=engine,
        max_archive_bytes=max_archive,
        max_entries=max_entries,
        max_file_bytes=max_file,
    )


def main() -> None:
    token = str(os.environ.get("OPENTULPA_SANDBOX_RPC_TOKEN") or "").strip()
    if len(token) < 32:
        raise SystemExit("OPENTULPA_SANDBOX_RPC_TOKEN must contain at least 32 characters")
    host = str(os.environ.get("HOST") or "127.0.0.1").strip()
    port = int(os.environ.get("PORT") or os.environ.get("OPENTULPA_SANDBOX_PORT") or 8787)
    app = create_sandbox_worker_app(service=build_default_worker_service(), token=token)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        ws="none",
        timeout_graceful_shutdown=10,
    )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()


__all__ = [
    "DevProcessEngine",
    "RestrictedProcessEngine",
    "SandboxArchiveRequest",
    "SandboxArchiveResult",
    "SandboxExecuteRequest",
    "SandboxExecuteResult",
    "SandboxExecutionEngine",
    "SandboxHealthResult",
    "SandboxSecretFile",
    "SandboxWorkerError",
    "SandboxWorkerService",
    "SandboxWorkspaceCreateRequest",
    "SandboxWorkspaceResult",
    "build_default_worker_service",
    "create_sandbox_worker_app",
    "main",
]
