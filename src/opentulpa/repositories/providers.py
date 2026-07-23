"""Repository sandbox providers behind one Deep Agents backend contract."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)

from opentulpa.deep_agent.sandbox import TenantContainerPolicy
from opentulpa.repositories.models import RepositoryProvider, RepositoryWorkspace

SecretResolver = Callable[[str, str], str | None]


class RepositorySandboxError(RuntimeError):
    """Sanitized repository sandbox failure."""


class RepositorySandboxUnavailableError(RepositorySandboxError):
    pass


class RepositorySandboxProvider(Protocol):
    name: RepositoryProvider

    def available(self, tenant_id: str) -> bool: ...

    def create(self, workspace: RepositoryWorkspace, *, github_token: str | None) -> str: ...

    def backend(self, workspace: RepositoryWorkspace) -> SandboxBackendProtocol: ...

    def stop(self, workspace: RepositoryWorkspace) -> None: ...

    def push(self, workspace: RepositoryWorkspace, *, github_token: str) -> None: ...


def _translate_path(path: str, *, root: str) -> str:
    value = str(path or "").strip()
    if not value.startswith("/"):
        raise ValueError("repository workspace paths must be absolute")
    source = PurePosixPath(value)
    if ".." in source.parts:
        raise ValueError("repository workspace path traversal is not allowed")
    parts = source.parts
    if parts[:2] == ("/", "workspace"):
        parts = ("/", *parts[2:])
    relative = PurePosixPath(*parts).relative_to("/")
    target = PurePosixPath(root) / relative
    return str(target)


class RepositoryRootedBackend(SandboxBackendProtocol):
    """Expose a repository checkout as `/workspace` regardless of provider layout."""

    def __init__(
        self,
        *,
        delegate: SandboxBackendProtocol,
        file_root: str,
        execution_root: str,
    ) -> None:
        self._delegate = delegate
        self._file_root = file_root.rstrip("/") or "/"
        self._execution_root = execution_root.rstrip("/") or "/"

    @property
    def id(self) -> str:
        return f"repository-root:{self._delegate.id}"

    def _path(self, path: str) -> str:
        return _translate_path(path, root=self._file_root)

    def _public_path(self, path: str | None) -> str | None:
        if path is None:
            return None
        source = PurePosixPath(path)
        root = PurePosixPath(self._file_root)
        try:
            relative = source.relative_to(root)
        except ValueError:
            return "/workspace"
        return str(PurePosixPath("/workspace") / relative)

    def _public_file_info(self, value: FileInfo) -> FileInfo:
        return cast(
            "FileInfo",
            {
                **value,
                "path": self._public_path(str(value.get("path") or "")) or "/workspace",
            },
        )

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        wrapped = f"cd {shlex.quote(self._execution_root)} && ({command})"
        return self._delegate.execute(wrapped, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        wrapped = f"cd {shlex.quote(self._execution_root)} && ({command})"
        return await self._delegate.aexecute(wrapped, timeout=timeout)

    def ls(self, path: str) -> LsResult:
        result = self._delegate.ls(self._path(path))
        return LsResult(
            error=result.error,
            entries=(
                [self._public_file_info(item) for item in result.entries]
                if result.entries is not None
                else None
            ),
        )

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._delegate.read(self._path(file_path), offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        result = self._delegate.write(self._path(file_path), content)
        return WriteResult(
            error=result.error,
            path=self._public_path(result.path),
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        result = self._delegate.edit(
            self._path(file_path),
            old_string,
            new_string,
            replace_all,
        )
        return EditResult(
            error=result.error,
            path=self._public_path(result.path),
            occurrences=result.occurrences,
        )

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        translated = self._path(path or "/workspace")
        result = self._delegate.glob(pattern, translated)
        return GlobResult(
            error=result.error,
            matches=(
                [self._public_file_info(item) for item in result.matches]
                if result.matches is not None
                else None
            ),
        )

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        translated = self._path(path or "/workspace")
        result = self._delegate.grep(pattern, translated, glob)
        return GrepResult(
            error=result.error,
            matches=(
                [
                    cast(
                        "GrepMatch",
                        {
                            **item,
                            "path": (
                                self._public_path(str(item.get("path") or ""))
                                or "/workspace"
                            ),
                        },
                    )
                    for item in result.matches
                ]
                if result.matches is not None
                else None
            ),
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        translated = [(self._path(path), content) for path, content in files]
        responses = self._delegate.upload_files(translated)
        return [
            FileUploadResponse(path=original[0], error=response.error)
            for original, response in zip(files, responses, strict=True)
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses = self._delegate.download_files([self._path(path) for path in paths])
        return [
            FileDownloadResponse(path=path, content=response.content, error=response.error)
            for path, response in zip(paths, responses, strict=True)
        ]


class LocalRepositoryBackend(SandboxBackendProtocol):
    """Persistent checkout with commands isolated in a short-lived rootless OCI process."""

    def __init__(
        self,
        *,
        workspace_id: str,
        root: Path,
        policy: TenantContainerPolicy,
        container_cli: str,
    ) -> None:
        self._workspace_id = workspace_id
        self._root = root.resolve()
        self._policy = policy
        self._container_cli = container_cli
        self._files = FilesystemBackend(
            root_dir=self._root,
            virtual_mode=True,
            max_file_size_mb=max(1, policy.max_file_bytes // (1024 * 1024)),
        )
        self._lock = threading.Lock()

    @property
    def id(self) -> str:
        return f"repository-local:{self._workspace_id}"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not str(command or "").strip():
            return ExecuteResponse(output="command is required", exit_code=2)
        requested = self._policy.timeout_seconds if timeout is None else max(1, int(timeout))
        effective_timeout = min(requested, self._policy.timeout_seconds)
        container_name = f"opentulpa-repo-{self._workspace_id[-20:]}"
        argv = [
            self._container_cli,
            "run",
            "--rm",
            "--name",
            container_name,
            "--init",
            "--pull",
            "never",
            "--security-opt",
            "no-new-privileges:true",
            "--cap-drop",
            "ALL",
            "--read-only",
            "--network",
            "bridge",
            "--cpus",
            self._policy.cpu_limit,
            "--memory",
            self._policy.memory_limit,
            "--memory-swap",
            self._policy.memory_limit,
            "--pids-limit",
            str(self._policy.pid_limit),
            "--mount",
            f"type=bind,src={self._root},dst=/workspace",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=128m,mode=1777",
            "--workdir",
            "/workspace",
            "--user",
            self._container_user(),
            "--env",
            "HOME=/tmp",
            self._policy.image,
            "/bin/sh",
            "-lc",
            command,
        ]
        with self._lock:
            try:
                completed = subprocess.run(
                    argv,
                    check=False,
                    capture_output=True,
                    timeout=effective_timeout,
                    env={"PATH": os.environ.get("PATH", "")},
                )
            except subprocess.TimeoutExpired:
                with suppress(OSError):
                    subprocess.run(
                        [self._container_cli, "rm", "--force", container_name],
                        check=False,
                        capture_output=True,
                        timeout=10,
                    )
                return ExecuteResponse(output="command timed out", exit_code=124)
            except OSError:
                return ExecuteResponse(output="repository sandbox unavailable", exit_code=127)
        raw = completed.stdout + completed.stderr
        truncated = len(raw) > self._policy.max_output_bytes
        output = raw[: self._policy.max_output_bytes].decode("utf-8", errors="replace")
        return ExecuteResponse(
            output=output,
            exit_code=completed.returncode,
            truncated=truncated,
        )

    @staticmethod
    def _container_user() -> str:
        uid = os.getuid() if hasattr(os, "getuid") else 65_532
        gid = os.getgid() if hasattr(os, "getgid") else 65_532
        if uid == 0:
            uid = gid = 65_532
        return f"{uid}:{gid}"

    def ls(self, path: str) -> LsResult:
        return self._files.ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._files.read(file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._files.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._files.edit(file_path, old_string, new_string, replace_all)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self._files.glob(pattern, path)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return self._files.grep(pattern, path, glob)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._files.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._files.download_files(paths)


def _git_askpass_environment(token: str) -> tuple[dict[str, str], Path]:
    descriptor, raw_path = tempfile.mkstemp(prefix="opentulpa-git-askpass-")
    path = Path(raw_path)
    os.write(
        descriptor,
        (
            b"#!/bin/sh\n"
            b"case \"$1\" in\n"
            b"  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
            b"  *) printf '%s\\n' \"$OPENTULPA_GIT_TOKEN\" ;;\n"
            b"esac\n"
        ),
    )
    os.close(descriptor)
    path.chmod(0o700)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "GIT_ASKPASS": str(path),
        "GIT_TERMINAL_PROMPT": "0",
        "OPENTULPA_GIT_TOKEN": token,
    }
    return environment, path


class LocalRepositoryProvider:
    name = RepositoryProvider.LOCAL

    def __init__(
        self,
        *,
        root: str | Path,
        policy: TenantContainerPolicy,
        container_cli: str,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._policy = policy
        self._container_cli = str(container_cli or "docker")
        self._backends: dict[str, RepositoryRootedBackend] = {}
        self._lock = threading.Lock()

    def _checkout(self, workspace: RepositoryWorkspace) -> Path:
        return self._root / workspace.id / "repo"

    def available(self, tenant_id: str) -> bool:
        del tenant_id
        return shutil.which(self._container_cli) is not None

    def create(self, workspace: RepositoryWorkspace, *, github_token: str | None) -> str:
        if not self.available(workspace.tenant_id):
            raise RepositorySandboxUnavailableError("local OCI runtime is unavailable")
        checkout = self._checkout(workspace)
        if checkout.exists():
            raise RepositorySandboxError("repository workspace already exists")
        checkout.parent.mkdir(parents=True, exist_ok=False, mode=0o700)
        command = [
            "git",
            "clone",
            "--branch",
            workspace.base_ref,
            "--single-branch",
            "--",
            workspace.repository_url,
            str(checkout),
        ]
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_TERMINAL_PROMPT": "0",
        }
        askpass: Path | None = None
        if github_token:
            environment, askpass = _git_askpass_environment(github_token)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=300,
                env=environment,
            )
        finally:
            if askpass is not None:
                askpass.unlink(missing_ok=True)
        if completed.returncode != 0:
            shutil.rmtree(checkout.parent, ignore_errors=True)
            raise RepositorySandboxError("repository clone failed")
        return workspace.id

    def backend(self, workspace: RepositoryWorkspace) -> SandboxBackendProtocol:
        checkout = self._checkout(workspace)
        if not checkout.is_dir():
            raise RepositorySandboxUnavailableError("local repository workspace is unavailable")
        with self._lock:
            backend = self._backends.get(workspace.id)
            if backend is None:
                raw = LocalRepositoryBackend(
                    workspace_id=workspace.id,
                    root=checkout,
                    policy=self._policy,
                    container_cli=self._container_cli,
                )
                backend = RepositoryRootedBackend(
                    delegate=raw,
                    file_root="/",
                    execution_root="/workspace",
                )
                self._backends[workspace.id] = backend
            return backend

    def stop(self, workspace: RepositoryWorkspace) -> None:
        with self._lock:
            self._backends.pop(workspace.id, None)

    def push(self, workspace: RepositoryWorkspace, *, github_token: str) -> None:
        environment, askpass = _git_askpass_environment(github_token)
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self._checkout(workspace)),
                    "push",
                    "--porcelain",
                    "origin",
                    f"refs/heads/{workspace.branch}:refs/heads/{workspace.branch}",
                ],
                check=False,
                capture_output=True,
                timeout=300,
                env=environment,
            )
        finally:
            askpass.unlink(missing_ok=True)
        if completed.returncode != 0:
            raise RepositorySandboxError("repository branch push failed")


class DaytonaRepositoryProvider:
    name = RepositoryProvider.DAYTONA

    def __init__(
        self,
        *,
        token_resolver: SecretResolver,
        api_url: str,
        target: str | None,
        snapshot: str | None,
    ) -> None:
        self._token_resolver = token_resolver
        self._api_url = api_url
        self._target = target
        self._snapshot = snapshot
        self._backends: dict[str, tuple[str, RepositoryRootedBackend]] = {}
        self._lock = threading.Lock()

    def _token(self, tenant_id: str) -> str:
        token = self._token_resolver(tenant_id, "daytona.manage")
        if not token:
            raise RepositorySandboxUnavailableError(
                "hosted repository sandbox is not configured; paste DAYTONA_API_KEY=<key>"
            )
        return token

    def _client(self, tenant_id: str) -> tuple[Any, str]:
        from daytona import Daytona, DaytonaConfig

        token = self._token(tenant_id)
        digest = hashlib.sha256(token.encode()).hexdigest()
        return (
            Daytona(
                DaytonaConfig(
                    api_key=token,
                    api_url=self._api_url,
                    target=self._target,
                )
            ),
            digest,
        )

    def available(self, tenant_id: str) -> bool:
        return bool(self._token_resolver(tenant_id, "daytona.manage"))

    def create(self, workspace: RepositoryWorkspace, *, github_token: str | None) -> str:
        from daytona import CreateSandboxFromSnapshotParams

        client, _ = self._client(workspace.tenant_id)
        sandbox = client.create(
            CreateSandboxFromSnapshotParams(
                name=f"opentulpa-{workspace.id}",
                snapshot=self._snapshot,
                auto_stop_interval=15,
                labels={
                    "opentulpa.workspace": workspace.id,
                    "opentulpa.tenant": hashlib.sha256(
                        workspace.tenant_id.encode()
                    ).hexdigest()[:16],
                },
                network_block_all=False,
            ),
            timeout=180,
        )
        try:
            sandbox.git.clone(
                workspace.repository_url,
                "/workspace/repo",
                branch=workspace.base_ref,
                username="x-access-token" if github_token else None,
                password=github_token,
                request_timeout=300,
            )
        except Exception as exc:
            with suppress(Exception):
                client.delete(sandbox, wait=False)
            raise RepositorySandboxError("repository clone failed") from exc
        return str(sandbox.id)

    def backend(self, workspace: RepositoryWorkspace) -> SandboxBackendProtocol:
        from langchain_daytona import DaytonaSandbox  # type: ignore[import-untyped]

        provider_id = str(workspace.provider_workspace_id or "")
        if not provider_id:
            raise RepositorySandboxUnavailableError("hosted repository workspace is unavailable")
        client, credential_digest = self._client(workspace.tenant_id)
        with self._lock:
            cached = self._backends.get(workspace.id)
            if cached is not None and cached[0] == credential_digest:
                return cached[1]
            try:
                sandbox = client.get(provider_id)
                client.start(sandbox, timeout=120)
            except Exception as exc:
                raise RepositorySandboxUnavailableError(
                    "hosted repository workspace could not be resumed"
                ) from exc
            backend = RepositoryRootedBackend(
                delegate=DaytonaSandbox(sandbox=sandbox, timeout=1800),
                file_root="/workspace/repo",
                execution_root="/workspace/repo",
            )
            self._backends[workspace.id] = (credential_digest, backend)
            return backend

    def stop(self, workspace: RepositoryWorkspace) -> None:
        provider_id = str(workspace.provider_workspace_id or "")
        if not provider_id:
            return
        client, _ = self._client(workspace.tenant_id)
        try:
            sandbox = client.get(provider_id)
            client.stop(sandbox, timeout=120)
        except Exception as exc:
            raise RepositorySandboxError("hosted repository workspace could not be stopped") from exc
        finally:
            with self._lock:
                self._backends.pop(workspace.id, None)

    def push(self, workspace: RepositoryWorkspace, *, github_token: str) -> None:
        provider_id = str(workspace.provider_workspace_id or "")
        if not provider_id:
            raise RepositorySandboxUnavailableError("hosted repository workspace is unavailable")
        client, _ = self._client(workspace.tenant_id)
        try:
            sandbox = client.get(provider_id)
            client.start(sandbox, timeout=120)
            sandbox.git.push(
                "/workspace/repo",
                username="x-access-token",
                password=github_token,
                branch=workspace.branch,
                remote="origin",
                set_upstream=True,
                request_timeout=300,
            )
        except Exception as exc:
            raise RepositorySandboxError("repository branch push failed") from exc


class RepositoryProviderRegistry:
    """Select and recover a provider without exposing credentials to the model."""

    def __init__(
        self,
        *,
        providers: list[RepositorySandboxProvider],
        default: str = "auto",
    ) -> None:
        self._providers = {provider.name: provider for provider in providers}
        self._default = str(default or "auto").casefold()

    def select(
        self,
        *,
        tenant_id: str,
        requested: str | None,
    ) -> RepositorySandboxProvider:
        choice = str(requested or self._default or "auto").casefold()
        if choice != "auto":
            provider = self._providers.get(RepositoryProvider(choice))
            if provider is None or not provider.available(tenant_id):
                raise RepositorySandboxUnavailableError(
                    f"{choice} repository sandbox is unavailable"
                )
            return provider
        for name in (RepositoryProvider.DAYTONA, RepositoryProvider.LOCAL):
            provider = self._providers.get(name)
            if provider is not None and provider.available(tenant_id):
                return provider
        raise RepositorySandboxUnavailableError(
            "no repository sandbox is configured; paste DAYTONA_API_KEY=<key> "
            "or install a local OCI runtime"
        )

    def for_workspace(self, workspace: RepositoryWorkspace) -> RepositorySandboxProvider:
        provider = self._providers.get(workspace.provider)
        if provider is None:
            raise RepositorySandboxUnavailableError("repository sandbox provider is unavailable")
        return provider


__all__ = [
    "DaytonaRepositoryProvider",
    "LocalRepositoryProvider",
    "RepositoryProviderRegistry",
    "RepositoryRootedBackend",
    "RepositorySandboxError",
    "RepositorySandboxProvider",
    "RepositorySandboxUnavailableError",
    "SecretResolver",
]
