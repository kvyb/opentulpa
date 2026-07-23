"""Route Deep Agents filesystem and shell operations into the active repository."""

from __future__ import annotations

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

from opentulpa.repositories.service import RepositoryWorkspaceService


class RepositoryRoutingSandbox(SandboxBackendProtocol):
    """Use the active thread checkout or fall back to the tenant workspace."""

    def __init__(
        self,
        *,
        repositories: RepositoryWorkspaceService,
        fallback: SandboxBackendProtocol,
        route_files: bool = True,
    ) -> None:
        self._repositories = repositories
        self._fallback = fallback
        self._route_files = route_files

    @property
    def id(self) -> str:
        mode = "workspace" if self._route_files else "execution"
        return f"repository-routing-sandbox:{mode}"

    def _backend(self) -> SandboxBackendProtocol:
        runtime = get_runtime()
        context = getattr(runtime, "context", None)
        tenant_id = str(getattr(context, "tenant_id", "") or "").strip()
        thread_id = str(getattr(context, "thread_id", "") or "").strip()
        if not tenant_id or not thread_id:
            raise RuntimeError("trusted repository routing context is unavailable")
        return (
            self._repositories.backend_for_thread(
                tenant_id=tenant_id,
                thread_id=thread_id,
            )
            or self._fallback
        )

    def _files_backend(self) -> SandboxBackendProtocol:
        return self._backend() if self._route_files else self._fallback

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._backend().execute(command, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return await self._backend().aexecute(command, timeout=timeout)

    def ls(self, path: str) -> LsResult:
        return self._files_backend().ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._files_backend().read(file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._files_backend().write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._files_backend().edit(file_path, old_string, new_string, replace_all)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self._files_backend().glob(pattern, path)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return self._files_backend().grep(pattern, path, glob)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._files_backend().upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._files_backend().download_files(paths)


__all__ = ["RepositoryRoutingSandbox"]
