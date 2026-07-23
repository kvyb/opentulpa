from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from opentulpa.api.routes.v2_repositories import register_v2_repository_routes
from opentulpa.repositories.models import (
    RepositoryProvider,
    RepositoryWorkspace,
    RepositoryWorkspaceStatus,
)
from opentulpa.repositories.providers import (
    RepositoryProviderRegistry,
    RepositoryRootedBackend,
)
from opentulpa.repositories.routing import RepositoryRoutingSandbox
from opentulpa.repositories.service import (
    RepositoryPublishError,
    RepositoryWorkspaceConflictError,
    RepositoryWorkspaceError,
    RepositoryWorkspaceService,
)
from opentulpa.repositories.store import RepositoryWorkspaceStore

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


class FakeBackend:
    id = "fake"

    def __init__(self) -> None:
        self.head = BASE_SHA
        self.dirty = ""
        self.remote = "https://github.com/acme/project.git"
        self.commands: list[str] = []
        self.paths: list[str] = []

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        del timeout
        self.commands.append(command)
        if command.startswith("git rev-parse"):
            return ExecuteResponse(output=f"{self.head}\n", exit_code=0)
        if command == "git status --porcelain=v1":
            return ExecuteResponse(output=self.dirty, exit_code=0)
        if command == "git branch --show-current":
            return ExecuteResponse(output="opentulpa/change\n", exit_code=0)
        if command == "git remote get-url origin":
            return ExecuteResponse(
                output=f"{self.remote}\n",
                exit_code=0,
            )
        if command.startswith("git rev-list --count"):
            return ExecuteResponse(output="1\n", exit_code=0)
        return ExecuteResponse(output="", exit_code=0)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self.execute(command, timeout=timeout)

    def ls(self, path: str) -> LsResult:
        self.paths.append(path)
        return []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        del offset, limit
        self.paths.append(file_path)
        return "content"

    def write(self, file_path: str, content: str) -> WriteResult:
        del content
        self.paths.append(file_path)
        return WriteResult(error=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        del old_string, new_string, replace_all
        self.paths.append(file_path)
        return EditResult(error=None)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        del pattern
        self.paths.append(str(path))
        return []

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        del pattern, glob
        self.paths.append(str(path))
        return []

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self.paths.extend(path for path, _ in files)
        return [FileUploadResponse(path=path) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        self.paths.extend(paths)
        return [FileDownloadResponse(path=path, content=b"data") for path in paths]


class FakeProvider:
    name = RepositoryProvider.DAYTONA

    def __init__(self) -> None:
        self.sandbox = FakeBackend()
        self.fail_backend = False
        self.created: list[str] = []
        self.pushed: list[tuple[str, str]] = []
        self.stopped: list[str] = []

    def available(self, tenant_id: str) -> bool:
        return tenant_id == "tenant-1"

    def create(self, workspace: RepositoryWorkspace, *, github_token: str | None) -> str:
        assert github_token == "github-secret"
        self.created.append(workspace.id)
        return f"sandbox-{workspace.id}"

    def backend(self, workspace: RepositoryWorkspace) -> FakeBackend:
        assert workspace.provider_workspace_id
        if self.fail_backend:
            raise RuntimeError("provider setup failed")
        return self.sandbox

    def stop(self, workspace: RepositoryWorkspace) -> None:
        self.stopped.append(workspace.id)

    def push(self, workspace: RepositoryWorkspace, *, github_token: str) -> None:
        self.pushed.append((workspace.id, github_token))


def _service(tmp_path: Any) -> tuple[RepositoryWorkspaceService, FakeProvider]:
    provider = FakeProvider()
    service = RepositoryWorkspaceService(
        store=RepositoryWorkspaceStore(tmp_path / "repositories.db"),
        providers=RepositoryProviderRegistry(providers=[provider]),
        github_token_resolver=lambda tenant_id, scope: (
            "github-secret" if tenant_id == "tenant-1" else None
        ),
    )
    service._open_pull_request = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "https://github.com/acme/project/pull/7"
    )
    return service, provider


@pytest.mark.asyncio
async def test_workspace_open_binds_thread_and_publishes_exact_commit(tmp_path: Any) -> None:
    service, provider = _service(tmp_path)

    workspace = await service.open(
        tenant_id="tenant-1",
        thread_id="thread-1",
        repository_url="https://github.com/acme/project",
        base_ref="main",
        branch="opentulpa/change",
        provider="auto",
    )

    assert workspace.status is RepositoryWorkspaceStatus.READY
    assert workspace.base_sha == BASE_SHA
    assert (await service.active(tenant_id="tenant-1", thread_id="thread-1")) == workspace
    provider.sandbox.head = HEAD_SHA

    status = await service.status(
        tenant_id="tenant-1",
        thread_id="thread-1",
    )
    assert status["head_sha"] == HEAD_SHA
    assert status["clean"] is True

    result = await service.publish(
        tenant_id="tenant-1",
        thread_id="thread-1",
        workspace_id=None,
        expected_head_sha=HEAD_SHA,
        title="Improve the project",
        body="Verified in the repository sandbox.",
        draft=True,
    )

    assert result["head_sha"] == HEAD_SHA
    assert result["pull_request_url"] == "https://github.com/acme/project/pull/7"
    assert provider.pushed == [(workspace.id, "github-secret")]
    assert "github-secret" not in repr(result)


@pytest.mark.asyncio
async def test_workspace_open_stops_provider_after_setup_failure(tmp_path: Any) -> None:
    service, provider = _service(tmp_path)
    provider.fail_backend = True

    with pytest.raises(RepositoryWorkspaceError, match="could not be created"):
        await service.open(
            tenant_id="tenant-1",
            thread_id="thread-1",
            repository_url="https://github.com/acme/project",
        )

    assert provider.stopped == provider.created
    [workspace] = await service.list(tenant_id="tenant-1", include_closed=True)
    assert workspace.status is RepositoryWorkspaceStatus.FAILED
    assert workspace.last_error == "repository workspace operation failed"


@pytest.mark.asyncio
async def test_publish_rejects_changed_head_and_dirty_tree(tmp_path: Any) -> None:
    service, provider = _service(tmp_path)
    await service.open(
        tenant_id="tenant-1",
        thread_id="thread-1",
        repository_url="https://github.com/acme/project.git",
        branch="opentulpa/change",
    )
    provider.sandbox.head = HEAD_SHA

    with pytest.raises(RepositoryWorkspaceConflictError):
        await service.publish(
            tenant_id="tenant-1",
            thread_id="thread-1",
            workspace_id=None,
            expected_head_sha="c" * 40,
            title="Unsafe",
            body="",
            draft=True,
        )

    provider.sandbox.dirty = " M README.md\n"
    with pytest.raises(RepositoryPublishError, match="uncommitted"):
        await service.publish(
            tenant_id="tenant-1",
            thread_id="thread-1",
            workspace_id=None,
            expected_head_sha=HEAD_SHA,
            title="Unsafe",
            body="",
            draft=True,
        )
    assert provider.pushed == []

    provider.sandbox.dirty = ""
    provider.sandbox.remote = "https://github.com/other/project.git"
    with pytest.raises(RepositoryPublishError, match="origin changed"):
        await service.publish(
            tenant_id="tenant-1",
            thread_id="thread-1",
            workspace_id=None,
            expected_head_sha=HEAD_SHA,
            title="Unsafe",
            body="",
            draft=True,
        )
    assert provider.pushed == []


@pytest.mark.asyncio
async def test_workspace_store_enforces_tenant_ownership(tmp_path: Any) -> None:
    service, _ = _service(tmp_path)
    workspace = await service.open(
        tenant_id="tenant-1",
        thread_id="thread-1",
        repository_url="https://github.com/acme/project",
    )

    assert await service.active(tenant_id="tenant-2", thread_id="thread-1") is None
    assert all(
        item.id != workspace.id
        for item in await service.list(tenant_id="tenant-2", include_closed=True)
    )


def test_rooted_backend_maps_workspace_paths_without_rewriting_content() -> None:
    delegate = FakeBackend()
    backend = RepositoryRootedBackend(
        delegate=delegate,
        file_root="/workspace/repo",
        execution_root="/workspace/repo",
    )

    backend.read("/workspace/src/app.py")
    backend.write("/workspace/README.md", "complete file")
    backend.download_files(["/workspace/README.md"])
    backend.execute("git status")

    assert delegate.paths == [
        "/workspace/repo/src/app.py",
        "/workspace/repo/README.md",
        "/workspace/repo/README.md",
    ]
    assert delegate.commands == ["cd /workspace/repo && (git status)"]
    with pytest.raises(ValueError, match="traversal"):
        backend.read("/workspace/../secrets")


def test_runtime_router_uses_active_thread_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    active = FakeBackend()
    fallback = FakeBackend()
    repositories = SimpleNamespace(
        backend_for_thread=lambda **kwargs: (
            active
            if kwargs == {"tenant_id": "tenant-1", "thread_id": "thread-1"}
            else None
        )
    )
    router = RepositoryRoutingSandbox(repositories=repositories, fallback=fallback)
    monkeypatch.setattr(
        "opentulpa.repositories.routing.get_runtime",
        lambda: SimpleNamespace(
            context=SimpleNamespace(tenant_id="tenant-1", thread_id="thread-1")
        ),
    )

    router.execute("git status")

    assert active.commands == ["git status"]
    assert fallback.commands == []

    execution_router = RepositoryRoutingSandbox(
        repositories=repositories,
        fallback=fallback,
        route_files=False,
    )
    execution_router.write("/scratch.txt", "ephemeral")

    assert active.paths == []
    assert fallback.paths == ["/scratch.txt"]


def test_repository_api_resolves_tenant_from_authentication(tmp_path: Any) -> None:
    service, _ = _service(tmp_path)
    app = FastAPI()

    async def principal(request: Request) -> Any:
        return SimpleNamespace(
            tenant_id=request.headers.get("x-tenant-id", ""),
            actor_id="owner",
            trust_class="owner",
            scopes=frozenset({"*"}),
        )

    register_v2_repository_routes(
        app,
        get_repositories=lambda: service,
        resolve_principal=principal,
    )
    client = TestClient(app)

    created = client.post(
        "/v2/repositories/workspaces",
        headers={"x-tenant-id": "tenant-1"},
        json={
            "thread_id": "thread-1",
            "repository_url": "https://github.com/acme/project",
        },
    )
    assert created.status_code == 201
    assert "tenant_id" not in created.json()

    own = client.get(
        "/v2/repositories/workspaces",
        headers={"x-tenant-id": "tenant-1"},
    )
    other = client.get(
        "/v2/repositories/workspaces",
        headers={"x-tenant-id": "tenant-2"},
    )
    assert len(own.json()["workspaces"]) == 1
    assert other.json()["workspaces"] == []
