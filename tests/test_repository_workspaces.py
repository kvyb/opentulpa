from __future__ import annotations

import base64
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
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
from opentulpa.deep_agent.sandbox import TenantContainerPolicy
from opentulpa.evolution.sandbox import CandidateProcessBackend
from opentulpa.repositories.models import (
    RepositoryProvider,
    RepositoryWorkspace,
    RepositoryWorkspaceStatus,
    utc_now,
)
from opentulpa.repositories.providers import (
    DaytonaRepositoryProvider,
    LocalRepositoryProvider,
    RepositoryProviderRegistry,
    RepositoryRootedBackend,
    RepositorySandboxUnavailableError,
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


class FakeLocalProvider(FakeProvider):
    name = RepositoryProvider.LOCAL


class RealGitBackend(FakeBackend):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.commands.append(command)
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ExecuteResponse(output=result.stdout + result.stderr, exit_code=result.returncode)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            relative = path.removeprefix("/workspace/")
            responses.append(
                FileDownloadResponse(path=path, content=(self.root / relative).read_bytes())
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            relative = path.removeprefix("/workspace/")
            target = self.root / relative
            target.write_bytes(content)
            responses.append(FileUploadResponse(path=path))
        return responses


class ExistingRepositoryProvider(FakeProvider):
    def __init__(self, backend: RealGitBackend) -> None:
        super().__init__()
        self.sandbox = backend


class FakeGitHubProxy:
    def __init__(self, *, head_tree_sha: str, corrupt_commit: bool = False) -> None:
        self.head_tree_sha = head_tree_sha
        self.corrupt_commit = corrupt_commit
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _commit_sha(body: dict[str, Any]) -> str:
        def identity(label: str) -> str:
            value = body[label]
            parsed = datetime.fromisoformat(value["date"])
            return (
                f"{label} {value['name']} <{value['email']}> "
                f"{int(parsed.timestamp())} {parsed.strftime('%z')}"
            )

        raw = (
            f"tree {body['tree']}\n"
            f"parent {body['parents'][0]}\n"
            f"{identity('author')}\n"
            f"{identity('committer')}\n"
            f"\n{body['message']}"
        ).encode()
        prefix = f"commit {len(raw)}\0".encode()
        return hashlib.sha1(prefix + raw, usedforsecurity=False).hexdigest()

    def request(
        self,
        *,
        tenant_id: str,
        method: str,
        endpoint: str,
        body: object | None = None,
    ) -> tuple[int, Any]:
        assert tenant_id == "tenant-1"
        call = {"method": method, "endpoint": endpoint, "body": body}
        self.calls.append(call)
        if endpoint.endswith("/git/blobs"):
            assert isinstance(body, dict)
            content = base64.b64decode(str(body["content"]), validate=True)
            prefix = f"blob {len(content)}\0".encode()
            sha = hashlib.sha1(prefix + content, usedforsecurity=False).hexdigest()
            return 201, {"sha": sha}
        if endpoint.endswith("/git/trees"):
            return 201, {"sha": self.head_tree_sha}
        if endpoint.endswith("/git/commits"):
            assert isinstance(body, dict)
            sha = self._commit_sha(body)
            return 201, {"sha": ("f" * 40 if self.corrupt_commit else sha)}
        if "/git/ref/heads/" in endpoint:
            return 404, {"message": "Not Found"}
        if endpoint.endswith("/git/refs"):
            assert isinstance(body, dict)
            return 201, {"object": {"sha": body["sha"]}}
        if endpoint.endswith("/pulls"):
            return 201, {"html_url": "https://github.com/acme/project/pull/9"}
        raise AssertionError(f"unexpected GitHub request: {method} {endpoint}")


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


def _workspace() -> RepositoryWorkspace:
    timestamp = utc_now()
    return RepositoryWorkspace(
        id="repo-process-test",
        tenant_id="tenant-1",
        repository_url="https://github.com/acme/project.git",
        provider=RepositoryProvider.LOCAL,
        provider_workspace_id="repo-process-test",
        base_ref="main",
        branch="opentulpa/change",
        status=RepositoryWorkspaceStatus.READY,
        created_at=timestamp,
        updated_at=timestamp,
        last_used_at=timestamp,
    )


def _composio_publish_service(
    tmp_path: Path,
    *,
    corrupt_commit: bool = False,
) -> tuple[
    RepositoryWorkspaceService,
    ExistingRepositoryProvider,
    FakeGitHubProxy,
    bytes,
    str,
]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/acme/project.git"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "OpenTulpa"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "opentulpa@localhost"],
        cwd=checkout,
        check=True,
    )
    (checkout / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "base"],
        cwd=checkout,
        check=True,
    )
    base_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
    ).strip()
    subprocess.run(
        ["git", "checkout", "-qb", "opentulpa/change"],
        cwd=checkout,
        check=True,
    )
    complete_content = ("complete README line\n" * 1400).encode()
    (checkout / "README.md").write_bytes(complete_content)
    subprocess.run(["git", "add", "README.md"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "docs: replace the complete README",
        ],
        cwd=checkout,
        check=True,
    )
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
    ).strip()
    head_tree_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=checkout,
        text=True,
    ).strip()

    backend = RealGitBackend(checkout)
    provider = ExistingRepositoryProvider(backend)
    proxy = FakeGitHubProxy(
        head_tree_sha=head_tree_sha,
        corrupt_commit=corrupt_commit,
    )
    store = RepositoryWorkspaceStore(tmp_path / "repositories.db")
    now = utc_now()
    workspace = RepositoryWorkspace(
        id="repo-composio",
        tenant_id="tenant-1",
        repository_url="https://github.com/acme/project.git",
        provider=RepositoryProvider.DAYTONA,
        provider_workspace_id="sandbox-composio",
        base_ref="main",
        branch="opentulpa/change",
        base_sha=base_sha,
        head_sha=head_sha,
        status=RepositoryWorkspaceStatus.READY,
        created_at=now,
        updated_at=now,
        last_used_at=now,
    )
    store.create(workspace)
    store.bind(
        tenant_id="tenant-1",
        thread_id="thread-1",
        workspace_id=workspace.id,
        bound_at=now,
    )
    service = RepositoryWorkspaceService(
        store=store,
        providers=RepositoryProviderRegistry(providers=[provider]),
        github_token_resolver=lambda tenant_id, scope: None,
        github_api_proxy=proxy,
    )
    return service, provider, proxy, complete_content, head_sha


@pytest.mark.asyncio
async def test_import_verified_patch_creates_one_clean_commit_from_recorded_base(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "OpenTulpa"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "opentulpa@localhost"],
        cwd=checkout,
        check=True,
    )
    (checkout / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=checkout, check=True)
    base_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    subprocess.run(["git", "checkout", "-qb", "opentulpa/change"], cwd=checkout, check=True)
    patch = (
        b"diff --git a/README.md b/README.md\n"
        b"--- a/README.md\n"
        b"+++ b/README.md\n"
        b"@@ -1 +1 @@\n"
        b"-base\n"
        b"+evolved\n"
    )
    digest = hashlib.sha256(patch).hexdigest()
    backend = RealGitBackend(checkout)
    provider = ExistingRepositoryProvider(backend)
    store = RepositoryWorkspaceStore(tmp_path / "repositories.db")
    now = utc_now()
    workspace = RepositoryWorkspace(
        id="repo-import",
        tenant_id="tenant-1",
        repository_url="https://github.com/acme/project.git",
        provider=RepositoryProvider.DAYTONA,
        provider_workspace_id="sandbox-import",
        base_ref="main",
        branch="opentulpa/change",
        base_sha=base_sha,
        head_sha=base_sha,
        status=RepositoryWorkspaceStatus.READY,
        created_at=now,
        updated_at=now,
        last_used_at=now,
    )
    store.create(workspace)
    service = RepositoryWorkspaceService(
        store=store,
        providers=RepositoryProviderRegistry(providers=[provider]),
        github_token_resolver=lambda tenant_id, scope: None,
    )

    imported = await service.import_verified_patch(
        tenant_id="tenant-1",
        thread_id="thread-1",
        workspace_id=workspace.id,
        patch=patch,
        expected_sha256=digest,
        message="Apply tested self-evolution",
    )

    assert imported["base_sha"] == base_sha
    assert imported["head_sha"] != base_sha
    assert imported["patch_sha256"] == digest
    assert (checkout / "README.md").read_text(encoding="utf-8") == "evolved\n"
    assert subprocess.check_output(
        ["git", "rev-list", "--count", f"{base_sha}..HEAD"],
        cwd=checkout,
        text=True,
    ).strip() == "1"
    assert subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=checkout,
        text=True,
    ).strip() == ""


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
async def test_composio_publishes_complete_verified_commit_without_git_token(
    tmp_path: Path,
) -> None:
    service, provider, proxy, complete_content, head_sha = _composio_publish_service(tmp_path)

    result = await service.publish(
        tenant_id="tenant-1",
        thread_id="thread-1",
        workspace_id=None,
        expected_head_sha=head_sha,
        title="Complete README",
        body="The full file was tested in the repository workspace.",
        draft=True,
    )

    blob_call = next(call for call in proxy.calls if call["endpoint"].endswith("/git/blobs"))
    assert isinstance(blob_call["body"], dict)
    assert base64.b64decode(blob_call["body"]["content"], validate=True) == complete_content
    assert len(complete_content) > 16_000
    assert result["head_sha"] == head_sha
    assert result["credential_source"] == "composio"
    assert result["pull_request_url"] == "https://github.com/acme/project/pull/9"
    assert provider.pushed == []


@pytest.mark.asyncio
async def test_composio_publish_fails_before_branch_when_commit_sha_changes(
    tmp_path: Path,
) -> None:
    service, _, proxy, _, head_sha = _composio_publish_service(
        tmp_path,
        corrupt_commit=True,
    )

    with pytest.raises(RepositoryPublishError, match="approved commit"):
        await service.publish(
            tenant_id="tenant-1",
            thread_id="thread-1",
            workspace_id=None,
            expected_head_sha=head_sha,
            title="Unsafe",
            body="",
            draft=True,
        )

    assert not any(call["endpoint"].endswith("/git/refs") for call in proxy.calls)


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


def test_auto_provider_prefers_zero_config_local_sandbox() -> None:
    local = FakeLocalProvider()
    hosted = FakeProvider()
    registry = RepositoryProviderRegistry(providers=[hosted, local])

    assert registry.select(tenant_id="tenant-1", requested="auto") is local


def test_hosted_provider_fails_cleanly_when_optional_adapter_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "opentulpa.repositories.providers.importlib.util.find_spec",
        lambda _: None,
    )
    provider = DaytonaRepositoryProvider(
        token_resolver=lambda tenant_id, scope: "configured",
        api_url="https://app.daytona.io/api",
        target=None,
        snapshot=None,
    )

    assert provider.available("tenant-1") is False
    with pytest.raises(RepositorySandboxUnavailableError, match="hosted-sandbox extra"):
        provider.backend(_workspace())


def test_local_provider_uses_process_sandbox_without_oci(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "workspaces" / "repo-process-test" / "repo"
    checkout.mkdir(parents=True)

    class ProcessBackend(FakeBackend):
        @staticmethod
        def supported() -> bool:
            return True

        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            self.root = str(kwargs["workspace"])

        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            del timeout
            self.commands.append(command)
            return ExecuteResponse(
                output=f"{self.root}/README.md\n",
                exit_code=0,
            )

    monkeypatch.setattr(
        "opentulpa.repositories.providers.CandidateProcessBackend",
        ProcessBackend,
    )
    monkeypatch.setattr("opentulpa.repositories.providers.shutil.which", lambda _: None)
    provider = LocalRepositoryProvider(
        root=tmp_path / "workspaces",
        policy=TenantContainerPolicy(network_enabled=True, timeout_seconds=600),
        container_cli="missing-oci",
    )

    assert provider.available("tenant-1") is True
    backend = provider.backend(_workspace())
    result = backend.execute("pwd")

    assert result.output == "/workspace/README.md\n"


@pytest.mark.skipif(
    not CandidateProcessBackend.is_supported(),
    reason=CandidateProcessBackend.unavailable_reason()
    or "root Linux bubblewrap isolation is required",
)
def test_process_repository_backend_runs_git_without_host_credentials(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    checkout = private_root / "workspaces" / "repo-process-test" / "repo"
    checkout.mkdir(parents=True)
    private_root.chmod(0o700)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "README.md").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=OpenTulpa",
            "-c",
            "user.email=opentulpa@localhost",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )
    monkeypatch.setenv("OPENTULPA_SMOKE_SECRET", "must-not-leak")
    monkeypatch.setenv("COMPOSIO_API_KEY", "must-not-leak")
    provider = LocalRepositoryProvider(
        root=private_root / "workspaces",
        policy=TenantContainerPolicy(network_enabled=True, timeout_seconds=600),
        container_cli="missing-oci",
    )

    result = provider.backend(_workspace()).execute(
        "printf 'after\\n' > README.md && "
        "git add README.md && "
        "git -c user.name=OpenTulpa -c user.email=opentulpa@localhost "
        "commit -qm update && "
        'printf \'%s|%s|%s\' "$(id -u)" "${OPENTULPA_SMOKE_SECRET-unset}" '
        '"${COMPOSIO_API_KEY-unset}"'
    )

    assert result.exit_code == 0
    uid, smoke_secret, composio_secret = result.output.split("|")
    assert uid != "0"
    assert smoke_secret == "unset"
    assert composio_secret == "unset"
    assert (checkout / "README.md").read_text(encoding="utf-8") == "after\n"
    assert private_root.stat().st_mode & 0o777 == 0o700


def test_runtime_router_uses_active_thread_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    active = FakeBackend()
    fallback = FakeBackend()
    repositories = SimpleNamespace(
        backend_for_thread=lambda **kwargs: (
            active if kwargs == {"tenant_id": "tenant-1", "thread_id": "thread-1"} else None
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

    router.upload_files_for_context(
        tenant_id="tenant-1",
        thread_id="thread-1",
        files=[("/workspace/.opentulpa/attachments/run/file.bin", b"data")],
    )

    assert active.paths == ["/workspace/.opentulpa/attachments/run/file.bin"]
    assert any(".opentulpa/attachments/" in command for command in active.commands)


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
