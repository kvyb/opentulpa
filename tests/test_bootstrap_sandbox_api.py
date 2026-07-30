from __future__ import annotations

import asyncio
import json
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import pytest
from deepagents.backends.protocol import ExecuteResponse
from fastapi import FastAPI

from opentulpa.bootstrap.sandbox_api import (
    SandboxExecutionClient,
    SandboxExecutionError,
    SandboxExecutionLease,
    TenantSandboxExecutionService,
    register_sandbox_execution_api,
)
from opentulpa.deep_agent import sandbox
from opentulpa.deep_agent.sandbox import TenantContainerBackend, TenantContainerPolicy


class _FakeSandboxService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ExecuteResponse:
        self.calls.append(kwargs)
        return ExecuteResponse(output="safe output", exit_code=0, truncated=False)


def _lease(epoch: int = 7) -> SandboxExecutionLease:
    return SandboxExecutionLease(release_id="release-a", lease_epoch=epoch)


@pytest.mark.asyncio
async def test_private_sandbox_api_requires_token_and_active_release_lease() -> None:
    app = FastAPI()
    service = _FakeSandboxService()
    leases: list[tuple[str, int, str]] = []

    async def authorize(release_id: str, lease_epoch: int, control_token: str) -> None:
        leases.append((release_id, lease_epoch, control_token))

    register_sandbox_execution_api(
        app,
        service=service,  # type: ignore[arg-type]
        token="s" * 48,
        authorize_lease=authorize,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bootstrap") as client:
        denied = await client.post(
            "/bootstrap/internal/v1/sandbox/execute",
            json={"tenant_id": "tenant-a", "command": "pwd", "timeout": 5},
        )
        accepted = await client.post(
            "/bootstrap/internal/v1/sandbox/execute",
            headers={
                "X-OpenTulpa-Sandbox-Token": "s" * 48,
                "X-OpenTulpa-Release-ID": "release-a",
                "X-OpenTulpa-Lease-Epoch": "7",
                "X-OpenTulpa-Control-Token": "c" * 48,
            },
            json={"tenant_id": "tenant-a", "command": "pwd", "timeout": 5},
        )

    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {"output": "safe output", "exit_code": 0, "truncated": False}
    assert leases == [("release-a", 7, "c" * 48)]
    assert service.calls == [
        {
            "lease": _lease(),
            "tenant_id": "tenant-a",
            "command": "pwd",
            "timeout": 5,
        }
    ]


def test_sandbox_client_sends_hidden_lease_identity_and_validates_response() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"output": "hello", "exit_code": 0, "truncated": False},
        )

    client = SandboxExecutionClient(
        base_url="http://host.docker.internal:8000/bootstrap/internal/v1/sandbox",
        token="s" * 48,
        release_id="release-a",
        lease_epoch=9,
        control_token="c" * 48,
        max_response_bytes=100_000,
        transport=httpx.MockTransport(handler),
    )

    result = client.execute(tenant_id="tenant-a", command="printf hello", timeout=8)

    assert result == ExecuteResponse(output="hello", exit_code=0, truncated=False)
    assert captured["body"] == {
        "tenant_id": "tenant-a",
        "command": "printf hello",
        "timeout": 8,
    }
    assert captured["headers"]["x-opentulpa-release-id"] == "release-a"
    assert captured["headers"]["x-opentulpa-lease-epoch"] == "9"
    assert captured["headers"]["x-opentulpa-control-token"] == "c" * 48
    assert captured["headers"]["x-opentulpa-sandbox-token"] == "s" * 48


@pytest.mark.asyncio
async def test_stable_service_derives_isolated_tenant_workspace_and_caps_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str, int | None, str]] = []

    async def execute(
        backend: TenantContainerBackend,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        calls.append((backend.workspace, command, timeout, backend._policy.image))  # noqa: SLF001
        return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    monkeypatch.setattr(TenantContainerBackend, "aexecute", execute)
    image = f"sha256:{'a' * 64}"
    service = TenantSandboxExecutionService(
        workspaces_root=tmp_path / "tenant-workspaces",
        allowed_root=tmp_path,
        policy=TenantContainerPolicy(image=image, timeout_seconds=10),
        container_cli="docker",
    )
    lease = _lease()
    await service.reconcile_lease(lease)

    await service.execute(lease=lease, tenant_id="tenant-a", command="one", timeout=8)
    await service.execute(lease=lease, tenant_id="tenant-b", command="two", timeout=9)

    assert calls[0][0] != calls[1][0]
    assert all(path.is_relative_to(tmp_path / "tenant-workspaces") for path, *_ in calls)
    assert [(command, timeout, used_image) for _, command, timeout, used_image in calls] == [
        ("one", 8, image),
        ("two", 9, image),
    ]


@pytest.mark.asyncio
async def test_lease_transition_rejects_commit_that_has_not_linearized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempting_commit = threading.Event()
    allow_commit = threading.Event()
    committed = threading.Event()

    async def execute(
        backend: TenantContainerBackend,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        del command, timeout

        def commit() -> None:
            attempting_commit.set()
            assert allow_commit.wait(timeout=2)
            authority = backend._commit_authority  # noqa: SLF001
            assert authority is not None
            with authority():
                committed.set()

        worker = asyncio.create_task(asyncio.to_thread(commit))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as cancellation:
            allow_commit.set()
            with suppress(Exception):
                await worker
            raise cancellation
        return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    monkeypatch.setattr(TenantContainerBackend, "aexecute", execute)
    service = TenantSandboxExecutionService(
        workspaces_root=tmp_path / "tenant-workspaces",
        allowed_root=tmp_path,
        policy=TenantContainerPolicy(image=f"sha256:{'a' * 64}"),
        container_cli="docker",
    )
    old_lease = _lease(10)
    new_lease = _lease(11)
    await service.reconcile_lease(old_lease)
    execution = asyncio.create_task(
        service.execute(
            lease=old_lease,
            tenant_id="tenant-a",
            command="mutate",
            timeout=5,
        )
    )
    assert await asyncio.to_thread(attempting_commit.wait, 1)

    await service.reconcile_lease(new_lease)

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert not committed.is_set()
    with pytest.raises(SandboxExecutionError, match="lease changed"):
        await service.execute(
            lease=old_lease,
            tenant_id="tenant-a",
            command="stale",
            timeout=5,
        )


@pytest.mark.asyncio
async def test_lease_transition_waits_for_commit_that_already_linearized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit_started = threading.Event()
    finish_commit = threading.Event()

    async def execute(
        backend: TenantContainerBackend,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        del command, timeout

        def commit() -> None:
            authority = backend._commit_authority  # noqa: SLF001
            assert authority is not None
            with authority():
                commit_started.set()
                assert finish_commit.wait(timeout=2)

        await asyncio.to_thread(commit)
        return ExecuteResponse(output="committed", exit_code=0, truncated=False)

    monkeypatch.setattr(TenantContainerBackend, "aexecute", execute)
    service = TenantSandboxExecutionService(
        workspaces_root=tmp_path / "tenant-workspaces",
        allowed_root=tmp_path,
        policy=TenantContainerPolicy(image=f"sha256:{'a' * 64}"),
        container_cli="docker",
    )
    old_lease = _lease(20)
    new_lease = _lease(21)
    await service.reconcile_lease(old_lease)
    execution = asyncio.create_task(
        service.execute(
            lease=old_lease,
            tenant_id="tenant-a",
            command="mutate",
            timeout=5,
        )
    )
    assert await asyncio.to_thread(commit_started.wait, 1)
    transition = asyncio.create_task(service.reconcile_lease(new_lease))
    await asyncio.sleep(0)
    assert not transition.done()

    finish_commit.set()
    with suppress(asyncio.CancelledError):
        await execution
    await transition


@pytest.mark.asyncio
async def test_restarted_sandbox_service_is_closed_until_current_lease_reconciles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def execute(
        backend: TenantContainerBackend,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        del backend, command, timeout
        return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    monkeypatch.setattr(TenantContainerBackend, "aexecute", execute)
    service = TenantSandboxExecutionService(
        workspaces_root=tmp_path / "tenant-workspaces",
        allowed_root=tmp_path,
        policy=TenantContainerPolicy(image=f"sha256:{'a' * 64}"),
        container_cli="docker",
    )
    lease = _lease(30)

    with pytest.raises(SandboxExecutionError, match="lease changed"):
        await service.execute(
            lease=lease,
            tenant_id="tenant-a",
            command="before-reconcile",
            timeout=5,
        )
    await service.reconcile_lease(lease)
    result = await service.execute(
        lease=lease,
        tenant_id="tenant-a",
        command="after-reconcile",
        timeout=5,
    )
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_lease_transition_timeout_leaves_sandbox_admissions_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = asyncio.Event()

    async def execute(
        backend: TenantContainerBackend,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        del backend, command, timeout
        while not blocked.is_set():
            try:
                await blocked.wait()
            except asyncio.CancelledError:
                continue
        return ExecuteResponse(output="stopped", exit_code=0, truncated=False)

    monkeypatch.setattr(TenantContainerBackend, "aexecute", execute)
    service = TenantSandboxExecutionService(
        workspaces_root=tmp_path / "tenant-workspaces",
        allowed_root=tmp_path,
        policy=TenantContainerPolicy(image=f"sha256:{'a' * 64}"),
        container_cli="docker",
    )
    service._transition_timeout_seconds = 0.05  # noqa: SLF001
    old_lease = _lease(40)
    new_lease = _lease(41)
    await service.reconcile_lease(old_lease)
    execution = asyncio.create_task(
        service.execute(
            lease=old_lease,
            tenant_id="tenant-a",
            command="ignore-cancellation",
            timeout=5,
        )
    )
    await asyncio.sleep(0)

    with pytest.raises(SandboxExecutionError, match="did not drain"):
        await service.reconcile_lease(new_lease)
    with pytest.raises(SandboxExecutionError, match="lease changed"):
        await service.execute(
            lease=new_lease,
            tenant_id="tenant-a",
            command="must-remain-closed",
            timeout=5,
        )

    blocked.set()
    await execution
    await service.reconcile_lease(new_lease)
    result = await service.execute(
        lease=new_lease,
        tenant_id="tenant-a",
        command="reopened-after-drain",
        timeout=5,
    )
    assert result.output == "stopped"


@pytest.mark.asyncio
async def test_lease_transition_timeout_while_commit_authority_is_held_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def execute(
        backend: TenantContainerBackend,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        del backend, command, timeout
        return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    monkeypatch.setattr(TenantContainerBackend, "aexecute", execute)
    service = TenantSandboxExecutionService(
        workspaces_root=tmp_path / "tenant-workspaces",
        allowed_root=tmp_path,
        policy=TenantContainerPolicy(image=f"sha256:{'a' * 64}"),
        container_cli="docker",
    )
    service._transition_timeout_seconds = 0.05  # noqa: SLF001
    old_lease = _lease(50)
    new_lease = _lease(51)
    await service.reconcile_lease(old_lease)
    assert service._commit_lock.acquire(blocking=False)  # noqa: SLF001
    try:
        with pytest.raises(SandboxExecutionError, match="could not fence"):
            await service.reconcile_lease(new_lease)
        with pytest.raises(SandboxExecutionError, match="lease changed"):
            await service.execute(
                lease=old_lease,
                tenant_id="tenant-a",
                command="must-remain-closed",
                timeout=5,
            )
    finally:
        service._commit_lock.release()  # noqa: SLF001

    await service.reconcile_lease(new_lease)
    result = await service.execute(
        lease=new_lease,
        tenant_id="tenant-a",
        command="reopened-after-fence",
        timeout=5,
    )
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_stable_service_discards_invalid_transaction_before_next_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def execute_container(argv: list[str], **kwargs: Any) -> Any:
        nonlocal calls
        del kwargs
        calls += 1
        mount = argv[argv.index("--mount") + 1]
        staged = Path(
            next(part.removeprefix("src=") for part in mount.split(",") if part.startswith("src="))
        )
        if calls == 1:
            (staged / ".env").write_text("TOKEN=secret", encoding="utf-8")
        else:
            (staged / "managed.txt").write_text("safe", encoding="utf-8")
        return sandbox.subprocess.CompletedProcess(argv, 0, stdout=b"ok")

    monkeypatch.setattr(sandbox.subprocess, "run", execute_container)
    service = TenantSandboxExecutionService(
        workspaces_root=tmp_path / "tenant-workspaces",
        allowed_root=tmp_path,
        policy=TenantContainerPolicy(image=f"sha256:{'a' * 64}"),
        container_cli="docker",
    )
    lease = _lease()
    await service.reconcile_lease(lease)

    rejected = await service.execute(
        lease=lease,
        tenant_id="tenant-a",
        command="poison",
        timeout=5,
    )
    recovered = await service.execute(
        lease=lease,
        tenant_id="tenant-a",
        command="recover",
        timeout=5,
    )

    assert rejected.exit_code == 126
    assert recovered.exit_code == 0
    tenant_workspace = next(
        path
        for path in (tmp_path / "tenant-workspaces").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    assert not (tenant_workspace / ".env").exists()
    assert (tenant_workspace / "managed.txt").read_text(encoding="utf-8") == "safe"


@pytest.mark.asyncio
async def test_stable_service_recovers_previous_workspace_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tenant-workspaces"
    backend = TenantContainerBackend(tenant_id="tenant-a", workspaces_root=root)
    backend.write("/notes.txt", "original")
    transaction = root / f".{backend.workspace.name}.transaction-crashed"
    transaction.mkdir(mode=0o700)
    backend._write_transaction_journal(  # noqa: SLF001
        transaction,
        phase="previous",
        container_name=None,
    )
    sandbox.os.replace(backend.workspace, transaction / "previous")

    observed: list[str] = []

    async def inspect_recovered(
        recovered: TenantContainerBackend,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        del command, timeout
        observed.append((recovered.workspace / "notes.txt").read_text(encoding="utf-8"))
        return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    monkeypatch.setattr(TenantContainerBackend, "aexecute", inspect_recovered)
    service = TenantSandboxExecutionService(
        workspaces_root=root,
        allowed_root=tmp_path,
        policy=TenantContainerPolicy(image=f"sha256:{'a' * 64}"),
        container_cli="docker",
    )
    lease = _lease()
    await service.reconcile_lease(lease)

    result = await service.execute(
        lease=lease,
        tenant_id="tenant-a",
        command="pwd",
        timeout=5,
    )

    assert result.exit_code == 0
    assert observed == ["original"]
    assert not any(".transaction-" in path.name for path in root.iterdir())


@pytest.mark.asyncio
async def test_stable_service_rejects_replaced_workspace_root(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    release_root.mkdir()
    workspaces = release_root / "deepagents" / "workspaces"
    service = TenantSandboxExecutionService(
        workspaces_root=workspaces,
        allowed_root=release_root,
        policy=TenantContainerPolicy(image=f"sha256:{'a' * 64}"),
        container_cli="docker",
    )
    lease = _lease()
    await service.reconcile_lease(lease)
    outside = tmp_path / "outside"
    outside.mkdir()
    workspaces.rmdir()
    workspaces.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="failed validation"):
        await service.execute(
            lease=lease,
            tenant_id="tenant-a",
            command="pwd",
            timeout=5,
        )


def test_tenant_backend_uses_remote_provider_without_invoking_local_oci(tmp_path: Path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.workspace_was_ready = False

        def execute(self, **kwargs: Any) -> ExecuteResponse:
            self.workspace_was_ready = kwargs["workspace"].is_dir()
            self.calls.append(kwargs)
            return ExecuteResponse(output="remote", exit_code=0, truncated=False)

    provider = Provider()
    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
        policy=TenantContainerPolicy(timeout_seconds=7),
        execution_provider=provider,
    )

    result = backend.execute("printf remote", timeout=999)

    assert result.output == "remote"
    assert len(provider.calls) == 1
    assert provider.calls[0]["tenant_id"] == "tenant-a"
    assert provider.calls[0]["command"] == "printf remote"
    assert provider.calls[0]["timeout"] == 7
    assert provider.workspace_was_ready is True
