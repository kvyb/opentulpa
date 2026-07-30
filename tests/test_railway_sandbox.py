from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
from deepagents.backends.protocol import ExecuteResponse

from opentulpa.deep_agent.railway_sandbox import (
    RailwaySandboxExecutionError,
    RailwaySandboxExecutionProvider,
)
from opentulpa.deep_agent.sandbox import TenantContainerBackend


def _archive(files: dict[str, str]) -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            payload = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))
    return base64.b64encode(buffer.getvalue()).decode()


def _provider(runner: Any) -> RailwaySandboxExecutionProvider:
    return RailwaySandboxExecutionProvider(
        token="project-token",
        environment_id="environment-id",
        max_output_bytes=4_096,
        max_workspace_archive_bytes=1_000_000,
        max_workspace_entries=100,
        max_file_bytes=100_000,
        runner=runner,
    )


def test_provider_synchronizes_workspace_and_reuses_tenant_sandbox(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    def run(request: dict[str, Any], timeout: int) -> dict[str, Any]:
        assert timeout == 7
        requests.append(request)
        with tarfile.open(
            fileobj=io.BytesIO(base64.b64decode(request["workspaceArchive"])),
            mode="r:gz",
        ) as archive:
            note = archive.extractfile("./note.txt")
            assert note is not None
            assert note.read() == b"local"
        return {
            "ok": True,
            "sandboxId": "sandbox-1",
            "workspaceArchive": _archive(
                {"note.txt": "local", "created-remotely.txt": "remote"}
            ),
            "workspaceSynchronized": True,
            "output": "remote output",
            "exitCode": 0,
            "truncated": False,
        }

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("local", encoding="utf-8")
    provider = _provider(run)

    first = provider.execute(
        tenant_id="tenant-a",
        command="touch created-remotely.txt",
        timeout=7,
        workspace=workspace,
    )
    second = provider.execute(
        tenant_id="tenant-a",
        command="pwd",
        timeout=7,
        workspace=workspace,
    )

    assert first.output == "remote output"
    assert second.exit_code == 0
    assert (workspace / "created-remotely.txt").read_text(encoding="utf-8") == "remote"
    assert requests[0]["sandboxId"] is None
    assert requests[1]["sandboxId"] == "sandbox-1"
    assert all("project-token" not in str(request) for request in requests)


def test_provider_rejects_unsafe_remote_workspace_archive(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)

    def run(request: dict[str, Any], timeout: int) -> dict[str, Any]:
        del request, timeout
        return {
            "ok": True,
            "sandboxId": "sandbox-1",
            "workspaceArchive": base64.b64encode(buffer.getvalue()).decode(),
            "workspaceSynchronized": True,
            "output": "",
            "exitCode": 0,
            "truncated": False,
        }

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "safe.txt").write_text("safe", encoding="utf-8")

    with pytest.raises(
        RailwaySandboxExecutionError,
        match="unsupported entry",
    ):
        _provider(run).execute(
            tenant_id="tenant-a",
            command="true",
            timeout=7,
            workspace=workspace,
        )

    assert (workspace / "safe.txt").read_text(encoding="utf-8") == "safe"


def test_provider_preserves_workspace_when_remote_command_times_out(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "before.txt").write_text("before", encoding="utf-8")

    def run(request: dict[str, Any], timeout: int) -> dict[str, Any]:
        del request, timeout
        return {
            "ok": True,
            "sandboxId": "sandbox-1",
            "workspaceArchive": None,
            "workspaceSynchronized": False,
            "output": "command timed out after 5s",
            "exitCode": 124,
            "truncated": False,
        }

    response = _provider(run).execute(
        tenant_id="tenant-a",
        command="sleep 10",
        timeout=5,
        workspace=workspace,
    )

    assert response.exit_code == 124
    assert (workspace / "before.txt").read_text(encoding="utf-8") == "before"


def test_tenant_backend_commits_remote_workspace_transaction(tmp_path: Path) -> None:
    class Provider:
        def execute(
            self,
            *,
            tenant_id: str,
            command: str,
            timeout: int,
            workspace: Path | None = None,
        ) -> ExecuteResponse:
            assert (tenant_id, command, timeout) == ("tenant-a", "make file", 9)
            assert workspace is not None
            assert (workspace / "local.txt").read_text(encoding="utf-8") == "local"
            (workspace / "remote.txt").write_text("remote", encoding="utf-8")
            return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
        execution_provider=Provider(),
    )
    (backend.workspace / "local.txt").write_text("local", encoding="utf-8")

    response = backend.execute("make file", timeout=9)

    assert response.exit_code == 0
    assert (backend.workspace / "remote.txt").read_text(encoding="utf-8") == "remote"


def test_tenant_backend_discards_failed_remote_workspace_transaction(
    tmp_path: Path,
) -> None:
    class Provider:
        def execute(
            self,
            *,
            tenant_id: str,
            command: str,
            timeout: int,
            workspace: Path | None = None,
        ) -> ExecuteResponse:
            del tenant_id, command, timeout
            assert workspace is not None
            (workspace / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("remote failed")

    backend = TenantContainerBackend(
        tenant_id="tenant-a",
        workspaces_root=tmp_path / "workspaces",
        execution_provider=Provider(),
    )
    (backend.workspace / "local.txt").write_text("local", encoding="utf-8")

    response = backend.execute("fail", timeout=9)

    assert response.exit_code == 127
    assert (backend.workspace / "local.txt").read_text(encoding="utf-8") == "local"
    assert not (backend.workspace / "partial.txt").exists()
