from __future__ import annotations

import base64
import io
import shlex
import sys
import tarfile
from pathlib import Path

import httpx
import pytest

from opentulpa.core.config import Settings
from opentulpa.sandbox.supervisor import SandboxWorkerSupervisor
from opentulpa.sandbox.worker import (
    DevProcessEngine,
    SandboxWorkerError,
    SandboxWorkerService,
    create_sandbox_worker_app,
)

TOKEN = "sandbox-token-with-at-least-thirty-two-characters"


def _service(tmp_path: Path, *, max_output_bytes: int = 20_000) -> SandboxWorkerService:
    return SandboxWorkerService(
        root=tmp_path / "worker",
        engine=DevProcessEngine(max_output_bytes=max_output_bytes),
        max_archive_bytes=500_000,
        max_entries=100,
        max_file_bytes=100_000,
    )


def _headers() -> dict[str, str]:
    return {"X-OpenTulpa-Sandbox-Token": TOKEN}


def test_sandbox_supervisor_worker_environment_excludes_app_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_env = {
        "OPENAI_COMPATIBLE_API_KEY": "model-secret",
        "TELEGRAM_BOT_TOKEN": "telegram-secret",
        "COMPOSIO_API_KEY": "composio-secret",
        "GITHUB_TOKEN": "github-secret",
        "RAILWAY_TOKEN": "railway-secret",
        "DAYTONA_API_KEY": "daytona-secret",
    }
    for name, value in secret_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("OPENTULPA_SANDBOX_RPC_URL", raising=False)
    monkeypatch.delenv("OPENTULPA_SANDBOX_RPC_TOKEN", raising=False)

    supervisor = SandboxWorkerSupervisor(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        settings=Settings(_env_file=None),
    )
    environment = supervisor._worker_environment()  # noqa: SLF001

    assert "OPENTULPA_SANDBOX_RPC_TOKEN" in environment
    assert environment["OPENTULPA_SANDBOX_WORKER_ROOT"] == str(tmp_path / "data" / "sandbox_worker")
    for name, value in secret_env.items():
        assert name not in environment
        assert value not in environment.values()


@pytest.mark.asyncio
async def test_sandbox_worker_api_requires_private_token(tmp_path: Path) -> None:
    app = create_sandbox_worker_app(service=_service(tmp_path), token=TOKEN)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as client:
        denied = await client.get("/internal/v1/sandbox/health")
        accepted = await client.get("/internal/v1/sandbox/health", headers=_headers())

    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["ok"] is True
    assert accepted.json()["checks"]["execute"] is True


@pytest.mark.asyncio
async def test_sandbox_worker_executes_in_workspace_and_preserves_files(tmp_path: Path) -> None:
    app = create_sandbox_worker_app(service=_service(tmp_path), token=TOKEN)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as client:
        created = await client.post(
            "/internal/v1/sandbox/workspaces",
            headers=_headers(),
            json={"kind": "scratch", "tenant_id": "tenant-a"},
        )
        workspace_id = created.json()["workspace_id"]
        first = await client.post(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/execute",
            headers=_headers(),
            json={"command": "printf hello > note.txt && cat note.txt", "timeout": 5},
        )
        second = await client.post(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/execute",
            headers=_headers(),
            json={"command": "cat note.txt", "timeout": 5},
        )

    assert created.status_code == 201
    assert first.json() == {"output": "hello", "exit_code": 0, "truncated": False}
    assert second.json() == {"output": "hello", "exit_code": 0, "truncated": False}


@pytest.mark.asyncio
async def test_sandbox_worker_rejects_unsafe_archive_paths(tmp_path: Path) -> None:
    app = create_sandbox_worker_app(service=_service(tmp_path), token=TOKEN)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as client:
        created = await client.post(
            "/internal/v1/sandbox/workspaces",
            headers=_headers(),
            json={"kind": "scratch", "tenant_id": "tenant-a"},
        )
        workspace_id = created.json()["workspace_id"]
        response = await client.post(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/archive",
            headers=_headers(),
            json={"archive": _tar_archive_with_symlink()},
        )

    assert response.status_code == 422


def test_sandbox_worker_workspace_ids_cannot_escape_root(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(SandboxWorkerError, match="workspace id is invalid"):
        service.delete_workspace("../escape")


@pytest.mark.asyncio
async def test_sandbox_worker_rejects_traversal_archive_paths(tmp_path: Path) -> None:
    app = create_sandbox_worker_app(service=_service(tmp_path), token=TOKEN)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as client:
        created = await client.post(
            "/internal/v1/sandbox/workspaces",
            headers=_headers(),
            json={"kind": "scratch", "tenant_id": "tenant-a"},
        )
        workspace_id = created.json()["workspace_id"]
        response = await client.post(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/archive",
            headers=_headers(),
            json={"archive": _tar_archive_with_file("../escape.txt", b"bad")},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_sandbox_worker_truncates_command_output(tmp_path: Path) -> None:
    app = create_sandbox_worker_app(service=_service(tmp_path, max_output_bytes=32), token=TOKEN)
    transport = httpx.ASGITransport(app=app)
    python = shlex.quote(sys.executable)
    script = shlex.quote("import sys; sys.stdout.write('x' * 200)")

    async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as client:
        created = await client.post(
            "/internal/v1/sandbox/workspaces",
            headers=_headers(),
            json={"kind": "scratch", "tenant_id": "tenant-a"},
        )
        workspace_id = created.json()["workspace_id"]
        response = await client.post(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/execute",
            headers=_headers(),
            json={"command": f"{python} -c {script}", "timeout": 5},
        )

    payload = response.json()
    assert payload["truncated"] is True
    assert payload["output"] == "x" * 32


@pytest.mark.asyncio
async def test_sandbox_worker_timeout_kills_process_group(tmp_path: Path) -> None:
    app = create_sandbox_worker_app(service=_service(tmp_path), token=TOKEN)
    transport = httpx.ASGITransport(app=app)
    python = shlex.quote(sys.executable)
    script = shlex.quote(
        "import subprocess, time; "
        "subprocess.Popen(['sh', '-c', 'sleep 4; touch child-survived']); "
        "time.sleep(30)"
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as client:
        created = await client.post(
            "/internal/v1/sandbox/workspaces",
            headers=_headers(),
            json={"kind": "scratch", "tenant_id": "tenant-a"},
        )
        workspace_id = created.json()["workspace_id"]
        timed_out = await client.post(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/execute",
            headers=_headers(),
            json={"command": f"{python} -c {script}", "timeout": 1},
        )
        probe = await client.post(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/execute",
            headers=_headers(),
            json={"command": "sleep 5; test ! -e child-survived", "timeout": 8},
        )

    assert timed_out.json()["exit_code"] == 124
    assert probe.json()["exit_code"] == 0


@pytest.mark.asyncio
async def test_sandbox_worker_secret_mounts_are_one_shot_and_redacted(
    tmp_path: Path,
) -> None:
    app = create_sandbox_worker_app(service=_service(tmp_path), token=TOKEN)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as client:
        created = await client.post(
            "/internal/v1/sandbox/workspaces",
            headers=_headers(),
            json={"kind": "scratch", "tenant_id": "tenant-a"},
        )
        workspace_id = created.json()["workspace_id"]
        execution = await client.post(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/execute",
            headers=_headers(),
            json={
                "command": "cat \"$SSH_KEY_FILE\"",
                "timeout": 5,
                "secret_files": [
                    {
                        "name": "id_ed25519",
                        "content": "super-secret-private-key",
                        "env": "SSH_KEY_FILE",
                    }
                ],
            },
        )
        archive = await client.get(
            f"/internal/v1/sandbox/workspaces/{workspace_id}/archive",
            headers=_headers(),
        )

    assert execution.status_code == 200
    assert execution.json()["output"] == "[redacted]"
    assert "super-secret-private-key" not in archive.text
    assert ".opentulpa_secret_mounts" not in _tar_member_names(archive.json()["archive"])


def _tar_archive_with_symlink() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _tar_archive_with_file(name: str, content: bytes) -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _tar_member_names(encoded: str) -> list[str]:
    raw = base64.b64decode(encoded)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        return [member.name for member in archive.getmembers()]
