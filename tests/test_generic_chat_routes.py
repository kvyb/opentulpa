from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.config import get_settings


class _StreamingRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.update_sender: Any | None = None
        self.file_sender: Any | None = None

    async def register_interactive_update_sender(self, *, thread_id: str, sender: Any) -> None:
        assert thread_id
        assert sender is not None
        self.update_sender = sender

    async def clear_interactive_update_sender(self, *, thread_id: str, sender: Any | None = None) -> None:
        assert thread_id
        if sender is None or sender is self.update_sender:
            self.update_sender = None

    async def register_interactive_file_sender(self, *, thread_id: str, sender: Any) -> None:
        assert thread_id
        assert sender is not None
        self.file_sender = sender

    async def clear_interactive_file_sender(self, *, thread_id: str, sender: Any | None = None) -> None:
        assert thread_id
        if sender is None or sender is self.file_sender:
            self.file_sender = None

    async def astream_text(self, **kwargs: Any):
        self.calls.append(kwargs)
        assert self.update_sender is not None
        result = self.update_sender("Checking context.")
        if hasattr(result, "__await__"):
            await result
        assert self.file_sender is not None
        file_result = self.file_sender({"id": "file_123", "original_filename": "demo.pdf"})
        if hasattr(file_result, "__await__"):
            await file_result
        yield "Hello"
        yield "Hello from web."


def _client(monkeypatch: Any, tmp_path: Any) -> tuple[TestClient, _StreamingRuntime]:
    monkeypatch.setenv("OPENTULPA_GENERIC_API_SECRET", "generic-secret")
    get_settings.cache_clear()
    runtime = _StreamingRuntime()
    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.db")
    app = create_app(agent_runtime=runtime, file_vault_service=vault)
    return TestClient(app), runtime


def test_web_chat_rejects_missing_bearer(monkeypatch: Any, tmp_path: Any) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    response = client.post(
        "/web/chat/turns",
        json={"customer_id": "telegram_1", "thread_id": "dashboard-owner-1", "text": "hi"},
    )
    assert response.status_code == 401


def test_web_chat_streams_owner_updates_files_and_final(monkeypatch: Any, tmp_path: Any) -> None:
    client, runtime = _client(monkeypatch, tmp_path)
    with client.stream(
        "POST",
        "/web/chat/turns",
        headers={"authorization": "Bearer generic-secret"},
        json={
            "customer_id": "telegram_1",
            "thread_id": "dashboard-owner-1",
            "text": "hi",
        },
    ) as response:
        text = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "event: status" in text
    assert "event: owner_update" in text
    assert "Checking context." in text
    assert "event: file" in text
    assert "/web/files/file_123/content" in text
    assert "event: delta" in text
    assert "Hello from web." in text
    assert "event: final" in text
    assert runtime.calls[0]["customer_id"] == "telegram_1"
    assert runtime.calls[0]["thread_id"] == "dashboard-owner-1"


def test_web_file_upload_and_content_are_bearer_protected(monkeypatch: Any, tmp_path: Any) -> None:
    client, _ = _client(monkeypatch, tmp_path)

    upload = client.post(
        "/web/files/upload",
        headers={"authorization": "Bearer generic-secret"},
        data={"customer_id": "telegram_1", "thread_id": "dashboard-owner-1", "kind": "document"},
        files={"file": ("hello.txt", b"hello world", "text/plain")},
    )
    assert upload.status_code == 200
    file_id = upload.json()["file"]["id"]

    unauth = client.get(
        f"/web/files/{file_id}/content",
        params={"customer_id": "telegram_1"},
    )
    assert unauth.status_code == 401

    content = client.get(
        f"/web/files/{file_id}/content",
        headers={"authorization": "Bearer generic-secret"},
        params={"customer_id": "telegram_1"},
    )
    assert content.status_code == 200
    assert content.content == b"hello world"
    assert content.headers["content-type"].startswith("text/plain")
