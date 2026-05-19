from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes import files as file_routes
from opentulpa.api.routes.files import register_file_routes
from opentulpa.context.file_vault import FileVaultService


class _UnusedVault:
    pass


class _FakeTelegramChat:
    def find_session_slots(self, customer_id: str) -> list[dict[str, Any]]:
        assert customer_id == "telegram_123"
        return [{"chat_id": 12345}]


class _RecordingTelegramClient:
    def __init__(self) -> None:
        self.sent_files: list[dict[str, Any]] = []

    async def send_file(self, **kwargs: Any) -> bool:
        self.sent_files.append(kwargs)
        return True


class _RecordingRuntime:
    def __init__(self) -> None:
        self.files: list[dict[str, Any]] = []

    async def emit_interactive_file(self, *, file: dict[str, Any]) -> dict[str, Any]:
        self.files.append(file)
        return {"ok": True, "sent": True}


def test_send_local_returns_delivery_marker_after_telegram_accepts_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tulpa_stuff = tmp_path / "tulpa_stuff"
    tulpa_stuff.mkdir()
    local_file = tulpa_stuff / "sample_delivery_report.txt"
    local_file.write_text("done", encoding="utf-8")

    monkeypatch.setattr(file_routes, "TULPA_STUFF_DIR", tulpa_stuff.resolve())

    app = FastAPI()
    telegram_client = _RecordingTelegramClient()
    register_file_routes(
        app,
        get_file_vault=lambda: _UnusedVault(),
        get_telegram_chat=lambda: _FakeTelegramChat(),
        get_telegram_client=lambda: telegram_client,
        get_agent_runtime=lambda: object(),
        telegram_enabled=True,
    )

    with TestClient(app) as client:
        response = client.post(
            "/internal/files/send_local",
            json={
                "customer_id": "telegram_123",
                "path": "tulpa_stuff/sample_delivery_report.txt",
                "caption": "Sample delivery report",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["delivered_to_chat"] is True
    assert payload["path"] == "tulpa_stuff/sample_delivery_report.txt"
    assert payload["chat_id"] == 12345
    assert "DELIVERED_TO_CHAT" in payload["model_instruction"]
    assert "Do not call the file-send tool again" in payload["model_instruction"]
    assert telegram_client.sent_files == [
        {
            "chat_id": 12345,
            "filename": "sample_delivery_report.txt",
            "raw_bytes": b"done",
            "kind": "document",
            "mime_type": "text/plain",
            "caption": "Sample delivery report",
            "parse_mode": "HTML",
        }
    ]


def test_send_web_image_delivers_to_registered_web_chat_when_telegram_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def _download_image_from_web_url(url: str, *, max_bytes: int) -> dict[str, Any]:
        assert url == "https://example.com/chipmunk.png"
        assert max_bytes == 10_000_000
        return {
            "filename": "chipmunk.png",
            "content_type": "image/png",
            "raw_bytes": b"png-bytes",
        }

    monkeypatch.setattr(file_routes, "download_image_from_web_url", _download_image_from_web_url)

    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.db")
    runtime = _RecordingRuntime()
    app = FastAPI()
    register_file_routes(
        app,
        get_file_vault=lambda: vault,
        get_telegram_chat=lambda: object(),
        get_telegram_client=lambda: object(),
        get_agent_runtime=lambda: runtime,
        telegram_enabled=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/internal/files/send_web_image",
            json={
                "customer_id": "telegram_123",
                "url": "https://example.com/chipmunk.png",
                "caption": "A chipmunk",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["delivered_to_chat"] is True
    assert payload["file_id"]
    assert "current chat" in payload["model_instruction"]
    assert runtime.files[0]["id"] == payload["file_id"]
    assert runtime.files[0]["kind"] == "photo"
    assert runtime.files[0]["mime_type"] == "image/png"
    assert runtime.files[0]["caption"] == "A chipmunk"
