from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from opentulpa.agent.runtime import AgentStreamEvent
from opentulpa.api.app import create_app
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.config import get_settings


class _StreamingRuntime:
    def __init__(self, *, incremental_chunks: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.update_sender: Any | None = None
        self.file_sender: Any | None = None
        self.incremental_chunks = incremental_chunks or ["Hello", " from web."]

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
        if kwargs.get("stream_incremental_deltas"):
            if kwargs.get("stream_status_events"):
                yield AgentStreamEvent(
                    event="status",
                    payload={"status": "active", "message": "Compacting chat history..."},
                )
                yield AgentStreamEvent(
                    event="reasoning",
                    payload={"status": "active", "message": "Reasoning..."},
                )
                yield AgentStreamEvent(
                    event="tool_call",
                    payload={
                        "status": "started",
                        "message": "Searching the web...",
                        "tool_names": ["web_search"],
                        "tool_call_count": 1,
                    },
                )
            for chunk in self.incremental_chunks:
                yield chunk
            return
        yield "Hello"
        yield "Hello from web."


def _sse_payloads(text: str, event_name: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        if f"event: {event_name}" not in block:
            continue
        data_lines = [
            line.removeprefix("data:").strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        assert data_lines
        payloads.append(json.loads("\n".join(data_lines)))
    return payloads


def _client(monkeypatch: Any, tmp_path: Any) -> tuple[TestClient, _StreamingRuntime]:
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    get_settings.cache_clear()
    runtime = _StreamingRuntime()
    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.db")
    app = create_app(agent_runtime=runtime, file_vault_service=vault)
    return TestClient(app), runtime


def _client_with_runtime(
    monkeypatch: Any,
    tmp_path: Any,
    runtime: _StreamingRuntime,
) -> TestClient:
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    get_settings.cache_clear()
    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.db")
    return TestClient(create_app(agent_runtime=runtime, file_vault_service=vault))


def _client_with_runtime_and_app(
    monkeypatch: Any,
    tmp_path: Any,
    runtime: _StreamingRuntime,
) -> tuple[TestClient, Any]:
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    get_settings.cache_clear()
    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.db")
    app = create_app(agent_runtime=runtime, file_vault_service=vault)
    return TestClient(app), app


def test_web_chat_rejects_missing_bearer(monkeypatch: Any, tmp_path: Any) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    response = client.post(
        "/web/chat/turns",
        json={"customer_id": "telegram_1", "thread_id": "dashboard-owner-1", "text": "hi"},
    )
    assert response.status_code == 401


def test_web_chat_has_typed_request_validation(monkeypatch: Any, tmp_path: Any) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    response = client.post(
        "/web/chat/turns",
        headers={"authorization": "Bearer web-secret"},
        json={"customer_id": "telegram_1", "thread_id": "dashboard-owner-1", "text": ""},
    )
    assert response.status_code == 422

    openapi = client.get("/openapi.json").json()
    operation = openapi["paths"]["/web/chat/turns"]["post"]
    schema_ref = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert schema_ref.endswith("/WebChatTurnRequest")


def test_web_chat_streams_owner_updates_files_and_final(monkeypatch: Any, tmp_path: Any) -> None:
    client, runtime = _client(monkeypatch, tmp_path)
    with client.stream(
        "POST",
        "/web/chat/turns",
        headers={"authorization": "Bearer web-secret"},
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
    assert {"status": "active", "message": "Compacting chat history..."} in _sse_payloads(
        text, "status"
    )
    assert "event: reasoning" in text
    assert "private reasoning" not in text
    assert _sse_payloads(text, "reasoning") == [{"status": "active", "message": "Reasoning..."}]
    assert _sse_payloads(text, "tool_call") == [
        {
            "status": "started",
            "message": "Searching the web...",
            "tool_names": ["web_search"],
            "tool_call_count": 1,
        }
    ]
    assert "event: delta" in text
    deltas = _sse_payloads(text, "delta")
    assert [delta["text"] for delta in deltas] == ["Hello", " from web."]
    assert [delta["append"] for delta in deltas] == [True, True]
    assert [delta["seq"] for delta in deltas] == [1, 2]
    assert all(isinstance(delta["server_received_at_ms"], int) for delta in deltas)
    assert "Hello from web." in text
    assert "event: final" in text
    assert runtime.calls[0]["customer_id"] == "telegram_1"
    assert runtime.calls[0]["thread_id"] == "dashboard-owner-1"
    assert runtime.calls[0]["stream_precommit_seconds"] == 0.0
    assert runtime.calls[0]["stream_incremental_deltas"] is True
    assert runtime.calls[0]["stream_status_events"] is True


def test_web_chat_preserves_whitespace_only_incremental_deltas(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    runtime = _StreamingRuntime(incremental_chunks=["Hello", " ", "world", "\n\n", "Again"])
    client = _client_with_runtime(monkeypatch, tmp_path, runtime)

    with client.stream(
        "POST",
        "/web/chat/turns",
        headers={"authorization": "Bearer web-secret"},
        json={
            "customer_id": "telegram_1",
            "thread_id": "dashboard-owner-1",
            "text": "hi",
        },
    ) as response:
        text = response.read().decode("utf-8")

    assert response.status_code == 200
    deltas = _sse_payloads(text, "delta")
    assert [delta["text"] for delta in deltas] == ["Hello", " ", "world", "\n\n", "Again"]
    assert [delta["append"] for delta in deltas] == [True, True, True, True, True]
    assert [delta["seq"] for delta in deltas] == [1, 2, 3, 4, 5]
    assert all(isinstance(delta["server_received_at_ms"], int) for delta in deltas)
    assert _sse_payloads(text, "final") == [{"text": "Hello world\n\nAgain"}]


def test_web_chat_streams_workflow_setup_status_events(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    runtime = _StreamingRuntime(incremental_chunks=["Workflow", " ready."])
    client, app = _client_with_runtime_and_app(monkeypatch, tmp_path, runtime)
    app.state.intake_workflow_setup.begin_session(
        customer_id="telegram_1",
        thread_id="dashboard-owner-1",
        mode="create",
    )

    with client.stream(
        "POST",
        "/web/chat/turns",
        headers={"authorization": "Bearer web-secret"},
        json={
            "customer_id": "telegram_1",
            "thread_id": "dashboard-owner-1",
            "text": "continue setup",
        },
    ) as response:
        text = response.read().decode("utf-8")

    assert response.status_code == 200
    assert _sse_payloads(text, "reasoning") == [{"status": "active", "message": "Reasoning..."}]
    assert _sse_payloads(text, "tool_call")
    assert _sse_payloads(text, "final") == [{"text": "Workflow ready."}]
    assert runtime.calls[0]["turn_mode"] == "workflow_setup"
    assert runtime.calls[0]["stream_status_events"] is True


def test_web_chat_resolves_bound_telegram_alias(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    get_settings.cache_clear()
    runtime = _StreamingRuntime()
    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.db")
    profiles = CustomerProfileService(tmp_path / "profiles.db")
    profiles.bind_telegram_user_id(user_id="usr_default", telegram_user_id="1")
    app = create_app(
        agent_runtime=runtime,
        file_vault_service=vault,
        customer_profile_service=profiles,
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/web/chat/turns",
        headers={"authorization": "Bearer web-secret"},
        json={
            "customer_id": "telegram_1",
            "thread_id": "dashboard-owner-1",
            "text": "hi",
        },
    ) as response:
        _ = response.read()

    assert response.status_code == 200
    assert runtime.calls[0]["customer_id"] == "usr_default"


def test_web_file_upload_and_content_are_bearer_protected(monkeypatch: Any, tmp_path: Any) -> None:
    client, _ = _client(monkeypatch, tmp_path)

    upload = client.post(
        "/web/files/upload",
        headers={"authorization": "Bearer web-secret"},
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
        headers={"authorization": "Bearer web-secret"},
        params={"customer_id": "telegram_1"},
    )
    assert content.status_code == 200
    assert content.content == b"hello world"
    assert content.headers["content-type"].startswith("text/plain")
