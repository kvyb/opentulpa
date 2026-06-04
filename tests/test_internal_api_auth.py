from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.skills.service import SkillStoreService


def _mk_client(
    tmp_path: Path,
    *,
    client_host: str = "127.0.0.1",
    agent_runtime: Any | None = None,
) -> TestClient:
    store = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(skill_store_service=store, agent_runtime=agent_runtime)
    return TestClient(app, client=(client_host, 50000))


def test_internal_routes_allow_server_local_traffic(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="127.0.0.1") as client:
        no_header = client.post("/internal/skills/list", json={"customer_id": "telegram_1"})
        assert no_header.status_code == 200
    get_settings.cache_clear()


def test_internal_routes_blocked_from_public_clients(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8") as client:
        response = client.post("/internal/skills/list", json={"customer_id": "telegram_1"})
        assert response.status_code == 403

        healthz = client.get("/healthz")
        assert healthz.status_code == 200
        assert "started_at" in healthz.json()

        agent_healthz = client.get("/agent/healthz")
        assert agent_healthz.status_code == 200
        assert agent_healthz.json()["backend"] == "langgraph"
        assert "started_at" in agent_healthz.json()
    get_settings.cache_clear()


def test_healthz_reports_draining_during_shutdown(tmp_path: Path) -> None:
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8") as client:
        healthz = client.get("/healthz")
        assert healthz.status_code == 200

        cast(Any, client.app).state.shutdown_drain.start_draining()
        draining = client.get("/healthz")

    assert draining.status_code == 503
    assert draining.json()["status"] == "draining"
    assert draining.json()["active_turns"] == 0
    assert "started_at" in draining.json()
    get_settings.cache_clear()


def test_webhook_route_public_with_telegram_auth(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8") as client:
        response = client.post(
            "/webhook/telegram",
            json={},
            headers={"x-telegram-bot-api-secret-token": "tg-secret"},
        )
        assert response.status_code == 200
    get_settings.cache_clear()


def test_web_events_route_public_with_bearer_auth(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8") as client:
        rejected = client.get("/web/events")
        accepted = client.get(
            "/web/events?after_id=999999",
            headers={"authorization": "Bearer web-secret"},
        )

    assert rejected.status_code == 403
    assert accepted.status_code == 200
    assert accepted.json() == {"events": [], "next_cursor": 999999}
    get_settings.cache_clear()


def test_generic_web_chat_route_is_public_but_bearer_protected(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8") as client:
        no_header = client.post(
            "/web/chat/turns",
            json={"customer_id": "telegram_1", "thread_id": "dashboard-owner-1", "text": "hi"},
        )
        bad_body = client.post(
            "/web/chat/turns",
            headers={"authorization": "Bearer web-secret"},
            json={"customer_id": "telegram_1", "thread_id": "dashboard-owner-1", "text": ""},
        )

    assert no_header.status_code == 401
    assert bad_body.status_code == 422
    get_settings.cache_clear()


def test_telegram_webhook_status_route_is_public_but_bearer_protected(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def fake_get_webhook_info(self: TelegramClient) -> dict[str, object]:
        _ = self
        return {
            "url": "https://app.example.com/webhook/telegram",
            "pending_update_count": 0,
            "allowed_updates": [
                "message",
                "edited_message",
                "callback_query",
                "my_chat_member",
                "business_connection",
                "business_message",
                "edited_business_message",
                "deleted_business_messages",
            ],
        }

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    monkeypatch.setattr(TelegramClient, "get_webhook_info", fake_get_webhook_info)
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8") as client:
        no_header = client.get("/web/telegram/status")
        accepted = client.get(
            "/web/telegram/status",
            headers={"authorization": "Bearer web-secret"},
        )

    assert no_header.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["ready"] is True
    assert accepted.json()["requires_webhook_reset"] is False
    get_settings.cache_clear()


def test_web_usage_route_public_with_bearer_auth(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    trace_path = tmp_path / "llm_call_traces.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "ts": "2026-06-04T00:00:00+00:00",
                "customer_id": "telegram_1",
                "thread_id": "dashboard-owner-1",
                "trace_id": "trace_usage",
                "native_tokens_prompt": 12,
                "native_cost_usd": 0.004,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = type("Runtime", (), {"_llm_call_trace_path": trace_path})()
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    get_settings.cache_clear()
    with _mk_client(tmp_path, client_host="8.8.8.8", agent_runtime=runtime) as client:
        no_header = client.get("/web/usage")
        missing_customer = client.get(
            "/web/usage",
            headers={"authorization": "Bearer web-secret"},
        )
        bad_window = client.get(
            "/web/usage?customer_id=telegram_1&since=2026-06-05T00:00:00Z&until=2026-06-04T00:00:00Z",
            headers={"authorization": "Bearer web-secret"},
        )
        accepted = client.get(
            "/web/usage?customer_id=telegram_1",
            headers={"authorization": "Bearer web-secret"},
        )

    assert no_header.status_code == 401
    assert missing_customer.status_code == 400
    assert missing_customer.json() == {"detail": "customer_id is required"}
    assert bad_window.status_code == 400
    assert bad_window.json() == {"detail": "since must be before or equal to until"}
    assert accepted.status_code == 200
    body = accepted.json()
    assert body["summary"]["generation_count"] == 1
    assert body["summary"]["input_tokens"] == 12
    assert body["summary"]["cost_usd"] == 0.004
    assert body["generations"][0]["trace_id"] == "trace_usage"
    get_settings.cache_clear()
