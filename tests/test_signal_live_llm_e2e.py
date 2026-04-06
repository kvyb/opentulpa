from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api import app as app_module
from opentulpa.api.app import create_app
from opentulpa.context.signals import SignalInboxService
from opentulpa.core.config import get_settings
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService, build_skill_markdown
from opentulpa.tasks import sandbox as sandbox_module


LIVE_FLAG = "OPENTULPA_ENABLE_LIVE_SIGNAL_E2E"
OWNER_CUSTOMER_ID = "telegram_owner_live_signal"
OWNER_THREAD_ID = "inbox_manychat_owner_live_001"
EXTERNAL_SUBJECT_ID = "mc_contact_live_001"
EXTERNAL_CONVERSATION_ID = "conv_live_001"

if str(os.getenv(LIVE_FLAG, "")).strip().lower() not in {"1", "true", "yes"}:
    pytest.skip(
        f"set {LIVE_FLAG}=1 to run live signal e2e test",
        allow_module_level=True,
    )

_settings = get_settings()
if not str(_settings.openrouter_api_key or "").strip():
    pytest.skip("OPENROUTER_API_KEY is required for live signal e2e", allow_module_level=True)


def _wait_for(predicate: object, timeout_seconds: float = 45.0) -> bool:
    deadline = time.time() + max(1.0, float(timeout_seconds))
    while time.time() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.2)
    return bool(predicate())


def _patch_project_root(monkeypatch: pytest.MonkeyPatch, project_root: Path) -> None:
    tulpa_dir = (project_root / "tulpa_stuff").resolve()
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sandbox_module, "TULPA_STUFF_DIR", tulpa_dir)
    monkeypatch.setattr(sandbox_module, "CATALOG_PATH", (tulpa_dir / ".tulpa_catalog.json").resolve())
    monkeypatch.setattr(sandbox_module, "CATALOG_README_PATH", (tulpa_dir / "README.md").resolve())
    monkeypatch.setitem(sandbox_module.ALLOWED_TERMINAL_DIRS, "tulpa_stuff", tulpa_dir)
    monkeypatch.setitem(sandbox_module.ALLOWED_READ_DIRS, "tulpa_stuff", tulpa_dir)


def _patch_runtime_internal_api(
    *,
    runtime: OpenTulpaLangGraphRuntime,
    app: Any,
    calls: list[dict[str, Any]],
) -> None:
    async def _request_with_backoff(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int = 2,
    ) -> httpx.Response:
        _ = (timeout, retries)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.request(
                method=method,
                url=path,
                params=params,
                json=json_body,
            )
        calls.append(
            {
                "method": str(method).upper(),
                "path": path,
                "params": params or {},
                "json_body": json_body or {},
                "status_code": int(response.status_code),
            }
        )
        return response

    runtime._request_with_backoff = _request_with_backoff  # type: ignore[method-assign]


def test_live_signal_webhook_to_outbox_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project"
    tulpa_dir = project_root / "tulpa_stuff"
    tulpa_dir.mkdir(parents=True)
    (tulpa_dir / "__init__.py").write_text('"""Agent-created integrations and skills."""\n', encoding="utf-8")
    (tulpa_dir / "business_info.md").write_text(
        "# Business Info\n\nWe are open Monday through Friday, 9 AM to 5 PM Pacific.\n",
        encoding="utf-8",
    )
    (tulpa_dir / "manychat_test.py").write_text(
        "from fastapi import APIRouter, Request\n"
        "\n"
        "public_router = APIRouter()\n"
        "\n"
        "@public_router.post('/incoming')\n"
        "async def incoming(request: Request):\n"
        "    body = await request.json()\n"
        "    customer_id = str(body.get('owner_customer_id') or '').strip()\n"
        "    thread_id = str(body.get('owner_thread_id') or '').strip() or f'inbox-{customer_id}'\n"
        "    text = str(body.get('text') or '').strip()\n"
        "    external_subject_id = str(body.get('external_subject_id') or '').strip()\n"
        "    external_conversation_id = str(body.get('external_conversation_id') or '').strip()\n"
        "    dispatch = {\n"
        "        'external_subject_id': external_subject_id,\n"
        "        'external_conversation_id': external_conversation_id,\n"
        "    }\n"
        "    signal = request.app.state.signal_inbox.ingest_signal(\n"
        "        source='manychat',\n"
        "        customer_id=customer_id,\n"
        "        thread_id=thread_id,\n"
        "        event_type='message',\n"
        "        text=text,\n"
        "        payload={\n"
        "            'external_subject_id': external_subject_id,\n"
        "            'external_conversation_id': external_conversation_id,\n"
        "            'raw': body,\n"
        "        },\n"
        "        dispatch=dispatch,\n"
        "    )\n"
        "    queue_id = await request.app.state.wake_queue.enqueue({\n"
        "        'type': 'signal_event',\n"
        "        'source': 'manychat',\n"
        "        'customer_id': customer_id,\n"
        "        'thread_id': thread_id,\n"
        "        'signal_id': signal['id'],\n"
        "    })\n"
        "    return {'ok': True, 'signal_id': signal['id'], 'queue_id': queue_id}\n",
        encoding="utf-8",
    )

    _patch_project_root(monkeypatch, project_root)

    signals = SignalInboxService(db_path=tmp_path / "signals.db")
    signals.upsert_rule(
        source="manychat",
        customer_id=OWNER_CUSTOMER_ID,
        thread_id=OWNER_THREAD_ID,
        wake_mode="always",
        batch_window_seconds=0,
        auto_reply=True,
        handler_skill_name="manychat-incoming-handler",
        guidance_text=(
            "Use the business info for this reply. "
            "Reply in one sentence. "
            "State that the business is open Monday through Friday, 9 AM to 5 PM Pacific."
        ),
    )
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    skills.upsert_skill(
        scope="user",
        customer_id=OWNER_CUSTOMER_ID,
        name="manychat-incoming-handler",
        skill_markdown=build_skill_markdown(
            name="manychat-incoming-handler",
            description="Handle manychat business-hours questions.",
            instructions=(
                "Use the business info for this reply. "
                "Reply in one sentence. "
                "State that the business is open Monday through Friday, 9 AM to 5 PM Pacific."
            ),
        ),
        source="test",
        enabled=True,
    )
    behavior_log = tmp_path / "signal_behavior.jsonl"
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key=str(_settings.openrouter_api_key or "").strip(),
        openrouter_base_url=str(_settings.openrouter_base_url or "").strip(),
        model_name=str(_settings.llm_model or "").strip(),
        wake_classifier_model_name=str(_settings.llm_model or "").strip(),
        guardrail_classifier_model_name=str(_settings.llm_model or "").strip(),
        checkpoint_db_path=str(tmp_path / "signal_checkpoints.sqlite"),
        behavior_log_enabled=True,
        behavior_log_path=str(behavior_log),
    )
    app = create_app(
        scheduler=SchedulerService(),
        agent_runtime=runtime,
        signal_inbox_service=signals,
        skill_store_service=skills,
    )

    with TestClient(app) as client:
        response = client.post(
            "/webhook/tulpa/manychat_test/incoming",
            json={
                "owner_customer_id": OWNER_CUSTOMER_ID,
                "owner_thread_id": OWNER_THREAD_ID,
                "external_subject_id": EXTERNAL_SUBJECT_ID,
                "external_conversation_id": EXTERNAL_CONVERSATION_ID,
                "text": "Hi, what are your business hours?",
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

        assert _wait_for(lambda: bool(signals.list_outbox(source="manychat")), timeout_seconds=60.0)

    outbox = signals.list_outbox(source="manychat")
    assert len(outbox) == 1
    reply = str(outbox[0]["text"] or "").strip()
    assert reply
    assert "monday" in reply.lower()
    assert "friday" in reply.lower()
    assert "9" in reply
    assert "5" in reply
    assert outbox[0]["dispatch"]["external_subject_id"] == EXTERNAL_SUBJECT_ID
    assert outbox[0]["dispatch"]["external_conversation_id"] == EXTERNAL_CONVERSATION_ID


def test_live_agent_can_create_connector_then_process_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    tulpa_dir = project_root / "tulpa_stuff"
    tulpa_dir.mkdir(parents=True)
    (tulpa_dir / "__init__.py").write_text('"""Agent-created integrations and skills."""\n', encoding="utf-8")
    (tulpa_dir / "business_info.md").write_text(
        "# Business Info\n\nWe are open Monday through Friday, 9 AM to 5 PM Pacific.\n",
        encoding="utf-8",
    )
    _patch_project_root(monkeypatch, project_root)

    signals = SignalInboxService(db_path=tmp_path / "signals.db")
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    behavior_log = tmp_path / "signal_setup_behavior.jsonl"
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://testserver",
        openrouter_api_key=str(_settings.openrouter_api_key or "").strip(),
        openrouter_base_url=str(_settings.openrouter_base_url or "").strip(),
        model_name=str(_settings.llm_model or "").strip(),
        wake_classifier_model_name=str(_settings.llm_model or "").strip(),
        guardrail_classifier_model_name=str(_settings.llm_model or "").strip(),
        checkpoint_db_path=str(tmp_path / "signal_setup_checkpoints.sqlite"),
        behavior_log_enabled=True,
        behavior_log_path=str(behavior_log),
    )
    app = create_app(
        scheduler=SchedulerService(),
        agent_runtime=runtime,
        signal_inbox_service=signals,
        skill_store_service=skills,
    )
    internal_calls: list[dict[str, Any]] = []
    _patch_runtime_internal_api(runtime=runtime, app=app, calls=internal_calls)

    setup_prompt = (
        "Set up a thin webhook inbox connector exactly once.\n"
        "Requirements:\n"
        "1. First call skill_get(name='signal-integration-operator').\n"
        "2. Create tulpa_stuff/live_signal_connector.py.\n"
        "3. Export public_router only.\n"
        "4. POST /incoming must parse JSON and normalize source='manychat_live'.\n"
        "5. It must read owner_customer_id, owner_thread_id, external_subject_id, external_conversation_id, and text from the body.\n"
        "6. It must call await request.app.state.signal_ingest({...}) instead of mapping customer identity itself.\n"
        "7. Pass source, owner_customer_id, owner_thread_id, external_subject_id, external_conversation_id, and text to that helper.\n"
        "8. The connector must not derive customer_id from the external sender.\n"
        "9. It must return JSON with ok, signal_id, and queue_id.\n"
        "10. After writing, call tulpa_validate_file and tulpa_reload.\n"
        "11. Create or update a user skill named 'manychat-live-incoming-handler' that says the business is open Monday through Friday, 9 AM to 5 PM Pacific and replies in one sentence.\n"
        "12. Then call signal_rule_upsert with source='manychat_live', "
        f"customer_id='{OWNER_CUSTOMER_ID}', thread_id='manychat_conv_live_002', "
        "wake_mode='always', batch_window_seconds=0, auto_reply=true, handler_skill_name='manychat-live-incoming-handler', "
        "and optional concise guidance_text.\n"
        "13. Then call signal_rule_list and confirm the matching rule exists before you finish.\n"
        "14. Do not use tulpa_run_terminal.\n"
        "15. Keep the connector thin; do not reimplement batching or orchestration.\n"
        "16. Reply with one short success sentence only after the connector is written, validated, reloaded, the handler skill is saved, and the rule is saved and verified."
    )

    with TestClient(app) as client:
        response = client.post(
            "/internal/chat",
            json={
                "customer_id": OWNER_CUSTOMER_ID,
                "thread_id": "chat-live-signal-setup",
                "text": setup_prompt,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True

        connector_path = tulpa_dir / "live_signal_connector.py"
        assert connector_path.exists()
        connector_text = connector_path.read_text(encoding="utf-8")
        assert "public_router" in connector_text
        assert "public_router.post" in connector_text
        assert "/incoming" in connector_text
        assert "owner_customer_id" in connector_text
        assert "external_subject_id" in connector_text
        assert "external_conversation_id" in connector_text
        assert "request.app.state.signal_ingest" in connector_text

        webhook = client.post(
            "/webhook/tulpa/live_signal_connector/incoming",
            json={
                "owner_customer_id": OWNER_CUSTOMER_ID,
                "owner_thread_id": "manychat_conv_live_002",
                "external_subject_id": "mc_contact_live_002",
                "external_conversation_id": "conv_live_002",
                "text": "Hello, what are your business hours?",
            },
        )
        assert webhook.status_code == 200
        assert webhook.json()["ok"] is True
        assert _wait_for(lambda: bool(signals.list_outbox(source="manychat_live")), timeout_seconds=60.0)

    called_paths = [str(item.get("path", "")) for item in internal_calls]
    assert "/internal/skills/get" in called_paths
    assert "/internal/tulpa/write_file" in called_paths
    assert "/internal/tulpa/validate_file" in called_paths
    assert "/internal/tulpa/reload" in called_paths
    assert "/internal/signals/rules/upsert" in called_paths
    assert "/internal/signals/rules" in called_paths
    assert "/internal/tulpa/run_terminal" not in called_paths

    rules = signals.list_rules(source="manychat_live", customer_id=OWNER_CUSTOMER_ID, limit=5)
    assert len(rules) == 1
    assert rules[0]["thread_id"] == "manychat_conv_live_002"
    assert rules[0]["wake_mode"] == "always"

    outbox = signals.list_outbox(source="manychat_live")
    assert len(outbox) == 1
    reply = str(outbox[0]["text"] or "").strip()
    assert reply
    assert "monday" in reply.lower()
    assert "friday" in reply.lower()
    assert "9" in reply
    assert "5" in reply
    assert outbox[0]["dispatch"]["external_subject_id"] == "mc_contact_live_002"
    assert outbox[0]["dispatch"]["external_conversation_id"] == "conv_live_002"
