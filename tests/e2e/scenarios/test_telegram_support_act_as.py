from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from harness.runner import build_harness, close_harness
from mocks.composio_instagram import FakeComposioInstagramService
from mocks.telegram import FakeTelegramClient

from opentulpa.api import app as app_module
from opentulpa.api.app import create_app
from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.config import get_settings
from opentulpa.interfaces.telegram import attachments as attachments_module
from opentulpa.interfaces.telegram import chat_service as chat_module
from opentulpa.interfaces.telegram import relay as relay_module
from opentulpa.interfaces.telegram.state_store import TelegramStateStore
from opentulpa.scheduler.service import SchedulerService
from opentulpa.tasks import sandbox as sandbox_module
from tests.workbook_fixtures import (
    SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME,
    SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE,
    write_sample_vehicle_services_xlsx,
)

pytestmark = [pytest.mark.e2e, pytest.mark.telegram]


class _SupportRuntime:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []
        self.ainvoke_calls: list[dict[str, Any]] = []

    def healthy(self) -> bool:
        return True

    async def ainvoke_text(self, **kwargs: Any) -> str:
        self.ainvoke_calls.append(kwargs)
        return f"Wake for {kwargs.get('customer_id')}"

    async def classify_wake_event(self, **kwargs: Any) -> dict[str, Any]:
        self.ainvoke_calls.append({"classify_wake_event": kwargs})
        return {"notify_user": True}


def _telegram_message(*, chat_id: int, user_id: int, username: str, text: str) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "message": {
            "message_id": int(time.time() * 1000) % 100000,
            "date": int(time.time()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": username},
            "text": text,
        },
    }


def _telegram_document_message(
    *,
    chat_id: int,
    user_id: int,
    username: str,
    caption: str,
    file_id: str,
    file_name: str,
    mime_type: str,
    file_size: int,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "message": {
            "message_id": int(time.time() * 1000) % 100000,
            "date": int(time.time()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": username},
            "caption": caption,
            "document": {
                "file_id": file_id,
                "file_unique_id": f"unique_{file_id}",
                "file_name": file_name,
                "mime_type": mime_type,
                "file_size": int(file_size),
            },
        },
    }


def _post_telegram(client: TestClient, *, chat_id: int, user_id: int, username: str, text: str) -> int:
    response = client.post(
        "/webhook/telegram",
        headers={"x-telegram-bot-api-secret-token": "test-secret"},
        json=_telegram_message(chat_id=chat_id, user_id=user_id, username=username, text=text),
    )
    return int(response.status_code)


def _wait_until(predicate: Any, timeout_seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.05)
    return bool(predicate())


def _sent_texts_for_chat(telegram_client: FakeTelegramClient, chat_id: int) -> list[str]:
    return [
        str(message.get("text", "") or "")
        for message in telegram_client.sent_messages
        if int(message["chat_id"]) == chat_id
    ]


@pytest.fixture()
def support_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    fake_tg = FakeTelegramClient("fake-token")
    runtime = _SupportRuntime()
    project_root = tmp_path / "project_root"
    project_root.mkdir(parents=True)
    state_store = TelegramStateStore(project_root / ".opentulpa" / "telegram_state.json")
    file_vault = FileVaultService(
        root_dir=project_root / ".opentulpa" / "file_vault",
        db_path=project_root / ".opentulpa" / "file_vault.db",
    )
    composio = FakeComposioInstagramService()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "101,202")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERNAMES", "")
    monkeypatch.setenv("TELEGRAM_SUPPORT_USER_IDS", "900")
    monkeypatch.setenv("TELEGRAM_SUPPORT_USERNAMES", "")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / "links.sqlite"))
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(chat_module, "STATE_STORE", state_store)
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(attachments_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(chat_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda _token: fake_tg)

    async def _fake_stream(**kwargs: Any) -> tuple[str, bool]:
        runtime.stream_calls.append(kwargs)
        customer_id = str(kwargs.get("customer_id", "") or "")
        text = str(kwargs.get("text", "") or "")
        if "create support workflow" in text.lower():
            app = payload["app"]
            app.state.intake_workflows.upsert_workflow(
                customer_id=customer_id,
                name="Support Created Workflow",
                channel="telegram_business_dm",
                provider="telegram_bot_api",
                source_config={"business_connection_id": f"bc_{customer_id}"},
                intent_description="Support-created test workflow",
                required_fields=["name"],
                sink_type="local_csv",
                sink_config={"file_path": f"tmp/{customer_id}.csv"},
                enabled=True,
            )
        await fake_tg.send_message(
            chat_id=kwargs["chat_id"],
            text=f"Handled for {customer_id}",
            parse_mode="HTML",
        )
        return f"Handled for {customer_id}", False

    monkeypatch.setattr(chat_module, "stream_langgraph_reply_to_telegram", _fake_stream)
    get_settings.cache_clear()
    app = create_app(
        agent_runtime=runtime,
        scheduler=SchedulerService(db_path=tmp_path / "scheduler.sqlite"),
        composio_service=composio,
        file_vault_service=file_vault,
    )
    payload: dict[str, Any] = {
        "app": app,
        "client": TestClient(app),
        "fake_tg": fake_tg,
        "runtime": runtime,
        "state_store": state_store,
        "file_vault": file_vault,
        "composio": composio,
    }
    with payload["client"] as client:
        payload["client"] = client
        yield payload
    get_settings.cache_clear()


def _seed_customer_signals(payload: dict[str, Any]) -> None:
    client: TestClient = payload["client"]
    file_vault: FileVaultService = payload["file_vault"]
    telegram_business = payload["app"].state.telegram_business

    assert _post_telegram(client, chat_id=2202, user_id=202, username="owner202", text="/fresh") == 200
    assert _post_telegram(client, chat_id=1101, user_id=101, username="owner101", text="/fresh") == 200
    telegram_business.upsert_connection(
        {
            "id": "bc_telegram_202",
            "user_chat_id": 2202,
            "is_enabled": True,
            "user": {"id": 202, "is_bot": False, "username": "owner202", "first_name": "Owner"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_telegram_101",
            "user_chat_id": 1101,
            "is_enabled": True,
            "user": {"id": 101, "is_bot": False, "username": "owner101", "first_name": "Owner"},
            "rights": {"can_reply": True},
        }
    )
    for customer_id in ("telegram_202", "telegram_101"):
        payload["app"].state.intake_workflows.upsert_workflow(
            customer_id=customer_id,
            name=f"Workflow {customer_id}",
            channel="instagram_dm",
            provider="composio",
            source_config={"connected_account_id": f"ig_{customer_id}", "conversation_id": "conv"},
            intent_description="Seeded workflow",
            required_fields=["name"],
            sink_type="local_csv",
            sink_config={"file_path": f"tmp/{customer_id}.csv"},
            enabled=True,
        )
        file_vault.ingest_file(
            customer_id=customer_id,
            chat_id=1101 if customer_id == "telegram_101" else 2202,
            kind="document",
            telegram_file_id=None,
            original_filename=f"{customer_id}.txt",
            mime_type="text/plain",
            caption="seed",
            raw_bytes=b"seed",
        )


def test_support_binding_preserves_customer_invisibility_and_thread_boundaries(
    support_app: dict[str, Any],
) -> None:
    _seed_customer_signals(support_app)
    client: TestClient = support_app["client"]
    fake_tg: FakeTelegramClient = support_app["fake_tg"]
    runtime: _SupportRuntime = support_app["runtime"]

    owner_message_count = {
        1101: len([m for m in fake_tg.sent_messages if int(m["chat_id"]) == 1101]),
        2202: len([m for m in fake_tg.sent_messages if int(m["chat_id"]) == 2202]),
    }

    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="/support_customers") == 200
    customer_reply = fake_tg.sent_messages[-1]["text"]
    assert "telegram_101" in customer_reply
    assert "owner=@owner101" in customer_reply
    assert "business=connected" in customer_reply
    assert "composio=connected" in customer_reply
    assert "workflows=1" in customer_reply

    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="/support_bind 1") == 200
    assert "Support bound to telegram_101" in fake_tg.sent_messages[-1]["text"]
    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="/support_whoami") == 200
    assert "Bound customer: telegram_101" in fake_tg.sent_messages[-1]["text"]

    support_thread_101 = support_app["state_store"].load()["support_bindings"]["9900"]["thread_id"]
    owner_thread_101 = support_app["state_store"].load()["sessions"]["1101"]["thread_id"]
    assert support_thread_101 != owner_thread_101

    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="List workflows.") == 200
    assert runtime.stream_calls[-1]["customer_id"] == "telegram_101"
    assert runtime.stream_calls[-1]["thread_id"] == support_thread_101
    assert "telegram_900" not in str(support_app["state_store"].load())

    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="/fresh") == 200
    fresh_thread_101 = support_app["state_store"].load()["support_bindings"]["9900"]["thread_id"]
    assert fresh_thread_101 != support_thread_101
    assert support_app["state_store"].load()["sessions"]["1101"]["thread_id"] == owner_thread_101

    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="/support_bind telegram_202") == 200
    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="List workflows again.") == 200
    assert runtime.stream_calls[-1]["customer_id"] == "telegram_202"
    assert runtime.stream_calls[-1]["thread_id"] != fresh_thread_101

    for chat_id, count in owner_message_count.items():
        assert len([m for m in fake_tg.sent_messages if int(m["chat_id"]) == chat_id]) == count


def test_support_workflow_edit_is_invisible_but_proactive_routes_correctly(
    support_app: dict[str, Any],
) -> None:
    _seed_customer_signals(support_app)
    client: TestClient = support_app["client"]
    fake_tg: FakeTelegramClient = support_app["fake_tg"]

    owner_count = len([m for m in fake_tg.sent_messages if int(m["chat_id"]) == 1101])
    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="/support_bind telegram_101") == 200
    assert _post_telegram(client, chat_id=9900, user_id=900, username="support", text="Create support workflow") == 200
    workflows = support_app["app"].state.intake_workflows.list_workflows(
        customer_id="telegram_101",
        include_disabled=True,
    )
    assert any(item["name"] == "Support Created Workflow" for item in workflows)
    assert len([m for m in fake_tg.sent_messages if int(m["chat_id"]) == 1101]) == owner_count

    wake_response = client.post(
        "/internal/wake",
        json={
            "type": "task_event",
            "customer_id": "telegram_101",
            "task_id": "task_support_e2e",
            "event_type": "needs_input",
            "payload": {"message": "owner-visible task"},
        },
    )
    assert wake_response.status_code == 200
    assert _wait_until(
        lambda: len([m for m in fake_tg.sent_messages if int(m["chat_id"]) == 1101]) == owner_count + 1
    )
    owner_messages = [m for m in fake_tg.sent_messages if int(m["chat_id"]) == 1101]
    support_messages = [m for m in fake_tg.sent_messages if int(m["chat_id"]) == 9900]
    assert len(owner_messages) == owner_count + 1
    assert not any(str(m.get("text", "")).startswith("Wake for telegram_101") for m in support_messages)


def test_normal_owners_remain_separate_and_support_commands_are_rejected(
    support_app: dict[str, Any],
) -> None:
    client: TestClient = support_app["client"]
    fake_tg: FakeTelegramClient = support_app["fake_tg"]

    assert _post_telegram(client, chat_id=1101, user_id=101, username="owner101", text="/fresh") == 200
    assert _post_telegram(client, chat_id=2202, user_id=202, username="owner202", text="/fresh") == 200
    state = support_app["state_store"].load()
    assert state["sessions"]["1101"]["customer_id"] == "telegram_101"
    assert state["sessions"]["2202"]["customer_id"] == "telegram_202"
    assert state["sessions"]["1101"]["thread_id"] != state["sessions"]["2202"]["thread_id"]

    assert _post_telegram(client, chat_id=1101, user_id=101, username="owner101", text="/support_customers") == 200
    assert "restricted" in str(fake_tg.sent_messages[-1]["text"]).lower()


@pytest.mark.live_llm
def test_live_llm_support_can_setup_autospa_workflow_for_bound_customer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    composio_instagram_fixture: FakeComposioInstagramService,
) -> None:
    monkeypatch.setenv("TELEGRAM_SUPPORT_USER_IDS", "900")
    harness = build_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario_name="support_autospa_live",
        composio_service=composio_instagram_fixture,
    )
    try:
        owner_chat_id = 1101
        support_chat_id = 9900
        customer_id = "telegram_101"
        business_connection_id = "bc_support_autospa_live"
        assert _post_telegram(
            harness.client,
            chat_id=owner_chat_id,
            user_id=101,
            username="owner101",
            text="/fresh",
        ) == 200
        harness.client.app.state.telegram_business.upsert_connection(
            {
                "id": business_connection_id,
                "user_chat_id": owner_chat_id,
                "is_enabled": True,
                "user": {"id": 101, "is_bot": False, "username": "owner101", "first_name": "Owner"},
                "rights": {"can_reply": True},
            }
        )
        owner_messages_before = len(
            [item for item in harness.telegram_client.sent_messages if int(item["chat_id"]) == owner_chat_id]
        )
        assert _post_telegram(
            harness.client,
            chat_id=support_chat_id,
            user_id=900,
            username="support",
            text="/support_bind 1",
        ) == 200

        asset = write_sample_vehicle_services_xlsx(
            harness.status_report_path.parent / SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME
        )
        registered = harness.telegram_client.register_file(
            file_id="tg_file_support_autospa_price",
            path=asset,
            filename=SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME,
            mime_type=SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE,
        )
        upload_text = (
            "Хочу создать workflow для Telegram Business входящих сообщений для клиента. "
            "Вот прайс AutoSpa. Используй только категории Мойка и Шиномонтаж, "
            "подготовь scoped business knowledge и запиши бронирования в Google Sheets через fake Composio. "
            "Сначала предложи workflow и жди подтверждения."
        )
        response = harness.client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": "test-secret"},
            json=_telegram_document_message(
                chat_id=support_chat_id,
                user_id=900,
                username="support",
                caption=upload_text,
                file_id=str(registered["file_id"]),
                file_name=str(registered["filename"]),
                mime_type=str(registered["mime_type"]),
                file_size=int(registered["file_size"]),
            ),
        )
        assert response.status_code == 200
        assert _wait_until(
            lambda: len([m for m in harness.telegram_client.sent_messages if int(m["chat_id"]) == support_chat_id])
            >= 2,
            timeout_seconds=90.0,
        )
        details = (
            "Workflow назови «AutoSpa Мойка и Шиномонтаж». Канал Telegram Business DM, "
            f"business_connection_id={business_connection_id}. Интент: вопросы о цене и запись на мойку "
            "или шиномонтаж. Собери service_category, service_name, car_class, desired_date, "
            "desired_time, client_name, phone, quoted_price. Sink: google_sheets_composio, "
            "spreadsheetId=sheet_support_autospa, sheetName=Bookings. Подготовь предложение."
        )
        assert _post_telegram(
            harness.client,
            chat_id=support_chat_id,
            user_id=900,
            username="support",
            text=details,
        ) == 200
        assert _wait_until(
            lambda: any(
                "AutoSpa Мойка и Шиномонтаж" in text
                and (
                    "Предложение" in text
                    or "готово к активации" in text
                    or "подтвержд" in text.lower()
                )
                for text in _sent_texts_for_chat(harness.telegram_client, support_chat_id)
            ),
            timeout_seconds=300.0,
        )
        assert _post_telegram(
            harness.client,
            chat_id=support_chat_id,
            user_id=900,
            username="support",
            text="Подтверждаю. Сохрани и активируй workflow сейчас.",
        ) == 200
        assert _wait_until(
            lambda: bool(
                harness.client.app.state.intake_workflows.list_workflows(
                    customer_id=customer_id,
                    include_disabled=True,
                )
            ),
            timeout_seconds=120.0,
        )
        workflows = harness.client.app.state.intake_workflows.list_workflows(
            customer_id=customer_id,
            include_disabled=True,
        )
        assert workflows
        assert all(str(item.get("customer_id")) == customer_id for item in workflows)
        assert len(
            [item for item in harness.telegram_client.sent_messages if int(item["chat_id"]) == owner_chat_id]
        ) == owner_messages_before
    finally:
        close_harness(harness)
