from __future__ import annotations

from pathlib import Path
import sys
import types

from fastapi.testclient import TestClient

apscheduler_module = types.ModuleType("apscheduler")
schedulers_module = types.ModuleType("apscheduler.schedulers")
asyncio_module = types.ModuleType("apscheduler.schedulers.asyncio")
triggers_module = types.ModuleType("apscheduler.triggers")
cron_module = types.ModuleType("apscheduler.triggers.cron")
date_module = types.ModuleType("apscheduler.triggers.date")
mem0_module = types.ModuleType("mem0")


class _DummyAsyncIOScheduler:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


class _DummyCronTrigger:
    @classmethod
    def from_crontab(cls, value: str) -> "_DummyCronTrigger":
        _ = value
        return cls()


class _DummyDateTrigger:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


class _DummyMemory:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


asyncio_module.AsyncIOScheduler = _DummyAsyncIOScheduler
cron_module.CronTrigger = _DummyCronTrigger
date_module.DateTrigger = _DummyDateTrigger
apscheduler_module.schedulers = schedulers_module
apscheduler_module.triggers = triggers_module
schedulers_module.asyncio = asyncio_module
triggers_module.cron = cron_module
triggers_module.date = date_module
mem0_module.Memory = _DummyMemory
sys.modules.setdefault("apscheduler", apscheduler_module)
sys.modules.setdefault("apscheduler.schedulers", schedulers_module)
sys.modules.setdefault("apscheduler.schedulers.asyncio", asyncio_module)
sys.modules.setdefault("apscheduler.triggers", triggers_module)
sys.modules.setdefault("apscheduler.triggers.cron", cron_module)
sys.modules.setdefault("apscheduler.triggers.date", date_module)
sys.modules.setdefault("mem0", mem0_module)

from opentulpa.api.app import create_app
from opentulpa.context.signals import SignalInboxService
from opentulpa.skills.service import SkillStoreService, build_skill_markdown


def _mk_client(tmp_path: Path) -> tuple[TestClient, SignalInboxService]:
    signals = SignalInboxService(db_path=tmp_path / "signals.db")
    store = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(skill_store_service=store, signal_inbox_service=signals)
    return TestClient(app), signals


def _upsert_handler_skill(tmp_path: Path, *, customer_id: str, name: str) -> None:
    store = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    store.upsert_skill(
        scope="user",
        customer_id=customer_id,
        name=name,
        skill_markdown=build_skill_markdown(
            name=name,
            description="Handle incoming contact messages for this source.",
            instructions=(
                "Ask focused follow-up questions, capture required details, and reply briefly."
            ),
        ),
        source="test",
        enabled=True,
    )


def test_signal_ingest_uses_resolved_rule_defaults(tmp_path: Path) -> None:
    client, signals = _mk_client(tmp_path)
    _upsert_handler_skill(tmp_path, customer_id="mc_123", name="manychat-incoming-handler")
    signals.upsert_rule(
        source="manychat",
        customer_id="mc_123",
        thread_id="chat-mc_123",
        wake_mode="always",
        batch_window_seconds=45,
        auto_reply=True,
        handler_skill_name="manychat-incoming-handler",
        guidance_text="Use business_info.md for answers.",
    )

    with client:
        response = client.post(
            "/internal/signals/ingest",
            json={
                "source": "manychat",
                "customer_id": "mc_123",
                "text": "Hi, what are your business hours?",
                "dispatch": {"conversation_id": "conv_1"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["signal"]["thread_id"] == "chat-mc_123"
    assert payload["signal"]["batch_window_seconds"] == 45
    assert payload["queue_id"]
    assert payload["rule"]["wake_mode"] == "always"


def test_signal_ingest_autowires_owner_identity_and_generic_external_ids(tmp_path: Path) -> None:
    client, signals = _mk_client(tmp_path)

    with client:
        response = client.post(
            "/internal/signals/ingest",
            json={
                "source": "manychat",
                "owner_customer_id": "telegram_owner_1",
                "owner_thread_id": "inbox_manychat_owner_1",
                "external_subject_id": "contact_42",
                "external_conversation_id": "conv_42",
                "text": "Need pricing",
                "channel": "instagram",
                "lead_stage": "warm",
                "custom": {"topic": "pricing"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["signal"]["customer_id"] == "telegram_owner_1"
    assert payload["signal"]["thread_id"] == "inbox_manychat_owner_1"

    claimed = signals.claim_ready_batch(
        source="manychat",
        customer_id="telegram_owner_1",
        thread_id="inbox_manychat_owner_1",
    )
    assert len(claimed) == 1
    assert claimed[0]["payload"]["external_subject_id"] == "contact_42"
    assert claimed[0]["payload"]["external_conversation_id"] == "conv_42"
    assert claimed[0]["payload"]["channel"] == "instagram"
    assert claimed[0]["payload"]["lead_stage"] == "warm"
    assert claimed[0]["payload"]["custom"]["topic"] == "pricing"
    assert claimed[0]["dispatch"]["external_subject_id"] == "contact_42"
    assert claimed[0]["dispatch"]["external_conversation_id"] == "conv_42"


def test_signal_ingest_manychat_derives_stable_thread_from_contact_id(tmp_path: Path) -> None:
    client, signals = _mk_client(tmp_path)

    with client:
        first = client.post(
            "/internal/signals/ingest",
            json={
                "source": "manychat",
                "owner_customer_id": "telegram_owner_1",
                "external_subject_id": "contact_42",
                "external_conversation_id": "conv_42",
                "text": "First question",
            },
        )
        second = client.post(
            "/internal/signals/ingest",
            json={
                "source": "manychat",
                "owner_customer_id": "telegram_owner_1",
                "external_subject_id": "contact_42",
                "external_conversation_id": "conv_99",
                "text": "Second question",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["signal"]["thread_id"] == "inbox_manychat_contact_42"
    assert second_payload["signal"]["thread_id"] == "inbox_manychat_contact_42"

    claimed = signals.claim_ready_batch(
        source="manychat",
        customer_id="telegram_owner_1",
        thread_id="inbox_manychat_contact_42",
    )
    assert [item["text"] for item in claimed] == ["First question", "Second question"]
    assert claimed[0]["dispatch"]["external_subject_id"] == "contact_42"
    assert claimed[1]["dispatch"]["external_subject_id"] == "contact_42"


def test_signal_ingest_manychat_falls_back_to_conversation_id_when_contact_missing(tmp_path: Path) -> None:
    client, _signals = _mk_client(tmp_path)

    with client:
        response = client.post(
            "/internal/signals/ingest",
            json={
                "source": "manychat",
                "owner_customer_id": "telegram_owner_1",
                "external_conversation_id": "conv_42",
                "text": "Need pricing",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["signal"]["thread_id"] == "inbox_manychat_conv_conv_42"


def test_signal_outbox_list_and_mark_sent(tmp_path: Path) -> None:
    client, signals = _mk_client(tmp_path)
    outbound = signals.create_outbound_message(
        source="manychat",
        customer_id="mc_123",
        thread_id="chat-mc_123",
        text="Thanks for your message.",
        dispatch={"conversation_id": "conv_1"},
        signal_ids=[1, 2],
    )

    with client:
        listed = client.get("/internal/signals/outbox", params={"source": "manychat"})
        assert listed.status_code == 200
        items = listed.json()["outbox"]
        assert len(items) == 1
        assert items[0]["id"] == outbound["id"]
        assert items[0]["status"] == "pending"

        marked = client.post(f"/internal/signals/outbox/{outbound['id']}/sent")
        assert marked.status_code == 200
        assert marked.json()["outbox"]["status"] == "sent"


def test_signal_rule_upsert_and_list_roundtrip_handler_skill_name(tmp_path: Path) -> None:
    client, _signals = _mk_client(tmp_path)
    _upsert_handler_skill(tmp_path, customer_id="owner_1", name="manychat-incoming-handler")

    with client:
        created = client.post(
            "/internal/signals/rules/upsert",
            json={
                "source": "manychat",
                "customer_id": "owner_1",
                "thread_id": "inbox_manychat_contact_42",
                "wake_mode": "always",
                "auto_reply": True,
                "handler_skill_name": "manychat-incoming-handler",
                "guidance_text": "Keep answers concise.",
            },
        )
        listed = client.get(
            "/internal/signals/rules",
            params={"source": "manychat", "customer_id": "owner_1"},
        )

    assert created.status_code == 200
    assert listed.status_code == 200
    assert created.json()["rule"]["handler_skill_name"] == "manychat-incoming-handler"
    assert listed.json()["rules"][0]["handler_skill_name"] == "manychat-incoming-handler"


def test_signal_rule_upsert_requires_handler_skill_name_when_auto_reply_enabled(tmp_path: Path) -> None:
    client, _signals = _mk_client(tmp_path)

    with client:
        created = client.post(
            "/internal/signals/rules/upsert",
            json={
                "source": "manychat",
                "customer_id": "owner_1",
                "thread_id": "inbox_manychat_contact_42",
                "wake_mode": "always",
                "auto_reply": True,
            },
        )

    assert created.status_code == 400
    assert "handler_skill_name is required" in created.json()["detail"]


def test_signal_rule_upsert_requires_existing_handler_skill(tmp_path: Path) -> None:
    client, _signals = _mk_client(tmp_path)

    with client:
        created = client.post(
            "/internal/signals/rules/upsert",
            json={
                "source": "manychat",
                "customer_id": "owner_1",
                "thread_id": "inbox_manychat_contact_42",
                "wake_mode": "always",
                "auto_reply": True,
                "handler_skill_name": "missing-handler",
            },
        )

    assert created.status_code == 400
    assert "does not resolve to an existing user/global skill" in created.json()["detail"]
