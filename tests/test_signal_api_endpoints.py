from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.context.signals import SignalInboxService
from opentulpa.skills.service import SkillStoreService


def _mk_client(tmp_path: Path) -> tuple[TestClient, SignalInboxService]:
    signals = SignalInboxService(db_path=tmp_path / "signals.db")
    store = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(skill_store_service=store, signal_inbox_service=signals)
    return TestClient(app), signals


def test_signal_ingest_uses_resolved_rule_defaults(tmp_path: Path) -> None:
    client, signals = _mk_client(tmp_path)
    signals.upsert_rule(
        source="manychat",
        customer_id="mc_123",
        thread_id="chat-mc_123",
        wake_mode="always",
        batch_window_seconds=45,
        auto_reply=True,
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
