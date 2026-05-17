from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.profiles import register_profile_routes
from opentulpa.context.customer_profiles import CustomerProfileService


class _MemoryRecorder:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def add_text(self, text: str, *, user_id: str, metadata: dict[str, Any]) -> None:
        self.entries.append(
            {
                "text": text,
                "user_id": user_id,
                "metadata": metadata,
            }
        )


def _mk_client(tmp_path: Path) -> tuple[TestClient, CustomerProfileService, _MemoryRecorder]:
    app = FastAPI()
    profiles = CustomerProfileService(tmp_path / "customer_profiles.sqlite")
    memory = _MemoryRecorder()
    register_profile_routes(
        app,
        get_profiles=lambda: profiles,
        get_memory=lambda: memory,
    )
    return TestClient(app), profiles, memory


def _wait_for(condition: Any, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(condition()):
            return
        time.sleep(0.01)
    raise AssertionError("condition not satisfied before timeout")


def test_profile_routes_use_pydantic_bodies_and_responses(tmp_path: Path) -> None:
    client, _, memory = _mk_client(tmp_path)

    directive_set = client.post(
        "/internal/directive/set",
        json={"customer_id": " telegram_1 ", "directive": " stay concise ", "source": " ui "},
    )
    assert directive_set.status_code == 200
    assert directive_set.json() == {"ok": True, "customer_id": "telegram_1"}

    directive_get = client.post("/internal/directive/get", json={"customer_id": "telegram_1"})
    assert directive_get.status_code == 200
    assert directive_get.json() == {"customer_id": "telegram_1", "directive": "stay concise"}

    time_set = client.post(
        "/internal/time_profile/set",
        json={"customer_id": "telegram_1", "utc_offset": "+08:00", "source": "ui"},
    )
    assert time_set.status_code == 200
    assert time_set.json() == {"ok": True, "customer_id": "telegram_1", "utc_offset": "+08:00"}

    _wait_for(lambda: len(memory.entries) == 2)
    assert memory.entries[0]["metadata"]["kind"] == "directive_fact"
    assert memory.entries[1]["metadata"]["kind"] == "life_fact"


def test_profile_routes_reject_invalid_payloads_via_pydantic(tmp_path: Path) -> None:
    client, _, _ = _mk_client(tmp_path)

    missing_customer = client.post("/internal/directive/get", json={})
    assert missing_customer.status_code == 422

    bad_offset = client.post(
        "/internal/time_profile/set",
        json={"customer_id": "telegram_1", "utc_offset": "UTC+8"},
    )
    assert bad_offset.status_code == 422


def test_profile_routes_bind_telegram_alias_to_generic_storage(tmp_path: Path) -> None:
    client, profiles, _ = _mk_client(tmp_path)
    profiles.set_directive("usr_default", "use generic storage", source="test")

    bind = client.post(
        "/profiles/bind-telegram",
        json={"user_id": "usr_default", "telegram_user_id": "123"},
    )
    assert bind.status_code == 200
    assert bind.json() == {"ok": True, "customer_id": "usr_default"}

    assert profiles.resolve_customer_id("telegram_123") == "usr_default"
    assert profiles.get_directive("telegram_123") == "use generic storage"

    listed = client.get("/profiles")
    assert listed.status_code == 200
    body = listed.json()
    assert body["profiles"] == [
        {
            "user_id": "usr_default",
            "storage_user_id": "usr_default",
            "telegram_user_id": "123",
            "aliases": ["usr_default", "telegram_123"],
        }
    ]
    assert body["bindings"][1]["alias_user_id"] == "telegram_123"


def test_profile_binding_uses_legacy_telegram_storage_when_it_exists_first(tmp_path: Path) -> None:
    _, profiles, _ = _mk_client(tmp_path)
    profiles.set_directive("telegram_123", "legacy directive", source="test")

    profiles.bind_telegram_user_id(user_id="usr_default", telegram_user_id="123")

    assert profiles.resolve_customer_id("usr_default") == "telegram_123"
    assert profiles.get_directive("usr_default") == "legacy directive"


def test_profile_binding_rejects_separate_existing_profiles(tmp_path: Path) -> None:
    _, profiles, _ = _mk_client(tmp_path)
    profiles.set_directive("usr_default", "generic directive", source="test")
    profiles.set_directive("telegram_123", "telegram directive", source="test")

    try:
        profiles.bind_telegram_user_id(user_id="usr_default", telegram_user_id="123")
    except ValueError as exc:
        assert "manual merge" in str(exc)
    else:
        raise AssertionError("expected separate profile bind to be rejected")


def test_profile_binding_rejects_existing_generic_profile_when_telegram_alias_is_bound(
    tmp_path: Path,
) -> None:
    _, profiles, _ = _mk_client(tmp_path)
    profiles.set_directive("telegram_123", "legacy directive", source="test")
    profiles.bind_telegram_user_id(user_id="usr_first", telegram_user_id="123")
    profiles.set_directive("usr_second", "separate generic directive", source="test")

    try:
        profiles.bind_telegram_user_id(user_id="usr_second", telegram_user_id="123")
    except ValueError as exc:
        assert "manual merge" in str(exc)
    else:
        raise AssertionError("expected conflicting generic profile bind to be rejected")

    assert profiles.resolve_customer_id("usr_second") == "usr_second"
    assert profiles.get_directive("usr_second") == "separate generic directive"


def test_profile_binding_rejects_existing_telegram_profile_when_generic_alias_is_bound(
    tmp_path: Path,
) -> None:
    _, profiles, _ = _mk_client(tmp_path)
    profiles.set_directive("usr_default", "generic directive", source="test")
    profiles.bind_telegram_user_id(user_id="usr_default", telegram_user_id="123")
    profiles.set_directive("telegram_456", "separate telegram directive", source="test")

    try:
        profiles.bind_telegram_user_id(user_id="usr_default", telegram_user_id="456")
    except ValueError as exc:
        assert "manual merge" in str(exc)
    else:
        raise AssertionError("expected conflicting telegram profile bind to be rejected")

    assert profiles.resolve_customer_id("telegram_456") == "telegram_456"
    assert profiles.get_directive("telegram_456") == "separate telegram directive"
