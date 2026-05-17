from __future__ import annotations

from typing import Any

import pytest

from opentulpa.interfaces.telegram import chat_service as chat_module


class _FakeStateStore:
    def __init__(self, initial: dict[str, Any]) -> None:
        self.state = initial

    def load(self) -> dict[str, Any]:
        return self.state

    def update(self, mutator: Any) -> Any:
        return mutator(self.state)

    def touch_assistant_message(self, chat_id: int | str) -> None:
        key = str(chat_id)
        binding = self.state.get("support_bindings", {}).get(key)
        if isinstance(binding, dict):
            binding["last_assistant_message_at"] = "touched"


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def healthy(self) -> bool:
        return True

    async def ainvoke_text(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"customer={kwargs.get('customer_id')} thread={kwargs.get('thread_id')}"


class _FakeTelegramClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token

    async def set_my_commands(
        self,
        *,
        commands: list[dict[str, str]],
        scope: dict[str, Any] | None = None,
    ) -> bool:
        self.calls.append(
            {
                "bot_token": self.bot_token,
                "commands": commands,
                "scope": scope or {},
            }
        )
        return True

    async def aclose(self) -> None:
        return None


def _body(*, chat_id: int, user_id: int, text: str, username: str | None = None) -> dict[str, Any]:
    from_user: dict[str, Any] = {"id": user_id}
    if username:
        from_user["username"] = username
    return {"message": {"chat": {"id": chat_id}, "from": from_user, "text": text}}


def _customers() -> list[dict[str, Any]]:
    return [
        {
            "customer_id": "telegram_101",
            "owner_username": "owner101",
            "owner_chat_id": "1101",
            "telegram_business_connected": True,
            "composio_connected": True,
            "workflow_count": 1,
            "file_count": 2,
            "last_activity": "2026-04-27T00:00:00+00:00",
        },
        {
            "customer_id": "telegram_202",
            "owner_username": "owner202",
            "owner_chat_id": "2202",
            "telegram_business_connected": False,
            "composio_connected": True,
            "workflow_count": 3,
            "file_count": 0,
            "last_activity": "2026-04-26T00:00:00+00:00",
        },
    ]


@pytest.mark.asyncio
async def test_support_commands_rejected_when_support_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 1, "sessions": {}, "pending_key_by_chat": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)

    text = await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="support", text="/support_customers"),
        support_customer_listing=_customers,
    )

    assert "restricted" in str(text).lower()
    assert fake_store.state.get("support_bindings") in (None, {})


@pytest.mark.asyncio
async def test_owner_fresh_uses_bound_generic_customer_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 1, "sessions": {}, "pending_key_by_chat": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)

    text = await chat_module.handle_telegram_text(
        body=_body(chat_id=1000, user_id=123, username="owner", text="/fresh"),
        allowed_user_ids_csv="123",
        resolve_telegram_customer_id=lambda user_id: "usr_default",
    )

    assert "Started a fresh chat context" in str(text)
    assert fake_store.state["sessions"]["1000"]["customer_id"] == "usr_default"


@pytest.mark.asyncio
async def test_support_bind_by_id_and_route_runtime_to_bound_customer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 1, "sessions": {}, "pending_key_by_chat": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    runtime = _Runtime()

    listed = await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="support", text="/support_customers"),
        support_user_ids_csv="900",
        support_customer_listing=_customers,
    )
    bound = await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="support", text="/support_bind 1"),
        support_user_ids_csv="900",
        support_customer_listing=_customers,
    )
    reply = await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="support", text="List workflows."),
        bot_token=None,
        support_user_ids_csv="900",
        agent_runtime=runtime,
        support_customer_listing=_customers,
    )

    assert "telegram_101" in str(listed)
    assert "business=connected" in str(listed)
    assert "Support bound to telegram_101" in str(bound)
    assert "customer=telegram_101" in str(reply)
    assert runtime.calls[0]["customer_id"] == "telegram_101"
    assert str(runtime.calls[0]["thread_id"]).startswith("chat_")
    assert "9000" not in fake_store.state.get("sessions", {})
    assert fake_store.state["support_bindings"]["9000"]["bound_customer_id"] == "telegram_101"


@pytest.mark.asyncio
async def test_username_support_configures_command_menu_after_first_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 1, "sessions": {}, "pending_key_by_chat": {}})
    _FakeTelegramClient.calls = []
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "TelegramClient", _FakeTelegramClient)

    await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="agent", text="/support_customers"),
        bot_token="123:abc",
        support_usernames_csv="agent",
        support_customer_listing=_customers,
    )
    await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="agent", text="/support_whoami"),
        bot_token="123:abc",
        support_usernames_csv="agent",
        support_customer_listing=_customers,
    )

    assert len(_FakeTelegramClient.calls) == 1
    call = _FakeTelegramClient.calls[0]
    assert call["scope"] == {"type": "chat", "chat_id": 9000}
    commands = call["commands"]
    assert any(
        str(item.get("command", "")).strip() == "support_bind"
        for item in commands
        if isinstance(item, dict)
    )
    assert fake_store.state["support_command_chats"]["9000"]


@pytest.mark.asyncio
async def test_support_bind_by_username_raw_customer_and_fresh_is_support_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore(
        {
            "admin_user_id": 1,
            "pending_key_by_chat": {},
            "sessions": {
                "1101": {
                    "user_id": 101,
                    "username": "owner101",
                    "customer_id": "telegram_101",
                    "thread_id": "chat_owner",
                    "wake_thread_id": "wake_owner",
                    "last_user_message_at": "2026-04-27T00:00:00+00:00",
                    "last_assistant_message_at": None,
                }
            },
        }
    )
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)

    await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="agent", text="/support_bind telegram_202"),
        support_usernames_csv="agent",
        support_customer_listing=_customers,
    )
    before = fake_store.state["support_bindings"]["9000"]["thread_id"]
    text = await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="agent", text="/fresh"),
        support_usernames_csv="agent",
        support_customer_listing=_customers,
    )
    after = fake_store.state["support_bindings"]["9000"]["thread_id"]

    assert "fresh support chat context" in str(text).lower()
    assert before != after
    assert fake_store.state["support_bindings"]["9000"]["bound_customer_id"] == "telegram_202"
    assert fake_store.state["sessions"]["1101"]["thread_id"] == "chat_owner"


@pytest.mark.asyncio
async def test_unbound_support_turn_does_not_create_support_customer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 1, "sessions": {}, "pending_key_by_chat": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)

    text = await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="support", text="Set up a workflow."),
        support_user_ids_csv="900",
        support_customer_listing=_customers,
    )

    assert "support_bind" in str(text)
    assert fake_store.state.get("sessions", {}) == {}
    assert not str(fake_store.state).count("telegram_900")


@pytest.mark.asyncio
async def test_invalid_customer_does_not_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 1, "sessions": {}, "pending_key_by_chat": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)

    text = await chat_module.handle_telegram_text(
        body=_body(chat_id=9000, user_id=900, username="support", text="/support_bind telegram_missing"),
        support_user_ids_csv="900",
        support_customer_listing=_customers,
    )

    assert "Customer not found" in str(text)
    assert fake_store.state.get("support_bindings") in (None, {})
