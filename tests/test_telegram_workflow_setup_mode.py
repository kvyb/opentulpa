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
        self.state["last_touched_chat_id"] = str(chat_id)


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke_text(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"turn_mode={kwargs.get('turn_mode')}"


def _body(*, chat_id: int, user_id: int, text: str, username: str | None = None) -> dict[str, Any]:
    from_user: dict[str, Any] = {"id": user_id}
    if username:
        from_user["username"] = username
    return {"message": {"chat": {"id": chat_id}, "from": from_user, "text": text}}


@pytest.mark.asyncio
async def test_telegram_owner_turn_uses_workflow_setup_mode_when_session_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 101, "sessions": {}, "pending_key_by_chat": {}})
    runtime = _Runtime()
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")

    text = await chat_module.handle_telegram_text(
        body=_body(chat_id=1101, user_id=101, username="owner", text="Продолжим настройку workflow."),
        bot_token=None,
        agent_runtime=runtime,
        workflow_setup_status=lambda **_: {"status": "active"},
    )

    assert text == "turn_mode=workflow_setup"
    assert runtime.calls[0]["customer_id"] == "telegram_101"
    assert runtime.calls[0]["thread_id"] == "chat-1101"
    assert runtime.calls[0]["turn_mode"] == "workflow_setup"


@pytest.mark.asyncio
async def test_telegram_streaming_turn_passes_workflow_setup_mode_when_session_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 101, "sessions": {}, "pending_key_by_chat": {}})
    captured: dict[str, Any] = {}
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        captured.update(kwargs)
        return "ok", False

    monkeypatch.setattr(chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram)

    text = await chat_module.handle_telegram_text(
        body=_body(chat_id=1101, user_id=101, username="owner", text="Продолжим настройку workflow."),
        bot_token="123:abc",
        agent_runtime=object(),
        workflow_setup_status=lambda **_: {"status": "active"},
    )

    assert text is None
    assert captured["customer_id"] == "telegram_101"
    assert captured["thread_id"] == "chat-1101"
    assert captured["turn_mode"] == "workflow_setup"
    assert fake_store.state["last_touched_chat_id"] == "1101"


@pytest.mark.asyncio
async def test_support_bound_telegram_turn_uses_bound_customer_workflow_setup_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore(
        {
            "admin_user_id": 101,
            "sessions": {},
            "pending_key_by_chat": {},
            "support_bindings": {
                "9900": {
                    "support_user_id": 900,
                    "support_username": "support",
                    "bound_customer_id": "telegram_101",
                    "thread_id": "chat_support_101",
                    "wake_thread_id": "wake_support_101",
                }
            },
        }
    )
    runtime = _Runtime()
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")

    text = await chat_module.handle_telegram_text(
        body=_body(chat_id=9900, user_id=900, username="support", text="Продолжим setup клиента."),
        bot_token=None,
        support_user_ids_csv="900",
        agent_runtime=runtime,
        workflow_setup_status=lambda **_: {"status": "active"},
    )

    assert text == "turn_mode=workflow_setup"
    assert runtime.calls[0]["customer_id"] == "telegram_101"
    assert runtime.calls[0]["thread_id"] == "chat_support_101"
    assert runtime.calls[0]["turn_mode"] == "workflow_setup"
