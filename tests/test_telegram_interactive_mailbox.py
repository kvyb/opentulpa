from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.interfaces.telegram import chat_service as chat_module


class _FakeStateStore:
    def __init__(self, initial: dict[str, Any]) -> None:
        self.state = initial
        self.assistant_touches: list[int] = []

    def update(self, mutator: Any) -> Any:
        return mutator(self.state)

    def touch_assistant_message(self, chat_id: int) -> None:
        self.assistant_touches.append(chat_id)


class _InteractiveRuntime:
    def __init__(self) -> None:
        self.registered_thread_ids: list[str] = []
        self.cleared_thread_ids: list[str] = []
        self.update_senders: dict[str, Any] = {}

    async def register_interactive_session(self, *, thread_id: str, session: Any) -> None:
        del session
        self.registered_thread_ids.append(thread_id)

    async def clear_interactive_session(
        self, *, thread_id: str, session: Any | None = None
    ) -> None:
        del session
        self.cleared_thread_ids.append(thread_id)

    async def register_interactive_update_sender(self, *, thread_id: str, sender: Any) -> None:
        self.update_senders[thread_id] = sender

    async def clear_interactive_update_sender(
        self,
        *,
        thread_id: str,
        sender: Any | None = None,
    ) -> None:
        if sender is None or self.update_senders.get(thread_id) is sender:
            self.update_senders.pop(thread_id, None)

    async def emit_registered_update(self, *, thread_id: str, text: str) -> dict[str, Any]:
        sender = self.update_senders[thread_id]
        return await sender(text)

    def healthy(self) -> bool:
        return True


class _FakeTelegramClient:
    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token
        self.message_calls: list[dict[str, Any]] = []

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.message_calls.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
            }
        )
        return {"ok": True}

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_telegram_interactive_owner_reply_to_bot_photo_adds_reply_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    runtime = _InteractiveRuntime()
    service = chat_module.TelegramChatService(
        bot_token="123:abc",
        file_vault=object(),
        memory=None,
    )
    captured_turn_texts: list[str] = []

    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "get_openai_compatible_api_key_from_env", lambda: "key")
    monkeypatch.setattr(chat_module, "is_user_allowed", lambda **kwargs: True)

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        captured_turn_texts.append(str(kwargs.get("text", "")))
        return "done", False

    monkeypatch.setattr(
        chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram
    )

    result = await service.handle_update(
        body={
            "message": {
                "chat": {"id": 1},
                "from": {"id": 100},
                "text": "Use this image for the landing screen.",
                "reply_to_message": {
                    "message_id": 44,
                    "from": {"id": 200, "is_bot": True, "username": "OpenTulpaBot"},
                    "caption": "Generated hero image for the car wash workflow.",
                    "photo": [
                        {
                            "file_id": "small",
                            "file_unique_id": "small_unique",
                            "width": 90,
                            "height": 90,
                            "file_size": 100,
                        },
                        {
                            "file_id": "large",
                            "file_unique_id": "large_unique",
                            "width": 1024,
                            "height": 768,
                            "file_size": 500,
                        },
                    ],
                },
            }
        },
        agent_runtime=runtime,
    )

    assert result is None
    assert captured_turn_texts and len(captured_turn_texts) == 1
    turn_text = captured_turn_texts[0]
    assert "the user replied to one of OpenTulpa's earlier messages" in turn_text
    assert "- replied_message_id: 44" in turn_text
    assert "Generated hero image for the car wash workflow." in turn_text
    assert "type=photo file_unique_id=large_unique size=1024x768" in turn_text
    assert "Current user message:\nUse this image for the landing screen." in turn_text


@pytest.mark.asyncio
async def test_telegram_group_mention_reply_adds_quoted_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    runtime = _InteractiveRuntime()
    service = chat_module.TelegramChatService(
        bot_token="123:abc",
        file_vault=object(),
        memory=None,
    )
    captured_turn_texts: list[str] = []

    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "get_openai_compatible_api_key_from_env", lambda: "key")
    monkeypatch.setattr(chat_module, "is_user_allowed", lambda **kwargs: True)

    async def _fake_resolve_bot_username(bot_token: str | None) -> str:
        assert bot_token == "123:abc"
        return "OpenTulpaBot"

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        captured_turn_texts.append(str(kwargs.get("text", "")))
        return "done", False

    monkeypatch.setattr(chat_module, "_resolve_bot_username", _fake_resolve_bot_username)
    monkeypatch.setattr(
        chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram
    )

    result = await service.handle_update(
        body={
            "message": {
                "chat": {"id": -1001, "type": "supergroup"},
                "from": {"id": 100, "username": "owner"},
                "text": "@OpenTulpaBot summarize this",
                "entities": [{"type": "mention", "offset": 0, "length": 14}],
                "reply_to_message": {
                    "message_id": 88,
                    "from": {"id": 200, "is_bot": False, "username": "alice"},
                    "text": "We should move the meeting to 14:30 and bring the price sheet.",
                },
            }
        },
        agent_runtime=runtime,
    )

    assert result is None
    assert captured_turn_texts and len(captured_turn_texts) == 1
    turn_text = captured_turn_texts[0]
    assert "Telegram quoted message context" in turn_text
    assert "replied_message_id: 88" in turn_text
    assert "move the meeting to 14:30" in turn_text
    assert "Current user message:\nsummarize this" in turn_text
    assert "@OpenTulpaBot" not in turn_text
    assert runtime.registered_thread_ids == ["chat--1001"]
    assert runtime.cleared_thread_ids == ["chat--1001"]
    assert fake_store.assistant_touches == [-1001]


@pytest.mark.asyncio
async def test_telegram_group_message_without_bot_mention_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    runtime = _InteractiveRuntime()
    service = chat_module.TelegramChatService(
        bot_token="123:abc",
        file_vault=object(),
        memory=None,
    )
    captured_turn_texts: list[str] = []

    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "get_openai_compatible_api_key_from_env", lambda: "key")
    monkeypatch.setattr(chat_module, "is_user_allowed", lambda **kwargs: True)

    async def _fake_resolve_bot_username(bot_token: str | None) -> str:
        del bot_token
        return "OpenTulpaBot"

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        captured_turn_texts.append(str(kwargs.get("text", "")))
        return "done", False

    monkeypatch.setattr(chat_module, "_resolve_bot_username", _fake_resolve_bot_username)
    monkeypatch.setattr(
        chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram
    )

    result = await service.handle_update(
        body={
            "message": {
                "chat": {"id": -1001, "type": "supergroup"},
                "from": {"id": 100, "username": "owner"},
                "text": "summarize this",
                "reply_to_message": {
                    "message_id": 88,
                    "from": {"id": 200, "is_bot": False, "username": "alice"},
                    "text": "We should move the meeting to 14:30.",
                },
            }
        },
        agent_runtime=runtime,
    )

    assert result is None
    assert captured_turn_texts == []
    assert runtime.registered_thread_ids == []
    assert fake_store.assistant_touches == []


@pytest.mark.asyncio
async def test_telegram_allowed_username_auto_binds_generic_owner_from_group_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    runtime = _InteractiveRuntime()
    bindings: dict[str, str] = {}
    stream_customers: list[str] = []
    service = chat_module.TelegramChatService(
        bot_token="123:abc",
        file_vault=object(),
        memory=None,
        owner_customer_id="usr_default",
        resolve_telegram_customer_id=lambda user_id: bindings.get(
            f"telegram_{user_id}", f"telegram_{user_id}"
        ),
        bind_telegram_customer_id=lambda **kwargs: bindings.__setitem__(
            f"telegram_{kwargs['telegram_user_id']}", str(kwargs["user_id"])
        ),
    )

    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "get_openai_compatible_api_key_from_env", lambda: "key")

    async def _fake_resolve_bot_username(bot_token: str | None) -> str:
        del bot_token
        return "OpenTulpaBot"

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        stream_customers.append(str(kwargs.get("customer_id", "")))
        return "done", False

    monkeypatch.setattr(chat_module, "_resolve_bot_username", _fake_resolve_bot_username)
    monkeypatch.setattr(
        chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram
    )

    result = await service.handle_update(
        body={
            "message": {
                "chat": {"id": -1001, "type": "supergroup"},
                "from": {"id": 100, "username": "owner"},
                "text": "@OpenTulpaBot list my sheets",
                "entities": [{"type": "mention", "offset": 0, "length": 14}],
            }
        },
        allowed_usernames_csv="owner",
        agent_runtime=runtime,
    )

    assert result is None
    assert bindings == {"telegram_100": "usr_default"}
    assert stream_customers == ["usr_default"]
    assert fake_store.state["sessions"]["-1001"]["customer_id"] == "usr_default"


def test_telegram_allowed_username_auto_bind_skips_telegram_owner_id() -> None:
    calls: list[dict[str, Any]] = []

    chat_module._maybe_auto_bind_allowed_username(
        owner_customer_id="telegram_100",
        allowed_usernames_csv="owner",
        username="owner",
        user_id=100,
        bind_telegram_customer_id=lambda **kwargs: calls.append(dict(kwargs)),
    )

    assert calls == []


def test_telegram_allowed_username_auto_bind_requires_single_matching_username() -> None:
    calls: list[dict[str, Any]] = []

    chat_module._maybe_auto_bind_allowed_username(
        owner_customer_id="usr_default",
        allowed_usernames_csv="owner,helper",
        username="owner",
        user_id=100,
        bind_telegram_customer_id=lambda **kwargs: calls.append(dict(kwargs)),
    )
    chat_module._maybe_auto_bind_allowed_username(
        owner_customer_id="usr_default",
        allowed_usernames_csv="owner",
        username="other",
        user_id=101,
        bind_telegram_customer_id=lambda **kwargs: calls.append(dict(kwargs)),
    )

    assert calls == []


@pytest.mark.asyncio
async def test_telegram_interactive_failed_voice_reply_does_not_stream_reply_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = chat_module.InteractiveSession(
        chat_id=1,
        customer_id="telegram_100",
        thread_id="chat-1",
    )
    submission, _ = await session.enqueue()

    async def _fake_ingest_attachments_with_typing(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    monkeypatch.setattr(
        chat_module, "_ingest_attachments_with_typing", _fake_ingest_attachments_with_typing
    )

    await chat_module._materialize_interactive_submission(
        session=session,
        submission=submission,
        text="",
        reply_context=(
            "Context: the user replied to one of OpenTulpa's earlier messages.\n"
            "- replied_message_id: 44"
        ),
        caption=None,
        attachments=[SimpleNamespace(kind="voice")],
        bot_token="123:abc",
        file_vault=object(),
        memory=None,
        agent_runtime=object(),
        customer_id="telegram_100",
        chat_id=1,
    )

    [result] = await session.consume_ready_batch()
    assert result.fragment is None
    assert result.direct_reply == (
        "I received your voice message but couldn't transcribe it. "
        "Please resend a shorter/clearer voice note or send text."
    )


@pytest.mark.asyncio
async def test_telegram_interactive_inbox_merges_slow_media_then_followup_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    runtime = _InteractiveRuntime()
    service = chat_module.TelegramChatService(
        bot_token="123:abc",
        file_vault=object(),
        memory=None,
    )
    captured_turn_texts: list[str] = []

    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "get_openai_compatible_api_key_from_env", lambda: "key")
    monkeypatch.setattr(chat_module, "is_user_allowed", lambda **kwargs: True)

    async def _fake_ingest_attachments_with_typing(**kwargs: Any) -> list[dict[str, Any]]:
        if kwargs.get("attachments"):
            await asyncio.sleep(0.05)
        else:
            return []
        return [
            {
                "id": "file_1",
                "original_filename": "cat.jpg",
                "kind": "photo",
                "stored_path": "vault/cat.jpg",
                "local_path": "tulpa_stuff/uploads/telegram_100/file_1_cat.jpg",
                "created_at": "2026-04-12T00:00:00Z",
                "summary": "orange cat sleeping on a chair",
            }
        ]

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        captured_turn_texts.append(str(kwargs.get("text", "")))
        return "done", False

    monkeypatch.setattr(
        chat_module, "_ingest_attachments_with_typing", _fake_ingest_attachments_with_typing
    )
    monkeypatch.setattr(
        chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram
    )

    image_body = {
        "message": {
            "chat": {"id": 1},
            "from": {"id": 100},
            "document": {"file_id": "doc1", "file_name": "cat.jpg", "mime_type": "image/jpeg"},
        }
    }
    text_body = {
        "message": {
            "chat": {"id": 1},
            "from": {"id": 100},
            "text": "The important part is the sleeping cat on the chair.",
        }
    }

    async def _send_image() -> str | None:
        return await service.handle_update(body=image_body, agent_runtime=runtime)

    async def _send_text() -> str | None:
        await asyncio.sleep(0.01)
        return await service.handle_update(body=text_body, agent_runtime=runtime)

    image_result, text_result = await asyncio.gather(_send_image(), _send_text())

    assert image_result is None
    assert text_result is None
    assert captured_turn_texts and len(captured_turn_texts) == 1
    merged = captured_turn_texts[0]
    assert "User uploaded one or more files without extra text." not in merged
    assert "If intent is unclear, ask what the user wants done" not in merged
    assert "Do not infer intent from filenames or content" not in merged
    assert "orange cat sleeping on a chair" in merged
    assert "sleeping cat on the chair" in merged
    assert runtime.registered_thread_ids == ["chat-1"]
    assert runtime.cleared_thread_ids == ["chat-1"]
    assert fake_store.assistant_touches == [1]


def test_build_effective_telegram_text_keeps_captioned_media_instruction_clean() -> None:
    effective_text, direct_reply = chat_module._build_effective_telegram_text(
        user_text="What is the best team for my available heroes?",
        attachments=[],
        ingested_files=[
            {
                "id": "file_1",
                "original_filename": "roster.jpg",
                "kind": "photo",
                "summary": "Roster: Diana, Rin, Cassius, Nia.",
            }
        ],
    )

    assert direct_reply is None
    assert "What is the best team for my available heroes?" in effective_text
    assert "Roster: Diana, Rin, Cassius, Nia." in effective_text
    assert "User uploaded one or more files without extra text." not in effective_text
    assert "If intent is unclear, ask what the user wants done" not in effective_text


@pytest.mark.asyncio
async def test_telegram_interactive_session_allows_explicit_owner_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    runtime = _InteractiveRuntime()
    service = chat_module.TelegramChatService(
        bot_token="123:abc",
        file_vault=object(),
        memory=None,
    )
    fake_client = _FakeTelegramClient("123:abc")

    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(chat_module, "get_openai_compatible_api_key_from_env", lambda: "key")
    monkeypatch.setattr(chat_module, "is_user_allowed", lambda **kwargs: True)

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        await kwargs["agent_runtime"].emit_registered_update(
            thread_id=kwargs["thread_id"],
            text="Проверяю прайс и подготовлю черновик.",
        )
        return "Черновик готов.", False

    monkeypatch.setattr(
        chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram
    )

    result = await service.handle_update(
        body={
            "message": {
                "chat": {"id": 1},
                "from": {"id": 100},
                "text": "Настрой workflow.",
            }
        },
        agent_runtime=runtime,
    )

    assert result is None
    assert fake_client.message_calls == [
        {
            "chat_id": 1,
            "text": "Проверяю прайс и подготовлю черновик.",
            "parse_mode": "HTML",
            "reply_markup": None,
        }
    ]
    assert runtime.update_senders == {}
    assert fake_store.assistant_touches == [1, 1]


@pytest.mark.asyncio
async def test_graph_agent_injects_interactive_fragments_before_second_model_call() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_model_messages: list[list[Any]] = []
    drain_calls = 0

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            return {"status": "ok"}

    async def _live_time(customer_id: str) -> dict[str, str]:
        del customer_id
        return {
            "server_time_local_iso": "2026-04-12T10:00:00+08:00",
            "server_time_utc_iso": "2026-04-12T02:00:00+00:00",
            "server_utc_offset": "+08:00",
            "user_time_local_iso": "2026-04-12T10:00:00+08:00",
            "user_utc_offset": "+08:00",
            "user_time_source": "profile",
        }

    async def _directive(customer_id: str) -> str | None:
        del customer_id
        return None

    async def _memory_grounding(**kwargs: Any) -> str:
        del kwargs
        return ""

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        captured_model_messages.append(list(messages))
        if len(captured_model_messages) == 1:
            return AIMessage(
                content="Let me inspect that.",
                tool_calls=[{"id": "call_1", "name": "fake_tool", "args": {}}],
            )
        return AIMessage(content="That looks like a sleeping orange cat on a chair.")

    async def _drain_interactive_fragments(*, thread_id: str) -> list[str]:
        nonlocal drain_calls
        del thread_id
        drain_calls += 1
        if drain_calls == 2:
            return ["The key detail is the sleeping cat on the chair."]
        return []

    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime.drain_interactive_fragments = _drain_interactive_fragments  # type: ignore[method-assign]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Please inspect the image I sent.")],
            "customer_id": "telegram_test",
            "thread_id": "chat_test",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_test",
        },
        config={"configurable": {"thread_id": "chat_test"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "That looks like a sleeping orange cat on a chair."
    assert len(captured_model_messages) == 2
    second_call = captured_model_messages[1]
    assert any(
        isinstance(message, SystemMessage)
        and "user steers with message:" in str(getattr(message, "content", "")).lower()
        and "sleeping cat on the chair" in str(getattr(message, "content", "")).lower()
        for message in second_call
    )
    assert any(
        isinstance(message, ToolMessage)
        and "status" in str(getattr(message, "content", "")).lower()
        for message in second_call
    )
