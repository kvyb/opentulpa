from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.lc_messages import AIMessage, HumanMessage
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

    async def clear_interactive_session(self, *, thread_id: str, session: Any | None = None) -> None:
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

    monkeypatch.setattr(chat_module, "_ingest_attachments_with_typing", _fake_ingest_attachments_with_typing)
    monkeypatch.setattr(chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram)

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
    assert "User uploaded one or more files without extra text." in merged
    assert "orange cat sleeping on a chair" in merged
    assert "sleeping cat on the chair" in merged
    assert runtime.registered_thread_ids == ["chat-1"]
    assert runtime.cleared_thread_ids == ["chat-1"]
    assert fake_store.assistant_touches == [1]


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

    monkeypatch.setattr(chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram)

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
    runtime._has_retrieval_evidence = lambda **kwargs: False  # type: ignore[assignment]
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
        isinstance(message, HumanMessage)
        and "sleeping cat on the chair" in str(getattr(message, "content", "")).lower()
        for message in second_call
    )
