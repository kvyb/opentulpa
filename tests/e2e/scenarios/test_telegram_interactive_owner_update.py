from __future__ import annotations

import asyncio
import contextvars
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from harness.runner import E2EHarness, build_harness, close_harness, load_jsonl
from langgraph.checkpoint.memory import InMemorySaver
from mocks.telegram import FakeTelegramClient

from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.lc_messages import AIMessage, SystemMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.agent.runtime_context_provider import RuntimeContextSourceProvider
from opentulpa.agent.runtime_input import ThreadInputCoordinator
from opentulpa.agent.tools_registry import register_runtime_tools
from opentulpa.api import app as app_module
from opentulpa.api.app import create_app
from opentulpa.api.routes import wake_search as wake_search_routes
from opentulpa.core.config import get_settings
from opentulpa.interfaces.telegram import attachments as attachments_module
from opentulpa.interfaces.telegram import chat_service as chat_module
from opentulpa.interfaces.telegram import relay as relay_module
from opentulpa.interfaces.telegram.interactive_inbox import TelegramInteractiveInbox
from opentulpa.interfaces.telegram.state_store import TelegramStateStore
from opentulpa.scheduler.service import SchedulerService
from opentulpa.tasks import sandbox as sandbox_module

pytestmark = [pytest.mark.e2e, pytest.mark.telegram]


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


def _wait_until(predicate: Any, timeout_seconds: float = 60.0) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while time.monotonic() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.2)
    return bool(predicate())


async def _wait_until_async(predicate: Any, timeout_seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while time.monotonic() < deadline:
        if bool(await predicate()):
            return True
        await asyncio.sleep(0.05)
    return bool(await predicate())


class _RecordingMemory:
    user_id = "default"

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []

    def add(
        self,
        messages: list[dict[str, Any]],
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        retries: int = 1,
    ) -> dict[str, Any]:
        del infer, retries
        text = "\n".join(
            str(message.get("content", "") or "").strip()
            for message in messages
            if isinstance(message, dict) and str(message.get("content", "") or "").strip()
        ).strip()
        record = {
            "id": f"mem_{len(self.entries) + 1}",
            "memory": text,
            "text": text,
            "score": 0.99,
            "metadata": {"kind": "preference_fact", **dict(metadata or {})},
            "user_id": str(user_id or self.user_id),
        }
        self.entries.append(record)
        return {"results": [record]}

    def add_text(
        self,
        text: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        retries: int = 1,
    ) -> dict[str, Any]:
        return self.add(
            [{"role": "user", "content": text}],
            user_id=user_id,
            metadata=metadata,
            infer=infer,
            retries=retries,
        )

    def search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.searches.append(
            {
                "query": str(query or ""),
                "user_id": str(user_id or self.user_id),
                "limit": int(limit),
                "metadata": metadata or {},
            }
        )
        kinds = metadata.get("kind") if isinstance(metadata, dict) else None
        allowed_kinds = {str(item) for item in kinds} if isinstance(kinds, list) else None
        out: list[dict[str, Any]] = []
        for entry in reversed(self.entries):
            kind = str(entry.get("metadata", {}).get("kind", "") or "")
            if allowed_kinds is not None and kind not in allowed_kinds:
                continue
            out.append(entry)
            if len(out) >= int(limit):
                break
        return out


async def _live_time(customer_id: str) -> dict[str, str]:
    del customer_id
    return {
        "server_time_local_iso": "2026-04-27T10:00:00+08:00",
        "server_time_utc_iso": "2026-04-27T02:00:00+00:00",
        "server_utc_offset": "+08:00",
        "user_time_local_iso": "2026-04-27T10:00:00+08:00",
        "user_utc_offset": "+08:00",
        "user_time_source": "profile",
    }


def _build_deterministic_runtime() -> tuple[OpenTulpaLangGraphRuntime, list[list[Any]]]:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model_calls: list[list[Any]] = []
    behavior_events: list[dict[str, Any]] = []

    async def _noop_start() -> None:
        return None

    async def _noop_shutdown() -> None:
        return None

    async def _list_available_skills(customer_id: str) -> list[dict[str, Any]]:
        del customer_id
        return []

    async def _load_skill_context_by_names(
        *, customer_id: str, skill_names: list[str]
    ) -> dict[str, Any]:
        del customer_id, skill_names
        return {"skill_names": [], "context": ""}

    async def _empty_memory_grounding(**kwargs: Any) -> str:
        del kwargs
        return ""

    async def _directive(customer_id: str) -> str | None:
        del customer_id
        return None

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        model_calls.append(list(messages))
        if len(model_calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_update",
                        "name": "send_owner_update",
                        "args": {
                            "message": "Ищу свежие данные и потом дам короткий вывод.",
                            "dedupe_key": "search_started",
                        },
                    },
                    {
                        "id": "call_search",
                        "name": "tool_group_exec",
                        "args": {
                            "group": "web",
                            "command": "web_search",
                            "args_json": {
                                "query": "OpenTulpa interactive owner update search test"
                            },
                        },
                    },
                ],
            )
        return AIMessage(
            content=(
                "Готово: поиск вернул тестовые данные по интерактивному update. "
                "Новый промежуточный ответ был отправлен до финального сообщения."
            )
        )

    runtime.start = _noop_start  # type: ignore[method-assign]
    runtime.shutdown = _noop_shutdown  # type: ignore[method-assign]
    runtime.model_name = "deterministic-test-model"
    runtime.recursion_limit = 12
    runtime._checkpointer = InMemorySaver()
    runtime._checkpointer_cm = None
    runtime._model = object()
    runtime._wake_execution_model = runtime._model
    runtime._model_with_tools = object()
    runtime._wake_execution_model_with_tools = runtime._model_with_tools
    runtime._graph = None
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._thread_rollup_service = None
    runtime._link_alias_service = None
    runtime._langfuse_tracer = None
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime._max_user_reply_chars = 4000
    runtime._interactive_sessions_lock = asyncio.Lock()
    runtime._interactive_sessions = {}
    runtime._interactive_update_senders_lock = asyncio.Lock()
    runtime._interactive_update_senders = {}
    runtime._interactive_update_sent_keys = {}
    runtime._active_customer_id_ctx = contextvars.ContextVar("test_customer_id", default="")
    runtime._active_thread_id_ctx = contextvars.ContextVar("test_thread_id", default="")
    runtime._active_customer_id = ""
    runtime._active_thread_id = ""
    runtime._behavior_log_enabled = False
    runtime._tools = {}

    runtime._list_available_skills = _list_available_skills  # type: ignore[method-assign]
    runtime._load_skill_context_by_names = _load_skill_context_by_names  # type: ignore[method-assign]
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _empty_memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._load_thread_rollup_sections = lambda thread_id: {}  # type: ignore[assignment]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.expand_link_aliases = lambda **kwargs: str(kwargs.get("text", ""))  # type: ignore[assignment]
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.astream_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: runtime._model_with_tools  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: behavior_events.append(kwargs)  # type: ignore[assignment]

    async def _request_with_backoff(*args: Any, **kwargs: Any) -> httpx.Response:
        raise RuntimeError("internal API requester was not bound")

    runtime._request_with_backoff = _request_with_backoff  # type: ignore[method-assign]
    runtime._context_source_provider = RuntimeContextSourceProvider(runtime)
    runtime._tools = register_runtime_tools(runtime)
    runtime._graph = build_runtime_graph(runtime)
    return runtime, model_calls


def test_telegram_interactive_chat_can_send_owner_update_before_search_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_tg = FakeTelegramClient("fake-token")
    runtime, model_calls = _build_deterministic_runtime()
    project_root = tmp_path / "project_root"
    project_root.mkdir(parents=True)
    search_queries: list[str] = []
    internal_calls: list[dict[str, Any]] = []

    async def _fake_run_web_search(query: str) -> dict[str, Any]:
        search_queries.append(query)
        return {
            "answer": f"Fake search answer for: {query}",
            "source_count": 2,
            "sources": [{"url": "https://example.com/opentulpa", "domain": "example.com"}],
            "model": "test-double",
        }

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "100")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERNAMES", "")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / "links.sqlite"))
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        chat_module,
        "STATE_STORE",
        TelegramStateStore(project_root / ".opentulpa" / "telegram_state.json"),
    )
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(attachments_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(chat_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(wake_search_routes, "run_web_search", _fake_run_web_search)
    get_settings.cache_clear()

    app = create_app(
        agent_runtime=runtime,
        scheduler=SchedulerService(db_path=tmp_path / "scheduler.sqlite"),
    )

    async def _request_with_backoff(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int = 2,
    ) -> httpx.Response:
        del timeout, retries
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.request(method=method, url=path, params=params, json=json_body)
        internal_calls.append(
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

    def owner_messages() -> list[dict[str, Any]]:
        return [
            item
            for item in fake_tg.sent_messages
            if int(item.get("chat_id", 0)) == 1100 and not item.get("business_connection_id")
        ]

    with TestClient(app) as client:
        response = client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": "test-secret"},
            json=_telegram_message(
                chat_id=1100,
                user_id=100,
                username="owner",
                text="Найди свежую информацию и сделай краткий вывод.",
            ),
        )
        assert response.status_code == 200
        assert _wait_until(lambda: len(owner_messages()) >= 2, timeout_seconds=10.0)

    get_settings.cache_clear()

    assert response.status_code == 200
    assert [item["text"] for item in owner_messages()] == [
        "Ищу свежие данные и потом дам короткий вывод.",
        (
            "Готово: поиск вернул тестовые данные по интерактивному update. "
            "Новый промежуточный ответ был отправлен до финального сообщения."
        ),
    ]
    assert search_queries == ["OpenTulpa interactive owner update search test"]
    assert any(item["path"] == "/internal/web_search" for item in internal_calls)
    assert len(model_calls) == 2


@pytest.mark.asyncio
async def test_telegram_interactive_message_steers_active_graph_after_tool_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_tg = FakeTelegramClient("fake-token")
    runtime, model_calls = _build_deterministic_runtime()
    first_tool_started = asyncio.Event()
    steering_ready = asyncio.Event()
    chat_id = 1200
    user_id = 100
    inbox = TelegramInteractiveInbox()

    class _WaitForSteeringTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            first_tool_started.set()
            await asyncio.wait_for(steering_ready.wait(), timeout=5.0)
            return {"status": "ok", "checkpoint": "after_first_model_call"}

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        model_calls.append(list(messages))
        if len(model_calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_wait",
                        "name": "wait_for_steering",
                        "args": {},
                    }
                ],
            )
        return AIMessage(content="I used the green mug detail from the steering message.")

    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.astream_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.tools_for_turn_mode = lambda turn_mode: [SimpleNamespace(name="wait_for_steering")]  # type: ignore[assignment]
    runtime._tools["wait_for_steering"] = _WaitForSteeringTool()

    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    monkeypatch.setattr(
        chat_module,
        "STATE_STORE",
        TelegramStateStore(tmp_path / ".opentulpa" / "telegram_state.json"),
    )
    monkeypatch.setattr(chat_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda _token: fake_tg)

    first_task = asyncio.create_task(
        chat_module.handle_telegram_text(
            body=_telegram_message(
                chat_id=chat_id,
                user_id=user_id,
                username="owner",
                text="Inspect the scene and answer after the tool checkpoint.",
            ),
            bot_token="test-bot-token",
            allowed_user_ids_csv=str(user_id),
            agent_runtime=runtime,
            interactive_inbox=inbox,
        )
    )
    assert await _wait_until_async(
        lambda: asyncio.sleep(0, result=first_tool_started.is_set() or first_task.done()),
        timeout_seconds=5.0,
    )
    if first_task.done():
        raise AssertionError(
            {
                "first_result": first_task.result(),
                "sent_messages": fake_tg.sent_messages,
                "model_calls": len(model_calls),
                "second_call_messages": [
                    str(getattr(message, "content", "") or "")[:500]
                    for message in (model_calls[1] if len(model_calls) > 1 else [])
                ],
            }
        )

    second_result = await chat_module.handle_telegram_text(
        body=_telegram_message(
            chat_id=chat_id,
            user_id=user_id,
            username="owner",
            text="Actually mention the green mug detail.",
        ),
        bot_token="test-bot-token",
        allowed_user_ids_csv=str(user_id),
        agent_runtime=runtime,
        interactive_inbox=inbox,
    )
    assert second_result is None

    async def _queued_steering_ready() -> bool:
        async with inbox._lock:
            session = inbox._sessions.get(str(chat_id))
        if session is None:
            return False
        async with session._condition:
            return any(
                item.ready and "green mug" in str(item.result.fragment or "").lower()
                for item in session._queue
            )

    assert await _wait_until_async(_queued_steering_ready, timeout_seconds=5.0)
    steering_ready.set()

    first_result = await asyncio.wait_for(first_task, timeout=10.0)
    assert first_result is None
    assert len(model_calls) == 2
    second_model_messages = model_calls[1]
    assert any(
        isinstance(message, SystemMessage)
        and "user steers with message:" in str(getattr(message, "content", "")).lower()
        and "green mug" in str(getattr(message, "content", "")).lower()
        for message in second_model_messages
    )
    assert [item["text"] for item in fake_tg.sent_messages if item.get("chat_id") == chat_id] == [
        "I used the green mug detail from the steering message."
    ]


@pytest.mark.live_llm
def test_live_telegram_interactive_chat_can_use_owner_update_while_searching(
    e2e_harness: E2EHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_user_id = 654
    owner_chat_id = 1654
    search_queries: list[str] = []
    internal_start = e2e_harness.count_internal_api_calls()

    async def _fake_run_web_search(query: str) -> dict[str, Any]:
        search_queries.append(query)
        return {
            "answer": (
                "Test search result: OpenTulpa Telegram interactive owner updates let the agent "
                "send a short interim owner message, continue tool work, and then send a final answer."
            ),
            "source_count": 2,
            "sources": [
                {"url": "https://example.com/opentulpa-owner-update", "domain": "example.com"},
                {"url": "https://example.com/telegram-agent-loop", "domain": "example.com"},
            ],
            "model": "test-double",
        }

    monkeypatch.setattr(wake_search_routes, "run_web_search", _fake_run_web_search)

    fresh_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username="owner654",
            text="/fresh",
        )
    )
    assert fresh_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == owner_chat_id
            for item in e2e_harness.telegram_client.sent_messages
        )
    )

    start_index = len(e2e_harness.telegram_client.sent_messages)
    live_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username="owner654",
            text=(
                "Найди в вебе свежую информацию про тестовый запрос "
                "«OpenTulpa Telegram interactive owner update». Перед тем как начнешь поиск, "
                "отправь мне отдельное короткое сообщение, что проверяешь источники. "
                "Потом сделай финальный краткий вывод по найденному."
            ),
        )
    )
    assert live_status == 200

    def owner_messages() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.telegram_client.sent_messages[start_index:]
            if int(item.get("chat_id", 0)) == owner_chat_id
            and not str(item.get("business_connection_id", "") or "").strip()
        ]

    assert _wait_until(lambda: len(owner_messages()) >= 2, timeout_seconds=180.0), owner_messages()
    assert _wait_until(lambda: bool(search_queries), timeout_seconds=5.0)

    messages = owner_messages()
    first_text = str(messages[0].get("text", "") or "").strip().lower()
    final_text = str(messages[-1].get("text", "") or "").strip()
    internal_calls = e2e_harness.internal_api_calls_since(internal_start)
    behavior = load_jsonl(e2e_harness.behavior_log_path)

    assert any(item.get("path") == "/internal/web_search" for item in internal_calls)
    assert any(str(item.get("event", "")) == "interactive_owner_update_sent" for item in behavior)
    assert any(marker in first_text for marker in ("провер", "ищ", "смотр", "нач"))
    assert final_text
    assert final_text != str(messages[0].get("text", "") or "").strip()

    e2e_harness.recorder.add(
        "live_owner_update_search_e2e",
        owner_chat_id=owner_chat_id,
        owner_message_count=len(messages),
        owner_messages=[item.get("text") for item in messages],
        search_queries=search_queries,
        internal_web_search_calls=[
            item for item in internal_calls if item.get("path") == "/internal/web_search"
        ],
    )


@pytest.mark.live_llm
def test_live_telegram_interactive_chat_remembers_and_honors_style_preference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = get_settings()
    if not str(settings.openai_compatible_api_key or "").strip():
        pytest.skip("OPENAI_COMPATIBLE_API_KEY (or OPENROUTER_API_KEY) required for live LLM e2e")

    memory = _RecordingMemory()
    harness = build_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario_name="interactive_style_memory",
        memory_service=memory,
    )
    owner_user_id = 777
    owner_chat_id = 1777

    def owner_messages_since(start_index: int) -> list[dict[str, Any]]:
        return [
            item
            for item in harness.telegram_client.sent_messages[start_index:]
            if int(item.get("chat_id", 0)) == owner_chat_id
            and not str(item.get("business_connection_id", "") or "").strip()
        ]

    try:
        fresh_status = harness.post_telegram(
            body=_telegram_message(
                chat_id=owner_chat_id,
                user_id=owner_user_id,
                username="owner777",
                text="/fresh",
            )
        )
        assert fresh_status == 200
        assert _wait_until(
            lambda: any(
                int(item.get("chat_id", 0)) == owner_chat_id
                for item in harness.telegram_client.sent_messages
            ),
            timeout_seconds=30.0,
        )

        internal_start = harness.count_internal_api_calls()
        style_start = len(harness.telegram_client.sent_messages)
        style_status = harness.post_telegram(
            body=_telegram_message(
                chat_id=owner_chat_id,
                user_id=owner_user_id,
                username="owner777",
                text=(
                    "Remember this as my normal Telegram chat writing style: write naturally. "
                    "Do not make answers look like Markdown documents. No headings, no horizontal "
                    "rules, no bold or italic marker style. Lists are okay when they fit naturally. "
                    "Reply with one short sentence after you save it."
                ),
            )
        )
        assert style_status == 200
        assert _wait_until(
            lambda: bool(owner_messages_since(style_start)) and bool(memory.entries),
            timeout_seconds=180.0,
        ), {
            "owner_messages": owner_messages_since(style_start),
            "memory_entries": memory.entries,
            "internal_calls": harness.internal_api_calls_since(internal_start),
        }

        stored_text = "\n".join(str(item.get("text", "") or "") for item in memory.entries)
        assert "Markdown" in stored_text or "markdown" in stored_text
        assert "natur" in stored_text.lower() or "естествен" in stored_text.lower()
        assert any(
            item.get("path") in {"/internal/memory/add", "/internal/directive/set"}
            for item in harness.internal_api_calls_since(internal_start)
        )

        second_internal_start = harness.count_internal_api_calls()
        second_start = len(harness.telegram_client.sent_messages)
        second_status = harness.post_telegram(
            body=_telegram_message(
                chat_id=owner_chat_id,
                user_id=owner_user_id,
                username="owner777",
                text=(
                    "Назови три идеи TikTok роликов для dark feminine personal brand. "
                    "Ответь по-русски, коротко, без объяснения моих стилевых правил."
                ),
            )
        )
        assert second_status == 200
        assert _wait_until(
            lambda: bool(owner_messages_since(second_start)),
            timeout_seconds=180.0,
        ), {
            "owner_messages": owner_messages_since(second_start),
            "memory_entries": memory.entries,
            "memory_searches": memory.searches,
        }

        reply_text = str(owner_messages_since(second_start)[-1].get("text", "") or "").strip()
        second_internal_calls = harness.internal_api_calls_since(second_internal_start)
        assert reply_text
        assert any(item.get("path") == "/internal/memory/search" for item in second_internal_calls)
        assert memory.searches
        assert not re.search(r"(?m)^\s*#{1,6}\s+", reply_text)
        assert not re.search(r"(?m)^\s*[-*_]{3,}\s*$", reply_text)
        assert "**" not in reply_text
        assert "__" not in reply_text

        harness.recorder.add(
            "live_interactive_style_memory_e2e",
            owner_chat_id=owner_chat_id,
            stored_memories=memory.entries,
            memory_searches=memory.searches,
            reply_text=reply_text,
            internal_memory_calls=[
                item
                for item in harness.internal_api_calls_since(internal_start)
                if item.get("path") in {"/internal/memory/add", "/internal/memory/search"}
            ],
        )
    finally:
        close_harness(harness)
