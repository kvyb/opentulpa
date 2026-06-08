from __future__ import annotations

import httpx
import pytest

from opentulpa.interfaces.telegram import client as telegram_client_module
from opentulpa.interfaces.telegram.client import TelegramClient


class _AlwaysTimeoutClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        raise httpx.ReadTimeout("timeout")


class _RetryThenSuccessClient:
    attempts = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        _RetryThenSuccessClient.attempts += 1
        if _RetryThenSuccessClient.attempts < 3:
            raise httpx.TransportError("temporary transport failure")
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 123}})


class _NotModifiedEditClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return httpx.Response(
            400,
            json={
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: message is not modified",
            },
        )


class _RetryAfterThenSuccessClient:
    attempts = 0

    async def post(self, *args, **kwargs):
        del args, kwargs
        _RetryAfterThenSuccessClient.attempts += 1
        if _RetryAfterThenSuccessClient.attempts == 1:
            return httpx.Response(
                429,
                json={
                    "ok": False,
                    "error_code": 429,
                    "description": "Too Many Requests: retry after 3",
                    "parameters": {"retry_after": 3},
                },
            )
        return httpx.Response(200, json={"ok": True, "result": True})


class _RecordingSendClient:
    payloads: list[dict]

    def __init__(self):
        self.payloads = []

    async def post(self, _url, *, json, timeout):
        del timeout
        self.payloads.append(dict(json))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(self.payloads)}})


class _FailSecondSendClient:
    payloads: list[dict]

    def __init__(self):
        self.payloads = []

    async def post(self, _url, *, json, timeout):
        del timeout
        self.payloads.append(dict(json))
        if len(self.payloads) >= 2:
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(self.payloads)}})


@pytest.mark.asyncio
async def test_post_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telegram_client_module.httpx, "AsyncClient", _AlwaysTimeoutClient)
    tg = TelegramClient("dummy")
    result = await tg._post("sendMessage", {"chat_id": 1, "text": "hello"})
    assert result is None


@pytest.mark.asyncio
async def test_post_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _RetryThenSuccessClient.attempts = 0
    monkeypatch.setattr(telegram_client_module.httpx, "AsyncClient", _RetryThenSuccessClient)
    tg = TelegramClient("dummy")
    result = await tg._post("sendMessage", {"chat_id": 1, "text": "hello"})
    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert _RetryThenSuccessClient.attempts == 3


@pytest.mark.asyncio
async def test_post_treats_not_modified_edit_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telegram_client_module.httpx, "AsyncClient", _NotModifiedEditClient)
    tg = TelegramClient("dummy")
    result = await tg._post("editMessageText", {"chat_id": 1, "message_id": 10, "text": "..."})
    assert isinstance(result, dict)
    assert result.get("ok") is True


@pytest.mark.asyncio
async def test_post_honors_retry_after_for_send_chat_action_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    _RetryAfterThenSuccessClient.attempts = 0
    monkeypatch.setattr(telegram_client_module.httpx, "AsyncClient", _RetryAfterThenSuccessClient)
    monkeypatch.setattr(telegram_client_module.asyncio, "sleep", _fake_sleep)
    tg = TelegramClient("dummy")

    result = await tg._post("sendChatAction", {"chat_id": 1, "action": "typing"})

    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert sleeps == [3.0]
    assert _RetryAfterThenSuccessClient.attempts == 2


@pytest.mark.asyncio
async def test_send_message_splits_long_markdown_into_formatted_messages() -> None:
    recorder = _RecordingSendClient()
    tg = TelegramClient("dummy")
    tg._client = recorder
    text = (
        "## ВОПРОС 1\n\n"
        "**1. «I retired my nice girl era» — Before/After**\n"
        "- Хук: резкий переход в чёрное платье\n"
        '- Титр: *"I retired my nice girl era"*\n'
        "- Почему вирусный: трансформация хорошо репостится\n\n"
    ) * 80

    result = await tg.send_message(chat_id=1, text=text, parse_mode="HTML")

    assert isinstance(result, dict)
    assert isinstance(result.get("results"), list)
    assert len(result["results"]) == len(recorder.payloads)
    assert len(recorder.payloads) > 1
    assert all(payload.get("parse_mode") == "HTML" for payload in recorder.payloads)
    assert all(len(str(payload.get("text", ""))) <= 3800 for payload in recorder.payloads)
    assert all("## " not in str(payload.get("text", "")) for payload in recorder.payloads)
    assert all("**" not in str(payload.get("text", "")) for payload in recorder.payloads)
    assert all(
        "[Truncated to fit Telegram.]" not in str(payload.get("text", ""))
        for payload in recorder.payloads
    )


@pytest.mark.asyncio
async def test_send_message_reports_failure_when_later_split_chunk_fails() -> None:
    recorder = _FailSecondSendClient()
    tg = TelegramClient("dummy")
    tg._client = recorder
    text = "## Heading\n\n**Important**\n- item\n\n" * 500

    result = await tg.send_message(chat_id=1, text=text, parse_mode="HTML")

    assert result is None
    assert len(recorder.payloads) >= 2
