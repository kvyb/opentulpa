from __future__ import annotations

import json

import httpx
import pytest

from opentulpa.capability_workers.telegram_api import (
    TelegramAPIError,
    TelegramAttachment,
    TelegramBotAPI,
    extract_attachments,
)


def test_extract_attachments_uses_largest_photo_and_document_metadata() -> None:
    attachments = extract_attachments(
        {
            "document": {
                "file_id": "doc",
                "file_name": "notes.txt",
                "mime_type": "text/plain",
                "file_size": 12,
            },
            "photo": [
                {"file_id": "small", "file_size": 10},
                {"file_id": "large", "file_unique_id": "photo-1", "file_size": 20},
            ],
        }
    )

    assert [(item.kind, item.file_id) for item in attachments] == [
        ("document", "doc"),
        ("photo", "large"),
    ]
    assert attachments[1].filename == "photo-1.jpg"


@pytest.mark.asyncio
async def test_bot_api_explicitly_disables_webhook_without_dropping_updates() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": True})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TelegramBotAPI(token="private-token", client=http)

    await client.delete_webhook()

    assert requests[0].url.path.endswith("/deleteWebhook")
    assert json.loads(requests[0].content) == {"drop_pending_updates": False}
    await http.aclose()


@pytest.mark.asyncio
async def test_bot_api_registers_native_slash_commands() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": True})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TelegramBotAPI(token="private-token", client=http)

    await client.set_my_commands(
        [{"command": "model", "description": "Show or select the model"}]
    )

    assert requests[0].url.path.endswith("/setMyCommands")
    assert json.loads(requests[0].content) == {
        "commands": [
            {"command": "model", "description": "Show or select the model"}
        ]
    }
    await http.aclose()


@pytest.mark.asyncio
async def test_bot_api_renders_markdown_as_telegram_html() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/sendMessage"):
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 12}})
        return httpx.Response(200, json={"ok": True, "result": True})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TelegramBotAPI(token="private-token", client=http)

    assert await client.send_message(
        chat_id=7,
        text="**Title**\n\n- Item",
    ) == [12]
    await client.edit_message_text(
        chat_id=7,
        message_id=12,
        text="## Update\n`code`",
    )

    sent = json.loads(requests[0].content)
    assert sent["parse_mode"] == "HTML"
    assert sent["text"] == "<b>Title</b>\n\n• Item"
    edited = json.loads(requests[1].content)
    assert edited["parse_mode"] == "HTML"
    assert edited["text"] == "<b>Update</b>\n<code>code</code>"
    await http.aclose()


@pytest.mark.asyncio
async def test_bot_api_falls_back_to_plain_text_only_for_invalid_markup() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "ok": False,
                    "description": "Bad Request: can't parse entities",
                },
            )
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 12}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TelegramBotAPI(token="private-token", client=http)

    assert await client.send_message(chat_id=7, text="**Title**") == [12]

    formatted = json.loads(requests[0].content)
    assert formatted["parse_mode"] == "HTML"
    assert formatted["text"] == "<b>Title</b>"
    fallback = json.loads(requests[1].content)
    assert "parse_mode" not in fallback
    assert fallback["text"] == "Title"
    await http.aclose()


@pytest.mark.asyncio
async def test_bot_api_long_poll_sorts_updates_and_never_exposes_token_in_error() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(
                200,
                json={"ok": True, "result": [{"update_id": 2}, {"update_id": 1}]},
            )
        return httpx.Response(401, json={"ok": False, "description": "private-token"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TelegramBotAPI(token="private-token", client=http)
    updates = await client.get_updates(offset=1, timeout_seconds=2)

    assert [item["update_id"] for item in updates] == [1, 2]
    assert json.loads(requests[0].content)["offset"] == 1
    with pytest.raises(TelegramAPIError) as captured:
        await client.get_me()
    assert "private-token" not in str(captured.value)
    await http.aclose()


@pytest.mark.asyncio
async def test_attachment_download_rejects_declared_oversize_before_network() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TelegramBotAPI(token="private-token", client=http, max_attachment_bytes=100)
    attachment = TelegramAttachment(
        kind="document",
        file_id="file_1",
        filename="large.bin",
        mime_type=None,
        declared_size=101,
    )

    with pytest.raises(TelegramAPIError, match="size limit"):
        await client.download_attachment(attachment)
    assert calls == 0
    await http.aclose()
