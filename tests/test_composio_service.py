from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import opentulpa.integrations.composio as composio_module
from opentulpa.integrations.composio import ComposioService


class _FakeComposioService(ComposioService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.calls: list[dict[str, object]] = []

    def inspect_instagram_reply_target(self, **kwargs):  # type: ignore[override]
        return {
            "ok": True,
            "matched": True,
            "recipient_id_verified": True,
            "recipient_id": kwargs.get("recipient_id"),
            "conversation_id": "conv_1",
            "latest_inbound_message_created_time": "2026-04-06T11:46:16+0000",
            "reply_window_status": "unconfirmed",
        }

    def _sdk_execute_tool(self, *, slug, arguments, connected_account_id, user_id, text=None):  # type: ignore[override]
        self.calls.append(
            {
                "slug": slug,
                "arguments": dict(arguments),
                "connected_account_id": connected_account_id,
                "user_id": user_id,
                "text": text,
            }
        )
        if len(self.calls) == 1:
            return {
                "successful": False,
                "error": (
                    'Failed to send message (status 400). Response: {"error":{"message":"Invalid parameter",'
                    '"error_user_title":"Invalid Message ID","error_subcode":2534002}}'
                ),
                "data": {"status_code": 400},
            }
        return {
            "successful": True,
            "error": None,
            "data": {"id": "mid_2"},
        }


class _FakeGoogleSheetsComposioService(ComposioService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.calls: list[dict[str, object]] = []

    def search_tools(self, *, query="", toolkits=None, limit=20):  # type: ignore[override]
        self.calls.append(
            {
                "method": "search_tools",
                "query": query,
                "toolkits": list(toolkits or []),
                "limit": limit,
            }
        )
        return {
            "ok": True,
            "items": [
                {
                    "slug": "GOOGLESHEETS_GET_SHEET_NAMES",
                    "toolkit_slug": "googlesheets",
                }
            ],
        }

    def execute_tool(  # type: ignore[override]
        self,
        *,
        customer_id,
        tool_slug,
        arguments=None,
        connected_account_id=None,
        text=None,
    ):
        self.calls.append(
            {
                "method": "execute_tool",
                "customer_id": customer_id,
                "tool_slug": tool_slug,
                "arguments": dict(arguments or {}),
                "connected_account_id": connected_account_id,
                "text": text,
            }
        )
        return {
            "successful": True,
            "error": None,
            "data": {"sheet_names": ["Заявки", "Архив", "Заявки"]},
        }


class _PartialInstagramComposioService(ComposioService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.calls: list[dict[str, object]] = []

    def _sdk_execute_tool(self, *, slug, arguments, connected_account_id, user_id, text=None):  # type: ignore[override]
        self.calls.append(
            {
                "slug": slug,
                "arguments": dict(arguments),
                "connected_account_id": connected_account_id,
                "user_id": user_id,
                "text": text,
            }
        )
        if slug == "INSTAGRAM_LIST_ALL_CONVERSATIONS":
            return {
                "successful": True,
                "error": None,
                "data": {"data": [{"id": "conv_ok"}, {"id": "conv_bad"}]},
            }
        if arguments.get("conversation_id") == "conv_bad":
            return {"successful": False, "error": "Unsupported get request", "data": {}}
        return {
            "successful": True,
            "error": None,
            "data": {
                "id": "conv_ok",
                "participants": {
                    "data": [
                        {"id": "business_1", "username": "biz"},
                        {"id": "lead_1", "username": "lead"},
                    ]
                },
                "messages": {
                    "data": [
                        {
                            "id": "msg_1",
                            "created_time": "2026-05-13T08:00:00+0000",
                            "from": {"id": "lead_1", "username": "lead"},
                            "to": {"data": [{"id": "business_1", "username": "biz"}]},
                            "message": "Hello",
                        }
                    ]
                },
                "updated_time": "2026-05-13T08:00:00+0000",
            },
        }


def test_composio_hot_loads_and_rotates_a_vault_backed_api_key(monkeypatch) -> None:
    keys: list[str] = []
    active = {"value": ""}

    class _Client:
        def __init__(self, *, api_key: str, provider: Any) -> None:
            del provider
            keys.append(api_key)

    monkeypatch.setattr(
        composio_module,
        "_load_composio_sdk",
        lambda: (_Client, lambda: object()),
    )
    service = ComposioService(
        api_key="",
        api_key_provider=lambda: active["value"],
    )

    assert service.enabled is False
    active["value"] = "first-composio-key"
    first = service._sdk()  # noqa: SLF001
    assert service.enabled is True
    assert service._sdk() is first  # noqa: SLF001

    active["value"] = "rotated-composio-key"
    second = service._sdk()  # noqa: SLF001

    assert second is not first
    assert keys == ["first-composio-key", "rotated-composio-key"]


def test_list_toolkits_clamps_provider_limit_to_fifty(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class _Session:
        def toolkits(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(items=[], next_cursor=None, total_pages=0)

    monkeypatch.setattr(
        ComposioService,
        "_session",
        lambda self, **kwargs: _Session(),
    )
    service = ComposioService(api_key="test-key")

    result = service.list_toolkits(customer_id="tenant-1", limit=100)

    assert result["ok"] is True
    assert calls == [
        {
            "toolkits": None,
            "is_connected": None,
            "limit": 50,
            "search": None,
        }
    ]


def test_composio_proxy_uses_connected_account_without_exposing_headers(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def proxy(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            status=201,
            data={"sha": "a" * 40},
            headers={"authorization": "Bearer provider-secret"},
        )

    monkeypatch.setattr(
        ComposioService,
        "_sdk",
        lambda self: SimpleNamespace(tools=SimpleNamespace(proxy=proxy)),
    )
    service = ComposioService(api_key="test-key")

    result = service.proxy_tool(
        endpoint="https://api.github.com/repos/acme/project/git/blobs",
        method="POST",
        connected_account_id="connection-1",
        body={"content": "ZGF0YQ==", "encoding": "base64"},
    )

    assert result == {"status": 201, "data": {"sha": "a" * 40}}
    assert calls == [
        {
            "endpoint": "https://api.github.com/repos/acme/project/git/blobs",
            "method": "POST",
            "body": {"content": "ZGF0YQ==", "encoding": "base64"},
            "connected_account_id": "connection-1",
        }
    ]
    assert "provider-secret" not in repr(result)


def test_instagram_send_retries_without_reply_to_message_id_on_invalid_mid() -> None:
    service = _FakeComposioService(api_key="test-key")

    result = service.execute_tool(
        customer_id="telegram_1",
        tool_slug="INSTAGRAM_SEND_TEXT_MESSAGE",
        arguments={
            "recipient_id": "rcp_1",
            "text": "hello",
            "reply_to_message_id": "mid_1",
        },
        connected_account_id="acct_1",
    )

    assert result["successful"] is True
    assert len(service.calls) == 2
    assert service.calls[0]["arguments"]["reply_to_message_id"] == "mid_1"
    assert "reply_to_message_id" not in service.calls[1]["arguments"]
    assert result["data"]["retried_without_reply_to_message_id"] is True


def test_list_instagram_conversations_skips_unreadable_threads() -> None:
    service = _PartialInstagramComposioService(api_key="test-key")

    result = service.list_instagram_conversations(
        customer_id="telegram_1",
        connected_account_id="acct_1",
    )

    assert result["ok"] is True
    assert [item["conversation_id"] for item in result["items"]] == ["conv_ok"]
    assert result["warnings"] == [
        {"conversation_id": "conv_bad", "error": "Unsupported get request"}
    ]


def test_list_google_sheets_tab_names_uses_composio_sheet_discovery_tool() -> None:
    service = _FakeGoogleSheetsComposioService(api_key="test-key")

    result = service.list_google_sheets_tab_names(
        customer_id="telegram_1",
        spreadsheet_id="sheet_123",
        connected_account_id="acct_1",
    )

    assert result == {
        "ok": True,
        "spreadsheet_id": "sheet_123",
        "sheet_names": ["Заявки", "Архив"],
        "tool_slug": "GOOGLESHEETS_GET_SHEET_NAMES",
    }
    execute_calls = [call for call in service.calls if call["method"] == "execute_tool"]
    assert execute_calls[0]["arguments"] == {"spreadsheetId": "sheet_123"}
