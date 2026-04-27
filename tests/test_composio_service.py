from __future__ import annotations

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
