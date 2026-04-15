from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FakeComposioInstagramService:
    """Transport-only fake for Instagram read/write calls."""

    enabled: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)
    conversations: dict[str, dict[str, Any]] = field(default_factory=dict)
    reply_fail_once_for_invalid_mid: bool = True

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "enabled": True,
            "callback_url_configured": True,
            "default_callback_url": "https://example.test/callback",
            "resolved_callback_url": "https://example.test/callback",
        }

    def list_instagram_conversations(
        self,
        *,
        customer_id: str,
        connected_account_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "list_instagram_conversations",
                "customer_id": customer_id,
                "connected_account_id": connected_account_id,
                "conversation_id": conversation_id,
                "limit": limit,
            }
        )
        if conversation_id and conversation_id in self.conversations:
            items = [self.conversations[conversation_id]["summary"]]
        else:
            items = [v["summary"] for v in self.conversations.values()]
        return {
            "ok": True,
            "customer_id": customer_id,
            "items": items[: max(1, int(limit))],
            "next_cursor": None,
        }

    def get_instagram_conversation(
        self,
        *,
        customer_id: str,
        conversation_id: str,
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "get_instagram_conversation",
                "customer_id": customer_id,
                "conversation_id": conversation_id,
                "connected_account_id": connected_account_id,
            }
        )
        item = self.conversations.get(conversation_id)
        if not item:
            return {
                "ok": False,
                "error": f"conversation not found: {conversation_id}",
            }
        return {
            "ok": True,
            "summary": item["summary"],
            "conversation": item["conversation"],
        }

    def inspect_instagram_reply_target(
        self,
        *,
        customer_id: str,
        recipient_id: str | None = None,
        conversation_id: str | None = None,
        connected_account_id: str | None = None,
        scan_limit: int = 10,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "inspect_instagram_reply_target",
                "customer_id": customer_id,
                "recipient_id": recipient_id,
                "conversation_id": conversation_id,
                "connected_account_id": connected_account_id,
                "scan_limit": scan_limit,
            }
        )
        cid = str(conversation_id or next(iter(self.conversations.keys()), "conv_e2e_1"))
        summary = self.conversations.get(cid, {}).get("summary", {})
        return {
            "ok": True,
            "matched": True,
            "recipient_id_verified": True,
            "recipient_id": str(recipient_id or summary.get("recipient_id", "1789")),
            "conversation_id": cid,
            "latest_inbound_message_created_time": str(
                summary.get("latest_message_created_time", datetime.now(timezone.utc).isoformat())
            ),
            "reply_window_status": "unconfirmed",
        }

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        args = dict(arguments or {})
        self.calls.append(
            {
                "method": "execute_tool",
                "customer_id": customer_id,
                "tool_slug": tool_slug,
                "arguments": dict(args),
                "connected_account_id": connected_account_id,
                "text": text,
            }
        )
        if (
            str(tool_slug or "").upper() == "INSTAGRAM_SEND_TEXT_MESSAGE"
            and self.reply_fail_once_for_invalid_mid
            and str(args.get("reply_to_message_id", "")).strip()
        ):
            self.reply_fail_once_for_invalid_mid = False
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
            "data": {
                "id": "mid_e2e_sent_1",
                "echo_text": text or args.get("text", ""),
                "retried_without_reply_to_message_id": "reply_to_message_id" not in args,
            },
        }


def build_instagram_conversation(
    *,
    conversation_id: str,
    recipient_id: str,
    inbound_text: str,
) -> dict[str, Any]:
    created = "2026-04-14T10:20:30+0000"
    summary = {
        "conversation_id": conversation_id,
        "recipient_id": recipient_id,
        "latest_message_id": "mid_latest_in_1",
        "latest_message_created_time": created,
        "message_count": 2,
    }
    conversation = {
        "id": conversation_id,
        "messages": {
            "data": [
                {
                    "id": "mid_prev_out_1",
                    "created_time": "2026-04-13T10:20:30+0000",
                    "from": {"id": "page_1", "username": "biz_account"},
                    "to": {"data": [{"id": recipient_id}]},
                    "message": "Hi! How can I help you?",
                },
                {
                    "id": "mid_latest_in_1",
                    "created_time": created,
                    "from": {"id": recipient_id, "username": "customer_1"},
                    "to": {"data": [{"id": "page_1"}]},
                    "message": inbound_text,
                },
            ]
        },
    }
    return {"summary": summary, "conversation": conversation}
