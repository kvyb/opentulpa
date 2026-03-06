"""
Slack API client for the agent (list channels, read history, post messages).

Uses SLACK_BOT_TOKEN. Required OAuth scopes (bot):
  channels:read, channels:history, chat:write, groups:read, groups:history (for private),
  im:read, im:history, mpim:read, mpim:history (for DMs) if you need them.

Write consent: posting is disallowed until the user grants permission in chat.
Consent is stored per customer_id; grant/check/revoke via the functions below.
"""

import logging
from typing import Any

from slack_sdk.web.async_client import AsyncWebClient

logger = logging.getLogger(__name__)

# In-memory set of customer_ids who have granted Slack write permission.
_slack_write_consent: set[str] = set()


def grant_slack_write_consent(customer_id: str) -> None:
    """Record that this user has allowed the agent to post to Slack."""
    _slack_write_consent.add(str(customer_id))


def has_slack_write_consent(customer_id: str) -> bool:
    """Return True if the user has granted permission to post to Slack."""
    return str(customer_id) in _slack_write_consent


def revoke_slack_write_consent(customer_id: str) -> None:
    """Revoke Slack write consent for this user."""
    _slack_write_consent.discard(str(customer_id))


class SlackClient:
    """Async Slack client for listing channels, reading history, and posting."""

    def __init__(self, token: str) -> None:
        self._client = AsyncWebClient(token=token)

    async def list_channels(
        self,
        *,
        types: str = "public_channel",
        exclude_archived: bool = True,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List channels the bot is in or all public channels (depending on scope)."""
        try:
            out = await self._client.conversations_list(
                types=types,
                exclude_archived=exclude_archived,
                limit=min(limit, 200),
                cursor=cursor,
            )
            if not out.get("ok"):
                return {"ok": False, "error": out.get("error", "unknown"), "channels": []}
            return {
                "ok": True,
                "channels": [
                    {
                        "id": c["id"],
                        "name": c.get("name", ""),
                        "is_channel": c.get("is_channel", True),
                    }
                    for c in out.get("channels", [])
                ],
                "next_cursor": out.get("response_metadata", {}).get("next_cursor") or "",
            }
        except Exception as e:
            logger.exception("Slack conversations_list failed: %s", e)
            return {"ok": False, "error": str(e), "channels": []}

    async def channel_history(
        self,
        channel_id: str,
        *,
        limit: int = 20,
        oldest: str | None = None,
        latest: str | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Get message history for a channel."""
        try:
            kwargs: dict[str, Any] = {"channel": channel_id, "limit": min(limit, 200)}
            if oldest:
                kwargs["oldest"] = oldest
            if latest:
                kwargs["latest"] = latest
            if cursor:
                kwargs["cursor"] = cursor
            out = await self._client.conversations_history(**kwargs)
            if not out.get("ok"):
                return {"ok": False, "error": out.get("error", "unknown"), "messages": []}
            messages = []
            for m in out.get("messages", []):
                messages.append(
                    {
                        "ts": m.get("ts"),
                        "user": m.get("user", ""),
                        "text": m.get("text", ""),
                        "thread_ts": m.get("thread_ts"),
                    }
                )
            return {
                "ok": True,
                "messages": messages,
                "next_cursor": out.get("response_metadata", {}).get("next_cursor") or "",
            }
        except Exception as e:
            logger.exception("Slack conversations_history failed: %s", e)
            return {"ok": False, "error": str(e), "messages": []}

    async def post_message(
        self,
        channel_id: str,
        text: str,
        *,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Post a message to a channel (or thread if thread_ts provided)."""
        try:
            kwargs: dict[str, Any] = {"channel": channel_id, "text": text}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            out = await self._client.chat_postMessage(**kwargs)
            if not out.get("ok"):
                return {"ok": False, "error": out.get("error", "unknown"), "ts": None}
            return {"ok": True, "ts": out.get("ts")}
        except Exception as e:
            logger.exception("Slack chat_postMessage failed: %s", e)
            return {"ok": False, "error": str(e), "ts": None}
