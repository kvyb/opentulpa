"""Telegram adapter for approval challenges."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime

from opentulpa.approvals.models import ApprovalRecord
from opentulpa.interfaces.telegram.client import TelegramClient


class TelegramApprovalAdapter:
    name = "telegram"
    interactive = True

    def __init__(self, *, client: TelegramClient) -> None:
        self._client = client
        self._lock = asyncio.Lock()
        self._pending_by_chat: defaultdict[str, list[ApprovalRecord]] = defaultdict(list)

    @staticmethod
    def _format_expiry(expires_at: str) -> str:
        try:
            dt = datetime.fromisoformat(expires_at)
            return dt.strftime("%H:%M:%S")
        except Exception:
            return "soon"

    @staticmethod
    def _action_preview(approval: ApprovalRecord) -> str:
        summary = str(approval.summary or "").strip() or f"execute {approval.action_name}"
        try:
            args = json.loads(str(approval.action_args_json or "{}"))
        except Exception:
            args = {}
        if not isinstance(args, dict):
            args = {}

        action_name = str(approval.action_name or "").strip()
        preview = summary
        if action_name == "browser_use_run":
            task_text = str(args.get("task", "")).strip()
            if task_text:
                preview = f"run browser task:\n{task_text}"
        elif action_name == "tulpa_run_terminal":
            cmd = str(args.get("command", "")).strip()
            if cmd:
                preview = f"run terminal command:\n{cmd}"

        # Keep message below Telegram hard limits while avoiding aggressive truncation.
        max_chars = 2400
        if len(preview) > max_chars:
            preview = preview[: max_chars - 3].rstrip() + "..."
        return preview

    @classmethod
    def _render_challenge(cls, approval: ApprovalRecord) -> tuple[str, dict[str, list[list[dict[str, str]]]]]:
        expiry = cls._format_expiry(approval.expires_at)
        action_preview = cls._action_preview(approval)
        text = (
            "Approval needed for external-impact action.\n\n"
            f"Action: {action_preview}\n"
            f"Destination: {approval.recipient_scope}\n"
            f"Risk: {approval.impact_type}\n"
            f"Expires: {expiry}\n\n"
            f"ID: {approval.id}"
        )
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "Approve", "callback_data": f"approval:{approval.id}:approve"},
                    {"text": "Deny", "callback_data": f"approval:{approval.id}:deny"},
                ]
            ]
        }
        return text, reply_markup

    async def send_challenge(self, approval: ApprovalRecord) -> bool:
        chat_id = str(approval.origin_conversation_id or "").strip()
        if not chat_id:
            return False
        text, reply_markup = self._render_challenge(approval)
        return bool(await self._client.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        ))

    async def queue_challenge(self, approval: ApprovalRecord) -> bool:
        chat_id = str(approval.origin_conversation_id or "").strip()
        if not chat_id:
            return False
        async with self._lock:
            self._pending_by_chat[chat_id].append(approval)
        return True

    async def flush_challenges(self, *, chat_id: str | int, limit: int | None = None) -> int:
        chat_key = str(chat_id).strip()
        if not chat_key:
            return 0
        safe_limit = max(1, int(limit)) if limit is not None else None
        async with self._lock:
            queued = list(self._pending_by_chat.get(chat_key) or [])
            if not queued:
                return 0
            if safe_limit is not None:
                to_send = queued[:safe_limit]
                remaining = queued[safe_limit:]
            else:
                to_send = queued
                remaining = []
            if remaining:
                self._pending_by_chat[chat_key] = remaining
            else:
                self._pending_by_chat.pop(chat_key, None)

        delivered = 0
        undelivered: list[ApprovalRecord] = []
        for approval in to_send:
            if await self.send_challenge(approval):
                delivered += 1
            else:
                undelivered.append(approval)

        if undelivered:
            async with self._lock:
                prior = self._pending_by_chat.get(chat_key) or []
                self._pending_by_chat[chat_key] = [*undelivered, *prior]
        return delivered
