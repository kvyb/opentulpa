"""Tenant owner delivery through durable Telegram session bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from opentulpa.deep_agent import AgentApproval
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.state_store import TelegramStateStore


class TelegramApprovalDelivery(Protocol):
    async def __call__(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        run_id: str,
        approval: AgentApproval,
    ) -> None: ...


class TelegramOwnerDelivery:
    def __init__(
        self,
        *,
        client: TelegramClient,
        state: TelegramStateStore,
        deliver_approval: TelegramApprovalDelivery,
    ) -> None:
        self._client = client
        self._state = state
        self._deliver_approval = deliver_approval

    def _chat_id(self, tenant_id: str) -> int | None:
        slots = self._state.find_session_slots(str(tenant_id or "").strip())
        for slot in slots:
            try:
                return int(slot["chat_id"])
            except (KeyError, TypeError, ValueError):
                continue
        return None

    async def deliver_text(
        self,
        *,
        tenant_id: str,
        title: str,
        text: str,
        run_id: str | None,
        approval_required: bool,
        approvals: tuple[AgentApproval, ...],
    ) -> None:
        chat_id = self._chat_id(tenant_id)
        if chat_id is None:
            return
        await self._client.send_message(
            chat_id=chat_id,
            text=f"{title}\n\n{text}".strip(),
            parse_mode=None,
        )
        self._state.touch_assistant_message(chat_id)
        if not approval_required or not run_id:
            return
        for approval in approvals:
            await self._deliver_approval(
                chat_id=chat_id,
                tenant_id=tenant_id,
                run_id=run_id,
                approval=approval,
            )

    async def deliver_artifact(
        self,
        *,
        tenant_id: str,
        path: Path,
        filename: str,
        media_type: str | None = None,
        caption: str | None = None,
    ) -> dict[str, Any]:
        chat_id = self._chat_id(tenant_id)
        if chat_id is None:
            raise RuntimeError("tenant has no owner delivery channel")
        safe_path = path.expanduser().resolve()
        raw = safe_path.read_bytes()
        sent = await self._client.send_file(
            chat_id=chat_id,
            filename=str(filename or safe_path.name),
            raw_bytes=raw,
            mime_type=media_type,
            caption=caption,
        )
        if not sent:
            raise RuntimeError("artifact delivery failed")
        self._state.touch_assistant_message(chat_id)
        return {"channel": "telegram", "chat_id": str(chat_id), "delivered": True}


__all__ = ["TelegramApprovalDelivery", "TelegramOwnerDelivery"]
