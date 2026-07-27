"""Telegram Business webhook ingress, separate from the owner interface worker."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)
_BUSINESS_UPDATE_KEYS = frozenset(
    {
        "business_connection",
        "business_message",
        "edited_business_message",
        "deleted_business_messages",
    }
)


class TelegramBusinessServicePort(Protocol):
    def ingest_update(self, body: dict[str, Any]) -> dict[str, Any]: ...

    def complete_ingress(self, ingress_key: str) -> None: ...


class TelegramBusinessWorkflowPort(Protocol):
    def list_workflows(
        self,
        *,
        customer_id: str,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]: ...

    def _source_matches_workflow(
        self,
        *,
        workflow: dict[str, Any],
        business_connection_id: str,
        conversation_id: str,
    ) -> bool: ...

    async def enqueue_telegram_business_workflow_run(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str,
        owner_chat_id: str,
        event_type: str,
    ) -> dict[str, Any]: ...

    async def drain_due_pending_runs(self, *, limit: int = 10) -> int: ...


@dataclass(frozen=True, slots=True)
class AcceptedTelegramBusinessUpdate:
    result: dict[str, Any]


class TelegramBusinessRelay:
    """Durably ingest Telegram Business updates and enqueue matching intake."""

    def __init__(
        self,
        *,
        business: TelegramBusinessServicePort,
        workflows: TelegramBusinessWorkflowPort,
    ) -> None:
        self._business = business
        self._workflows = workflows

    async def accept_update(
        self,
        body: dict[str, Any],
    ) -> AcceptedTelegramBusinessUpdate:
        if not any(key in body for key in _BUSINESS_UPDATE_KEYS):
            return AcceptedTelegramBusinessUpdate(result={"handled": False})
        result = self._business.ingest_update(dict(body))
        if bool(result.get("trigger_workflows")) and bool(result.get("dispatch_pending")):
            await self._enqueue_intake(result)
            self._business.complete_ingress(str(result.get("ingress_key") or "").strip())
            result = {**result, "dispatch_pending": False}
        return AcceptedTelegramBusinessUpdate(result=result)

    async def process_update(self, accepted: AcceptedTelegramBusinessUpdate) -> None:
        if bool(accepted.result.get("handled")):
            await self._workflows.drain_due_pending_runs(limit=10)

    async def _enqueue_intake(self, result: dict[str, Any]) -> None:
        tenant_id = str(result.get("customer_id") or "").strip()
        connection_id = str(result.get("business_connection_id") or "").strip()
        conversation_id = str(result.get("chat_id") or "").strip()
        owner_chat_id = str(result.get("user_chat_id") or "").strip()
        if not tenant_id or not connection_id or not conversation_id:
            raise RuntimeError("Telegram Business intake identity is incomplete")
        workflows = self._workflows.list_workflows(
            customer_id=tenant_id,
            include_disabled=False,
        )
        for workflow in workflows:
            if (
                str(workflow.get("channel") or "") != "telegram_business_dm"
                or str(workflow.get("provider") or "") != "telegram_bot_api"
                or not self._workflows._source_matches_workflow(  # noqa: SLF001
                    workflow=workflow,
                    business_connection_id=connection_id,
                    conversation_id=conversation_id,
                )
            ):
                continue
            outcome = await self._workflows.enqueue_telegram_business_workflow_run(
                customer_id=tenant_id,
                workflow_id=str(workflow.get("workflow_id") or ""),
                conversation_id=conversation_id,
                owner_chat_id=owner_chat_id,
                event_type="telegram_business_webhook",
            )
            if not bool(outcome.get("ok")):
                logger.error(
                    "failed to queue Telegram Business intake workflow",
                    extra={
                        "tenant_id": tenant_id,
                        "workflow_id": str(workflow.get("workflow_id") or ""),
                        "conversation_id": conversation_id,
                    },
                )
                raise RuntimeError("Telegram Business intake could not be queued")


__all__ = ["AcceptedTelegramBusinessUpdate", "TelegramBusinessRelay"]
