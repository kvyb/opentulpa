"""Application orchestration for wake events."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN
from opentulpa.web.events import append_web_event


class WakeOrchestrator:
    """Processes wake payloads and routes notifications/backlog updates."""

    def __init__(
        self,
        *,
        settings: Any,
        get_context_events: Callable[[], Any],
        get_telegram_chat: Callable[[], Any],
        get_telegram_client: Callable[[], Any],
        get_agent_runtime: Callable[[], Any],
        get_intake_workflows: Callable[[], Any] | None = None,
        resolve_customer_id: Callable[[str], str] | None = None,
    ) -> None:
        self._settings = settings
        self._get_context_events = get_context_events
        self._get_telegram_chat = get_telegram_chat
        self._get_telegram_client = get_telegram_client
        self._get_agent_runtime = get_agent_runtime
        self._get_intake_workflows = get_intake_workflows
        self._resolve_customer_id = resolve_customer_id

    def _customer_id(self, value: Any) -> str:
        cid = str(value or "").strip()
        if not cid or self._resolve_customer_id is None:
            return cid
        resolved = str(self._resolve_customer_id(cid) or "").strip()
        return resolved or cid

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _backlog(self, *, customer_id: str, source: str, event_type: str, payload: dict[str, Any]) -> None:
        self._get_context_events().add_event(
            customer_id=customer_id,
            source=source,
            event_type=event_type,
            payload=payload,
        )

    def _record_routine_execution(
        self,
        *,
        customer_id: str,
        event_type: str,
        payload: dict[str, Any],
        notification_status: str,
        notification_error: str = "",
        notified_chat_ids: list[int] | None = None,
    ) -> None:
        event_payload: dict[str, Any] = {
            "routine_id": str(payload.get("routine_id", "") or "").strip(),
            "routine_name": str(payload.get("routine_name", "") or "").strip(),
            "execution_status": str(payload.get("execution_status", "") or "").strip(),
            "execution_summary": str(payload.get("execution_summary", "") or "").strip()[:1200],
            "execution_error": str(payload.get("execution_error", "") or "").strip()[:500],
            "notify_user": bool(payload.get("notify_user", False)),
            "notification_status": str(notification_status or "").strip(),
            "notification_error": str(notification_error or "").strip()[:500],
        }
        if notified_chat_ids:
            event_payload["notified_chat_ids"] = [int(chat_id) for chat_id in notified_chat_ids[:5]]
        with suppress(Exception):
            self._backlog(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload={key: value for key, value in event_payload.items() if value not in ("", None)},
            )
        summary = str(payload.get("execution_summary", "") or "").strip()
        if summary and summary != NO_NOTIFY_TOKEN and bool(payload.get("notify_user", False)):
            append_web_event(
                customer_id=customer_id,
                thread_id=str(payload.get("routine_id", "") or "").strip(),
                source="routine",
                kind="proactive_message",
                text=summary,
                metadata_json=json.dumps(
                    {
                        "event_type": event_type,
                        "routine_id": str(payload.get("routine_id", "") or "").strip(),
                        "routine_name": str(payload.get("routine_name", "") or "").strip(),
                    },
                    ensure_ascii=False,
                ),
            )

    @staticmethod
    def _compact_payload_summary(payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict) or not payload:
            return "{}"
        safe_payload = {
            str(key): value
            for key, value in payload.items()
            if str(key) not in {"instruction"}
        }
        text = json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)
        if len(text) <= 1200:
            return text
        return text[:1197].rstrip() + "..."

    @staticmethod
    def _direct_owner_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        direct_slots: list[dict[str, Any]] = []
        for slot in slots:
            if str(slot.get("role", "")).strip() == "support":
                continue
            with suppress(Exception):
                chat_id = int(slot["chat_id"])
                if chat_id > 0:
                    direct_slots.append(slot)
        return direct_slots

    async def handle_event(self, body: dict[str, Any]) -> None:
        wake_type = str(body.get("type", "")).strip()
        if wake_type not in {"task_event", "routine_event"}:
            return

        if wake_type == "task_event":
            await self._handle_task_event(body)
            return
        await self._handle_routine_event(body)

    def _backlog_task_event(
        self,
        *,
        customer_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self._backlog(
            customer_id=customer_id,
            source="task",
            event_type=event_type,
            payload=payload,
        )

    async def _task_event_should_notify(
        self,
        *,
        runtime: Any,
        customer_id: str,
        task_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> bool:
        if event_type == "needs_input":
            return True
        if not runtime or not hasattr(runtime, "classify_wake_event"):
            return False
        decision = await runtime.classify_wake_event(
            customer_id=customer_id,
            event_label=f"task/{event_type}",
            payload={"task_id": task_id, "payload": payload},
        )
        return bool(decision.get("notify_user", False))

    async def _send_task_event_replies(
        self,
        *,
        customer_id: str,
        task_id: str,
        event_type: str,
        replies: list[dict[str, Any]],
    ) -> None:
        for item in replies:
            sent = await self._get_telegram_client().send_message(
                chat_id=item["chat_id"],
                text=item["text"],
                parse_mode="HTML",
            )
            if sent:
                append_web_event(
                    customer_id=customer_id,
                    thread_id=task_id,
                    source="task",
                    kind="proactive_message",
                    text=str(item.get("text", "") or ""),
                    metadata_json=json.dumps({"event_type": event_type}, ensure_ascii=False),
                )

    async def _handle_task_event(self, body: dict[str, Any]) -> None:
        customer_id = self._customer_id(body.get("customer_id", ""))
        event_type = str(body.get("event_type", "")).strip()
        payload = self._payload(body.get("payload"))
        if not customer_id or event_type not in {"done", "failed", "needs_input", "worker_stopped"}:
            return

        runtime = self._get_agent_runtime()
        task_id = str(body.get("task_id", ""))
        backlog_payload = {"task_id": str(body.get("task_id", "")), **payload}
        if not await self._task_event_should_notify(
            runtime=runtime,
            customer_id=customer_id,
            task_id=task_id,
            event_type=event_type,
            payload=payload,
        ):
            self._backlog_task_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=backlog_payload,
            )
            return
        if not self._settings.telegram_bot_token:
            self._backlog_task_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=backlog_payload,
            )
            return
        try:
            replies = await self._get_telegram_chat().relay_task_event(
                customer_id=customer_id,
                task_id=task_id,
                event_type=event_type,
                payload=payload,
                agent_runtime=runtime,
            )
        except Exception:
            self._backlog_task_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=backlog_payload,
            )
            return
        if not replies:
            self._backlog_task_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=backlog_payload,
            )
            return
        await self._send_task_event_replies(
            customer_id=customer_id,
            task_id=task_id,
            event_type=event_type,
            replies=replies,
        )

    def _backlog_routine_event(
        self,
        *,
        customer_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self._backlog(
            customer_id=customer_id,
            source="routine",
            event_type=event_type,
            payload=payload,
        )

    def _record_invalid_routine_event(
        self,
        *,
        customer_id: str,
        event_type: str,
        payload: dict[str, Any],
        error: str,
    ) -> None:
        payload["execution_status"] = "invalid"
        payload["execution_error"] = error
        self._backlog_routine_event(
            customer_id=customer_id,
            event_type=event_type,
            payload=payload,
        )

    async def _record_routine_notification_result(
        self,
        *,
        customer_id: str,
        event_type: str,
        payload: dict[str, Any],
        notify_user: bool,
        execution_summary: str,
    ) -> None:
        if not notify_user or not self._settings.telegram_bot_token or execution_summary == NO_NOTIFY_TOKEN:
            self._record_routine_execution(
                customer_id=customer_id,
                event_type=event_type,
                payload=payload,
                notification_status="skipped",
                notification_error=(
                    "notify_user=false"
                    if not notify_user
                    else "telegram_bot_token_missing"
                    if not self._settings.telegram_bot_token
                    else "no_notify_token"
                ),
            )
            return
        notified_chat_ids = await self._notify_routine_owner_slots(
            customer_id=customer_id,
            text=execution_summary,
        )
        if not notified_chat_ids:
            self._record_routine_execution(
                customer_id=customer_id,
                event_type=event_type,
                payload=payload,
                notification_status="backlogged",
                notification_error="no_telegram_session_slots",
            )
            return
        self._record_routine_execution(
            customer_id=customer_id,
            event_type=event_type,
            payload=payload,
            notification_status="sent",
            notified_chat_ids=notified_chat_ids,
        )

    async def _handle_intake_workflow_routine_event(
        self,
        *,
        customer_id: str,
        event_type: str,
        workflow_id: str,
        notify_user: bool,
        queue_payload: dict[str, Any],
    ) -> None:
        if not workflow_id or self._get_intake_workflows is None:
            self._record_invalid_routine_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=queue_payload,
                error="intake workflow payload missing workflow_id",
            )
            return
        try:
            result = await self._get_intake_workflows().run_workflow(
                customer_id=customer_id,
                workflow_id=workflow_id,
                event_type=event_type,
            )
        except Exception as exc:
            queue_payload["execution_status"] = "failed"
            queue_payload["execution_error"] = str(exc)[:500]
            self._backlog_routine_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=queue_payload,
            )
            return
        execution_summary = str(result.get("summary", "") or "").strip() or NO_NOTIFY_TOKEN
        queue_payload["execution_status"] = "executed" if bool(result.get("ok", False)) else "failed"
        queue_payload["execution_summary"] = execution_summary[:2000]
        queue_payload["execution_result"] = result
        if queue_payload["execution_status"] != "executed":
            queue_payload["execution_error"] = (
                " | ".join(str(item) for item in _safe_error_list(result.get("errors")))[:500]
                or "workflow execution failed"
            )
        await self._record_routine_notification_result(
            customer_id=customer_id,
            event_type=event_type,
            payload=queue_payload,
            notify_user=notify_user,
            execution_summary=execution_summary,
        )

    async def _handle_agent_routine_event(
        self,
        *,
        customer_id: str,
        event_type: str,
        routine_id: str,
        routine_name: str,
        routine_instruction: str,
        notify_user: bool,
        payload: dict[str, Any],
        queue_payload: dict[str, Any],
        runtime: Any,
    ) -> None:
        execution_prompt = (
            "System update: a scheduled routine fired.\n"
            "Execute this routine now using tools/skills as needed.\n"
            "Treat this as background execution, not a normal chat reply.\n"
            f"- event: routine/{event_type}\n"
            f"- routine_id: {routine_id or 'unknown'}\n"
            f"- routine_name: {routine_name or 'unnamed'}\n"
            f"- instruction: {routine_instruction[:1500]}\n"
            f"- payload_summary: {self._compact_payload_summary(payload)}\n\n"
            "After execution, return a concise summary: what was done, outcome, and any blockers."
        )
        execution_thread_id = (
            f"routine_{routine_id}_{new_short_id('wake')}"
            if routine_id
            else f"routine_{customer_id}_{new_short_id('wake')}"
        )
        try:
            execution_text = await runtime.ainvoke_text(
                thread_id=execution_thread_id,
                customer_id=customer_id,
                text=execution_prompt,
                turn_mode="routine_wake",
                include_pending_context=False,
                prompt_mode_override="literal_chat",
            )
        except Exception as exc:
            queue_payload["execution_status"] = "failed"
            queue_payload["execution_error"] = str(exc)[:500]
            self._backlog_routine_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=queue_payload,
            )
            return
        execution_summary = str(execution_text or "").strip()
        if not execution_summary:
            execution_summary = "Routine executed, but no summary was produced."
        queue_payload["execution_status"] = "executed"
        queue_payload["execution_summary"] = execution_summary[:2000]
        await self._record_routine_notification_result(
            customer_id=customer_id,
            event_type=event_type,
            payload=queue_payload,
            notify_user=notify_user,
            execution_summary=execution_summary,
        )

    async def _handle_routine_event(self, body: dict[str, Any]) -> None:
        payload = self._payload(body.get("payload"))
        customer_id = self._customer_id(body.get("customer_id") or payload.get("customer_id") or "")
        if not customer_id:
            return
        event_type = str(body.get("event_type") or payload.get("event_type") or "scheduled").strip()
        notify_raw = body.get("notify_user", payload.get("notify_user", True))
        notify_user = not (
            notify_raw is False or str(notify_raw).strip().lower() in {"0", "false", "no", "off"}
        )
        routine_id = str(body.get("routine_id") or payload.get("routine_id") or "").strip()
        routine_name = str(body.get("routine_name") or payload.get("routine_name") or "").strip()
        queue_payload = {
            "routine_id": routine_id,
            "routine_name": routine_name,
            "event_type": event_type,
            "notify_user": bool(notify_user),
            "payload": payload,
        }

        runtime = self._get_agent_runtime()
        if runtime is None:
            self._backlog_routine_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=queue_payload,
            )
            return

        if str(payload.get("workflow_type", "")).strip() == "intake_workflow":
            workflow_id = str(payload.get("workflow_id", "")).strip()
            await self._handle_intake_workflow_routine_event(
                customer_id=customer_id,
                event_type=event_type,
                workflow_id=workflow_id,
                notify_user=notify_user,
                queue_payload=queue_payload,
            )
            return

        routine_instruction = str(payload.get("instruction", "")).strip()
        if not routine_instruction:
            self._record_invalid_routine_event(
                customer_id=customer_id,
                event_type=event_type,
                payload=queue_payload,
                error="routine payload missing required instruction",
            )
            return
        await self._handle_agent_routine_event(
            customer_id=customer_id,
            event_type=event_type,
            routine_id=routine_id,
            routine_name=routine_name,
            routine_instruction=routine_instruction,
            notify_user=notify_user,
            payload=payload,
            queue_payload=queue_payload,
            runtime=runtime,
        )

    async def _notify_routine_owner_slots(self, *, customer_id: str, text: str) -> list[int]:
        routine_slots: list[dict[str, Any]] = []
        with suppress(Exception):
            routine_slots = self._get_telegram_chat().find_session_slots(customer_id)
        notified_chat_ids: list[int] = []
        for slot in self._direct_owner_slots(routine_slots):
            chat_id = int(slot["chat_id"])
            sent = await self._get_telegram_client().send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML",
            )
            if not sent:
                continue
            with suppress(Exception):
                self._get_telegram_chat().touch_assistant_message(chat_id)
            notified_chat_ids.append(chat_id)
        return notified_chat_ids


def _safe_error_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]
