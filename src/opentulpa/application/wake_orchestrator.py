"""Application orchestration for wake events."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN


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
        get_approvals: Callable[[], Any] | None = None,
        get_signal_inbox: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._get_context_events = get_context_events
        self._get_telegram_chat = get_telegram_chat
        self._get_telegram_client = get_telegram_client
        self._get_agent_runtime = get_agent_runtime
        self._get_approvals = get_approvals
        self._get_signal_inbox = get_signal_inbox

    def _backlog(self, *, customer_id: str, source: str, event_type: str, payload: dict[str, Any]) -> None:
        self._get_context_events().add_event(
            customer_id=customer_id,
            source=source,
            event_type=event_type,
            payload=payload,
        )

    async def _flush_deferred_challenges(self, *, chat_id: int | str) -> None:
        if self._get_approvals is None:
            return
        with suppress(Exception):
            await self._get_approvals().flush_deferred_challenges(
                origin_interface="telegram",
                origin_conversation_id=str(chat_id),
            )

    @staticmethod
    def _signal_prompt(*, source: str, signals: list[dict[str, Any]], guidance_text: str) -> str:
        lines = [
            "External messages/signals arrived for this conversation.",
            "Treat the normalized message texts below as the latest inbound user content.",
            "Reply naturally and concisely if a reply is appropriate.",
            f"source={source}",
        ]
        if guidance_text:
            lines.append(f"guidance={guidance_text[:2000]}")
        lines.append("")
        lines.append("Signals:")
        for idx, item in enumerate(signals[:20], start=1):
            text = str(item.get("text", "")).strip()
            event_type = str(item.get("event_type", "message")).strip() or "message"
            payload = item.get("payload", {})
            if text:
                lines.append(f"{idx}. [{event_type}] {text[:2000]}")
                if isinstance(payload, dict) and payload:
                    lines.append(f"   metadata={str(payload)[:1200]}")
            else:
                lines.append(f"{idx}. [{event_type}] payload={payload}")
        lines.append("")
        lines.append("Use the existing thread context and any stored business context when answering.")
        return "\n".join(lines)

    @staticmethod
    def _pick_dispatch(signals: list[dict[str, Any]]) -> dict[str, Any]:
        for item in reversed(signals):
            dispatch = item.get("dispatch")
            if isinstance(dispatch, dict) and dispatch:
                return dispatch
        return {}

    async def handle_event(self, body: dict[str, Any]) -> None:
        wake_type = str(body.get("type", "")).strip()
        if wake_type not in {"task_event", "routine_event", "approval_event", "signal_event"}:
            return

        if wake_type == "approval_event":
            await self._handle_approval_event(body)
            return
        if wake_type == "task_event":
            await self._handle_task_event(body)
            return
        if wake_type == "signal_event":
            await self._handle_signal_event(body)
            return
        await self._handle_routine_event(body)

    async def _handle_signal_event(self, body: dict[str, Any]) -> None:
        if self._get_signal_inbox is None:
            return
        inbox = self._get_signal_inbox()
        source = str(body.get("source", "")).strip()
        customer_id = str(body.get("customer_id", "")).strip()
        thread_id = str(body.get("thread_id", "")).strip()
        if not source or not customer_id or not thread_id:
            return

        signals = inbox.claim_ready_batch(source=source, customer_id=customer_id, thread_id=thread_id)
        if not signals:
            return

        signal_ids = [int(item.get("id", 0) or 0) for item in signals if int(item.get("id", 0) or 0) > 0]
        rule = inbox.resolve_rule(source=source, customer_id=customer_id, thread_id=thread_id)
        wake_mode = str(rule.get("wake_mode", "classifier")).strip().lower() or "classifier"
        guidance_text = str(rule.get("guidance_text", "")).strip()
        auto_reply = bool(rule.get("auto_reply", True))

        runtime = self._get_agent_runtime()
        should_wake = wake_mode == "always"
        if wake_mode == "never":
            should_wake = False
        elif wake_mode == "classifier" and runtime and hasattr(runtime, "classify_wake_event"):
            decision = await runtime.classify_wake_event(
                customer_id=customer_id,
                event_label=f"signal/{source}",
                payload={
                    "thread_id": thread_id,
                    "signal_count": len(signals),
                    "guidance_text": guidance_text[:1000],
                    "signals": [
                        {
                            "event_type": str(item.get("event_type", ""))[:100],
                            "text": str(item.get("text", ""))[:500],
                            "payload": item.get("payload", {}),
                        }
                        for item in signals[:10]
                    ],
                },
            )
            should_wake = bool(decision.get("notify_user", False))

        if not should_wake or runtime is None or not hasattr(runtime, "ainvoke_text"):
            self._backlog(
                customer_id=customer_id,
                source=f"signal:{source}",
                event_type="batched",
                payload={
                    "thread_id": thread_id,
                    "signal_ids": signal_ids,
                    "signal_count": len(signals),
                    "signals": signals[:10],
                    "wake_mode": wake_mode,
                },
            )
            inbox.mark_done(signal_ids)
            return

        try:
            reply_text = await runtime.ainvoke_text(
                thread_id=thread_id,
                customer_id=customer_id,
                text=self._signal_prompt(
                    source=source,
                    signals=signals,
                    guidance_text=guidance_text,
                ),
                turn_mode="interactive",
                include_pending_context=True,
            )
        except Exception:
            inbox.release_processing(signal_ids)
            raise

        dispatch = self._pick_dispatch(signals)
        reply = str(reply_text or "").strip()
        if auto_reply and reply and dispatch:
            inbox.create_outbound_message(
                source=source,
                customer_id=customer_id,
                thread_id=thread_id,
                text=reply,
                dispatch=dispatch,
                signal_ids=signal_ids,
            )
        else:
            self._backlog(
                customer_id=customer_id,
                source=f"signal:{source}",
                event_type="processed",
                payload={
                    "thread_id": thread_id,
                    "signal_ids": signal_ids,
                    "signal_count": len(signals),
                    "reply": reply[:2000],
                    "auto_reply": auto_reply,
                    "dispatch_available": bool(dispatch),
                },
            )
        inbox.mark_done(signal_ids)

    async def _handle_approval_event(self, body: dict[str, Any]) -> None:
        customer_id = str(body.get("customer_id", "")).strip()
        payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
        event_type = str(body.get("event_type", payload.get("event_type", "approved"))).strip()
        if not customer_id:
            return
        queue_payload = {
            "approval_id": str(body.get("approval_id", payload.get("approval_id", ""))).strip(),
            "event_type": event_type,
            "payload": payload,
        }
        runtime = self._get_agent_runtime()
        if not self._settings.telegram_bot_token or runtime is None:
            self._backlog(
                customer_id=customer_id,
                source="approval",
                event_type=event_type,
                payload=queue_payload,
            )
            return

        try:
            replies = await self._get_telegram_chat().relay_event(
                customer_id=customer_id,
                event_label=f"approval/{event_type}",
                payload=queue_payload,
                agent_runtime=runtime,
            )
        except Exception:
            self._backlog(
                customer_id=customer_id,
                source="approval",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        if not replies:
            self._backlog(
                customer_id=customer_id,
                source="approval",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        for item in replies:
            await self._get_telegram_client().send_message(
                chat_id=item["chat_id"],
                text=item["text"],
                parse_mode="HTML",
            )
            with suppress(Exception):
                self._get_telegram_chat().touch_assistant_message(int(item["chat_id"]))
            await self._flush_deferred_challenges(chat_id=item["chat_id"])

    async def _handle_task_event(self, body: dict[str, Any]) -> None:
        customer_id = str(body.get("customer_id", "")).strip()
        event_type = str(body.get("event_type", "")).strip()
        payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
        if not customer_id or event_type not in {"done", "failed", "needs_input", "worker_stopped"}:
            return

        runtime = self._get_agent_runtime()
        should_notify = event_type == "needs_input"
        if not should_notify and runtime and hasattr(runtime, "classify_wake_event"):
            decision = await runtime.classify_wake_event(
                customer_id=customer_id,
                event_label=f"task/{event_type}",
                payload={
                    "task_id": str(body.get("task_id", "")),
                    "payload": payload,
                },
            )
            should_notify = bool(decision.get("notify_user", False))

        backlog_payload = {"task_id": str(body.get("task_id", "")), **payload}
        if not should_notify:
            self._backlog(
                customer_id=customer_id,
                source="task",
                event_type=event_type,
                payload=backlog_payload,
            )
            return
        if not self._settings.telegram_bot_token:
            self._backlog(
                customer_id=customer_id,
                source="task",
                event_type=event_type,
                payload=backlog_payload,
            )
            return
        try:
            replies = await self._get_telegram_chat().relay_task_event(
                customer_id=customer_id,
                task_id=str(body.get("task_id", "")),
                event_type=event_type,
                payload=payload,
                agent_runtime=runtime,
            )
        except Exception:
            self._backlog(
                customer_id=customer_id,
                source="task",
                event_type=event_type,
                payload=backlog_payload,
            )
            return
        if not replies:
            self._backlog(
                customer_id=customer_id,
                source="task",
                event_type=event_type,
                payload=backlog_payload,
            )
            return
        for item in replies:
            await self._get_telegram_client().send_message(
                chat_id=item["chat_id"],
                text=item["text"],
                parse_mode="HTML",
            )
            await self._flush_deferred_challenges(chat_id=item["chat_id"])

    async def _handle_routine_event(self, body: dict[str, Any]) -> None:
        payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
        customer_id = str(body.get("customer_id") or payload.get("customer_id") or "").strip()
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
            self._backlog(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return

        routine_instruction = str(payload.get("instruction", "")).strip()
        if not routine_instruction:
            queue_payload["execution_status"] = "invalid"
            queue_payload["execution_error"] = "routine payload missing required instruction"
            self._backlog(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        execution_prompt = (
            "System update: a scheduled routine fired.\n"
            "Execute this routine now using tools/skills as needed.\n"
            "Treat this as background execution, not a normal chat reply.\n"
            f"- event: routine/{event_type}\n"
            f"- routine_id: {routine_id or 'unknown'}\n"
            f"- routine_name: {routine_name or 'unnamed'}\n"
            f"- instruction: {routine_instruction[:3000]}\n"
            f"- payload: {payload}\n\n"
            "After execution, return a concise summary: what was done, outcome, and any blockers."
        )
        execution_thread_id = f"routine_{routine_id}" if routine_id else f"routine_{customer_id}"
        try:
            execution_text = await runtime.ainvoke_text(
                thread_id=execution_thread_id,
                customer_id=customer_id,
                text=execution_prompt,
                turn_mode="routine_wake",
                include_pending_context=False,
            )
        except Exception as exc:
            queue_payload["execution_status"] = "failed"
            queue_payload["execution_error"] = str(exc)[:500]
            self._backlog(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return

        execution_summary = str(execution_text or "").strip()
        if not execution_summary:
            execution_summary = "Routine executed, but no summary was produced."
        queue_payload["execution_status"] = "executed"
        queue_payload["execution_summary"] = execution_summary[:2000]

        if not notify_user:
            self._backlog(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return
        if not self._settings.telegram_bot_token or execution_summary == NO_NOTIFY_TOKEN:
            self._backlog(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return

        slots: list[dict[str, Any]] = []
        with suppress(Exception):
            slots = self._get_telegram_chat().find_session_slots(customer_id)
        if not slots:
            self._backlog(
                customer_id=customer_id,
                source="routine",
                event_type=event_type,
                payload=queue_payload,
            )
            return

        for slot in slots:
            chat_id = int(slot["chat_id"])
            await self._get_telegram_client().send_message(
                chat_id=chat_id,
                text=execution_summary,
                parse_mode="HTML",
            )
            with suppress(Exception):
                self._get_telegram_chat().touch_assistant_message(chat_id)
            await self._flush_deferred_challenges(chat_id=chat_id)
