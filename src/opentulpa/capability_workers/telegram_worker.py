"""Standalone Telegram interface worker backed only by the V2 Agent API."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import os
import secrets
import signal
from collections.abc import AsyncIterator, Callable, Mapping, MutableMapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from opentulpa.capability_workers.agent_api import (
    AgentAPIClient,
    AgentAPIError,
    AgentEvent,
    AgentNotification,
    AgentNotificationApproval,
)
from opentulpa.capability_workers.state import TelegramWorkerState
from opentulpa.capability_workers.telegram_api import (
    TelegramAPIError,
    TelegramAttachment,
    TelegramBotAPI,
    extract_attachments,
)

logger = logging.getLogger(__name__)

_DECISIONS = ("approve", "edit", "reject")
_EDIT_ARGUMENT_LIMIT_BYTES = 8 * 1024
_EDIT_ARGUMENT_MAX_DEPTH = 8
_EDIT_ARGUMENT_MAX_NODES = 512
_HIDDEN_EDIT_ARGUMENTS = {
    "actor_id",
    "channel",
    "correlation_id",
    "customer_id",
    "run_kind",
    "tenant_id",
    "thread_id",
}
_MAX_SECRET_BYTES = 16 * 1024


class WorkerConfigurationError(RuntimeError):
    """A worker setting is missing or unsafe."""


class AgentRunClient(Protocol):
    async def upload_file(
        self,
        *,
        filename: str,
        content: bytes,
        mime_type: str | None,
        kind: str,
        caption: str | None,
        source_event_id: str,
    ) -> str: ...

    def start_run(
        self,
        *,
        thread_id: str,
        text: str,
        file_ids: list[str],
        source_event_id: str,
    ) -> AsyncIterator[AgentEvent]: ...

    def resume_run(
        self,
        *,
        run_id: str,
        approval_id: str,
        decision: Literal["approve", "edit", "reject"],
        source_event_id: str,
        edited_arguments: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    def replay_run(
        self,
        *,
        run_id: str,
        after_sequence: int,
    ) -> AsyncIterator[AgentEvent]: ...

    async def list_notifications(
        self,
        *,
        after_id: int,
        limit: int = 100,
        wait_seconds: float = 0,
    ) -> list[AgentNotification]: ...

    async def acknowledge_notification(self, notification_id: int) -> None: ...


class TelegramTransport(Protocol):
    async def delete_webhook(self) -> None: ...

    async def get_me(self) -> dict[str, Any]: ...

    async def get_updates(
        self,
        *,
        offset: int,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]: ...

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None: ...

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str,
        show_alert: bool = False,
    ) -> None: ...

    async def download_attachment(self, attachment: TelegramAttachment) -> bytes: ...


class TelegramInterfaceWorker:
    """Long-poll one paired Telegram identity into the stable Agent API."""

    def __init__(
        self,
        *,
        telegram: TelegramTransport,
        agent: AgentRunClient,
        state: TelegramWorkerState,
        pairing_code: str | None,
        poll_timeout_seconds: int = 30,
        retry_delay_seconds: float = 1,
    ) -> None:
        if not 1 <= poll_timeout_seconds <= 50:
            raise ValueError("poll_timeout_seconds must be between 1 and 50")
        if retry_delay_seconds <= 0:
            raise ValueError("retry_delay_seconds must be positive")
        self._telegram = telegram
        self._agent = agent
        self._state = state
        self._pairing_code = str(pairing_code or "").strip() or None
        self._poll_timeout_seconds = poll_timeout_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._bot_id: int | None = None
        self._webhook_cleared = False

    async def run(
        self,
        stop: asyncio.Event,
        *,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        """Poll until stopped, cancelling an in-flight long poll on shutdown."""

        await self._run_until_stopped(self._initialize(), stop)
        if stop.is_set():
            return
        if on_ready is not None:
            on_ready()
        while not stop.is_set():
            try:
                await self._run_until_stopped(self.poll_once(), stop)
            except (AgentAPIError, TelegramAPIError) as exc:
                if stop.is_set():
                    break
                logger.warning("Telegram worker request failed: %s", exc)
                await self._sleep_until_stopped(stop)

    async def _initialize(self) -> None:
        await self._ensure_bot_identity()
        if not self._webhook_cleared:
            await self._telegram.delete_webhook()
            self._webhook_cleared = True
        try:
            await self.recover()
        except (AgentAPIError, TelegramAPIError) as exc:
            # The local Agent API does not accept requests until application startup
            # finishes. Readiness is still gated on Telegram identity validation;
            # durable recovery is retried by the normal polling loop.
            logger.warning("Telegram worker deferred startup recovery: %s", exc)

    async def poll_once(self) -> int:
        """Poll and serially handle one Telegram update batch."""

        await self._ensure_bot_identity()
        await self._deliver_undelivered_approvals()
        await self._deliver_notifications()
        updates = await self._telegram.get_updates(
            offset=self._state.next_update_id,
            timeout_seconds=self._poll_timeout_seconds,
        )
        handled = 0
        for update in updates:
            update_id = _nonnegative_int(update.get("update_id"))
            if update_id is None:
                continue
            source_event_id = self._source_event_id(update_id)
            if self._state.source_seen(source_event_id):
                self._state.complete_update(
                    update_id=update_id,
                    source_event_id=source_event_id,
                )
                handled += 1
                continue
            pending = self._state.pending_run(source_event_id)
            if pending is not None:
                await self._recover_pending_record(pending)
            else:
                await self._handle_update(
                    update=update,
                    update_id=update_id,
                    source_event_id=source_event_id,
                )
            handled += 1
        return handled

    async def recover(self) -> None:
        """Resume observed Agent API runs and redeliver persisted approvals."""

        for record in self._state.pending_runs():
            try:
                await self._recover_pending_record(record)
            except (AgentAPIError, TelegramAPIError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Telegram worker could not recover a pending run: %s", exc)
        await self._deliver_undelivered_approvals()
        await self._deliver_notifications()

    async def _recover_pending_record(self, record: Mapping[str, Any]) -> None:
        source_event_id = str(record["source_event_id"])
        update_id = int(record["update_id"])
        run_id = str(record["run_id"])
        chat_id = int(record["chat_id"])
        sequence = max(0, int(record.get("sequence") or 0))
        accumulated_text = str(record.get("accumulated_text") or "")
        paired = self._state.paired_identity()
        if paired is None or paired[1] != chat_id or not run_id or not source_event_id:
            raise ValueError("pending run is not bound to the paired Telegram identity")
        await self._consume_events(
            events=self._agent.replay_run(
                run_id=run_id,
                after_sequence=sequence,
            ),
            source_event_id=source_event_id,
            update_id=update_id,
            chat_id=chat_id,
            user_id=paired[0],
            initial_run_id=run_id,
            initial_sequence=sequence,
            initial_text=accumulated_text,
        )

    async def _handle_update(
        self,
        *,
        update: Mapping[str, Any],
        update_id: int,
        source_event_id: str,
    ) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            await self._handle_callback(
                callback=callback,
                update_id=update_id,
                source_event_id=source_event_id,
            )
            return
        raw_message = update.get("message") or update.get("edited_message")
        if not isinstance(raw_message, dict):
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        await self._handle_message(
            message=raw_message,
            update_id=update_id,
            source_event_id=source_event_id,
        )

    async def _handle_message(
        self,
        *,
        message: dict[str, Any],
        update_id: int,
        source_event_id: str,
    ) -> None:
        chat = message.get("chat")
        sender = message.get("from")
        chat_id = _telegram_chat_id(chat.get("id")) if isinstance(chat, dict) else None
        user_id = _positive_int(sender.get("id")) if isinstance(sender, dict) else None
        if chat_id is None or user_id is None:
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return

        text = str(message.get("text") or message.get("caption") or "").strip()
        paired = self._state.paired_identity()
        if paired is None:
            await self._handle_pairing(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                update_id=update_id,
                source_event_id=source_event_id,
            )
            return
        if paired != (user_id, chat_id):
            await self._telegram.send_message(chat_id=chat_id, text="Unauthorized.")
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return

        edit_token = self._state.awaiting_edit(chat_id)
        if edit_token:
            await self._handle_edited_arguments(
                token=edit_token,
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                update_id=update_id,
                source_event_id=source_event_id,
            )
            return

        command, argument = _command(text)
        if command == "/start":
            await self._telegram.send_message(
                chat_id=chat_id,
                text=(
                    "OpenTulpa is connected. Send a request or attach files. "
                    "Use /fresh to start a new conversation."
                ),
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        if command == "/fresh":
            self._state.reset_thread(chat_id)
            await self._telegram.send_message(
                chat_id=chat_id,
                text="Started a fresh conversation.",
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        if command == "/regenerate" and not argument:
            text = "/regenerate"

        attachments = extract_attachments(message)
        if not text and not attachments:
            await self._telegram.send_message(
                chat_id=chat_id,
                text="Send a message or a supported attachment.",
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        file_ids = await self._upload_attachments(
            attachments=attachments,
            caption=str(message.get("caption") or "").strip() or None,
            source_event_id=source_event_id,
        )
        if not text:
            text = "Please inspect and respond to the attached files."
        await self._consume_events(
            events=self._agent.start_run(
                thread_id=self._state.thread_id(chat_id),
                text=text,
                file_ids=file_ids,
                source_event_id=source_event_id,
            ),
            source_event_id=source_event_id,
            update_id=update_id,
            chat_id=chat_id,
            user_id=user_id,
        )

    async def _handle_pairing(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        update_id: int,
        source_event_id: str,
    ) -> None:
        command, argument = _command(text)
        valid = bool(
            command == "/start"
            and argument
            and self._pairing_code
            and hmac.compare_digest(argument, self._pairing_code)
        )
        if not valid:
            await self._telegram.send_message(
                chat_id=chat_id,
                text="Pairing required. Send /start followed by the one-time pairing code.",
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        if not self._state.pair(user_id=user_id, chat_id=chat_id):
            await self._telegram.send_message(chat_id=chat_id, text="Unauthorized.")
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        self._pairing_code = None
        await self._telegram.send_message(
            chat_id=chat_id,
            text=(
                "OpenTulpa is paired with this Telegram account. "
                "Send a request or attach files."
            ),
        )
        self._complete(update_id=update_id, source_event_id=source_event_id)

    async def _upload_attachments(
        self,
        *,
        attachments: list[TelegramAttachment],
        caption: str | None,
        source_event_id: str,
    ) -> list[str]:
        file_ids: list[str] = []
        for index, attachment in enumerate(attachments):
            content = await self._telegram.download_attachment(attachment)
            filename = _safe_filename(attachment.filename)
            file_id = await self._agent.upload_file(
                filename=filename,
                content=content,
                mime_type=TelegramBotAPI.inferred_mime_type(attachment),
                kind=attachment.kind,
                caption=caption,
                source_event_id=f"{source_event_id}:file:{index}",
            )
            file_ids.append(file_id)
        return file_ids

    async def _handle_callback(
        self,
        *,
        callback: dict[str, Any],
        update_id: int,
        source_event_id: str,
    ) -> None:
        callback_id = str(callback.get("id") or "").strip()
        message = callback.get("message")
        sender = callback.get("from")
        chat = message.get("chat") if isinstance(message, dict) else None
        chat_id = _telegram_chat_id(chat.get("id")) if isinstance(chat, dict) else None
        user_id = _positive_int(sender.get("id")) if isinstance(sender, dict) else None
        if not callback_id or chat_id is None or user_id is None:
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        if self._state.paired_identity() != (user_id, chat_id):
            await self._telegram.answer_callback_query(
                callback_query_id=callback_id,
                text="Unauthorized",
                show_alert=True,
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return

        parts = str(callback.get("data") or "").split(":")
        if len(parts) != 3 or parts[0] != "ot" or parts[2] not in _DECISIONS:
            await self._telegram.answer_callback_query(
                callback_query_id=callback_id,
                text="This control is no longer active.",
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        token = parts[1]
        decision = cast(Literal["approve", "edit", "reject"], parts[2])
        record = self._state.approval(token)
        if not _approval_matches(record, chat_id=chat_id, user_id=user_id):
            await self._telegram.answer_callback_query(
                callback_query_id=callback_id,
                text="Approval not found.",
                show_alert=True,
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        assert record is not None
        allowed = {str(item) for item in record.get("allowed_decisions", [])}
        if decision not in allowed:
            await self._telegram.answer_callback_query(
                callback_query_id=callback_id,
                text="That decision is not allowed.",
                show_alert=True,
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        if decision == "edit":
            self._state.await_edit(chat_id=chat_id, token=token)
            await self._telegram.answer_callback_query(
                callback_query_id=callback_id,
                text="Send the replacement arguments as JSON.",
            )
            await self._telegram.send_message(
                chat_id=chat_id,
                text=(
                    "Reply with one JSON object containing the complete replacement tool "
                    "arguments. OpenTulpa will not echo them. Use /cancel to stop editing."
                ),
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return

        await self._telegram.answer_callback_query(
            callback_query_id=callback_id,
            text="Processing approval...",
        )
        await self._consume_events(
            events=self._agent.resume_run(
                run_id=str(record["run_id"]),
                approval_id=str(record["approval_id"]),
                decision=decision,
                source_event_id=source_event_id,
            ),
            source_event_id=source_event_id,
            update_id=update_id,
            chat_id=chat_id,
            user_id=user_id,
            initial_run_id=str(record["run_id"]),
            consumed_approval_token=token,
        )

    async def _handle_edited_arguments(
        self,
        *,
        token: str,
        chat_id: int,
        user_id: int,
        text: str,
        update_id: int,
        source_event_id: str,
    ) -> None:
        record = self._state.approval(token)
        if not _approval_matches(record, chat_id=chat_id, user_id=user_id):
            self._state.clear_awaiting_edit(chat_id=chat_id, token=token)
            await self._telegram.send_message(
                chat_id=chat_id,
                text="This approval is no longer pending.",
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        if _command(text)[0] == "/cancel":
            self._state.clear_awaiting_edit(chat_id=chat_id, token=token)
            await self._telegram.send_message(
                chat_id=chat_id,
                text="Stopped editing. The approval is still pending.",
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        assert record is not None
        if "edit" not in record.get("allowed_decisions", []):
            self._state.clear_awaiting_edit(chat_id=chat_id, token=token)
            await self._telegram.send_message(
                chat_id=chat_id,
                text="Editing is not allowed for this approval.",
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        try:
            edited_arguments = parse_edited_arguments(text)
        except ValueError:
            await self._telegram.send_message(
                chat_id=chat_id,
                text=(
                    "Send one valid JSON object with the complete replacement arguments, "
                    "or use /cancel."
                ),
            )
            self._complete(update_id=update_id, source_event_id=source_event_id)
            return
        await self._consume_events(
            events=self._agent.resume_run(
                run_id=str(record["run_id"]),
                approval_id=str(record["approval_id"]),
                decision="edit",
                edited_arguments=edited_arguments,
                source_event_id=source_event_id,
            ),
            source_event_id=source_event_id,
            update_id=update_id,
            chat_id=chat_id,
            user_id=user_id,
            initial_run_id=str(record["run_id"]),
            consumed_approval_token=token,
        )

    async def _consume_events(
        self,
        *,
        events: AsyncIterator[AgentEvent],
        source_event_id: str,
        update_id: int,
        chat_id: int,
        user_id: int,
        initial_run_id: str = "",
        initial_sequence: int = 0,
        initial_text: str = "",
        consumed_approval_token: str | None = None,
    ) -> Literal["completed", "failed", "approval"]:
        run_id = initial_run_id
        sequence = max(0, initial_sequence)
        accumulated = initial_text[-200_000:]
        async for event in events:
            if run_id and event.run_id != run_id:
                raise AgentAPIError("Agent API changed run identifiers mid-stream.")
            if event.sequence <= sequence:
                continue
            run_id = event.run_id
            if event.type == "message.delta":
                accumulated = (accumulated + str(event.data.get("text") or ""))[-200_000:]
            if event.settles_stream:
                self._state.save_pending_run(
                    source_event_id=source_event_id,
                    update_id=update_id,
                    run_id=run_id,
                    chat_id=chat_id,
                    sequence=sequence,
                    accumulated_text=accumulated,
                )
                if event.type == "run.completed":
                    final_text = str(event.data.get("text") or "").strip() or accumulated.strip()
                    await self._telegram.send_message(
                        chat_id=chat_id,
                        text=final_text or "The run completed without a message.",
                    )
                    self._complete(
                        update_id=update_id,
                        source_event_id=source_event_id,
                        consumed_approval_token=consumed_approval_token,
                    )
                    return "completed"
                if event.type == "run.failed":
                    failure = str(event.data.get("message") or "Agent run failed.").strip()
                    await self._telegram.send_message(
                        chat_id=chat_id,
                        text=(failure or "Agent run failed.")[:2_000],
                    )
                    self._complete(
                        update_id=update_id,
                        source_event_id=source_event_id,
                        consumed_approval_token=consumed_approval_token,
                    )
                    return "failed"
                approval_token = self._record_approval(
                    event=event,
                    chat_id=chat_id,
                    user_id=user_id,
                )
                consume = (
                    consumed_approval_token
                    if consumed_approval_token != approval_token
                    else None
                )
                self._complete(
                    update_id=update_id,
                    source_event_id=source_event_id,
                    consumed_approval_token=consume,
                )
                await self._deliver_approval(approval_token)
                return "approval"
            sequence = event.sequence
            self._state.save_pending_run(
                source_event_id=source_event_id,
                update_id=update_id,
                run_id=run_id,
                chat_id=chat_id,
                sequence=sequence,
                accumulated_text=accumulated,
            )
        raise AgentAPIError("Agent API event stream ended before a durable terminal event.")

    def _record_approval(self, *, event: AgentEvent, chat_id: int, user_id: int) -> str:
        approval_id = str(event.data.get("approval_id") or "").strip()
        if not approval_id:
            raise AgentAPIError("Agent API returned an approval without an identifier.")
        existing = self._state.find_approval(run_id=event.run_id, approval_id=approval_id)
        if existing is not None:
            return existing[0]
        raw_allowed = event.data.get("allowed_decisions")
        if isinstance(raw_allowed, list | tuple):
            allowed = [str(item) for item in raw_allowed if str(item) in _DECISIONS]
        else:
            allowed = list(_DECISIONS)
        if not allowed:
            raise AgentAPIError("Agent API approval has no supported decisions.")
        token = secrets.token_hex(8)
        self._state.save_approval(
            token=token,
            run_id=event.run_id,
            approval_id=approval_id,
            chat_id=chat_id,
            user_id=user_id,
            allowed_decisions=allowed,
            tool_name=str(event.data.get("tool_name") or "action")[:120],
            description=str(event.data.get("description") or "Approval required")[:1_500],
        )
        return token

    async def _deliver_approval(self, token: str) -> None:
        record = self._state.approval(token)
        if record is None or bool(record.get("delivered")):
            return
        allowed = {str(item) for item in record.get("allowed_decisions", [])}
        labels = {"approve": "Approve", "edit": "Edit", "reject": "Reject"}
        buttons = [
            {
                "text": labels[decision],
                "callback_data": f"ot:{token}:{decision}",
            }
            for decision in _DECISIONS
            if decision in allowed
        ]
        markup = {"inline_keyboard": [buttons]}
        tool_name = str(record.get("tool_name") or "action")
        description = str(record.get("description") or "Approval required")
        await self._telegram.send_message(
            chat_id=int(record["chat_id"]),
            text=f"Approval required for {tool_name}: {description}",
            reply_markup=markup,
        )
        self._state.mark_approval_delivered(token)

    async def _deliver_undelivered_approvals(self) -> None:
        for token, _ in self._state.undelivered_approvals():
            await self._deliver_approval(token)

    async def _deliver_notifications(self) -> None:
        paired = self._state.paired_identity()
        if paired is None:
            return
        await self._flush_notification_acks()
        notifications = await self._agent.list_notifications(
            after_id=self._state.notification_cursor,
            limit=100,
            wait_seconds=0,
        )
        user_id, chat_id = paired
        for notification in notifications:
            await self._telegram.send_message(
                chat_id=chat_id,
                text=notification.text[:50_000],
            )
            for approval in notification.approvals:
                token = self._record_notification_approval(
                    notification=notification,
                    approval=approval,
                    chat_id=chat_id,
                    user_id=user_id,
                )
                await self._deliver_approval(token)
            self._state.mark_notification_delivered(notification.id)
            await self._agent.acknowledge_notification(notification.id)
            self._state.mark_notification_acknowledged(notification.id)

    async def _flush_notification_acks(self) -> None:
        for notification_id in self._state.pending_notification_acks():
            await self._agent.acknowledge_notification(notification_id)
            self._state.mark_notification_acknowledged(notification_id)

    def _record_notification_approval(
        self,
        *,
        notification: AgentNotification,
        approval: AgentNotificationApproval,
        chat_id: int,
        user_id: int,
    ) -> str:
        run_id = str(notification.run_id or "").strip()
        if not run_id:
            raise AgentAPIError("Agent API returned an approval without a run id.")
        existing = self._state.find_approval(
            run_id=run_id,
            approval_id=approval.approval_id,
        )
        if existing is not None:
            return existing[0]
        token = secrets.token_hex(8)
        self._state.save_approval(
            token=token,
            run_id=run_id,
            approval_id=approval.approval_id,
            chat_id=chat_id,
            user_id=user_id,
            allowed_decisions=list(approval.allowed_decisions),
            tool_name=approval.tool_name[:120],
            description=approval.description[:1_500],
        )
        return token

    async def _ensure_bot_identity(self) -> None:
        if self._bot_id is not None:
            return
        bot = await self._telegram.get_me()
        bot_id = _positive_int(bot.get("id"))
        if bot_id is None:
            raise TelegramAPIError("Telegram returned an invalid bot identity.")
        self._bot_id = bot_id

    def _source_event_id(self, update_id: int) -> str:
        if self._bot_id is None:
            raise RuntimeError("Telegram bot identity has not been loaded")
        return f"telegram:{self._bot_id}:{update_id}"

    def _complete(
        self,
        *,
        update_id: int,
        source_event_id: str,
        consumed_approval_token: str | None = None,
    ) -> None:
        self._state.complete_update(
            update_id=update_id,
            source_event_id=source_event_id,
            consumed_approval_token=consumed_approval_token,
        )

    async def _run_until_stopped(self, work: Any, stop: asyncio.Event) -> None:
        work_task = asyncio.create_task(work)
        stop_task = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait(
            {work_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done:
            work_task.cancel()
            await asyncio.gather(work_task, return_exceptions=True)
            return
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        await work_task

    async def _sleep_until_stopped(self, stop: asyncio.Event) -> None:
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=self._retry_delay_seconds)


def parse_edited_arguments(text: str) -> dict[str, Any]:
    """Parse a bounded JSON replacement without exposing hidden run context."""

    if not text or len(text.encode("utf-8")) > _EDIT_ARGUMENT_LIMIT_BYTES:
        raise ValueError("edited arguments are missing or too large")

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON numbers are not supported")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("edited arguments must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("edited arguments must be a JSON object")

    nodes = 0

    def validate(candidate: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _EDIT_ARGUMENT_MAX_NODES or depth > _EDIT_ARGUMENT_MAX_DEPTH:
            raise ValueError("edited arguments are too complex")
        if isinstance(candidate, dict):
            for key, item in candidate.items():
                if not isinstance(key, str):
                    raise ValueError("edited argument keys must be strings")
                if key.lower() in _HIDDEN_EDIT_ARGUMENTS:
                    raise ValueError("runtime context cannot be edited")
                validate(item, depth + 1)
        elif isinstance(candidate, list):
            for item in candidate:
                validate(item, depth + 1)
        elif isinstance(candidate, float) and not math.isfinite(candidate):
            raise ValueError("edited arguments contain a non-finite number")
        elif candidate is not None and not isinstance(candidate, str | int | float | bool):
            raise ValueError("edited arguments contain an unsupported value")

    validate(value, 0)
    return cast(dict[str, Any], value)


def read_secret(
    env_name: str,
    fd_env_name: str,
    *,
    environ: MutableMapping[str, str] | None = None,
    required: bool = True,
) -> str | None:
    """Read a secret from one environment value or inherited file descriptor."""

    environment = environ if environ is not None else os.environ
    direct = environment.pop(env_name, None)
    raw_fd = environment.pop(fd_env_name, None)
    if direct is not None and raw_fd is not None:
        raise WorkerConfigurationError(
            f"Configure only one of {env_name} and {fd_env_name}."
        )
    value: str | None = None
    if raw_fd is not None:
        try:
            descriptor = int(raw_fd)
        except ValueError as exc:
            raise WorkerConfigurationError(f"{fd_env_name} must be a file descriptor.") from exc
        chunks: list[bytes] = []
        total = 0
        try:
            while True:
                chunk = os.read(descriptor, min(4_096, _MAX_SECRET_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_SECRET_BYTES:
                    raise WorkerConfigurationError(f"{fd_env_name} secret is too large.")
        except OSError as exc:
            raise WorkerConfigurationError(f"Could not read {fd_env_name}.") from exc
        finally:
            with suppress(OSError):
                os.close(descriptor)
        try:
            value = b"".join(chunks).decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise WorkerConfigurationError(f"{fd_env_name} must contain UTF-8 text.") from exc
    elif direct is not None:
        if len(direct.encode("utf-8")) > _MAX_SECRET_BYTES:
            raise WorkerConfigurationError(f"{env_name} secret is too large.")
        value = direct.strip()
    if not value:
        if required:
            raise WorkerConfigurationError(f"Configure {env_name} or {fd_env_name}.")
        return None
    return value


async def run_from_environment(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Build and run the bundled worker without accepting credentials as arguments."""

    environment = environ if environ is not None else os.environ
    capability_config = _capability_config(environment)
    state_path = Path(
        environment.get(
            "OPENTULPA_TELEGRAM_STATE_PATH",
            str(
                capability_config.get("state_path")
                or "~/.opentulpa/telegram-worker.json"
            ),
        )
    )
    state = TelegramWorkerState(state_path)
    telegram_token = read_secret(
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN_FD",
        environ=environment,
    )
    agent_credential = read_secret(
        "OPENTULPA_AGENT_API_TOKEN",
        "OPENTULPA_AGENT_API_TOKEN_FD",
        environ=environment,
    )
    pairing_code = read_secret(
        "OPENTULPA_TELEGRAM_PAIRING_CODE",
        "OPENTULPA_TELEGRAM_PAIRING_CODE_FD",
        environ=environment,
        required=state.paired_identity() is None,
    )
    assert telegram_token is not None and agent_credential is not None
    poll_timeout = _bounded_environment_int(
        environment,
        "OPENTULPA_TELEGRAM_POLL_SECONDS",
        default=_config_int(capability_config, "poll_timeout_seconds", 30),
        minimum=1,
        maximum=50,
    )
    max_attachment_bytes = _bounded_environment_int(
        environment,
        "OPENTULPA_TELEGRAM_MAX_ATTACHMENT_BYTES",
        default=_config_int(capability_config, "max_attachment_bytes", 45_000_000),
        minimum=1,
        maximum=100_000_000,
    )
    telegram = TelegramBotAPI(
        token=telegram_token,
        base_url=environment.get("OPENTULPA_TELEGRAM_API_URL", "https://api.telegram.org"),
        max_attachment_bytes=max_attachment_bytes,
    )
    agent = AgentAPIClient(
        base_url=environment.get(
            "OPENTULPA_AGENT_API_URL",
            str(capability_config.get("agent_api_url") or "http://127.0.0.1:8000"),
        ),
        credential=agent_credential,
    )
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=pairing_code,
        poll_timeout_seconds=poll_timeout,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal_name, stop.set)
    try:
        await worker.run(
            stop,
            on_ready=lambda: _signal_worker_ready(environment),
        )
    finally:
        await agent.aclose()
        await telegram.aclose()


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    try:
        asyncio.run(run_from_environment())
    except WorkerConfigurationError as exc:
        logger.error("Telegram worker configuration failed: %s", exc)
        return 2
    except (AgentAPIError, TelegramAPIError) as exc:
        logger.error("Telegram worker startup failed: %s", exc)
        return 3
    except KeyboardInterrupt:
        return 130
    return 0


def _signal_worker_ready(environ: Mapping[str, str]) -> None:
    raw_path = str(environ.get("OPENTULPA_WORKER_READY_FILE") or "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    if not path.is_absolute() or len(raw_path) > 4_096:
        raise WorkerConfigurationError("Worker readiness path is invalid.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise WorkerConfigurationError("Worker readiness could not be recorded.") from exc


def _bounded_environment_int(
    environ: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(environ.get(name, str(default)))
    except ValueError as exc:
        raise WorkerConfigurationError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise WorkerConfigurationError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return value


def _capability_config(environ: Mapping[str, str]) -> dict[str, Any]:
    raw = str(environ.get("OPENTULPA_CAPABILITY_CONFIG") or "").strip()
    if not raw:
        return {}
    if len(raw.encode("utf-8")) > 64 * 1024:
        raise WorkerConfigurationError("OPENTULPA_CAPABILITY_CONFIG is too large.")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise WorkerConfigurationError(
            "OPENTULPA_CAPABILITY_CONFIG must be valid JSON."
        ) from exc
    if not isinstance(value, dict):
        raise WorkerConfigurationError(
            "OPENTULPA_CAPABILITY_CONFIG must be a JSON object."
        )
    allowed = {
        "agent_api_url",
        "max_attachment_bytes",
        "poll_timeout_seconds",
        "state_path",
    }
    if set(value).difference(allowed):
        raise WorkerConfigurationError(
            "OPENTULPA_CAPABILITY_CONFIG contains unsupported fields."
        )
    return value


def _config_int(config: Mapping[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerConfigurationError(f"Capability config field {name} must be an integer.") from exc


def _approval_matches(
    record: Mapping[str, Any] | None,
    *,
    chat_id: int,
    user_id: int,
) -> bool:
    if record is None:
        return False
    try:
        return int(record.get("chat_id") or 0) == chat_id and int(
            record.get("user_id") or 0
        ) == user_id
    except (TypeError, ValueError):
        return False


def _command(text: str) -> tuple[str, str]:
    parts = text.split(None, 1)
    if not parts:
        return "", ""
    command = parts[0].split("@", 1)[0].lower()
    argument = parts[1].strip() if len(parts) == 2 else ""
    return command, argument


def _safe_filename(value: str) -> str:
    filename = str(value or "file.bin").replace("\\", "/").rsplit("/", 1)[-1].strip()
    return (filename or "file.bin")[:255]


def _positive_int(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved >= 0 else None


def _telegram_chat_id(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved != 0 else None


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgentRunClient",
    "TelegramInterfaceWorker",
    "TelegramTransport",
    "WorkerConfigurationError",
    "main",
    "parse_edited_arguments",
    "read_secret",
    "run_from_environment",
]
