"""Telegram transport for Deep Agent owner runs and approvals."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.ids import new_short_id
from opentulpa.deep_agent.contracts import (
    AgentApproval,
    AgentRunEvent,
    AgentRunRequest,
    ApprovalDecision,
)
from opentulpa.interfaces.telegram.attachments import extract_attachments
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.security import is_user_allowed
from opentulpa.interfaces.telegram.state_store import TelegramStateStore
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling import AgentChannel, AgentRunContext, AgentRunKind

logger = logging.getLogger(__name__)
_BUSINESS_UPDATE_KEYS = {
    "business_connection",
    "business_message",
    "edited_business_message",
    "deleted_business_messages",
}
_APPROVAL_DECISIONS = {"approve", "edit", "reject"}
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


@dataclass(frozen=True)
class AcceptedTelegramUpdate:
    body: dict[str, Any]
    is_business: bool
    business_result: dict[str, Any] | None = None
    ingress_key: str | None = None
    should_process: bool = True


async def _with_first_event(
    first: AgentRunEvent,
    events: AsyncIterator[AgentRunEvent],
) -> AsyncIterator[AgentRunEvent]:
    yield first
    async for event in events:
        yield event


class TelegramAgentService(Protocol):
    def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]: ...

    def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]: ...


class DeepAgentTelegramRelay:
    """Map Telegram updates to tenant-scoped Deep Agent runs.

    Telegram remains a channel adapter. It does not alter prompts, checkpoints, or
    graph state while a run is active.
    """

    def __init__(
        self,
        *,
        agent: TelegramAgentService,
        client: TelegramClient,
        state: TelegramStateStore,
        profiles: CustomerProfileService,
        files: FileVaultService,
        bot_token: str,
        owner_tenant_id: str | None,
        allowed_user_ids: str | None,
        allowed_usernames: str | None,
        telegram_business: Any | None = None,
        intake_workflows: Any | None = None,
        resolve_agent_spec: Callable[[str, str], AgentSpecRef] | None = None,
    ) -> None:
        self._agent = agent
        self._client = client
        self._state = state
        self._profiles = profiles
        self._files = files
        self._bot_token = str(bot_token or "").strip()
        self._owner_tenant_id = str(owner_tenant_id or "").strip()
        self._allowed_user_ids = allowed_user_ids
        self._allowed_usernames = allowed_usernames
        self._telegram_business = telegram_business
        self._intake_workflows = intake_workflows
        self._resolve_agent_spec = resolve_agent_spec
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._approval_locks: dict[str, asyncio.Lock] = {}
        self._owner_ingress_lock = asyncio.Lock()
        self._inbox_dispatcher: asyncio.Task[None] | None = None
        self._accepted_approvals: dict[
            str,
            tuple[dict[str, Any], Literal["approve", "edit", "reject"]],
        ] = {}

    def healthy(self) -> bool:
        """Return whether the durable owner inbox dispatcher is running."""

        task = self._inbox_dispatcher
        return task is not None and not task.done()

    async def accept_update(self, body: dict[str, Any]) -> AcceptedTelegramUpdate:
        safe_body = dict(body)
        is_business = any(key in safe_body for key in _BUSINESS_UPDATE_KEYS)
        if not is_business:
            ingress_key, should_process = self._state.enqueue_owner_update(safe_body)
            return AcceptedTelegramUpdate(
                body=safe_body,
                is_business=False,
                ingress_key=ingress_key,
                should_process=should_process,
            )
        if self._telegram_business is None:
            raise RuntimeError("Telegram Business ingress is unavailable")
        result = self._telegram_business.ingest_update(safe_body)
        if not isinstance(result, dict) or not bool(result.get("handled")):
            return AcceptedTelegramUpdate(
                body=safe_body,
                is_business=True,
                business_result=dict(result) if isinstance(result, dict) else None,
            )
        if bool(result.get("trigger_workflows")) and bool(result.get("dispatch_pending")):
            await self._enqueue_business_intake(result)
            ingress_key = str(result.get("ingress_key") or "").strip()
            self._telegram_business.complete_ingress(ingress_key)
            result = {**result, "dispatch_pending": False}
        return AcceptedTelegramUpdate(
            body=safe_body,
            is_business=True,
            business_result=result,
        )

    async def process_update(self, accepted: AcceptedTelegramUpdate) -> None:
        if accepted.is_business:
            drain = getattr(self._intake_workflows, "drain_due_pending_runs", None)
            if callable(drain):
                await drain(limit=10)
            return
        if not accepted.should_process:
            return
        ingress_key = accepted.ingress_key
        if ingress_key is None:
            raise RuntimeError("Telegram owner ingress identity is unavailable")
        async with self._owner_ingress_lock:
            body = self._state.owner_update(ingress_key)
            if body is None:
                return
            await self._handle_owner_update(body)
            self._state.complete_owner_update(ingress_key)

    async def start(self) -> None:
        if self._inbox_dispatcher is not None and not self._inbox_dispatcher.done():
            return
        self._inbox_dispatcher = asyncio.create_task(
            self._dispatch_owner_inbox(),
            name="opentulpa-telegram-owner-inbox",
        )

    async def shutdown(self) -> None:
        task = self._inbox_dispatcher
        self._inbox_dispatcher = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _dispatch_owner_inbox(self) -> None:
        while True:
            for ingress_key, body in self._state.pending_owner_updates(limit=100):
                try:
                    await self.process_update(
                        AcceptedTelegramUpdate(
                            body=body,
                            is_business=False,
                            ingress_key=ingress_key,
                        )
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Persisted Telegram owner update processing failed")
            await asyncio.sleep(1.0)

    async def handle_update(self, body: dict[str, Any]) -> None:
        accepted = await self.accept_update(body)
        await self.process_update(accepted)

    async def _handle_owner_update(self, body: dict[str, Any]) -> None:
        callback = body.get("callback_query")
        if isinstance(callback, dict):
            await self._handle_callback(callback)
            return
        message = body.get("message") or body.get("edited_message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat")
        sender = message.get("from")
        if not isinstance(chat, dict) or not isinstance(sender, dict):
            return
        chat_id = self._positive_int(chat.get("id"))
        user_id = self._positive_int(sender.get("id"))
        username = str(sender.get("username") or "").strip().removeprefix("@") or None
        if chat_id is None or user_id is None:
            return
        if not is_user_allowed(
            user_id=user_id,
            username=username,
            allowed_user_ids_csv=self._allowed_user_ids,
            allowed_usernames_csv=self._allowed_usernames,
        ):
            await self._client.send_message(chat_id=chat_id, text="Unauthorized.", parse_mode=None)
            return

        text = str(message.get("text") or message.get("caption") or "").strip()
        if text.split(None, 1)[0:1] == ["/start"]:
            await self._client.send_message(
                chat_id=chat_id,
                text=(
                    "OpenTulpa is connected. Send a request or attach files. "
                    "Use /fresh to start a new conversation checkpoint."
                ),
                parse_mode=None,
            )
            return
        if text.split(None, 1)[0:1] == ["/fresh"]:
            tenant_id = self._resolve_tenant(user_id)
            self._reset_session(
                chat_id=chat_id,
                user_id=user_id,
                username=username,
                tenant_id=tenant_id,
            )
            await self._client.send_message(
                chat_id=chat_id,
                text="Started a fresh conversation.",
                parse_mode=None,
            )
            return

        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            tenant_id = self._resolve_tenant(user_id)
            if await self._handle_pending_approval_edit(
                chat_id=chat_id,
                user_id=user_id,
                tenant_id=tenant_id,
                text=text,
            ):
                return
            thread_id = self._session(
                chat_id=chat_id,
                user_id=user_id,
                username=username,
                tenant_id=tenant_id,
            )
            file_ids = await self._ingest_files(
                message=message,
                chat_id=chat_id,
                tenant_id=tenant_id,
            )
            if not text:
                text = "Please inspect and respond to the attached files."
            context = AgentRunContext(
                tenant_id=tenant_id,
                actor_id=f"telegram:{user_id}",
                thread_id=thread_id,
                channel=AgentChannel.TELEGRAM,
                run_kind=AgentRunKind.OWNER,
                correlation_id=f"telegram_{uuid4().hex}",
                origin=OriginRef(
                    interface="telegram",
                    source_id="owner-bot",
                    conversation_id=str(chat_id),
                    message_id=str(message.get("message_id") or "") or None,
                ),
                agent_spec=(
                    self._resolve_agent_spec(tenant_id, AgentRunKind.OWNER.value)
                    if self._resolve_agent_spec is not None
                    else AgentSpecRef(
                        tenant_id=tenant_id,
                        spec_id="owner",
                        revision=1,
                    )
                ),
                trust_class="owner",
            )
            await self._deliver_events(
                chat_id=chat_id,
                tenant_id=tenant_id,
                events=self._agent.stream(
                    AgentRunRequest(
                        context=context,
                        text=text,
                        file_ids=tuple(file_ids),
                    )
                ),
            )

    async def _deliver_events(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        events: AsyncIterator[AgentRunEvent],
    ) -> None:
        chunks: list[str] = []
        failed = ""
        interrupted = False
        async for event in events:
            if event.type == "message.delta":
                chunks.append(str(event.data.get("text") or ""))
            elif event.type == "approval.required":
                interrupted = True
                await self._send_approval(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    run_id=event.run_id,
                    approval=event.data,
                )
            elif event.type == "run.failed":
                failed = str(event.data.get("message") or "Agent run failed.")
            elif event.type == "run.completed":
                final = str(event.data.get("text") or "").strip()
                if final:
                    chunks = [final]
        reply = "".join(chunks).strip()
        if reply:
            await self._client.send_message(chat_id=chat_id, text=reply, parse_mode=None)
            self._state.touch_assistant_message(chat_id)
        elif failed:
            await self._client.send_message(chat_id=chat_id, text=failed, parse_mode=None)
        elif not interrupted:
            await self._client.send_message(
                chat_id=chat_id,
                text="The run completed without a message.",
                parse_mode=None,
            )

    async def _send_approval(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        run_id: str,
        approval: dict[str, Any],
    ) -> None:
        approval_id = str(approval.get("approval_id") or "").strip()
        token = new_short_id("approval", suffix_chars=12)
        raw_allowed = approval.get("allowed_decisions")
        if isinstance(raw_allowed, list | tuple):
            allowed_decisions = tuple(
                str(item) for item in raw_allowed if str(item) in _APPROVAL_DECISIONS
            )
        else:
            allowed_decisions = ("approve", "edit", "reject")

        def persist(state: dict[str, Any]) -> None:
            pending = state.get("pending_approvals")
            if not isinstance(pending, dict):
                pending = {}
            pending[token] = {
                "tenant_id": tenant_id,
                "run_id": run_id,
                "approval_id": approval_id,
                "chat_id": chat_id,
                "allowed_decisions": list(allowed_decisions),
                "created_at": datetime.now(UTC).isoformat(),
            }
            state["pending_approvals"] = pending

        self._state.update(persist)
        tool_name = str(approval.get("tool_name") or "action")
        description = str(approval.get("description") or "Approval required")
        labels = {"approve": "Approve", "edit": "Edit", "reject": "Reject"}
        buttons = [
            {"text": labels[decision], "callback_data": f"ot:{token}:{decision}"}
            for decision in ("approve", "edit", "reject")
            if decision in allowed_decisions
        ]
        markup = {"inline_keyboard": [buttons]}
        await self._client.send_message(
            chat_id=chat_id,
            text=f"Approval required for {tool_name}: {description}",
            parse_mode=None,
            reply_markup=markup,
        )

    async def deliver_approval(
        self,
        *,
        chat_id: int,
        tenant_id: str,
        run_id: str,
        approval: AgentApproval,
    ) -> None:
        """Deliver a scheduled run interrupt through the normal approval path."""

        if approval.status != "pending":
            return
        await self._send_approval(
            chat_id=chat_id,
            tenant_id=tenant_id,
            run_id=run_id,
            approval={
                "approval_id": approval.id,
                "tool_name": approval.tool_name,
                "description": approval.description,
                "allowed_decisions": list(approval.allowed_decisions),
            },
        )

    async def _handle_callback(self, callback: dict[str, Any]) -> None:
        callback_id = str(callback.get("id") or "").strip()
        data = str(callback.get("data") or "").strip()
        sender = callback.get("from")
        message = callback.get("message")
        if not isinstance(sender, dict) or not isinstance(message, dict):
            return
        chat = message.get("chat")
        if not isinstance(chat, dict):
            return
        chat_id = self._positive_int(chat.get("id"))
        user_id = self._positive_int(sender.get("id"))
        username = str(sender.get("username") or "").strip().removeprefix("@") or None
        if chat_id is None or user_id is None:
            return
        if not is_user_allowed(
            user_id=user_id,
            username=username,
            allowed_user_ids_csv=self._allowed_user_ids,
            allowed_usernames_csv=self._allowed_usernames,
        ):
            await self._client.answer_callback_query(
                callback_query_id=callback_id,
                text="Unauthorized",
                show_alert=True,
            )
            return
        prefix, token, decision = (data.split(":", 2) + ["", "", ""])[:3]
        if prefix != "ot" or decision not in _APPROVAL_DECISIONS:
            await self._client.answer_callback_query(
                callback_query_id=callback_id,
                text="This control is no longer active.",
            )
            return
        expected_tenant = self._resolve_tenant(user_id)
        lock = self._approval_locks.setdefault(token, asyncio.Lock())
        async with lock:
            record = self._get_approval(token)
            if not self._approval_matches(
                record,
                chat_id=chat_id,
                tenant_id=expected_tenant,
            ):
                await self._client.answer_callback_query(
                    callback_query_id=callback_id,
                    text="Approval not found.",
                    show_alert=True,
                )
                return
            assert record is not None
            accepted_decision = self._finish_accepted_approval(token=token, record=record)
            if accepted_decision is not None:
                callback_text = {
                    "approve": "Already approved",
                    "edit": "Edited arguments already accepted",
                    "reject": "Already rejected",
                }[accepted_decision]
                await self._client.answer_callback_query(
                    callback_query_id=callback_id,
                    text=callback_text,
                )
                return
            allowed = {
                str(item)
                for item in record.get("allowed_decisions", [])
                if str(item) in _APPROVAL_DECISIONS
            }
            if decision not in allowed:
                await self._client.answer_callback_query(
                    callback_query_id=callback_id,
                    text="That decision is not allowed for this approval.",
                    show_alert=True,
                )
                return
            if decision == "edit":
                if not self._begin_approval_edit(
                    token=token,
                    record=record,
                    user_id=user_id,
                ):
                    await self._client.answer_callback_query(
                        callback_query_id=callback_id,
                        text="Approval changed. Open the latest approval and try again.",
                        show_alert=True,
                    )
                    return
                await self._client.answer_callback_query(
                    callback_query_id=callback_id,
                    text="Send the replacement arguments as JSON.",
                )
                await self._client.send_message(
                    chat_id=chat_id,
                    text=(
                        "Reply with one JSON object containing the complete replacement "
                        "tool arguments. OpenTulpa will not echo the arguments. "
                        "Use /cancel to stop editing."
                    ),
                    parse_mode=None,
                )
                return
            await self._resume_approval(
                token=token,
                record=record,
                chat_id=chat_id,
                tenant_id=expected_tenant,
                callback_id=callback_id,
                decision=cast(Literal["approve", "reject"], decision),
            )

    async def _resume_approval(
        self,
        *,
        token: str,
        record: dict[str, Any],
        chat_id: int,
        tenant_id: str,
        callback_id: str | None,
        decision: Literal["approve", "edit", "reject"],
        edited_arguments: dict[str, Any] | None = None,
    ) -> bool:
        events = self._agent.resume(
            str(record["run_id"]),
            ApprovalDecision(
                approval_id=str(record["approval_id"]),
                decision=decision,
                edited_arguments=edited_arguments,
            ),
        )
        iterator = events.__aiter__()
        try:
            first_event = await anext(iterator)
        except StopAsyncIteration:
            await self._report_resume_failure(chat_id=chat_id, callback_id=callback_id)
            return False
        except Exception:
            logger.exception(
                "Telegram approval resume failed before acceptance: run_id=%s",
                record.get("run_id"),
            )
            await self._report_resume_failure(chat_id=chat_id, callback_id=callback_id)
            return False

        self._accepted_approvals[token] = (dict(record), decision)
        try:
            consumed = self._consume_approval(token, expected=record)
        except Exception:
            consumed = False
            logger.exception(
                "Telegram approval was accepted but its callback handle could not be removed: "
                "run_id=%s",
                record.get("run_id"),
            )
        if not consumed:
            logger.warning(
                "Telegram approval callback handle changed after resume acceptance: run_id=%s",
                record.get("run_id"),
            )
        else:
            self._accepted_approvals.pop(token, None)

        if callback_id:
            callback_text = {
                "approve": "Approved",
                "edit": "Edited arguments accepted",
                "reject": "Rejected",
            }[decision]
            await self._client.answer_callback_query(
                callback_query_id=callback_id,
                text=callback_text,
            )
        elif decision == "edit":
            await self._client.send_message(
                chat_id=chat_id,
                text="Edited arguments accepted.",
                parse_mode=None,
            )

        try:
            await self._deliver_events(
                chat_id=chat_id,
                tenant_id=tenant_id,
                events=_with_first_event(first_event, iterator),
            )
        except Exception:
            logger.exception(
                "Telegram could not deliver an accepted resumed run: run_id=%s",
                record.get("run_id"),
            )
            await self._client.send_message(
                chat_id=chat_id,
                text="The approval was accepted, but the resumed run could not be delivered.",
                parse_mode=None,
            )
        return True

    async def _report_resume_failure(self, *, chat_id: int, callback_id: str | None) -> None:
        if callback_id:
            await self._client.answer_callback_query(
                callback_query_id=callback_id,
                text="Could not resume. Try again.",
                show_alert=True,
            )
            return
        await self._client.send_message(
            chat_id=chat_id,
            text="The edit was not applied. Send the JSON again to retry.",
            parse_mode=None,
        )

    def _get_approval(self, token: str) -> dict[str, Any] | None:
        pending = self._state.load().get("pending_approvals")
        if not isinstance(pending, dict):
            return None
        record = pending.get(token)
        return dict(record) if isinstance(record, dict) else None

    @staticmethod
    def _approval_matches(
        record: dict[str, Any] | None,
        *,
        chat_id: int,
        tenant_id: str,
    ) -> bool:
        return bool(
            record is not None
            and int(record.get("chat_id") or 0) == chat_id
            and str(record.get("tenant_id") or "") == tenant_id
        )

    def _consume_approval(self, token: str, *, expected: dict[str, Any]) -> bool:
        def consume(state: dict[str, Any]) -> bool:
            pending = state.get("pending_approvals")
            if not isinstance(pending, dict):
                return False
            current = pending.get(token)
            if not isinstance(current, dict) or current != expected:
                return False
            pending.pop(token, None)
            state["pending_approvals"] = pending
            edits = state.get("pending_approval_edits")
            if isinstance(edits, dict):
                state["pending_approval_edits"] = {
                    key: value
                    for key, value in edits.items()
                    if not isinstance(value, dict) or str(value.get("token") or "") != token
                }
            return True

        return bool(self._state.update(consume))

    def _finish_accepted_approval(
        self,
        *,
        token: str,
        record: dict[str, Any],
    ) -> Literal["approve", "edit", "reject"] | None:
        accepted = self._accepted_approvals.get(token)
        if accepted is None or accepted[0] != record:
            return None
        try:
            consumed = self._consume_approval(token, expected=record)
        except Exception:
            consumed = False
            logger.exception(
                "Telegram could not finish removing an accepted approval handle: run_id=%s",
                record.get("run_id"),
            )
        if consumed:
            self._accepted_approvals.pop(token, None)
        return accepted[1]

    def _begin_approval_edit(
        self,
        *,
        token: str,
        record: dict[str, Any],
        user_id: int,
    ) -> bool:
        chat_id = int(record["chat_id"])

        def begin(state: dict[str, Any]) -> bool:
            pending = state.get("pending_approvals")
            current = pending.get(token) if isinstance(pending, dict) else None
            if not isinstance(current, dict) or current != record:
                return False
            edits = state.get("pending_approval_edits")
            if not isinstance(edits, dict):
                edits = {}
            edits[f"{chat_id}:{user_id}"] = {
                "token": token,
                "tenant_id": str(record["tenant_id"]),
                "chat_id": chat_id,
                "user_id": user_id,
                "created_at": datetime.now(UTC).isoformat(),
            }
            state["pending_approval_edits"] = edits
            return True

        return bool(self._state.update(begin))

    def _pending_approval_edit(self, *, chat_id: int, user_id: int) -> dict[str, Any] | None:
        edits = self._state.load().get("pending_approval_edits")
        marker = edits.get(f"{chat_id}:{user_id}") if isinstance(edits, dict) else None
        return dict(marker) if isinstance(marker, dict) else None

    def _clear_pending_approval_edit(self, *, chat_id: int, user_id: int, token: str) -> bool:
        key = f"{chat_id}:{user_id}"

        def clear(state: dict[str, Any]) -> bool:
            edits = state.get("pending_approval_edits")
            if not isinstance(edits, dict):
                return False
            marker = edits.get(key)
            if not isinstance(marker, dict) or str(marker.get("token") or "") != token:
                return False
            edits.pop(key, None)
            state["pending_approval_edits"] = edits
            return True

        return bool(self._state.update(clear))

    async def _handle_pending_approval_edit(
        self,
        *,
        chat_id: int,
        user_id: int,
        tenant_id: str,
        text: str,
    ) -> bool:
        marker = self._pending_approval_edit(chat_id=chat_id, user_id=user_id)
        if not self._approval_matches(marker, chat_id=chat_id, tenant_id=tenant_id):
            return False
        assert marker is not None
        token = str(marker.get("token") or "")
        record = self._get_approval(token)
        accepted_decision = (
            self._finish_accepted_approval(token=token, record=record)
            if record is not None
            else None
        )
        if accepted_decision is not None:
            await self._client.send_message(
                chat_id=chat_id,
                text="This approval was already accepted.",
                parse_mode=None,
            )
            return True
        if text.split(None, 1)[0:1] == ["/cancel"]:
            self._clear_pending_approval_edit(
                chat_id=chat_id,
                user_id=user_id,
                token=token,
            )
            await self._client.send_message(
                chat_id=chat_id,
                text="Stopped editing. The approval is still pending.",
                parse_mode=None,
            )
            return True
        try:
            edited_arguments = self._parse_edited_arguments(text)
        except ValueError:
            await self._client.send_message(
                chat_id=chat_id,
                text=(
                    "Send one valid JSON object with the complete replacement arguments, "
                    "or use /cancel."
                ),
                parse_mode=None,
            )
            return True

        lock = self._approval_locks.setdefault(token, asyncio.Lock())
        async with lock:
            record = self._get_approval(token)
            if not self._approval_matches(record, chat_id=chat_id, tenant_id=tenant_id):
                self._clear_pending_approval_edit(
                    chat_id=chat_id,
                    user_id=user_id,
                    token=token,
                )
                await self._client.send_message(
                    chat_id=chat_id,
                    text="This approval is no longer pending.",
                    parse_mode=None,
                )
                return True
            assert record is not None
            if "edit" not in record.get("allowed_decisions", []):
                self._clear_pending_approval_edit(
                    chat_id=chat_id,
                    user_id=user_id,
                    token=token,
                )
                await self._client.send_message(
                    chat_id=chat_id,
                    text="Editing is not allowed for this approval.",
                    parse_mode=None,
                )
                return True
            await self._resume_approval(
                token=token,
                record=record,
                chat_id=chat_id,
                tenant_id=tenant_id,
                callback_id=None,
                decision="edit",
                edited_arguments=edited_arguments,
            )
            return True

    @staticmethod
    def _parse_edited_arguments(text: str) -> dict[str, Any]:
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
        if {str(key).lower() for key in value} & _HIDDEN_EDIT_ARGUMENTS:
            raise ValueError("runtime context cannot be edited")

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

    def _resolve_tenant(self, user_id: int) -> str:
        if self._owner_tenant_id:
            return self._profiles.resolve_customer_id(self._owner_tenant_id)
        return self._profiles.resolve_telegram_customer_id(user_id)

    def _session(
        self,
        *,
        chat_id: int,
        user_id: int,
        username: str | None,
        tenant_id: str,
    ) -> str:
        existing = self._state.get_session_slot(chat_id)
        if existing and str(existing.get("customer_id") or "") == tenant_id:
            thread_id = str(existing.get("thread_id") or "").strip()
            if thread_id:
                self._touch_session(chat_id, user_id, username, tenant_id, thread_id)
                return thread_id
        return self._reset_session(
            chat_id=chat_id,
            user_id=user_id,
            username=username,
            tenant_id=tenant_id,
        )

    def _reset_session(
        self,
        *,
        chat_id: int,
        user_id: int,
        username: str | None,
        tenant_id: str,
    ) -> str:
        thread_id = new_short_id("chat", suffix_chars=12)
        self._touch_session(chat_id, user_id, username, tenant_id, thread_id)
        return thread_id

    def _touch_session(
        self,
        chat_id: int,
        user_id: int,
        username: str | None,
        tenant_id: str,
        thread_id: str,
    ) -> None:
        def update(state: dict[str, Any]) -> None:
            sessions = state.get("sessions")
            if not isinstance(sessions, dict):
                sessions = {}
            old = sessions.get(str(chat_id))
            slot = dict(old) if isinstance(old, dict) else {}
            slot.update(
                {
                    "user_id": user_id,
                    "username": username or "",
                    "customer_id": tenant_id,
                    "thread_id": thread_id,
                    "role": "owner",
                    "last_user_message_at": datetime.now(UTC).isoformat(),
                }
            )
            sessions[str(chat_id)] = slot
            state["sessions"] = sessions

        self._state.update(update)

    async def _ingest_files(
        self,
        *,
        message: dict[str, Any],
        chat_id: int,
        tenant_id: str,
    ) -> list[str]:
        ids: list[str] = []
        for attachment in extract_attachments(message):
            downloaded = await self._client.download_file(file_id=attachment.file_id)
            raw = downloaded.get("raw_bytes") if isinstance(downloaded, dict) else None
            if not isinstance(raw, bytes | bytearray) or not raw:
                continue
            record = self._files.ingest_file(
                customer_id=tenant_id,
                chat_id=chat_id,
                kind=attachment.kind,
                telegram_file_id=attachment.file_id,
                original_filename=attachment.filename or f"{attachment.kind}.bin",
                mime_type=attachment.mime_type,
                caption=str(message.get("caption") or "").strip() or None,
                raw_bytes=bytes(raw),
            )
            file_id = str(record.get("id") or "").strip()
            if file_id:
                ids.append(file_id)
        return ids

    async def _enqueue_business_intake(self, result: dict[str, Any]) -> None:
        if self._intake_workflows is None:
            raise RuntimeError("intake workflow service is unavailable")
        tenant_id = str(result.get("customer_id") or "").strip()
        connection_id = str(result.get("business_connection_id") or "").strip()
        conversation_id = str(result.get("chat_id") or "").strip()
        owner_chat_id = str(result.get("user_chat_id") or "").strip()
        if not tenant_id or not connection_id or not conversation_id:
            raise RuntimeError("Telegram Business intake identity is incomplete")
        workflows = self._intake_workflows.list_workflows(
            customer_id=tenant_id,
            include_disabled=False,
        )
        for workflow in workflows:
            if str(workflow.get("channel") or "") != "telegram_business_dm":
                continue
            if str(workflow.get("provider") or "") != "telegram_bot_api":
                continue
            if not self._intake_workflows._source_matches_workflow(  # noqa: SLF001
                workflow=workflow,
                business_connection_id=connection_id,
                conversation_id=conversation_id,
            ):
                continue
            outcome = await self._intake_workflows.enqueue_telegram_business_workflow_run(
                customer_id=tenant_id,
                workflow_id=str(workflow.get("workflow_id") or ""),
                conversation_id=conversation_id,
                owner_chat_id=owner_chat_id,
                event_type="telegram_business_webhook",
            )
            if not bool(outcome.get("ok")):
                logger.error(
                    "failed to queue Telegram Business intake workflow: %r",
                    outcome,
                    extra={
                        "tenant_id": tenant_id,
                        "workflow_id": str(workflow.get("workflow_id") or ""),
                        "conversation_id": conversation_id,
                    },
                )
                raise RuntimeError("Telegram Business intake could not be queued")

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None


__all__ = ["AcceptedTelegramUpdate", "DeepAgentTelegramRelay", "TelegramAgentService"]
