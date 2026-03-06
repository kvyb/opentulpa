"""Telegram webhook route registration."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from opentulpa.application.approval_execution import ApprovalExecutionOrchestrator
from opentulpa.interfaces.telegram.client import (
    parse_telegram_callback_query,
    parse_telegram_update,
)

logger = logging.getLogger(__name__)


def _parse_approval_callback_data(data: str) -> tuple[str, str] | None:
    raw = str(data or "").strip()
    # Format: approval:<approval_id>:approve|deny
    match = re.fullmatch(
        r"approval:([a-z0-9_-]{6,40}):(approve|deny)",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1), match.group(2).lower()


async def _execute_approved_action_and_summarize(
    *,
    get_agent_runtime: Callable[[], Any],
    get_context_events: Callable[[], Any],
    approval_id: str,
    decision_payload: dict[str, Any],
    chat_id: int,
) -> str:
    """Backward-compatible helper kept for tests/internal call sites."""
    orchestrator = ApprovalExecutionOrchestrator(
        get_agent_runtime=get_agent_runtime,
        get_context_events=get_context_events,
    )
    return await orchestrator.execute_approved_action_and_summarize(
        approval_id=approval_id,
        decision_payload=decision_payload,
        chat_id=chat_id,
    )


async def _emit_typing_until_done(
    *,
    client: Any,
    chat_id: int,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        with suppress(Exception):
            await client.send_chat_action(chat_id=chat_id, action="typing")
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)


async def _run_post_approval_execution_flow(
    *,
    get_telegram_client: Callable[[], Any],
    get_approvals: Callable[[], Any] | None,
    get_approval_execution_orchestrator: Callable[[], Any],
    approval_ids: list[str],
    decision_payload: dict[str, Any],
    chat_id: int,
    approval_message_id: int | None = None,
) -> None:
    safe_ids = [str(item).strip() for item in approval_ids if str(item).strip()]
    if not safe_ids:
        return
    client = get_telegram_client()
    with suppress(Exception):
        if isinstance(approval_message_id, int):
            await client.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=approval_message_id,
                reply_markup={"inline_keyboard": []},
            )

    loader_stop = asyncio.Event()
    loader_task = asyncio.create_task(
        _emit_typing_until_done(
            client=client,
            chat_id=chat_id,
            stop_event=loader_stop,
        )
    )
    final_outcome = "I couldn't execute the approved action due to an internal error."
    try:
        final_outcome = await get_approval_execution_orchestrator().execute_group_and_merge(
            approval_ids=safe_ids,
            decision_payload=decision_payload,
            chat_id=chat_id,
        )
    finally:
        if not loader_stop.is_set():
            loader_stop.set()
        with suppress(Exception):
            await loader_task

    with suppress(Exception):
        await client.send_message(chat_id=chat_id, text=final_outcome, parse_mode="HTML")

    if get_approvals is not None:
        with suppress(Exception):
            await get_approvals().flush_deferred_challenges(
                origin_interface="telegram",
                origin_conversation_id=str(chat_id),
            )


async def _run_post_denial_iteration_flow(
    *,
    get_telegram_client: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    get_approvals: Callable[[], Any] | None,
    decision_payload: dict[str, Any],
    chat_id: int,
) -> None:
    runtime = get_agent_runtime()
    customer_id = str(decision_payload.get("customer_id", "")).strip()
    thread_id = str(decision_payload.get("thread_id", "")).strip() or f"chat-{chat_id}"
    approval_id = str(
        decision_payload.get("id", decision_payload.get("approval_id", ""))
    ).strip()
    action_name = str(decision_payload.get("action_name", "")).strip()
    summary = str(decision_payload.get("summary", "")).strip()
    action_args = (
        decision_payload.get("action_args")
        if isinstance(decision_payload.get("action_args"), dict)
        else {}
    )
    fallback_text = (
        "Understood. You denied this action. Tell me what to change and I will revise "
        "the plan and resubmit for approval."
    )
    if runtime is None or not hasattr(runtime, "ainvoke_text") or not customer_id:
        with suppress(Exception):
            await get_telegram_client().send_message(
                chat_id=chat_id,
                text=fallback_text,
                parse_mode="HTML",
            )
        if get_approvals is not None:
            with suppress(Exception):
                await get_approvals().flush_deferred_challenges(
                    origin_interface="telegram",
                    origin_conversation_id=str(chat_id),
                )
        return
    prompt = (
        "System update: The user denied your planned action.\n"
        "Do not execute external actions now.\n"
        f"- approval_id: {approval_id or 'unknown'}\n"
        f"- action_name: {action_name or 'unknown'}\n"
        f"- summary: {summary or 'none'}\n"
        f"- action_args: {json.dumps(action_args, ensure_ascii=False)[:3000]}\n\n"
        "Respond to the user now in plain language.\n"
        "1) Acknowledge denial.\n"
        "2) Ask what should change.\n"
        "3) Offer to resubmit a revised action for approval."
    )
    try:
        reply = await runtime.ainvoke_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=prompt,
            include_pending_context=False,
            recursion_limit_override=36,
        )
    except Exception:
        reply = ""
    safe_reply = str(reply or "").strip() or fallback_text
    with suppress(Exception):
        await get_telegram_client().send_message(
            chat_id=chat_id,
            text=safe_reply,
            parse_mode="HTML",
        )
    if get_approvals is not None:
        with suppress(Exception):
            await get_approvals().flush_deferred_challenges(
                origin_interface="telegram",
                origin_conversation_id=str(chat_id),
            )


def register_telegram_webhook_routes(
    app: FastAPI,
    *,
    settings: Any,
    get_telegram_client: Callable[[], Any],
    get_telegram_chat: Callable[[], Any],
    get_approvals: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    get_approval_execution_orchestrator: Callable[[], Any],
    decide_approval_and_maybe_wake: Callable[..., Awaitable[dict[str, Any]]],
) -> None:
    """Register Telegram webhook with callback-driven approval support."""

    @app.post("/webhook/telegram")
    async def telegram_webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
        if not settings.telegram_bot_token:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        expected_secret = str(settings.telegram_webhook_secret or "").strip()
        if not expected_secret:
            return JSONResponse(
                status_code=503,
                content={"detail": "telegram webhook secret not configured"},
            )
        incoming_secret = str(request.headers.get("x-telegram-bot-api-secret-token", "") or "").strip()
        if not hmac.compare_digest(incoming_secret, expected_secret):
            return JSONResponse(status_code=403, content={"detail": "invalid telegram secret"})
        body = await request.json()

        # Immediate 200 OK, logic runs in background.
        background_tasks.add_task(_telegram_background_handler, body=body)
        return Response(status_code=200)

    async def _telegram_background_handler(body: dict[str, Any]) -> None:
        callback_id, callback_user_id, callback_chat_id, callback_data, callback_message_id = (
            parse_telegram_callback_query(body)
        )
        parsed_callback = _parse_approval_callback_data(callback_data or "")
        if parsed_callback and callback_id and callback_user_id and callback_chat_id:
            approval_id, requested_decision = parsed_callback
            result = await decide_approval_and_maybe_wake(
                approval_id=approval_id,
                decision=requested_decision,
                actor_interface="telegram",
                actor_id=str(callback_user_id),
                enqueue_wake=False,
            )
            approved = bool(result.get("ok")) and str(result.get("status", "")).strip() == "approved"
            denied = bool(result.get("ok")) and str(result.get("status", "")).strip() == "denied"
            if approved:
                ack_text = "Working on the task."
            elif denied:
                ack_text = "Action denied."
            else:
                ack_text = (
                    f"Approval {requested_decision} failed: {result.get('reason', 'not_allowed')}"
                )
            with suppress(Exception):
                await get_telegram_client().answer_callback_query(
                    callback_query_id=callback_id,
                    text=ack_text,
                    show_alert=False,
                )
            if approved:
                await _run_post_approval_execution_flow(
                    get_telegram_client=get_telegram_client,
                    get_approvals=get_approvals,
                    get_approval_execution_orchestrator=get_approval_execution_orchestrator,
                    approval_ids=[approval_id],
                    decision_payload=result if isinstance(result, dict) else {},
                    chat_id=callback_chat_id,
                    approval_message_id=callback_message_id if isinstance(callback_message_id, int) else None,
                )
            elif denied:
                if isinstance(callback_message_id, int):
                    with suppress(Exception):
                        await get_telegram_client().edit_message_text(
                            chat_id=callback_chat_id,
                            message_id=callback_message_id,
                            text=ack_text,
                            parse_mode="HTML",
                            reply_markup={"inline_keyboard": []},
                        )
                await _run_post_denial_iteration_flow(
                    get_telegram_client=get_telegram_client,
                    get_agent_runtime=get_agent_runtime,
                    get_approvals=get_approvals,
                    decision_payload=result if isinstance(result, dict) else {},
                    chat_id=callback_chat_id,
                )
            elif isinstance(callback_message_id, int):
                with suppress(Exception):
                    await get_telegram_client().edit_message_text(
                        chat_id=callback_chat_id,
                        message_id=callback_message_id,
                        text=ack_text,
                        parse_mode="HTML",
                        reply_markup={"inline_keyboard": []},
                    )
            return

        message = body.get("message") or body.get("edited_message") or {}
        chat_id = message.get("chat", {}).get("id")
        _ = parse_telegram_update(body)

        try:
            reply = await get_telegram_chat().handle_update(
                body=body,
                allowed_user_ids_csv=settings.telegram_allowed_user_ids,
                allowed_usernames_csv=settings.telegram_allowed_usernames,
                agent_runtime=get_agent_runtime(),
            )
        except Exception as exc:
            logger.exception("Unhandled Telegram background handler failure: %s", exc)
            if chat_id is not None:
                with suppress(Exception):
                    await get_telegram_client().send_message(
                        chat_id=chat_id,
                        text="I hit an internal error while processing your message. Please try again.",
                        parse_mode="HTML",
                    )
                with suppress(Exception):
                    await get_approvals().flush_deferred_challenges(
                        origin_interface="telegram",
                        origin_conversation_id=str(chat_id),
                    )
            return

        if reply and chat_id is not None:
            with suppress(Exception):
                await get_telegram_client().send_message(
                    chat_id=chat_id,
                    text=reply,
                    parse_mode="HTML",
                )
            with suppress(Exception):
                get_telegram_chat().touch_assistant_message(int(chat_id))

        if chat_id is not None:
            with suppress(Exception):
                await get_approvals().flush_deferred_challenges(
                    origin_interface="telegram",
                    origin_conversation_id=str(chat_id),
                )
