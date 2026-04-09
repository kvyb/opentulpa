"""Telegram reply streaming and wake-event relays."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import zlib
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from opentulpa.agent.runtime import (
    STREAM_APPROVAL_HANDOFF_SIGNAL,
    STREAM_PROGRESS_PREFIX,
    STREAM_WAIT_SIGNAL,
    MergedInputSuppressedError,
)
from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.constants import DEBUG_LOG_PATH, LOW_SIGNAL_REPLIES

logger = logging.getLogger(__name__)
NO_NOTIFY_TOKEN = "__NO_NOTIFY__"
STREAM_EDIT_MIN_INTERVAL_SECONDS = 0.2
STREAM_EDIT_MIN_CHAR_DELTA = 80
STREAM_INITIAL_VISIBLE_MIN_CHARS = 48
STREAM_INITIAL_VISIBLE_MAX_WAIT_SECONDS = 0.45
STREAM_FOLLOWUP_VISIBLE_MIN_CHARS = 24


def _clean_thread_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def normalize_reply_text(text: str) -> str:
    import re

    t = text.strip().lower()
    t = re.sub(r"[.!?]+$", "", t)
    return " ".join(t.split())


def is_low_signal_reply(text: str) -> bool:
    normalized = normalize_reply_text(text)
    if not normalized:
        return True
    return normalized in LOW_SIGNAL_REPLIES


def _is_progress_signal(partial: str) -> bool:
    if partial == STREAM_WAIT_SIGNAL:
        return True
    return partial.startswith(STREAM_PROGRESS_PREFIX)


def debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "runId": "telegram",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def _emit_typing_until_done(
    *,
    client: TelegramClient,
    chat_id: int,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        with suppress(Exception):
            await client.send_chat_action(chat_id=chat_id, action="typing")
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)


async def stream_langgraph_reply_to_telegram(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
) -> tuple[str | None, bool]:
    last_streamed = ""
    final_reply = None
    delivered_any = False
    live_delivery_text = ""
    live_delivery_at = 0.0
    client = TelegramClient(bot_token)
    draft_id = zlib.crc32(f"{thread_id}:{customer_id}:{chat_id}:{new_short_id('dft')}".encode("utf-8")) or 1
    draft_enabled = chat_id > 0
    waiting_for_segment = True
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        _emit_typing_until_done(client=client, chat_id=chat_id, stop_event=typing_stop)
    )
    suppressed = False
    first_token_timeout_s = 90.0
    first_token_retry_timeout_s = 180.0
    stream_idle_timeout_s = 180.0
    stream_idle_retry_timeout_s = 240.0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 2
    stream_started_at = time.monotonic()
    next_chunk_task: asyncio.Task[Any] | None = None
    logger.info(
        "telegram.stream start chat_id=%s thread_id=%s customer_id=%s text_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        len(str(text or "")),
    )

    async def _recover_after_stream_timeout() -> str | None:
        if not hasattr(agent_runtime, "ainvoke_text"):
            return None
        try:
            recovered = await asyncio.wait_for(
                agent_runtime.ainvoke_text(
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=text,
                    turn_mode="interactive",
                ),
                timeout=90.0,
            )
        except Exception:
            return None
        safe = str(recovered or "").strip()
        if not safe or is_low_signal_reply(safe):
            return None
        return safe

    async def _send_draft_reply(text: str, *, force: bool = False) -> None:
        nonlocal delivered_any, draft_enabled, final_reply, live_delivery_text, live_delivery_at
        current = str(text or "").strip()
        if not current:
            return
        now = time.monotonic()
        should_send = force or not live_delivery_text
        sentence_like = current.endswith((".", "!", "?", "\n"))
        if not live_delivery_text and not force:
            elapsed = now - stream_started_at
            if len(current) < STREAM_INITIAL_VISIBLE_MIN_CHARS and not sentence_like and elapsed < STREAM_INITIAL_VISIBLE_MAX_WAIT_SECONDS:
                final_reply = current
                return
        if not should_send:
            if current == live_delivery_text:
                final_reply = current
                return
            grew_by = max(0, len(current) - len(live_delivery_text))
            elapsed = now - live_delivery_at
            should_send = (
                (elapsed >= STREAM_EDIT_MIN_INTERVAL_SECONDS and grew_by >= STREAM_FOLLOWUP_VISIBLE_MIN_CHARS)
                or grew_by >= STREAM_EDIT_MIN_CHAR_DELTA
                or sentence_like
            )
        final_reply = current
        if not should_send:
            return
        if not draft_enabled:
            return
        if not await client.send_message_draft(
            chat_id=chat_id,
            draft_id=draft_id,
            text=current,
            parse_mode="HTML",
        ):
            draft_enabled = False
            logger.warning(
                "telegram.stream draft_disabled chat_id=%s thread_id=%s customer_id=%s",
                chat_id,
                thread_id,
                customer_id,
            )
            return
        delivered_any = True
        live_delivery_text = current
        live_delivery_at = now

    try:
        stream = agent_runtime.astream_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
            turn_mode="interactive",
        )
        stream_iter = stream.__aiter__()
        while True:
            if not last_streamed:
                timeout_s = (
                    first_token_timeout_s
                    if consecutive_timeouts == 0
                    else first_token_retry_timeout_s
                )
            else:
                timeout_s = (
                    stream_idle_timeout_s
                    if consecutive_timeouts == 0
                    else stream_idle_retry_timeout_s
                )
            try:
                if next_chunk_task is None:
                    next_chunk_task = asyncio.create_task(stream_iter.__anext__())
                partial = await asyncio.wait_for(
                    asyncio.shield(next_chunk_task),
                    timeout=timeout_s,
                )
                next_chunk_task = None
            except StopAsyncIteration:
                next_chunk_task = None
                break
            except TimeoutError:
                consecutive_timeouts += 1
                if consecutive_timeouts < max_consecutive_timeouts:
                    logger.warning(
                        "telegram.stream timeout_retry chat_id=%s thread_id=%s customer_id=%s stage=%s",
                        chat_id,
                        thread_id,
                        customer_id,
                        "first_token" if not last_streamed else "idle",
                    )
                    continue
                if next_chunk_task is not None and not next_chunk_task.done():
                    next_chunk_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        async with asyncio.timeout(1.0):
                            await next_chunk_task
                next_chunk_task = None
                with suppress(Exception):
                    async with asyncio.timeout(1.0):
                        await stream.aclose()
                recovered_text = await _recover_after_stream_timeout()
                if recovered_text:
                    logger.warning(
                        "telegram.stream timeout_recovered chat_id=%s thread_id=%s customer_id=%s stage=%s",
                        chat_id,
                        thread_id,
                        customer_id,
                        "first_token" if not last_streamed else "idle",
                    )
                    final_reply = recovered_text
                    break
                timeout_text = (
                    "Still working, but the model response timed out. "
                    "Please retry in a moment."
                )
                logger.error(
                    "telegram.stream timeout_fail chat_id=%s thread_id=%s customer_id=%s stage=%s",
                    chat_id,
                    thread_id,
                    customer_id,
                    "first_token" if not last_streamed else "idle",
                )
                final_reply = timeout_text
                break
            progress_text = partial if isinstance(partial, str) else ""
            if progress_text and _is_progress_signal(progress_text):
                if not waiting_for_segment:
                    waiting_for_segment = True
                    last_streamed = ""
                continue
            if isinstance(partial, str) and partial == STREAM_APPROVAL_HANDOFF_SIGNAL:
                # Approval UI delivery is handled out-of-band by approval adapters.
                draft_enabled = False
                suppressed = True
                final_reply = None
                break
            if not isinstance(partial, str):
                continue
            consecutive_timeouts = 0
            current = partial.strip()
            if not current or is_low_signal_reply(current) or current == last_streamed:
                continue
            # Defensive boundary handling for streams that reset partial text without explicit signal.
            if last_streamed and not current.startswith(last_streamed):
                waiting_for_segment = True
                last_streamed = ""
            if waiting_for_segment:
                waiting_for_segment = False
            last_streamed = current
            await _send_draft_reply(current)
    except MergedInputSuppressedError:
        logger.info(
            "telegram.stream suppressed_by_merge chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        suppressed = True
    except Exception:
        if next_chunk_task is not None and not next_chunk_task.done():
            next_chunk_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                async with asyncio.timeout(1.0):
                    await next_chunk_task
        if not typing_stop.is_set():
            typing_stop.set()
        with suppress(Exception):
            await typing_task
        raise
    if next_chunk_task is not None and not next_chunk_task.done():
        next_chunk_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            async with asyncio.timeout(1.0):
                await next_chunk_task
    if not typing_stop.is_set():
        typing_stop.set()
    with suppress(Exception):
        await typing_task
    if not suppressed and not final_reply:
        logger.error(
            "telegram.stream no_final_reply chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        final_reply = (
            "I couldn't produce a visible user-facing reply for that step "
            "(the model/tool loop ended without displayable output)."
        )
    if not suppressed and final_reply:
        sent = await client.send_message(
            chat_id=chat_id,
            text=final_reply,
            parse_mode="HTML",
        )
        if sent:
            delivered_any = True
        elif not delivered_any:
            final_reply = None
    logger.info(
        "telegram.stream complete chat_id=%s thread_id=%s customer_id=%s suppressed=%s final_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        suppressed,
        len(str(final_reply or "")),
    )
    return final_reply, suppressed


async def relay_task_event_via_main_agent(
    *,
    customer_id: str,
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
    state_store: Any,
    find_session_slots: Callable[[str], list[dict[str, Any]]],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    return await relay_event_via_main_agent(
        customer_id=customer_id,
        event_label=f"task/{event_type}",
        payload={
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload,
        },
        state_store=state_store,
        find_session_slots=find_session_slots,
        agent_runtime=agent_runtime,
    )


async def relay_event_via_main_agent(
    *,
    customer_id: str,
    event_label: str,
    payload: dict[str, Any],
    state_store: Any,
    find_session_slots: Callable[[str], list[dict[str, Any]]],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    slots = find_session_slots(customer_id)
    if not slots:
        return []
    if agent_runtime is None:
        raise RuntimeError("Agent runtime unavailable for wake relay")
    routine_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    routine_instruction = str(routine_payload.get("instruction", "")).strip()
    routine_name = str(payload.get("routine_name", "")).strip()
    proactive_heartbeat = bool(routine_payload.get("proactive_heartbeat", False))
    now_utc = datetime.now(UTC)
    replies: list[dict[str, Any]] = []
    for slot in slots:
        chat_id = int(slot["chat_id"])
        chat_key = str(chat_id)
        last_user_at = str(slot.get("last_user_message_at", "")).strip()
        last_assistant_at = str(slot.get("last_assistant_message_at", "")).strip()
        user_idle_hours = "unknown"
        assistant_idle_hours = "unknown"
        if last_user_at:
            with suppress(Exception):
                parsed = datetime.fromisoformat(last_user_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                user_idle_hours = f"{max(0.0, (now_utc - parsed).total_seconds() / 3600.0):.2f}"
        if last_assistant_at:
            with suppress(Exception):
                parsed = datetime.fromisoformat(last_assistant_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                assistant_idle_hours = f"{max(0.0, (now_utc - parsed).total_seconds() / 3600.0):.2f}"

        if (
            str(event_label).startswith("routine/")
            and proactive_heartbeat
            and hasattr(agent_runtime, "classify_wake_event")
        ):
            precheck_payload = {
                "event_label": event_label,
                "routine_name": routine_name,
                "routine_payload": routine_payload,
                "last_user_message_at_utc": last_user_at or "unknown",
                "last_assistant_message_at_utc": last_assistant_at or "unknown",
                "user_idle_hours": user_idle_hours,
                "assistant_idle_hours": assistant_idle_hours,
            }
            decision = {"notify_user": True}
            with suppress(Exception):
                decision = await agent_runtime.classify_wake_event(
                    customer_id=customer_id,
                    event_label="routine/heartbeat_precheck",
                    payload=precheck_payload,
                )
            if not bool(decision.get("notify_user", False)):
                continue

        if str(event_label).startswith("routine/"):
            instruction = (
                "System update: a scheduled routine woke you.\n"
                "Decide if the user should be messaged right now.\n"
                f"- event: {event_label}\n"
                f"- routine_name: {routine_name or 'unnamed'}\n"
                f"- routine_instruction: {routine_instruction[:3000] or '(none)'}\n"
                f"- last_user_message_at_utc: {last_user_at or 'unknown'}\n"
                f"- user_idle_hours: {user_idle_hours}\n"
                f"- last_assistant_message_at_utc: {last_assistant_at or 'unknown'}\n"
                f"- assistant_idle_hours: {assistant_idle_hours}\n"
                f"- now_utc: {now_utc.isoformat()}\n"
                f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}\n\n"
                f"If you decide to skip messaging this run, reply exactly: {NO_NOTIFY_TOKEN}\n"
                "If you decide to message, send one concise, natural message (no rigid status template)."
            )
        else:
            instruction = (
                "System update: a background event occurred.\n"
                "Respond with concise plain-language status, what happened, and next action.\n"
                f"- event: {event_label}\n"
                f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}"
            )

        def _ensure_wake_thread_id(state: dict[str, Any], _chat_key: str = chat_key) -> str:
            sessions = state.get("sessions")
            if not isinstance(sessions, dict):
                sessions = {}
            raw_slot = sessions.get(_chat_key)
            if not isinstance(raw_slot, dict):
                raw_slot = {}
            wake_thread_id = _clean_thread_id(raw_slot.get("wake_thread_id"))
            if not wake_thread_id or not wake_thread_id.lower().startswith("wake_"):
                wake_thread_id = new_short_id("wake")
                raw_slot["wake_thread_id"] = wake_thread_id
                sessions[_chat_key] = raw_slot
                state["sessions"] = sessions
            return wake_thread_id

        wake_thread_id = state_store.update(_ensure_wake_thread_id)
        try:
            text = await agent_runtime.ainvoke_text(
                thread_id=wake_thread_id,
                customer_id=customer_id,
                text=instruction,
                turn_mode="event_notification",
                include_pending_context=False,
                recursion_limit_override=36 if proactive_heartbeat else None,
            )
            safe = str(text or "").strip()
            if not safe:
                continue
            if safe == NO_NOTIFY_TOKEN:
                replies.append({"chat_id": chat_id, "text": NO_NOTIFY_TOKEN})
                continue
            replies.append({"chat_id": chat_id, "text": safe})
        except Exception:
            continue
    return replies
